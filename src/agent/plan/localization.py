import random
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from agent.contracts import CANONICAL_COMPONENTS, normalize_component_confidence_map, normalize_components


EXPERT_LIGHTWEIGHT = "lm"
EXPERT_LLM = "llm"
EXPERT_RANDOM = "rand"

DEFAULT_EXPERT_PRIORS = {
    EXPERT_LIGHTWEIGHT: (8.0, 2.0),
    EXPERT_LLM: (2.0, 2.0),
    EXPERT_RANDOM: (2.0, 8.0),
}


@dataclass
class LocalizationDecision:
    expert: str
    root_causes: List[str]
    confidence: Dict[str, float]
    reused: bool = False


class LocalizationExpertSelector:
    """Prior-guided Thompson sampler for EvoTune tuning-space localization."""

    def __init__(self, agent_config=None, rng=None):
        self.agent_config = agent_config or {}
        self.rng = rng or random
        self.priors = self._load_priors(self.agent_config.get("localization_expert_priors"))
        self.window = max(
            1,
            int(
                self.agent_config.get(
                    "localization_feedback_window",
                    self.agent_config.get("no_improvement_threshold", 1),
                )
            ),
        )

    def ensure_state(self, state):
        localization = state.setdefault("localization", {})
        experts = localization.setdefault("experts", {})
        for expert, (alpha, beta) in self.priors.items():
            expert_state = experts.setdefault(expert, {})
            expert_state.setdefault("alpha", float(alpha))
            expert_state.setdefault("beta", float(beta))
        localization.setdefault("active_expert", None)
        localization.setdefault("active_root", None)
        localization.setdefault("active_confidence", {})
        localization.setdefault("pending_trials", 0)
        localization.setdefault("pending_outcomes", [])
        localization.setdefault("last_expert", None)
        localization.setdefault("last_update", None)
        return localization

    def active_decision(self, state):
        localization = self.ensure_state(state)
        active_expert = localization.get("active_expert")
        active_root = normalize_components(localization.get("active_root"))
        if not active_expert or not active_root:
            return None
        if int(localization.get("pending_trials", 0)) >= self.window:
            return None
        return LocalizationDecision(
            expert=active_expert,
            root_causes=active_root,
            confidence=normalize_component_confidence_map(localization.get("active_confidence")),
            reused=True,
        )

    def select_expert(self, state, allowed_experts: Optional[Sequence[str]] = None):
        localization = self.ensure_state(state)
        allowed = [expert for expert in (allowed_experts or self.priors.keys()) if expert in self.priors]
        if not allowed:
            raise ValueError("No localization expert is available")

        samples = {}
        for expert in allowed:
            expert_state = localization["experts"].setdefault(expert, {})
            alpha = float(expert_state.get("alpha", self.priors[expert][0]))
            beta = float(expert_state.get("beta", self.priors[expert][1]))
            samples[expert] = self.rng.betavariate(alpha, beta)
        return max(samples, key=samples.get), samples

    def start_decision(self, state, expert, root_causes, confidence=None):
        localization = self.ensure_state(state)
        root_causes = normalize_components(root_causes)
        confidence = normalize_component_confidence_map(confidence)
        localization["active_expert"] = expert
        localization["active_root"] = root_causes
        localization["active_confidence"] = confidence
        localization["pending_trials"] = 0
        localization["pending_outcomes"] = []
        localization["last_expert"] = expert
        state["current_root"] = root_causes
        state["confidence"] = confidence
        return LocalizationDecision(expert=expert, root_causes=root_causes, confidence=confidence)

    def record_feedback(self, state, success):
        localization = self.ensure_state(state)
        active_expert = localization.get("active_expert")
        if not active_expert:
            return None

        outcomes = list(localization.get("pending_outcomes", []))
        outcomes.append(bool(success))
        localization["pending_outcomes"] = outcomes
        localization["pending_trials"] = int(localization.get("pending_trials", 0)) + 1

        if localization["pending_trials"] < self.window:
            return None

        impact = 1 if any(outcomes) else 0
        expert_state = localization["experts"].setdefault(active_expert, {})
        expert_state["alpha"] = float(expert_state.get("alpha", self.priors[active_expert][0])) + impact
        expert_state["beta"] = float(expert_state.get("beta", self.priors[active_expert][1])) + (1 - impact)
        localization["last_update"] = {
            "expert": active_expert,
            "impact": impact,
            "outcomes": outcomes,
            "alpha": expert_state["alpha"],
            "beta": expert_state["beta"],
        }
        localization["active_expert"] = None
        localization["active_root"] = None
        localization["active_confidence"] = {}
        localization["pending_trials"] = 0
        localization["pending_outcomes"] = []
        return localization["last_update"]

    def reset_active(self, state):
        localization = self.ensure_state(state)
        localization["active_expert"] = None
        localization["active_root"] = None
        localization["active_confidence"] = {}
        localization["pending_trials"] = 0
        localization["pending_outcomes"] = []

    def _load_priors(self, configured_priors):
        priors = dict(DEFAULT_EXPERT_PRIORS)
        if not isinstance(configured_priors, dict):
            return priors
        for expert, pair in configured_priors.items():
            if expert not in priors:
                continue
            try:
                alpha, beta = pair
                priors[expert] = (float(alpha), float(beta))
            except (TypeError, ValueError):
                continue
        return priors


def build_tuning_history(state):
    history = []
    attempts = state.get("component_failures", {}) or {}
    successes = state.get("component_successes", {}) or {}
    for label in CANONICAL_COMPONENTS:
        history.append(
            {
                "label": label,
                "success_count": int(successes.get(label, 0)),
                "fail_count": int(attempts.get(label, 0)),
            }
        )
    return history


def random_component_subset(exclude_roots: Optional[Iterable] = None):
    excluded = {frozenset(normalize_components(item)) for item in (exclude_roots or [])}
    candidates: List[Tuple[str, ...]] = []
    components = list(CANONICAL_COMPONENTS)
    for width in range(1, len(components) + 1):
        for subset in combinations(components, width):
            if frozenset(subset) not in excluded:
                candidates.append(subset)
    if not candidates:
        candidates = [(component,) for component in components]
    return list(random.choice(candidates))
