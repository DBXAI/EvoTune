import random
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from agent.contracts import CANONICAL_COMPONENTS, normalize_component_confidence_map, normalize_components


EXPERT_LIGHTWEIGHT = "lm"
EXPERT_LLM = "llm"

# The paper uses two diagnostic experts: the lightweight predictor and the
# LLM detector.  The random expert remains available only for explicit
# exploration ablations; it is not part of the default hybrid policy.
DEFAULT_EXPERT_PRIORS = {
    EXPERT_LIGHTWEIGHT: {"intercept": 0.20},
    EXPERT_LLM: {"intercept": 0.0},
}


@dataclass
class LocalizationDecision:
    expert: str
    root_causes: List[str]
    confidence: Dict[str, float]
    reused: bool = False


class LocalizationExpertSelector:
    """Contextual linear-Gaussian Thompson sampler for expert selection."""

    def __init__(self, agent_config=None, rng=None):
        self.agent_config = agent_config or {}
        self.rng = rng or random
        self.context_dim = max(1, int(self.agent_config.get("localization_context_dim", 32)))
        self.prior_variance = float(self.agent_config.get("localization_prior_variance", 1.0))
        self.reward_variance = float(self.agent_config.get("localization_reward_variance", 0.25))
        self.priors = self._load_priors(self.agent_config.get("localization_expert_priors"))

    def ensure_state(self, state):
        localization = state.setdefault("localization", {})
        experts = localization.setdefault("experts", {})
        for expert, prior in self.priors.items():
            expert_state = experts.setdefault(expert, {})
            # Discard the old Beta posterior format from pre-bandit checkpoints.
            if "mean" not in expert_state or "covariance" not in expert_state:
                mean = np.zeros(self.context_dim + 1, dtype=float)
                mean[0] = float(prior.get("intercept", 0.0))
                expert_state.clear()
                expert_state["mean"] = mean.tolist()
                expert_state["covariance"] = (
                    np.eye(self.context_dim + 1, dtype=float) * self.prior_variance
                ).tolist()
        localization.setdefault("active_expert", None)
        localization.setdefault("active_root", None)
        localization.setdefault("active_confidence", {})
        localization.setdefault("active_context", None)
        localization.setdefault("last_expert", None)
        localization.setdefault("last_update", None)
        return localization

    def active_decision(self, state):
        # The contextual bandit selects an expert for every observation.  The
        # previous implementation reused one localization for a feedback
        # window, which is not the workflow described in the paper.
        self.ensure_state(state)
        return None

    def _context_vector(self, context=None):
        values = np.asarray(context if context is not None else [], dtype=float).reshape(-1)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        values = values[: self.context_dim]
        if values.size < self.context_dim:
            values = np.pad(values, (0, self.context_dim - values.size))
        return np.concatenate(([1.0], values))

    def select_expert(
        self,
        state,
        allowed_experts: Optional[Sequence[str]] = None,
        context=None,
    ):
        localization = self.ensure_state(state)
        allowed = [
            expert for expert in (allowed_experts or self.priors.keys()) if expert in self.priors
        ]
        if not allowed:
            raise ValueError("No localization expert is available")

        x = self._context_vector(context)
        samples = {}
        for expert in allowed:
            expert_state = localization["experts"].setdefault(expert, {})
            mean = np.asarray(expert_state["mean"], dtype=float)
            covariance = np.asarray(expert_state["covariance"], dtype=float)
            sampled_theta = np.random.multivariate_normal(mean, covariance)
            samples[expert] = float(x @ sampled_theta)
        return max(samples, key=samples.get), samples

    def start_decision(self, state, expert, root_causes, confidence=None, context=None):
        localization = self.ensure_state(state)
        root_causes = normalize_components(root_causes)
        confidence = normalize_component_confidence_map(confidence)
        localization["active_expert"] = expert
        localization["active_root"] = root_causes
        localization["active_confidence"] = confidence
        localization["active_context"] = self._context_vector(context).tolist()
        localization["last_expert"] = expert
        state["current_root"] = root_causes
        state["confidence"] = confidence
        return LocalizationDecision(expert=expert, root_causes=root_causes, confidence=confidence)

    def record_feedback(self, state, reward=None, reliable=None, success=None):
        localization = self.ensure_state(state)
        active_expert = localization.get("active_expert")
        if not active_expert:
            return None

        # Backwards-compatible call shape for small unit tests and old callers.
        if reward is None:
            reward = 1.0 if success else 0.0
        if reliable is None:
            reliable = bool(success) or float(reward) != 0.0
        if not reliable:
            localization["active_expert"] = None
            localization["active_root"] = None
            localization["active_confidence"] = {}
            localization["active_context"] = None
            return None

        expert_state = localization["experts"].setdefault(active_expert, {})
        x = np.asarray(localization.get("active_context"), dtype=float)
        mean = np.asarray(expert_state["mean"], dtype=float)
        covariance = np.asarray(expert_state["covariance"], dtype=float)
        precision = np.linalg.inv(covariance)
        noise_precision = 1.0 / max(self.reward_variance, 1e-8)
        new_precision = precision + noise_precision * np.outer(x, x)
        new_covariance = np.linalg.inv(new_precision)
        new_mean = new_covariance @ (
            precision @ mean + noise_precision * float(reward) * x
        )
        expert_state["mean"] = new_mean.tolist()
        expert_state["covariance"] = new_covariance.tolist()
        localization["last_update"] = {
            "expert": active_expert,
            "reward": float(reward),
            "reliable": True,
        }
        localization["active_expert"] = None
        localization["active_root"] = None
        localization["active_confidence"] = {}
        localization["active_context"] = None
        return localization["last_update"]

    def reset_active(self, state):
        localization = self.ensure_state(state)
        localization["active_expert"] = None
        localization["active_root"] = None
        localization["active_confidence"] = {}
        localization["active_context"] = None

    def _load_priors(self, configured_priors):
        priors = dict(DEFAULT_EXPERT_PRIORS)
        if not isinstance(configured_priors, dict):
            return priors
        for expert, value in configured_priors.items():
            if expert not in priors:
                continue
            try:
                if isinstance(value, dict):
                    priors[expert] = {"intercept": float(value.get("intercept", 0.0))}
                else:
                    # Accept the old [alpha, beta] shape while migrating it
                    # to a neutral Gaussian prior.
                    priors[expert] = {"intercept": 0.0}
            except (TypeError, ValueError, AttributeError):
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
