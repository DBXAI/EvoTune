import itertools
import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd
import torch
from agents import Agent, Runner
from numpy.linalg import norm
from sklearn.cluster import HDBSCAN

from agent.contracts import (
    CANONICAL_COMPONENTS,
    normalize_component_label,
)
from agent.memory.localization_refinement import (
    build_component_labels,
    masked_bce_loss,
    parse_attempt_vector,
)
from agent.prompt import (
    build_case_reflection_prompt,
    build_experience_summary_prompt,
    get_reflection_agent_instructions,
    get_summary_agent_instructions,
)

# Import soft Q retriever
from agent.memory.soft_q_retriever import SoftQRetriever
from agent.plan.RCRank.utils.load_data import (
    load_dataset_valid as load_dataset_tensor_valid,
)
from utils.openai_agents import configure_openai_agents
from utils.types import CaseInfo, QueryInfo

logger = logging.getLogger(__name__)

# Ensure logger inherits root logger configuration
logger.setLevel(logging.INFO)
logger.propagate = True


@dataclass
class MemoryCase:
    id: str
    case_info: CaseInfo
    timestamp: float = field(default_factory=time.time)
    metrics: dict = field(default_factory=dict)
    sql_embedding: Optional[List[float]] = None
    action_embedding: Optional[List[float]] = None


@dataclass
class ClusterExperience:
    id: str
    case_ids: List[str]
    summary: str = ""
    prototype: Optional[List[float]] = None
    radius: float = 0.0

    def to_dict(self):
        return {
            "id": self.id,
            "case_ids": list(self.case_ids),
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            id=data["id"],
            case_ids=list(data.get("case_ids", [])),
            summary=data.get("summary", ""),
        )


class SummaryAgent:
    def __init__(self, summary_model):
        self.summaryAgent = Agent(
            name="SummaryAgent",
            model=summary_model,
            instructions=get_summary_agent_instructions(),
        )
        self.reflectionAgent = Agent(
            name="ReflectionAgent",
            model=summary_model,
            instructions=get_reflection_agent_instructions(),
        )

    def reflect_case(self, case_info):
        prompt = build_case_reflection_prompt(case_info)
        query_info = case_info.get("query_info", {}) or {}
        # logger.info(
        #     "[Prompt][Reflection][SQL %s]\n%s",
        #     query_info.get("query_id", "unknown"),
        #     prompt,
        # )
        try:
            result = Runner.run_sync(
                starting_agent=self.reflectionAgent,
                input=prompt,
            )
            return re.sub(r"\s+", " ", str(result.final_output or "")).strip()
        except Exception as exc:
            logger.warning(
                "Case reflection failed for SQL %s: %s",
                query_info.get("query_id", "unknown"),
                exc,
                exc_info=True,
            )
            return ""

    def summarize_cases(self, root_causes, cases, previous_summary=None):
        prompt = build_experience_summary_prompt(root_causes, cases, previous_summary)
        # logger.info(
        #     "[Prompt][ExperienceSummary][Cases %s]\n%s",
        #     [case.id for case in cases],
        #     prompt,
        # )
        try:
            result = Runner.run_sync(
                starting_agent=self.summaryAgent,
                input=prompt,
            )
            return re.sub(r"\s+", " ", str(result.final_output or "")).strip()
        except Exception as exc:
            logger.warning(
                "Experience summarization failed for cases %s: %s",
                [case.id for case in cases],
                exc,
                exc_info=True,
            )
            return previous_summary or ""


class MemoryManager:
    def __init__(self, config, predictor=None, planner_config=None, agent_config=None, api_settings=None):
        self.db_path = config.get("db_path")
        self.archive_size = config.get("archive_size")
        self.cases = {}
        self.archive = set()
        self.best_case_id = None
        self.agent_config = agent_config or {}
        self.api_settings = api_settings or {}

        self.enable_finetune = config.get("enable_finetune")
        self.finetune_threshold = config.get("finetune_threshold")
        self.finetune_epochs = config.get("finetune_epochs")
        self.finetune_batch_size = config.get("finetune_batch_size")
        self.finetune_device = config.get("finetune_device")
        self.model_output_dir = config.get("model_output_dir")

        # Exploration threshold for verified supervision
        self.eta = config.get("eta", 3)  # Minimum exploration threshold η

        self.bert_model_path = config.get("bert_model_path")
        self.experience_path = config.get("experience_path") or f"{self.db_path}_experiences.json"
        self.pending_buffer_threshold = config.get(
            "pending_buffer_threshold",
            config.get("experience_distill_threshold", 20),
        )
        self.hdbscan_min_cluster_size = config.get("hdbscan_min_cluster_size", 5)
        self.hdbscan_min_samples = config.get("hdbscan_min_samples", 5)
        self.experience_top_k = config.get("experience_top_k", 2)
        self.summary_agent_model = config.get("summary_agent_model")

        # Retrieval mode configuration: static or dynamic
        self.retrieval_mode = config.get("retrieval_mode", "dynamic")

        # Sample save configuration
        self.enable_save_samples = config.get("enable_save_samples", False)
        self.samples_save_path = config.get("samples_save_path")

        # Retriever model save/load path
        self.retriever_model_path = config.get("retriever_model_path")

        self.current_round = 1

        self.root_cause_types = list(CANONICAL_COMPONENTS)
        # Support multiple root cause combinations
        self.island_keys = self._generate_island_keys(self.root_cause_types)
        self.islands = {key: set() for key in self.island_keys}

        self._rcrank_predictor = predictor
        self._planner_config = planner_config  # Store planner config for lazy initialization

        # Initialize soft Q retriever
        self.retriever = SoftQRetriever(memory_manager=self)

        if self.api_settings:
            configure_openai_agents(self.api_settings)

        self.summary_agent = SummaryAgent(self.summary_agent_model)
        self.experiences = {}
        self.pending_case_ids = []
        self.case_to_experience = {}
        self.next_experience_index = 1
        self._case_embedding_cache = {}

        # Load the saved retriever model
        if os.path.exists(self.retriever_model_path):
            self.retriever.load_model(self.retriever_model_path)

        self._init_database()
        self.load()

    def _cases_to_dataframe(self):
        rows = []
        label_space = self.root_cause_types

        for cid, case in self.cases.items():
            info = case.case_info
            raw = info.get("query_info", {})
            query = raw.get("query", "")
            plan_json = raw.get("plan_json")

            # Unify to stringified plan JSON
            if isinstance(plan_json, dict):
                plan_json = json.dumps({"Plan": plan_json})
            external_metrics = raw.get("external_metrics", [])
            internal_metrics = raw.get("internal_metrics", [])

            roots = self._normalize_root_causes_list(info.get("root_causes"))
            label_vec = build_component_labels(roots, info.get("label", "positive"), label_space)
            selected_vec = [1 if r in roots else 0 for r in label_space]

            duration = info.get("new_time", 0.0)

            # Get tuning attempts per component: n_{i,b}
            # tuning_attempts is a dict mapping component (root_cause_type) to number of attempts
            tuning_attempts = info.get("tuning_attempts", {})
            tuning_attempts = self._normalize_tuning_attempts(tuning_attempts)
            if not tuning_attempts:
                # Fallback: if positive case, assume at least 1 attempt per root cause component
                # For negative cases, assume 0 attempts (unverified)
                tuning_attempts = {}
                if info.get("label") == "positive":
                    for rc in roots:
                        if rc in label_space:
                            tuning_attempts[rc] = tuning_attempts.get(rc, 0) + 1

            # Build attempts vector: [n_{i,b}] for each component b in label_space
            attempts_vec = [tuning_attempts.get(rc, 0) for rc in label_space]

            rows.append(
                {
                    "sql_file": self.db_path,
                    "query": query,
                    "plan_json": plan_json,
                    "internal_metrics": json.dumps(internal_metrics),
                    "external_metrics": json.dumps(external_metrics),
                    "multilabel": str(label_vec),
                    "duration": duration,
                    "error": None,
                    "case_label": info.get("label", "positive"),
                    "tuning_attempts": json.dumps(attempts_vec),  # n_{i,b} for each component b
                    "selected_components": json.dumps(selected_vec),
                }
            )
        return pd.DataFrame(rows)

    def _export_cases_csv(self, csv_path):
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df = self._cases_to_dataframe()
        df.to_csv(csv_path, index=False)
        return csv_path

    def _run_finetune(
        self,
        csv_path,
        output_dir,
        device,
        epochs,
        batch_size,
    ):
        # Use EvoTune's localization refinement objective: masked BCE over reliable labels.
        os.makedirs(output_dir, exist_ok=True)

        train_dataloader, _, _, train_len, _, _, _ = load_dataset_tensor_valid(
            csv_path, batch_size=batch_size, device=device
        )

        if not hasattr(self, "_rcrank_predictor") or self._rcrank_predictor is None:
            from agent.plan.online_predict import RCRankPredictor

            self._rcrank_predictor = RCRankPredictor(device=device, train_data_path=csv_path)
        predictor = self._rcrank_predictor
        model = predictor.model.to(device)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # Important parameter, try 1e-4 or 3e-4
        eta = self.eta  # Minimum exploration threshold

        def localization_refinement_loss(
            labels,
            pred_label,
            tuning_attempts,
            selected_components,
            eta=3,
        ):
            bce_loss = masked_bce_loss(pred_label, labels, selected_components, tuning_attempts, eta)
            auxiliary_loss = pred_label.sum() * 0.0
            return bce_loss, bce_loss, auxiliary_loss

        # 3) Training loop
        print("Starting finetune")
        for epoch_idx in range(epochs):
            epoch_total_loss = 0.0
            epoch_bce_loss = 0.0
            epoch_auxiliary_loss = 0.0

            print(f"Finetune epoch {epoch_idx}")
            for batch in train_dataloader:
                # batch keys: query(list[str]), plan(dict of tensors), log[tensor], timeseries[tensor], multilabel[tensor]
                sql_texts = batch["query"]
                plan = batch["plan"]
                logs = batch["log"].to(device)
                timeseries = batch["timeseries"].to(device)
                labels = batch["multilabel"].to(device).float()

                # Get tuning_attempts n_{i,b} for each component
                if "tuning_attempts" in batch:
                    tuning_attempts_list = [parse_attempt_vector(ta, labels.size(1)) for ta in batch["tuning_attempts"]]
                    tuning_attempts = torch.tensor(tuning_attempts_list, device=device, dtype=torch.float32)
                else:
                    tuning_attempts = torch.zeros_like(labels, device=device)

                if "selected_components" in batch:
                    selected_components_list = [
                        parse_attempt_vector(sc, labels.size(1)) for sc in batch["selected_components"]
                    ]
                    selected_components = torch.tensor(selected_components_list, device=device, dtype=torch.float32)
                else:
                    selected_components = labels.clone().to(device)

                # Text encoding
                sql = predictor.tokenizer(
                    sql_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                ).to(device)
                # Plan tensor preparation
                plan = {
                    "x": plan["x"].squeeze(1).to(device).to(torch.float32),
                    "attn_bias": plan["attn_bias"].squeeze(1).to(device),
                    "rel_pos": plan["rel_pos"].squeeze(1).to(device),
                    "heights": plan["heights"].squeeze(1).to(device),
                }

                optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    pred_label, _ = model(sql, plan, timeseries, logs)  # [B, K]
                    total_loss, bce_loss, auxiliary_loss = localization_refinement_loss(
                        labels,
                        pred_label,
                        tuning_attempts,
                        selected_components,
                        eta,
                    )
                    epoch_bce_loss += bce_loss.item()
                    epoch_auxiliary_loss += auxiliary_loss.item()

                    total_loss.backward()
                    optimizer.step()
                    epoch_total_loss += total_loss.item()

            # Record training logs
            avg_total_loss = epoch_total_loss / max(1, len(train_dataloader))
            avg_bce_loss = epoch_bce_loss / max(1, len(train_dataloader))
            avg_auxiliary_loss = epoch_auxiliary_loss / max(1, len(train_dataloader))

            print(
                f"Masked-BCE finetune epoch {epoch_idx}: total_loss={avg_total_loss:.6f}, "
                f"bce_loss={avg_bce_loss:.6f}, auxiliary_loss={avg_auxiliary_loss:.6f}"
            )

        # 4) Save weights
        best_path = os.path.join(output_dir, "best_model.pt")
        torch.save(model.state_dict(), best_path)
        return best_path

    def _reload_predictor(self, model_path, stats_data_path, device):
        if not hasattr(self, "_rcrank_predictor") or self._rcrank_predictor is None:
            from agent.plan.online_predict import RCRankPredictor

            self._rcrank_predictor = RCRankPredictor(
                model_path=model_path, train_data_path=stats_data_path, device=device
            )
            self._rcrank_predictor.model.eval()
            return

        predictor = self._rcrank_predictor
        predictor.device = device
        predictor.model_path = model_path
        predictor._compute_statistics(stats_data_path)
        predictor.model.load_state_dict(torch.load(model_path, map_location=device))
        predictor.model.to(device)
        predictor.model.eval()

    def _maybe_finetune(self):
        if not self.enable_finetune:
            return

        total = len(self.cases)
        if total - self._last_finetune_count < self.finetune_threshold:
            return

        # export CSV
        csv_path = os.path.join(self.model_output_dir, "memory_cases.csv")
        os.makedirs(self.model_output_dir, exist_ok=True)
        self._export_cases_csv(csv_path)

        # finetune
        best_model = self._run_finetune(
            csv_path,
            self.model_output_dir,
            self.finetune_device,
            self.finetune_epochs,
            self.finetune_batch_size,
        )

        # reload predictor
        self._reload_predictor(best_model, csv_path, self.finetune_device)
        self._last_finetune_count = total

    def _init_database(self):
        if not os.path.exists(self.db_path):
            with open(self.db_path, "w") as f:
                json.dump({}, f)

    def _generate_island_keys(self, root_cause_types):
        keys = []
        for i in range(1, len(root_cause_types) + 1):
            keys.extend([frozenset(pair) for pair in itertools.combinations(root_cause_types, i)])
        return keys

    def _build_case_reflection(self, case_info):
        return self.summary_agent.reflect_case(case_info)

    def _get_case_cluster_embedding(self, case):
        if case.id in self._case_embedding_cache:
            return self._case_embedding_cache[case.id]

        embedding = self.retriever._build_case_embedding(case.case_info)
        if hasattr(embedding, "detach"):
            embedding = embedding.detach().cpu().numpy().tolist()
        else:
            embedding = list(embedding)
        self._case_embedding_cache[case.id] = embedding
        return embedding

    def _infer_next_experience_index(self):
        max_idx = 0
        for exp_id in self.experiences.keys():
            match = re.match(r"^experience-(\d+)$", str(exp_id))
            if match:
                max_idx = max(max_idx, int(match.group(1)))
        return max_idx + 1

    def _load_experiences(self):
        self.experiences = {}
        self.pending_case_ids = []
        self.case_to_experience = {}
        self.next_experience_index = 1

        if not self.experience_path or not os.path.exists(self.experience_path):
            return

        with open(self.experience_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        self.experiences = {
            exp_id: ClusterExperience.from_dict(exp_info) for exp_id, exp_info in payload.get("experiences", {}).items()
        }
        self.next_experience_index = self._infer_next_experience_index()
        self._rebuild_case_to_experience_map()

    def _save_experiences(self):
        if not self.experience_path:
            return

        directory = os.path.dirname(self.experience_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        payload = {
            "experiences": {exp_id: experience.to_dict() for exp_id, experience in self.experiences.items()},
        }
        with open(self.experience_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    def _rebuild_case_to_experience_map(self):
        self.case_to_experience = {}
        for exp_id, experience in self.experiences.items():
            for case_id in experience.case_ids:
                self.case_to_experience[case_id] = exp_id

    def _refresh_experience(self, exp_id, experience=None):
        experience = experience or self.experiences[exp_id]
        cases = [self.cases[cid] for cid in experience.case_ids if cid in self.cases]
        if not cases:
            return None

        embeddings = [self._get_case_cluster_embedding(case) for case in cases]
        prototype = np.mean(np.asarray(embeddings, dtype=float), axis=0).astype(float).tolist()
        distances = [
            float(np.linalg.norm(np.asarray(embedding, dtype=float) - np.asarray(prototype, dtype=float)))
            for embedding in embeddings
        ]
        experience.prototype = prototype
        experience.radius = max(distances) if distances else 0.0
        self.experiences[exp_id] = experience
        self._rebuild_case_to_experience_map()
        return experience

    def _update_experience_summary(self, experience):
        cases = [self.cases[cid] for cid in experience.case_ids if cid in self.cases]
        if not cases:
            return experience
        merged_root_causes = []
        for case in cases:
            merged_root_causes.extend(case.case_info.get("root_causes", []))
        experience.summary = self.summary_agent.summarize_cases(
            self._normalize_root_causes_list(merged_root_causes),
            cases,
            experience.summary,
        )
        self.experiences[experience.id] = experience
        return experience

    def _create_experience(self, case_ids):
        experience = ClusterExperience(
            id=f"experience-{self.next_experience_index:04d}",
            case_ids=list(case_ids),
        )
        self.next_experience_index += 1
        experience = self._refresh_experience(experience.id, experience=experience)
        if experience is None:
            return None
        return self._update_experience_summary(experience)

    def _distill_case_ids(self, case_ids):
        candidate_case_ids = [
            case_id
            for case_id in dict.fromkeys(case_ids)
            if case_id in self.cases and case_id not in self.case_to_experience
        ]
        if len(candidate_case_ids) < self.hdbscan_min_cluster_size:
            return candidate_case_ids

        matrix = np.asarray(
            [self._get_case_cluster_embedding(self.cases[case_id]) for case_id in candidate_case_ids],
            dtype=float,
        )
        clusterer = HDBSCAN(
            min_cluster_size=self.hdbscan_min_cluster_size,
            min_samples=self.hdbscan_min_samples,
            metric="euclidean",
            allow_single_cluster=True,
        )
        labels = clusterer.fit_predict(matrix)

        unresolved = []
        cluster_groups = {}
        for idx, label in enumerate(labels):
            case_id = candidate_case_ids[idx]
            if int(label) == -1:
                unresolved.append(case_id)
                continue
            cluster_groups.setdefault(int(label), []).append(case_id)

        for member_ids in cluster_groups.values():
            if len(member_ids) < self.hdbscan_min_cluster_size:
                unresolved.extend(member_ids)
                continue
            self._create_experience(member_ids)

        self._rebuild_case_to_experience_map()
        return list(dict.fromkeys(unresolved))

    def _bootstrap_experiences(self):
        self._load_experiences()
        for exp_id in list(self.experiences.keys()):
            self._refresh_experience(exp_id)

        assigned_case_ids = set(self.case_to_experience.keys())
        self.pending_case_ids = [case_id for case_id in self.cases.keys() if case_id not in assigned_case_ids]

        if len(self.pending_case_ids) >= self.pending_buffer_threshold:
            unresolved = self._distill_case_ids(self.pending_case_ids)
            self.pending_case_ids = [case_id for case_id in unresolved if case_id not in self.case_to_experience]
            self._save_experiences()

    def _merge_case_into_nearest_experience(self, case):
        candidates = list(self.experiences.values())
        if not candidates:
            return False

        case_embedding = self._get_case_cluster_embedding(case)
        nearest_experience = None
        nearest_distance = float("inf")

        for experience in candidates:
            prototype = experience.prototype or []
            if not prototype:
                continue
            distance = float(
                np.linalg.norm(np.asarray(case_embedding, dtype=float) - np.asarray(prototype, dtype=float))
            )
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_experience = experience

        if nearest_experience is None or nearest_distance > nearest_experience.radius:
            return False

        if case.id not in nearest_experience.case_ids:
            nearest_experience.case_ids.append(case.id)
        refreshed = self._refresh_experience(nearest_experience.id, experience=nearest_experience)
        if refreshed is not None:
            self._update_experience_summary(refreshed)
        return True

    def _sync_case_to_experiences(self, case):
        try:
            if self._merge_case_into_nearest_experience(case):
                self._save_experiences()
                return

            if case.id not in self.pending_case_ids and case.id not in self.case_to_experience:
                self.pending_case_ids.append(case.id)
            if len(self.pending_case_ids) >= self.pending_buffer_threshold:
                unresolved = self._distill_case_ids(self.pending_case_ids)
                self.pending_case_ids = [case_id for case_id in unresolved if case_id not in self.case_to_experience]
                self._save_experiences()
        except Exception as exc:
            logger.warning("Failed to update experience clusters for case %s: %s", case.id, exc)

    def retrieve_experiences(self, case_ids):
        if not case_ids:
            return {"items": [], "summary": "None."}

        selected = []
        seen_experience_ids = set()
        for case_id in case_ids:
            experience_id = self.case_to_experience.get(case_id)
            if not experience_id or experience_id in seen_experience_ids:
                continue
            experience = self.experiences.get(experience_id)
            if experience is None:
                continue
            selected.append(experience)
            seen_experience_ids.add(experience_id)
            if len(selected) >= self.experience_top_k:
                break

        if not selected:
            return {"items": [], "summary": "None."}

        lines = []
        for idx, experience in enumerate(selected, 1):
            lines.append(f"\n    Experience {idx}: {experience.summary}")

        return {
            "items": selected,
            "summary": "".join(lines).strip() or "None.",
        }

    def add_case(self, case):
        self._validate_and_route_root_causes(case)
        self.cases[case.id] = case
        self._enforce_archive_limit()

        self._update_archive(case)
        self._update_best_case(case)

        root_causes_key = self._normalize_root_causes(case.case_info.get("root_causes"))
        if root_causes_key in self.islands:
            self.islands[root_causes_key].add(case.id)
        else:
            self.islands[root_causes_key] = set([case.id])

        self.save_to_memory(case)
        self._sync_case_to_experiences(case)
        logger.debug(f"Added case {case.id} to island {root_causes_key}")

        try:
            self._maybe_finetune()
        except Exception as e:
            print(f"Online finetune failed: {e}")
        return case.id

    def _build_used_case_snapshot(self, case_info):
        if not isinstance(case_info, dict):
            return {}

        query_info = case_info.get("query_info", {}) or {}
        return {
            "case_id": case_info.get("case_id"),
            "query_id": query_info.get("query_id"),
            "query": query_info.get("query"),
            "execution_time": query_info.get("execution_time"),
            "label": case_info.get("label"),
            "root_causes": self._normalize_root_causes_list(case_info.get("root_causes")),
            "approve_time": case_info.get("approve_time"),
            "fix_action": case_info.get("fix_action"),
        }

    def _normalize_used_cases(self, used_cases):
        if not isinstance(used_cases, dict):
            return {"positive": [], "negative": []}

        return {
            "positive": [self._build_used_case_snapshot(case_info) for case_info in used_cases.get("positive", [])],
            "negative": [self._build_used_case_snapshot(case_info) for case_info in used_cases.get("negative", [])],
        }

    def save_to_memory(self, new_case=None):
        if new_case is not None:
            try:
                with open(self.db_path, "r") as f:
                    data_to_save = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                data_to_save = {}

            data_to_save[new_case.id] = new_case.case_info

            with open(self.db_path, "w") as f:
                json.dump(data_to_save, f)
        else:
            # full save
            with open(self.db_path, "w") as f:
                data_to_save = {}
                for cid, case in self.cases.items():
                    data_to_save[cid] = case.case_info
                json.dump(data_to_save, f)

    def save_case(
        self,
        query_info,
        fix_action,
        root_causes,
        old_time,
        new_time,
        label,
        tuning_attempts=None,
        round_num=None,
        used_cases=None,
    ):
        """
        Save a case to memory

        Args:
            query_info: Query information
            fix_action: Fix action applied
            root_causes: Root causes (list or single value)
            old_time: Original execution time
            new_time: New execution time after fix
            label: "positive" or "negative"
            tuning_attempts: Dict[str, int] mapping component (root_cause_type) to number of attempts n_{i,b}
                           If None, will be estimated from root_causes
            round_num: Current round number (if None, uses self.current_round)
        """
        approve_time = round(old_time - new_time, 3)

        # If tuning_attempts not provided, estimate from root_causes
        if tuning_attempts is None:
            tuning_attempts = {}
            # If positive case, assume at least 1 attempt for each root cause component
            if label == "positive":
                root_causes_list = self._normalize_root_causes_list(root_causes)
                for rc in root_causes_list:
                    if rc in self.root_cause_types:
                        tuning_attempts[rc] = tuning_attempts.get(rc, 0) + 1
            # For negative cases, we can't reliably estimate, so leave as empty dict
        else:
            tuning_attempts = self._normalize_tuning_attempts(tuning_attempts)

        root_causes = self._normalize_root_causes_list(root_causes)

        # Use provided round_num or current_round
        if round_num is None:
            round_num = self.current_round

        query_id_str = str(query_info.query_id)
        match = re.match(r"^(\d+)(.*)$", query_id_str)
        if match:
            numeric_part = match.group(1)
            suffix = match.group(2)
            query_id_str = numeric_part.zfill(2) + suffix
        case_id = f"{round_num}-{query_id_str}"

        case_info = {
            "query_info": query_info.__dict__,
            "fix_action": fix_action,
            "root_causes": root_causes,
            "old_time": old_time,
            "new_time": new_time,
            "label": label,
            "approve_time": approve_time,
            "tuning_attempts": tuning_attempts,
            "used_cases": self._normalize_used_cases(used_cases),
        }
        case_info["reflection"] = self._build_case_reflection(case_info)
        case_info["reflection_source"] = "summary_agent"

        sql_embedding = self.build_sql_embedding(query_info)
        action_embedding = self.build_action_embedding(str(fix_action))

        case = MemoryCase(
            id=case_id,
            case_info=case_info,
            metrics={"performance": new_time, "approve_time": approve_time},
            sql_embedding=sql_embedding,
            action_embedding=action_embedding,
        )
        self.add_case(case)

        if self.retriever is not None:
            self.retriever.check_and_add_pending_experiences(case_id)

        return case.id

    def generate_next_query_info_id(self, query_id, round_num=None):
        """
        Generate next_query_info_id in format "round-query_id"

        Args:
            query_id: Query ID
            round_num: Next round number (if None, uses current_round + 1)

        Returns:
            next_query_info_id string, e.g., "2-01"
        """
        if round_num is None:
            round_num = self.current_round + 1

        query_id_str = str(query_id)
        match = re.match(r"^(\d+)(.*)$", query_id_str)
        if match:
            numeric_part = match.group(1)
            suffix = match.group(2)  # Could be empty or letters like 'c', 'b'
            query_id_str = numeric_part.zfill(2) + suffix
        return f"{round_num}-{query_id_str}"

    def set_current_round(self, round_num):
        """Set current round number"""
        self.current_round = round_num

    def get_case(self, case_id):
        return self.cases.get(case_id)

    def ids_to_cases(self, ids):
        return [self.cases[cid] for cid in ids]

    def load(self):
        if os.path.exists(self.db_path):
            with open(self.db_path, "r") as f:
                data = json.load(f)
                for cid, case_info in data.items():
                    case = self._build_memory_case_from_info(cid, case_info)
                    self.cases[cid] = case

                    root_causes = case.case_info.get("root_causes")
                    if root_causes:
                        island_key = self._normalize_root_causes(root_causes)
                        self.islands[island_key].add(cid)

                positive_cases = [c for c in self.cases.values() if c.case_info.get("label") == "positive"]
                positive_cases.sort(key=lambda c: c.metrics.get("approve_time", 0.0), reverse=True)
                self.archive = set([c.id for c in positive_cases[: self.archive_size]])
            self._last_finetune_count = len(self.cases)
            self._bootstrap_experiences()
            # self._last_finetune_count = 0
            # self._maybe_finetune()
        else:
            self._last_finetune_count = 0

    def _build_memory_case_from_info(self, case_id, case_info):
        case_info = dict(case_info)
        case_info["root_causes"] = self._normalize_root_causes_list(case_info.get("root_causes"))
        case_info["tuning_attempts"] = self._normalize_tuning_attempts(case_info.get("tuning_attempts"))
        if "reflection" not in case_info:
            case_info["reflection"] = self._build_case_reflection(case_info)
            case_info["reflection_source"] = "summary_agent"
        raw_info = case_info["query_info"]

        if "is_rewrite" not in raw_info:
            raw_info["is_rewrite"] = False
        query_info = QueryInfo(**raw_info)
        sql_embedding = self.build_sql_embedding(query_info)
        action_embedding = self.build_action_embedding(str(case_info["fix_action"]))

        approve_time = case_info.get("approve_time", 0.0)
        new_time = case_info.get("new_time", 0.0)
        metrics = {"performance": new_time, "approve_time": approve_time}

        return MemoryCase(
            id=case_id,
            case_info=case_info,
            metrics=metrics,
            sql_embedding=sql_embedding,
            action_embedding=action_embedding,
        )

    def _feature_coords_to_key(self, coords):
        return "-".join(str(c) for c in coords)

    def _is_better(self, case1, case2):
        return case1.metrics.get("approve_time", 0.0) > case2.metrics.get("approve_time", 0.0)

    # select the best case by approve_time
    def best_by_approve(self, cases):
        if not cases:
            return None
        return max(
            cases,
            key=lambda c: c.case_info.get("approve_time") or c.metrics.get("approve_time", 0.0),
        )

    def _update_archive(self, case):
        # only positive cases are allowed to enter archive
        if case.case_info.get("label") != "positive":
            return

        if len(self.archive) < self.archive_size:
            self.archive.add(case.id)
            return
        archive_cases = [self.cases[aid] for aid in self.archive]
        worst_case = min(archive_cases, key=lambda c: c.metrics.get("approve_time", 0.0))
        if self._is_better(case, worst_case):
            self.archive.remove(worst_case.id)
            self.archive.add(case.id)

    def _update_best_case(self, case):
        # only positive cases are allowed to become best_case
        if case.case_info.get("label") != "positive":
            return

        if self.best_case_id is None or self._is_better(case, self.cases[self.best_case_id]):
            self.best_case_id = case.id

    def _enforce_archive_limit(self):
        if len(self.archive) > self.archive_size:
            # sort by approve_time
            archive_cases = [self.cases[case_id] for case_id in self.archive if case_id in self.cases]
            sorted_archive_cases = sorted(archive_cases, key=lambda c: c.metrics.get("approve_time"))

            # remove the worst cases
            to_remove_count = len(self.archive) - self.archive_size
            to_remove_cases = sorted_archive_cases[:to_remove_count]

            for case in to_remove_cases:
                if case.id in self.archive:
                    self.archive.remove(case.id)

    def retrieve_cases(self, query_info, root_causes, pos_n=2, neg_n=2, mode="exploit"):
        """
        Unified case retrieval interface, selects static or dynamic retrieval based on configured retrieval_mode
        """
        if self.retrieval_mode == "static":
            retrieval = self.retrieve_static_cases(query_info, root_causes, pos_n, neg_n, mode)
        else:
            retrieval = self.retrieve_dynamic_cases(query_info, root_causes, pos_n, neg_n, mode)

        retrieved_case_ids = []
        for case_info in retrieval.get("positive", []) + retrieval.get("negative", []):
            case_id = case_info.get("case_id")
            if case_id:
                retrieved_case_ids.append(case_id)

        experience_payload = self.retrieve_experiences(retrieved_case_ids)
        retrieval["experiences"] = experience_payload.get("summary", "None.")
        return retrieval

    def retrieve_dynamic_cases(self, query_info, root_causes, pos_n=2, neg_n=2, mode="exploit"):
        root_causes = self._normalize_root_causes_list(root_causes)
        # Get all candidate cases
        all_cases = list(self.cases.values())
        if not all_cases:
            return {"mode": mode, "positive": [], "negative": []}

        # Filter candidate cases based on mode
        if mode == "exploit":
            # Exploit mode: select cases with same root cause
            candidate_cases = []
            for case in all_cases:
                case_root_causes = self._normalize_root_causes_list(case.case_info.get("root_causes"))
                if any(rc in case_root_causes for rc in root_causes):
                    case_info = dict(case.case_info)
                    case_info["case_id"] = case.id
                    candidate_cases.append(case_info)
        elif mode == "explore":
            # Explore mode: select cases with different root causes
            candidate_cases = []
            for case in all_cases:
                case_root_causes = self._normalize_root_causes_list(case.case_info.get("root_causes"))
                if not any(rc in case_root_causes for rc in root_causes):
                    case_info = dict(case.case_info)
                    case_info["case_id"] = case.id
                    candidate_cases.append(case_info)

        # If no candidate cases, return empty result
        if not candidate_cases:
            return {"mode": mode, "positive": [], "negative": []}

        # Use Soft Q Learning retriever to select cases
        selected_cases = self.retriever.select_cases(
            query_info.__dict__, candidate_cases, root_causes, top_k=pos_n + neg_n
        )

        # Classify by label
        positive_cases = []
        negative_cases = []

        for case_info, probability in selected_cases:
            if case_info.get("label") == "positive":
                positive_cases.append(case_info)
            else:
                negative_cases.append(case_info)

        # Limit quantity
        positive_cases = positive_cases[:pos_n]
        negative_cases = negative_cases[:neg_n]

        return {
            "mode": mode,
            "positive": positive_cases,
            "negative": negative_cases,
        }

    def retrieve_static_cases(self, query_info, root_causes, pos_n=2, neg_n=2, mode="exploit"):
        """
        Use traditional islands to construct sample pool and output by label:
        - exploitation pool: island corresponding to target root cause
        - exploration pool: all other islands merged
        Positive/Negative division based on case_info.label (fallback to approve_time>0 as positive if missing).
        Positive return rule: include at least 1 with same SQL and 1 with different SQL with max approve_time each, rest filled in descending order of improvement up to pos_n.
        Negative return rule: randomly sample at most neg_n from negative pool.
        Returns: {"mode": str, "positive": [case_info], "negative": [case_info]}
        """
        query_text = query_info.query.lower().strip()
        target_root = self._normalize_root_causes(root_causes)

        target_island_ids = list(self.islands.get(target_root, set()))
        other_island_ids = []
        for island_key, ids in self.islands.items():
            if island_key != target_root:
                other_island_ids.extend(list(ids))

        # Exploitation pool
        exploit_pool = self.ids_to_cases(target_island_ids)
        explore_pool = self.ids_to_cases(other_island_ids)

        # ε-greedy mode
        chosen_pool = explore_pool if mode == "explore" else exploit_pool

        # In exploit pool, prioritize same SQL cases
        def split_by_sql(cases):
            same_sql_list = []
            diff_sql_list = []
            for c in cases:
                stored_query = c.case_info.get("query_info").get("query")
                if stored_query.lower().strip() == query_text:
                    same_sql_list.append(c)
                else:
                    diff_sql_list.append(c)
            return same_sql_list, diff_sql_list

        # Split by label/approve_time
        same_sql_pool, diff_sql_pool = split_by_sql(chosen_pool)

        # Positive pool and negative pool
        pos_same = [c for c in same_sql_pool if c.case_info.get("label") == "positive"]
        pos_diff = [c for c in diff_sql_pool if c.case_info.get("label") == "positive"]
        neg_all = [c for c in chosen_pool if c.case_info.get("label") != "positive"]

        selected_pos = []
        best_same = self.best_by_approve(pos_same)
        if best_same is not None:
            selected_pos.append(best_same)
        best_diff = self.best_by_approve(pos_diff)
        if best_diff is not None:
            selected_pos.append(best_diff)

        # Remaining positive cases are sorted by approve_time
        remaining_pos = [c for c in (pos_same + pos_diff) if c not in selected_pos]
        remaining_pos_sorted = sorted(
            remaining_pos,
            key=lambda c: c.case_info.get("approve_time") or c.metrics.get("approve_time", 0.0),
            reverse=True,
        )
        selected_pos.extend(remaining_pos_sorted)
        selected_pos = selected_pos[: max(1, pos_n)]

        # Negative cases are sampled randomly
        if len(neg_all) > neg_n:
            selected_neg = random.sample(neg_all, neg_n)
        else:
            selected_neg = neg_all

        # Export case_info
        pos_infos = []
        for case in selected_pos:
            case_info = dict(case.case_info)
            case_info["case_id"] = case.id
            pos_infos.append(case_info)

        neg_infos = []
        for case in selected_neg:
            case_info = dict(case.case_info)
            case_info["case_id"] = case.id
            neg_infos.append(case_info)

        return {"mode": mode, "positive": pos_infos, "negative": neg_infos}

    def retrieve_similar_cases(self, query_info, pos_n=2, neg_n=2, top_n=None):
        """
        Static case retrieval without diagnosis/root-cause filtering.
        Retrieve globally similar historical cases based on SQL embedding,
        while prioritizing exact same-SQL matches when available.
        """
        if not self.cases:
            return {"mode": "static_simple", "positive": [], "negative": []}

        normalized_query = str(getattr(query_info, "query", "") or "").strip().lower()
        candidate_count = top_n or min(len(self.cases), max((pos_n + neg_n) * 6, 20))

        same_sql_cases = []
        for case in self.cases.values():
            stored_query = ((case.case_info.get("query_info") or {}).get("query") or "").strip().lower()
            if stored_query and stored_query == normalized_query:
                same_sql_cases.append(case)

        same_sql_cases.sort(
            key=lambda case: (
                case.case_info.get("label") == "positive",
                case.case_info.get("approve_time") or case.metrics.get("approve_time", 0.0),
            ),
            reverse=True,
        )

        similar_cases = self.retrieve_embedding(query_info, top_n=candidate_count)

        ranked_cases = []
        seen_case_ids = set()

        for case in same_sql_cases:
            if case.id not in seen_case_ids:
                ranked_cases.append(case)
                seen_case_ids.add(case.id)

        for case in similar_cases:
            if case.id not in seen_case_ids:
                ranked_cases.append(case)
                seen_case_ids.add(case.id)

        positive_cases = []
        negative_cases = []
        for case in ranked_cases:
            case_info = dict(case.case_info)
            case_info["case_id"] = case.id
            if case.case_info.get("label") == "positive":
                if len(positive_cases) < pos_n:
                    positive_cases.append(case_info)
            else:
                if len(negative_cases) < neg_n:
                    negative_cases.append(case_info)

            if len(positive_cases) >= pos_n and len(negative_cases) >= neg_n:
                break

        return {
            "mode": "static_simple",
            "positive": positive_cases,
            "negative": negative_cases,
        }

    def _normalize_plan_json_for_encoder(self, query_plan):
        if query_plan is None:
            return None

        if isinstance(query_plan, str):
            normalized = query_plan.strip()
            return normalized or None

        if isinstance(query_plan, dict):
            if "Plan" in query_plan:
                return json.dumps([[query_plan]], ensure_ascii=False)
            return json.dumps([[{"Plan": query_plan}]], ensure_ascii=False)

        if isinstance(query_plan, list):
            if not query_plan:
                return None
            if isinstance(query_plan[0], dict) and "Plan" in query_plan[0]:
                return json.dumps([query_plan], ensure_ascii=False)
            return json.dumps(query_plan, ensure_ascii=False)

        try:
            normalized = json.dumps(query_plan, ensure_ascii=False)
        except TypeError:
            normalized = str(query_plan).strip()
        return normalized or None

    def retrieve_embedding(self, query_info, top_n=5, metric="cosine", case_ids=None):
        try:
            query_embedding = self.build_sql_embedding(query_info)
        except Exception as exc:
            logger.warning(
                "Failed to build query embedding for SQL %s: %s",
                getattr(query_info, "query_id", "unknown"),
                exc,
                exc_info=True,
            )
            return []

        if not query_embedding:
            return []

        query_emb = np.array(query_embedding, dtype=float)
        if query_emb.size == 0 or norm(query_emb) == 0:
            return []

        # the embedding similarity
        case_sims = []
        if case_ids is not None:
            candidates = [self.get_case(cid) for cid in case_ids]
        else:
            candidates = list(self.cases.values())
        for case in candidates:
            if case is None or case.sql_embedding is None:
                continue
            case_emb = np.array(case.sql_embedding, dtype=float)
            if case_emb.size == 0:
                logger.warning(
                    "Skipping case %s in retrieval because sql_embedding is empty",
                    getattr(case, "id", "unknown"),
                )
                continue
            if case_emb.ndim != 1 or query_emb.ndim != 1:
                logger.warning(
                    "Skipping case %s in retrieval because embedding rank is invalid: query_ndim=%s case_ndim=%s",
                    getattr(case, "id", "unknown"),
                    query_emb.ndim,
                    case_emb.ndim,
                )
                continue
            if case_emb.shape[0] != query_emb.shape[0]:
                logger.warning(
                    "Skipping case %s in retrieval because embedding dimensions do not match: query_dim=%s case_dim=%s",
                    getattr(case, "id", "unknown"),
                    query_emb.shape[0],
                    case_emb.shape[0],
                )
                continue
            if not np.isfinite(case_emb).all():
                logger.warning(
                    "Skipping case %s in retrieval because sql_embedding contains non-finite values",
                    getattr(case, "id", "unknown"),
                )
                continue
            case_norm = norm(case_emb)
            if case_norm == 0:
                logger.warning(
                    "Skipping case %s in retrieval because sql_embedding norm is zero",
                    getattr(case, "id", "unknown"),
                )
                continue
            if metric == "cosine":
                sim = np.dot(query_emb, case_emb) / (norm(query_emb) * case_norm + 1e-8)
            elif metric == "euclidean":
                sim = -norm(query_emb - case_emb)
            else:
                raise ValueError("Unknown metric")
            case_sims.append((sim, case))

        # sort and return top-N
        case_sims.sort(reverse=True, key=lambda x: x[0])
        return [c for _, c in case_sims[:top_n]]

    def build_sql_embedding(self, query_info):
        # use the RCRankPredictor
        if not hasattr(self, "_rcrank_predictor") or self._rcrank_predictor is None:
            from agent.plan.online_predict import RCRankPredictor

            if self._planner_config is None:
                self._planner_config = {
                    "device": "cpu",
                    "model_path": "/root/DREAM/src/agent/plan/model_res/GateComDiffPretrainModel slow_sql_data_gen eta0.07/best_model.pt",
                    "train_data_path": "/root/DREAM/src/agent/plan/slow_sql_data_gen.csv",
                    "opt_threshold": 0.5,
                    "num_classes": 4,
                }
            self._rcrank_predictor = RCRankPredictor(self._planner_config)
        predictor = self._rcrank_predictor

        import pandas as pd

        from agent.plan.RCRank.model.modules.QueryFormer.utils import Encoding
        from agent.plan.RCRank.utils.plan_encoding import PlanEncoder

        encoding = Encoding(None, {"NA": 0})

        query = query_info.query
        normalized_plan_json = self._normalize_plan_json_for_encoder(query_info.plan_json)
        if not normalized_plan_json:
            logger.warning(
                "SQL %s has empty or invalid plan_json; skipping archive embedding retrieval",
                getattr(query_info, "query_id", "unknown"),
            )
            return []

        try:
            df = pd.DataFrame([{"plan_json": normalized_plan_json}])
            plan_encoder = PlanEncoder(df=df, encoding=encoding)
            encoded_df = plan_encoder.df
            if encoded_df.empty:
                logger.warning(
                    "Plan encoding returned no rows for SQL %s; skipping archive embedding retrieval",
                    getattr(query_info, "query_id", "unknown"),
                )
                return []
            plan_tensor = encoded_df["json_plan_tensor"].iloc[0]
        except Exception as exc:
            logger.warning(
                "Plan encoding failed for SQL %s: %s",
                getattr(query_info, "query_id", "unknown"),
                exc,
                exc_info=True,
            )
            return []

        sql = predictor.tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=512).to(
            predictor.device
        )

        plan = {
            "x": plan_tensor["x"].to(predictor.device).to(torch.float32),
            "attn_bias": plan_tensor["attn_bias"].to(predictor.device),
            "rel_pos": plan_tensor["rel_pos"].to(predictor.device),
            "heights": plan_tensor["heights"].to(predictor.device),
        }

        timeseries = torch.tensor(query_info.external_metrics)
        if len(timeseries.shape) == 2:
            timeseries = timeseries.unsqueeze(0)
        timeseries = (timeseries - predictor.timeseries_train_mean.unsqueeze(1)) / (
            predictor.timeseries_train_std.unsqueeze(1) + 1e-6
        )
        timeseries = timeseries.to(predictor.device)

        log = torch.tensor(query_info.internal_metrics)
        if len(log.shape) == 1:
            log = log.unsqueeze(0)
        log = (log - predictor.logs_train_mean) / (predictor.logs_train_std + 1e-6)
        log = log.to(predictor.device)

        # get the embedding
        emb = predictor.model.get_embedding(sql, plan, timeseries, log)
        return emb.squeeze(0).cpu().numpy().tolist()

    def build_action_embedding(self, action):
        from transformers import BertModel, BertTokenizer

        if not hasattr(self, "_bert_tokenizer"):
            bert_path = self.bert_model_path
            self._bert_tokenizer = BertTokenizer.from_pretrained(bert_path)
            self._bert_model = BertModel.from_pretrained(bert_path)
            self._bert_model.eval()
        tokenizer = self._bert_tokenizer
        model = self._bert_model

        inputs = tokenizer(action, return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
            cls_emb = outputs.last_hidden_state[:, 0, :]  # [CLS] vector, 768 dimensions
            # Directly truncate to 64 dimensions (take first 64 dimensions)
            truncated_emb = cls_emb[:, :64]  # 768 -> 64
        return truncated_emb.squeeze(0).cpu().numpy().tolist()

    def _normalize_root_causes(self, root_causes):
        if root_causes is None:
            return frozenset()
        normalized = self._normalize_root_causes_list(root_causes)
        return frozenset(normalized)

    def _normalize_root_causes_list(self, root_causes):
        if root_causes is None:
            return []
        if isinstance(root_causes, str):
            candidates = [root_causes]
        else:
            try:
                candidates = list(root_causes)
            except Exception:
                candidates = [root_causes]

        normalized = []
        seen = set()
        for candidate in candidates:
            label = normalize_component_label(candidate)
            if label in self.root_cause_types and label not in seen:
                seen.add(label)
                normalized.append(label)

        normalized.sort(key=lambda item: self.root_cause_types.index(item))
        return normalized

    def _normalize_tuning_attempts(self, tuning_attempts):
        if not isinstance(tuning_attempts, dict):
            return {}

        normalized = {}
        for label, count in tuning_attempts.items():
            component = normalize_component_label(label)
            if component not in self.root_cause_types:
                continue
            normalized[component] = normalized.get(component, 0) + int(count)
        return normalized

    def _infer_root_causes_from_action(self, fix_action, query_info):
        fix_action = str(fix_action).lower()
        query_text = query_info.query.lower().strip()
        inferred: Set[str] = set()
        if "index" in fix_action:
            inferred.add("index")
        if ("set" in fix_action) and ("=" in fix_action):
            inferred.add("knob")
        if ("/*+" in query_text) or ("*/" in query_text):
            inferred.add("hint")
        if query_info.is_rewrite:
            inferred.add("query")
        return frozenset(inferred) if inferred else frozenset()

    def _validate_and_route_root_causes(self, case):
        # validate and route the root causes
        declared = self._normalize_root_causes(case.case_info.get("root_causes"))
        raw_info = case.case_info.get("query_info")
        query_info = QueryInfo(**raw_info)
        inferred = self._infer_root_causes_from_action(case.case_info.get("fix_action"), query_info)

        if inferred and (inferred != declared):
            case.case_info["root_causes"] = self._normalize_root_causes_list(inferred)
        else:
            case.case_info["root_causes"] = self._normalize_root_causes_list(case.case_info.get("root_causes"))

        case.case_info["tuning_attempts"] = self._normalize_tuning_attempts(case.case_info.get("tuning_attempts"))

        key = self._normalize_root_causes(case.case_info.get("root_causes"))
        if key not in self.islands:
            self.islands[key] = set()

    def explore_root_from_archive(self, query_info, exclude_roots):
        """
        retrieve the most similar SQL root cause from archive
        return the most similar SQL root cause, if not found, return None
        """
        if not self.archive:
            return None

        exclude_roots = exclude_roots

        archive_cases = [self.cases[aid] for aid in self.archive]
        if not archive_cases:
            return None

        try:
            similar_cases = self.retrieve_embedding(query_info, top_n=len(archive_cases), case_ids=list(self.archive))
        except Exception as exc:
            logger.warning(
                "Archive exploration failed for SQL %s: %s",
                getattr(query_info, "query_id", "unknown"),
                exc,
                exc_info=True,
            )
            return None

        for case in similar_cases:
            case_root_str = str(self._normalize_root_causes(case.case_info.get("root_causes")))
            if case_root_str not in exclude_roots:
                return case.case_info.get("root_causes")

        return None

    def save_experience(self, query_info, case_info, root_causes, reward, next_query_info_id=None, done=False):
        """
        Save single experience to file (JSONL format, append mode)

        Experience tuple: (s_t, o_t, B_t, r_t, next_query_info_id, done)

        Args:
            query_info: s_t - current state (query_info)
            case_info: o_t - retrieved observation from memory
            root_causes: B_t - current bottleneck components
            reward: r_t - reward
            next_query_info_id: next_query_info_id in format "round-query_id", e.g., "2-01" for round 2, query 01
                               None or empty string if terminal
            done: Whether the episode is terminated
        """

        if not self.samples_save_path:
            return

        os.makedirs(os.path.dirname(self.samples_save_path), exist_ok=True)

        # Generate next_query_info_id if not provided
        if next_query_info_id is None and not done:
            next_query_info_id = self.generate_next_query_info_id(query_info.query_id)

        # Prepare current experience dict
        exp_dict = {
            "query_info": query_info.__dict__ if hasattr(query_info, "__dict__") else query_info,
            "case_info": case_info,
            "root_causes": self._normalize_root_causes_list(root_causes),
            "reward": reward,
            "next_query_info_id": next_query_info_id if next_query_info_id else "",
            "done": done,
        }

        # Save current experience to file
        with open(self.samples_save_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(exp_dict, ensure_ascii=False) + "\n")
