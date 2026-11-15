import os
import random
import hashlib
import json
import copy
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from agent.contracts import CANONICAL_COMPONENTS, normalize_components


@dataclass
class Experience:
    query_info: Dict[str, Any]
    root_causes: list
    case_info: Dict[str, Any]
    reward: float
    next_query_info_id: str  # Format: "round-query_id", e.g., "2-01" for round 2, query 01
    query_id: str = ""
    query_info_id: str = ""  # Current round and Query ID, e.g., "1-01"
    done: bool = False
    success_count: int = 0
    failure_count: int = 0
    history_case_ids: list = field(default_factory=list)


class SoftQRetriever:
    """Case retriever based on soft Q-learning"""

    def __init__(self, memory_manager=None):
        self.device = "cpu"

        # [query(32), localized components(4), success/failure(2),
        # history(101), case query/action(96), case components(4), reward(1)]
        # matches the state and demonstration encoding in Section V-B.
        self.embedding_dim = 240
        self.hidden_dim = 512
        self.output_dim = 1
        self.temperature = 1.0

        # Soft Q-learning parameters
        self.gamma = 0.99  # Discount factor γ
        self.alpha = 1.0  # Temperature parameter α for entropy regularization
        self.prior_beta = float(
            getattr(memory_manager, "retrieval_prior_beta", 0.8)
        )

        # Experience pool parameters
        self.batch_size = 64
        self.learning_rate = 1e-4
        
        # Memory case sampling
        self.max_sample_cases = 8

        # Store memory_manager reference
        self.memory_manager = memory_manager

        # Initialize network
        self._init_networks()

        # Experience pool (unlimited size, as experiences are persisted to file)
        self.experience_pool = deque()
        
        # Pending experiences: experiences that were skipped because next_query_info_id doesn't exist yet
        # Format: {next_query_info_id: [Experience, ...]}
        # When a case with matching case_id is created, these experiences will be added to experience_pool
        self.pending_experiences = {}

        # Optimizer
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)

        # Statistics
        self.total_experiences = 0
        self.update_count = 0

        # Embeddings are immutable for a case/query during one tuning run.
        # Supervised utility retrieval scores every eligible case, so caching
        # avoids recomputing the RCRank/BERT representation hundreds of times
        # for the same query.
        self._query_embedding_cache = {}
        self._case_embedding_cache = {}
        self._history_embedding_cache = {}

    def _init_networks(self):
        # Main network: two-layer MLP (outputs Q-value, not probability)
        # According to paper: Q^π(s_t, M_t, o_t) is a Q-value, not a probability
        # The softmax over Q-values is applied in select_cases, not in the network
        self.network = nn.Sequential(
            nn.Linear(self.embedding_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim // 2, self.output_dim),
            # No activation: output raw Q-value (can be any real number)
        ).to(self.device)

        # Target network (for stable training)
        self.target_network = nn.Sequential(
            nn.Linear(self.embedding_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim // 2, self.output_dim),
            # No activation: output raw Q-value
        ).to(self.device)

        self.target_network.load_state_dict(self.network.state_dict())
        self.target_network.eval()

    def _build_query_embedding(self, query_info) -> torch.Tensor:
        from utils.types import QueryInfo

        cache_key = self._embedding_cache_key(query_info)
        cached = self._query_embedding_cache.get(cache_key)
        if cached is not None:
            return cached.clone()

        query_obj = QueryInfo(**query_info)
        embedding_list = self.memory_manager.build_sql_embedding(query_obj)

        embedding = self._fixed_embedding(embedding_list, 32, "query")
        self._query_embedding_cache[cache_key] = embedding.detach().clone()
        return embedding

    def _build_case_embedding(self, case_info) -> torch.Tensor:
        from utils.types import QueryInfo

        cache_key = self._embedding_cache_key(case_info)
        cached = self._case_embedding_cache.get(cache_key)
        if cached is not None:
            return cached.clone()

        memory_case = getattr(self.memory_manager, "cases", {}).get(case_info.get("case_id"))
        if (memory_case is not None
                and memory_case.sql_embedding is not None and len(memory_case.sql_embedding)
                and memory_case.action_embedding is not None and len(memory_case.action_embedding)):
            query_embedding = memory_case.sql_embedding
            action_embedding = memory_case.action_embedding
        else:
            case_query_info = case_info.get("query_info")
            case_query_obj = QueryInfo(**case_query_info)
            query_embedding = self.memory_manager.build_sql_embedding(case_query_obj)
            fix_action = case_info.get("fix_action", "")
            action_embedding = self.memory_manager.build_action_embedding(fix_action)

        query_features = self._fixed_embedding(query_embedding, 32, "case query")
        action_features = self._fixed_embedding(action_embedding, 64, "case action")
        embedding = torch.cat((query_features, action_features))
        self._case_embedding_cache[cache_key] = embedding.detach().clone()
        return embedding

    def _fixed_embedding(self, values, width, name):
        """Keep the retriever input width stable when a plan cannot be encoded."""
        vector = torch.as_tensor(values, dtype=torch.float32, device=self.device).flatten()
        vector = torch.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
        if vector.numel() == 0:
            # RCRank skips plans with more filter predicates than its encoder
            # supports.  Retain the action and state features for this case.
            return torch.zeros(width, dtype=torch.float32, device=self.device)
        if vector.numel() < width:
            return F.pad(vector, (0, width - vector.numel()))
        return vector[:width]

    @staticmethod
    def _embedding_cache_key(info):
        """Return a stable key for query/case embedding memoization."""
        if not isinstance(info, dict):
            return str(info)
        if info.get("case_id"):
            return f"case:{info['case_id']}"
        # Query IDs persist across successful actions, while plans and metrics
        # change. Cache by the complete execution state Q_t.
        payload = json.dumps(info, sort_keys=True, default=str)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def _encode_root_causes(self, root_causes) -> torch.Tensor:
        root_cause_types = list(CANONICAL_COMPONENTS)
        root_causes = normalize_components(root_causes)

        root_cause_vector = torch.zeros(4)
        for i, root_cause_type in enumerate(root_cause_types):
            if root_cause_type in root_causes:
                root_cause_vector[i] = 1.0

        return root_cause_vector.to(self.device)

    def _compute_features(self, query_info, case_info, root_causes, state_counts=None):
        query_embedding = self._build_query_embedding(query_info)
        case_embedding = self._build_case_embedding(case_info)

        root_cause_vector = self._encode_root_causes(root_causes)

        state_counts = state_counts or {}
        counts_tensor = torch.tensor(
            [
                float(state_counts.get("success_count", 0)),
                float(state_counts.get("failure_count", 0)),
            ],
            dtype=torch.float32,
            device=self.device,
        )

        relative_reward = self._case_relative_reward(case_info)
        case_components = self._encode_root_causes(
            case_info.get("localized_components", case_info.get("root_causes", []))
        )
        history_embedding = self._build_history_embedding(state_counts.get("history_case_ids", []))
        features = torch.cat([
            query_embedding, root_cause_vector, counts_tensor, history_embedding,
            case_embedding, case_components,
            torch.tensor([relative_reward], dtype=torch.float32, device=self.device),
        ])

        return features

    def _build_history_embedding(self, case_ids):
        """Mean-pool prior observation embeddings for this query's history."""
        case_ids = tuple(case_ids or ())
        cached = self._history_embedding_cache.get(case_ids)
        if cached is not None:
            return cached.clone()
        embeddings = []
        for case_id in case_ids:
            case = getattr(self.memory_manager, "cases", {}).get(case_id)
            if case is None:
                continue
            info = dict(case.case_info, case_id=case.id)
            base = self._build_case_embedding(info)
            components = self._encode_root_causes(
                info.get("localized_components", info.get("root_causes", []))
            )
            reward = self._case_relative_reward(info)
            embeddings.append(torch.cat([
                base, components,
                torch.tensor([reward], dtype=torch.float32, device=self.device),
            ]))
        pooled = torch.stack(embeddings).mean(dim=0) if embeddings else torch.zeros(101, device=self.device)
        self._history_embedding_cache[case_ids] = pooled.detach().clone()
        return pooled

    def predict_case_value(self, query_info, case_info, root_causes, state_counts=None):
        """
        Predict Q-value for a single case
        
        Returns:
            Q-value (raw output from network, can be any real number)
        """
        was_training = self.network.training
        self.network.eval()
        with torch.no_grad():
            features = self._compute_features(query_info, case_info, root_causes, state_counts)
            features = features.unsqueeze(0)

            q_value = self.network(features)
            value = q_value.item()

        self.network.train(was_training)
        return value

    def predict_case_values(self, query_info, candidate_cases, root_causes, state_counts=None, batch_size=256):
        """Score all candidates with batched Q-network forwards, preserving input order."""
        if not candidate_cases:
            return []
        batch_size = max(1, int(batch_size))
        was_training = self.network.training
        self.network.eval()
        try:
            values = []
            with torch.no_grad():
                for start in range(0, len(candidate_cases), batch_size):
                    features = torch.stack([
                        self._compute_features(query_info, case_info, root_causes, state_counts)
                        for case_info in candidate_cases[start:start + batch_size]
                    ])
                    values.extend(self.network(features).reshape(-1).tolist())
            return values
        finally:
            self.network.train(was_training)

    def select_cases(self, query_info, candidate_cases, root_causes, top_k=4, state_counts=None):
        """Sample from the diagnosis-prior-guided Boltzmann policy (Eq. 15)."""
        if not candidate_cases:
            return []

        # Check alpha value for numerical stability
        if self.alpha <= 0:
            raise ValueError(f"alpha must be positive, got {self.alpha}")

        # Score the same full memory as before, but avoid one network call per case.
        cases_list = candidate_cases
        q_values_tensor = torch.tensor(
            self.predict_case_values(query_info, cases_list, root_causes, state_counts),
            dtype=torch.float32, device=self.device,
        )

        log_prior = self._diagnosis_log_prior(root_causes, cases_list)
        softmax_probs = torch.softmax(log_prior + q_values_tensor / self.alpha, dim=0)

        k = min(top_k, len(cases_list))
        sampled_indices = torch.multinomial(softmax_probs, num_samples=k, replacement=False)
        result = []
        for idx in sampled_indices:
            idx_int = idx.item()
            case_info = cases_list[idx_int]
            probability = softmax_probs[idx_int].item()
            result.append((case_info, probability))

        return result

    def _diagnosis_log_prior(self, root_causes, cases):
        """Normalized Jaccard prior over actual modified components (Eqs. 12-13)."""
        localized = set(normalize_components(root_causes))
        alignment = []
        for case in cases:
            modified = set(normalize_components(case.get("modified_components", [])))
            union = localized | modified
            alignment.append(len(localized & modified) / len(union) if union else 0.0)
        scores = torch.tensor(alignment, dtype=torch.float32, device=self.device)
        return torch.log_softmax(self.prior_beta * scores, dim=0)

    def add_experience(
        self,
        query_info,
        case_info,
        root_causes,
        reward,
        next_query_info_id,
        done=False,
        state_counts=None,
        query_info_id=None,
    ):
        """
        Add experience tuple (s_t, o_t, B_t, r_t, next_query_info_id, done)
        
        Only add experience to pool if:
        1. done=True (terminal state), or
        2. next_query_info_id is empty/None (terminal state), or
        3. next_query_info_id corresponds to a case that exists in memory
        
        This ensures that we only train on complete experiences where the next state is available.
        
        Args:
            query_info: s_t - current state (query_info, can be dict or QueryInfo object)
            case_info: o_t - retrieved observation from memory
            root_causes: B_t - current bottleneck components
            reward: r_t - reward
            next_query_info_id: next_query_info_id in format "round-query_id", e.g., "2-01" for round 2, query 01
                               None or empty string if terminal
            done: Whether the episode is terminated
        """
        # Identify the query by Query ID and the state by round plus Query ID.
        # SQL text remains part of the model input, not the state identifier.
        query_data = query_info.__dict__ if hasattr(query_info, '__dict__') else query_info
        query_id = str(query_data.get("query_id", ""))

        # Create experience
        # QueryInfo is updated later in the tuning loop when a rewrite is
        # trialed.  A transition must retain the accepted, pre-action state
        # even when that trial is rolled back.
        experience = Experience(
            query_info=copy.deepcopy(query_data),
            case_info=copy.deepcopy(case_info),
            root_causes=normalize_components(root_causes),
            reward=reward,
            next_query_info_id=next_query_info_id if next_query_info_id else "",
            query_id=query_id,
            query_info_id=str(query_info_id or query_id),
            done=done,
            success_count=int((state_counts or {}).get("success_count", 0)),
            failure_count=int((state_counts or {}).get("failure_count", 0)),
            history_case_ids=list((state_counts or {}).get("history_case_ids", [])),
        )

        # Only add experience to pool if it's complete (terminal or next state exists)
        if done or not experience.next_query_info_id:
            # Terminal state: always add
            self.experience_pool.append(experience)
            self.total_experiences += 1
        else:
            # Non-terminal state: check if next_query_info exists in memory
            next_query_info, _ = self._get_next_state_from_id(experience.next_query_info_id)
            if next_query_info is not None:
                # Next state exists: add to pool
                self.experience_pool.append(experience)
                self.total_experiences += 1
            else:
                # Next state doesn't exist yet: add to pending experiences
                # It will be added later when the next state is created (in the next round)
                if experience.next_query_info_id not in self.pending_experiences:
                    self.pending_experiences[experience.next_query_info_id] = []
                self.pending_experiences[experience.next_query_info_id].append(experience)
    
    def check_and_add_pending_experiences(self, case_id):
        """
        Check if there are pending experiences waiting for this case_id, and add them to experience_pool.
        
        This should be called when a new case is created, to add previously skipped experiences.
        
        Args:
            case_id: The case_id that was just created (format: "round-query_id")
        """

        if case_id in self.pending_experiences:
            pending_list = self.pending_experiences[case_id]
            for experience in pending_list:
                self.experience_pool.append(experience)
                self.total_experiences += 1
            # Remove from pending
            del self.pending_experiences[case_id]

    def finalize_pending_experiences(self, next_query_info_id):
        """Close transitions when a query meets an early-stop criterion."""
        pending = self.pending_experiences.pop(next_query_info_id, [])
        for experience in pending:
            experience.done = True
            experience.next_query_info_id = ""
            self.experience_pool.append(experience)
            self.total_experiences += 1
        if pending:
            self.update_network()

    def _get_next_memory_cases(self, next_query_info_id):
        """
        Get memory cases that were created before the next_query_info_id.
        
        Args:
            next_query_info_id: Format "round-query_id", e.g., "2-01"
        
        Returns:
            List of case_info that were created before the next_query_info_id
        """
        if self.memory_manager is None or not next_query_info_id:
            return []
        
        # Case archives preserve insertion order. The next case is recorded
        # after its retrieval decision, so it and later cases are excluded.
        candidate_cases = []
        for case in self.memory_manager.cases.values():
            if case.id == next_query_info_id:
                break
            candidate_cases.append(dict(case.case_info, case_id=case.id))
        return candidate_cases

    @staticmethod
    def _case_relative_reward(case_info):
        """Normalize a demonstration's observed latency change by its old latency."""
        try:
            old_time = float(case_info.get("old_time") or 0.0)
            trial_time = case_info.get("new_time")
            new_time = old_time if trial_time is None else float(trial_time)
        except (TypeError, ValueError):
            return 0.0
        return (old_time - new_time) / max(abs(old_time), 1e-8)

    @staticmethod
    def _case_improved(case_info):
        """Count accepted improvements, including gains below the RAWS delta."""
        if case_info.get("validation_status") == -1:
            return False
        try:
            return float(case_info["new_time"]) < float(case_info["old_time"])
        except (KeyError, TypeError, ValueError):
            return case_info.get("label") == "positive"

    def _compute_soft_bellman_target(
        self, next_query_info, next_root_causes, next_memory_cases, done,
        next_state_counts=None,
    ):
        """
        Compute the prior-guided soft state value in Eq. 16.
        
        Args:
            next_query_info: s_{t+1} - next state
            next_root_causes: B_{t+1} - next bottleneck components
            next_memory_cases: List of case_info in M_{t+1} (next memory)
            done: Whether episode is terminated
        
        Returns:
            TD target value
        """
        if done or len(next_memory_cases) == 0:
            return torch.tensor(0.0, device=self.device)
        
        query_id = str(next_query_info.get("query_id", ""))
        history = [
            case for case in next_memory_cases
            if str((case.get("query_info") or {}).get("query_id", "")) == query_id
        ]
        if next_state_counts is None:
            # Older case archives do not record the exact retrieval state.
            next_state_counts = {
                "success_count": sum(self._case_improved(case) for case in history),
                "failure_count": sum(not self._case_improved(case) for case in history),
                "history_case_ids": [case["case_id"] for case in history if case.get("case_id")],
            }

        # Chunk the Bellman target forwards as memory grows. No gradients flow
        # through the target network, but the case prior still covers all cases.
        q_value_chunks = []
        with torch.no_grad():
            for start in range(0, len(next_memory_cases), 256):
                features = torch.stack([
                    self._compute_features(next_query_info, case_info, next_root_causes, next_state_counts)
                    for case_info in next_memory_cases[start:start + 256]
                ])
                q_value_chunks.append(self.target_network(features).reshape(-1))
        q_values_tensor = torch.cat(q_value_chunks)
        
        log_prior = self._diagnosis_log_prior(next_root_causes, next_memory_cases)
        soft_value = self.alpha * torch.logsumexp(
            log_prior + q_values_tensor / self.alpha, dim=0
        )
        
        return soft_value

    def update_network(self, next_memory_cases_fn=None):
        if len(self.experience_pool) < self.batch_size:
            return
        # Randomly sample experiences
        batch_experiences = random.sample(self.experience_pool, self.batch_size)

        # Prepare training data
        features_list = []
        td_targets = []

        for experience in batch_experiences:
            # Compute features for (s_t, o_t, B_t)
            features = self._compute_features(
                experience.query_info,
                experience.case_info,
                experience.root_causes,
                {
                    "success_count": experience.success_count,
                    "failure_count": experience.failure_count,
                    "history_case_ids": experience.history_case_ids,
                },
            )
            features_list.append(features)
            
            # Compute TD target: y_t = r_t + γ * α * log(Σ_{o' in M_{t+1}} exp(Q_θ(s_{t+1}, M_{t+1}, o')/α))
            if experience.done or not experience.next_query_info_id:
                # Terminal state: y_t = r_t
                td_target = torch.tensor(experience.reward, dtype=torch.float32, device=self.device)
            else:
                # Get next_query_info and next_root_causes from memory_cases using next_query_info_id
                next_query_info, next_root_causes = self._get_next_state_from_id(experience.next_query_info_id)
                
                if next_query_info is None:
                    # If next_query_info not found, treat as terminal
                    td_target = torch.tensor(experience.reward, dtype=torch.float32, device=self.device)
                else:
                    # Get next memory cases M_{t+1}
                    next_memory_cases = self._get_next_memory_cases(experience.next_query_info_id)
                    
                    # Compute soft Bellman target
                    soft_value = self._compute_soft_bellman_target(
                        next_query_info,
                        next_root_causes,
                        next_memory_cases,
                        False,  # Not done since we have next state
                        next_state_counts=self._get_next_state_counts_from_id(
                            experience.next_query_info_id
                        ),
                    )
                    
                    # TD target: y_t = r_t + γ * soft_value
                    td_target = torch.tensor(experience.reward, dtype=torch.float32, device=self.device) + self.gamma * soft_value
            
            td_targets.append(td_target)

        # Convert to tensors
        features_tensor = torch.stack(features_list)
        td_targets_tensor = torch.stack(td_targets)

        # Forward pass: Q_θ(s_t, M_t, o_t)
        q_predictions = self.network(features_tensor).squeeze()

        # Compute TD loss: L_TD(θ) = E[(Q_θ(s_t, M_t, o_t) - y_t)^2]
        loss = F.mse_loss(q_predictions, td_targets_tensor)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1

        # Periodically update target network
        if self.update_count % 10 == 0:
            self.target_network.load_state_dict(self.network.state_dict())
    
    def _get_next_state_from_id(self, next_query_info_id):
        """
        Get next_query_info and next_root_causes from memory_cases using next_query_info_id.
        
        Args:
            next_query_info_id: Format "round-query_id", e.g., "2-01"
        
        Returns:
            (next_query_info, next_root_causes) tuple, or (None, None) if not found
        """
        if not next_query_info_id or self.memory_manager is None:
            return None, None
        
        # Split only the round prefix; a Query ID may itself contain '-'.
        try:
            round_text, separator, target_query_id = next_query_info_id.partition("-")
            if not separator or not target_query_id:
                return None, None
            
            target_round = int(round_text)
            
            # Find the case with matching round and query_id
            # Since case_id format is "round-query_id", there should be at most one matching case
            target_case_id = f"{target_round}-{target_query_id}"
            if target_case_id in self.memory_manager.cases:
                case = self.memory_manager.cases[target_case_id]
                case_info = case.case_info
                next_query_info = case_info.get("query_info")
                next_root_causes = case_info.get("root_causes", [])
                next_root_causes = normalize_components(next_root_causes)
                return next_query_info, next_root_causes
        except (ValueError, IndexError):
            pass
        
        return None, None

    def _get_next_state_counts_from_id(self, next_query_info_id):
        """Retrieve the exact history features captured at the next decision."""
        if self.memory_manager is None:
            return None
        case = self.memory_manager.cases.get(next_query_info_id)
        if case is None:
            return None
        counts = case.case_info.get("retrieval_state_counts")
        return counts if isinstance(counts, dict) else None

    def save_model(self, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        torch.save(self.network.state_dict(), save_path)
        print(f"SoftQRetriever model saved to: {save_path}")

    def load_model(self, load_path):
        state_dict = torch.load(load_path, map_location=self.device)

        # Legacy checkpoints order inputs as query, case, localized components,
        # approve_time, then (in 241-wide checkpoints) counts, history, case
        # components, and normalized reward. Reorder to [u_t; e(O)] and drop
        # approve_time, which is absent from the paper's encoding.
        old_first_layer = state_dict.get("0.weight")
        if old_first_layer is not None and old_first_layer.shape[1] != self.embedding_dim:
            expanded = torch.zeros(
                old_first_layer.shape[0], self.embedding_dim,
                dtype=old_first_layer.dtype,
            )
            if old_first_layer.shape[1] in (133, 241):
                expanded[:, :32] = old_first_layer[:, :32]
                expanded[:, 32:36] = old_first_layer[:, 128:132]
                expanded[:, 139:235] = old_first_layer[:, 32:128]
                if old_first_layer.shape[1] == 241:
                    expanded[:, 36:38] = old_first_layer[:, 133:135]
                    expanded[:, 38:139] = old_first_layer[:, 135:236]
                    expanded[:, 235:239] = old_first_layer[:, 236:240]
                    expanded[:, 239] = old_first_layer[:, 240]
            else:
                width = min(old_first_layer.shape[1], self.embedding_dim)
                expanded[:, :width] = old_first_layer[:, :width]
            state_dict["0.weight"] = expanded

        self.network.load_state_dict(state_dict)
        self.target_network.load_state_dict(state_dict)
        self.target_network.eval()

        print(f"SoftQRetriever model loaded from: {load_path}")
