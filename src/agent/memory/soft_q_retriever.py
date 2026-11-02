import os
import random
import hashlib
from collections import deque
from dataclasses import dataclass
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
    done: bool = False
    success_count: int = 0
    failure_count: int = 0


class SoftQRetriever:
    """Case retriever based on soft Q-learning"""

    def __init__(self, memory_manager=None):
        self.device = "cpu"

        # query(32) + observation(32+64) + localized components(4) +
        # candidate improvement(1) + current success/failure counts(2).
        self.embedding_dim = 135
        self.hidden_dim = 512
        self.output_dim = 1
        self.temperature = 1.0

        # Soft Q-learning parameters
        self.gamma = 0.99  # Discount factor γ
        self.alpha = 1.0  # Temperature parameter α for entropy regularization

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

    def _init_networks(self):
        # Main network: two-layer MLP (outputs Q-value, not probability)
        # According to paper: Q^π(s_t, M_t, o_t) is a Q-value, not a probability
        # The softmax over Q-values is applied in select_cases, not in the network
        self.network = nn.Sequential(
            nn.Linear(self.embedding_dim, self.hidden_dim),  # query_emb(32) + case_emb(64+32) + root_causes(4) + approve_time(1) = 133
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
            nn.Linear(self.embedding_dim, self.hidden_dim),  # 133-dimensional input
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim // 2, self.output_dim),
            # No activation: output raw Q-value
        ).to(self.device)

        self.target_network.load_state_dict(self.network.state_dict())

    def _build_query_embedding(self, query_info) -> torch.Tensor:
        from utils.types import QueryInfo

        cache_key = self._embedding_cache_key(query_info)
        cached = self._query_embedding_cache.get(cache_key)
        if cached is not None:
            return cached.clone()

        query_obj = QueryInfo(**query_info)
        embedding_list = self.memory_manager.build_sql_embedding(query_obj)

        embedding = torch.tensor(embedding_list, dtype=torch.float32)
        embedding = embedding.to(self.device)
        self._query_embedding_cache[cache_key] = embedding.detach().clone()
        return embedding

    def _build_case_embedding(self, case_info) -> torch.Tensor:
        from utils.types import QueryInfo

        cache_key = self._embedding_cache_key(case_info)
        cached = self._case_embedding_cache.get(cache_key)
        if cached is not None:
            return cached.clone()

        case_query_info = case_info.get("query_info")
        case_query_obj = QueryInfo(**case_query_info)
        query_embedding = self.memory_manager.build_sql_embedding(case_query_obj)

        fix_action = case_info.get("fix_action", "")
        action_embedding = self.memory_manager.build_action_embedding(fix_action)

        combined_embedding = query_embedding + action_embedding

        embedding = torch.tensor(combined_embedding, dtype=torch.float32)
        embedding = embedding.to(self.device)
        self._case_embedding_cache[cache_key] = embedding.detach().clone()
        return embedding

    @staticmethod
    def _embedding_cache_key(info):
        """Return a stable key for query/case embedding memoization."""
        if not isinstance(info, dict):
            return str(info)
        explicit_id = info.get("case_id") or info.get("query_id")
        if explicit_id:
            return str(explicit_id)
        query_info = info.get("query_info") if isinstance(info.get("query_info"), dict) else info
        query = str(query_info.get("query", ""))
        action = str(info.get("fix_action", ""))
        return hashlib.sha1(f"{query}\0{action}".encode("utf-8")).hexdigest()

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

        approve_time = case_info.get("approve_time", 0.0)
        approve_time_tensor = torch.tensor([approve_time], dtype=torch.float32).to(self.device)
        state_counts = state_counts or {}
        counts_tensor = torch.tensor(
            [
                float(state_counts.get("success_count", 0)),
                float(state_counts.get("failure_count", 0)),
            ],
            dtype=torch.float32,
            device=self.device,
        )

        features = torch.cat(
            [query_embedding, case_embedding, root_cause_vector, approve_time_tensor, counts_tensor]
        )

        return features

    def predict_case_value(self, query_info, case_info, root_causes, state_counts=None):
        """
        Predict Q-value for a single case
        
        Returns:
            Q-value (raw output from network, can be any real number)
        """
        with torch.no_grad():
            features = self._compute_features(query_info, case_info, root_causes, state_counts)
            features = features.unsqueeze(0)

            q_value = self.network(features)
            value = q_value.item()

            return value

    def select_cases(self, query_info, candidate_cases, root_causes, top_k=4, state_counts=None):
        """
        Select k cases using softmax over Q-values according to paper:
        μ*(o | s, M) = exp(Q*(s, M, o)/α) / Σ_{o' in M} exp(Q*(s, M, o')/α)
        
        Then sample k cases based on softmax probabilities (weighted sampling).
        """
        if not candidate_cases:
            return []

        # Check alpha value for numerical stability
        if self.alpha <= 0:
            raise ValueError(f"alpha must be positive, got {self.alpha}")

        # Calculate Q-value for each case
        case_q_values = []
        for case_info in candidate_cases:
            q_value = self.predict_case_value(query_info, case_info, root_causes, state_counts)
            case_q_values.append((case_info, q_value))

        # Convert to tensors for softmax computation
        cases_list = [case_info for case_info, _ in case_q_values]
        q_values_list = [q_val for _, q_val in case_q_values]
        q_values_tensor = torch.tensor(q_values_list, dtype=torch.float32, device=self.device)

        # Apply softmax over Q-values with temperature α
        # μ*(o | s, M) = exp(Q*(s, M, o)/α) / Σ_{o' in M} exp(Q*(s, M, o')/α)
        # Use log-sum-exp trick for numerical stability
        q_values_scaled = q_values_tensor / self.alpha
        q_max = torch.max(q_values_scaled)
        # Compute log-sum-exp: log(Σ exp(Q/α)) = max + log(Σ exp(Q/α - max))
        log_sum_exp = q_max + torch.log(torch.sum(torch.exp(q_values_scaled - q_max)) + 1e-12)
        # Compute softmax probabilities: exp(Q/α) / Σ exp(Q/α)
        softmax_probs = torch.exp(q_values_scaled - log_sum_exp)

        k = min(top_k, len(cases_list))
        sampled_indices = torch.multinomial(softmax_probs, num_samples=k, replacement=False)
        result = []
        for idx in sampled_indices:
            idx_int = idx.item()
            case_info = cases_list[idx_int]
            probability = softmax_probs[idx_int].item()
            result.append((case_info, probability))

        return result

    def add_experience(
        self,
        query_info,
        case_info,
        root_causes,
        reward,
        next_query_info_id,
        done=False,
        state_counts=None,
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
        # Create experience
        experience = Experience(
            query_info=query_info.__dict__ if hasattr(query_info, '__dict__') else query_info,
            case_info=case_info,
            root_causes=normalize_components(root_causes),
            reward=reward,
            next_query_info_id=next_query_info_id if next_query_info_id else "",
            done=done,
            success_count=int((state_counts or {}).get("success_count", 0)),
            failure_count=int((state_counts or {}).get("failure_count", 0)),
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
        
        # Parse next_query_info_id: "round-query_id"
        # Since both case_id and next_query_info_id use fixed-length format (round-query_id with zfill),
        # we can use direct string comparison
        try:
            candidate_cases = [case.case_info for case in self.memory_manager.cases.values() if case.id < next_query_info_id]
        except Exception:
            # If anything fails, return all cases
            candidate_cases = [case.case_info for case in self.memory_manager.cases.values()]
        
        # Random uniform sampling if number of cases exceeds max_sample_cases
        if len(candidate_cases) > self.max_sample_cases:
            sampled_cases = random.sample(candidate_cases, self.max_sample_cases)
            return sampled_cases
        else:
            return candidate_cases

    def _compute_soft_bellman_target(self, next_query_info, next_root_causes, next_memory_cases, done):
        """
        Compute soft Bellman target: y_t = r_t + γ * α * log(Σ_{o' in M_{t+1}} exp(Q_θ(s_{t+1}, M_{t+1}, o')/α))
        
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
        
        # Compute Q-values for all observations in next memory
        q_values_next = []
        for case_info in next_memory_cases:
            with torch.no_grad():
                features_next = self._compute_features(
                    next_query_info,
                    case_info,
                    next_root_causes,
                    {"success_count": 0, "failure_count": 0},
                )
                features_next = features_next.unsqueeze(0)
                q_value = self.target_network(features_next).squeeze()
                q_values_next.append(q_value)
        
        if len(q_values_next) == 0:
            return torch.tensor(0.0, device=self.device)
        
        # Stack Q-values: [num_cases]
        q_values_tensor = torch.stack(q_values_next)  # [num_cases]
        
        # Compute soft Bellman backup: α * log(Σ exp(Q/α))
        # For numerical stability, use log-sum-exp trick
        q_values_scaled = q_values_tensor / self.alpha
        q_max = torch.max(q_values_scaled)
        log_sum_exp = q_max + torch.log(torch.sum(torch.exp(q_values_scaled - q_max)) + 1e-12)
        soft_value = self.alpha * log_sum_exp
        
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
                        False  # Not done since we have next state
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
        
        # Parse next_query_info_id: "round-query_id"
        try:
            parts = next_query_info_id.split("-")
            if len(parts) != 2:
                return None, None
            
            target_round = int(parts[0])
            target_query_id = parts[1]
            
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

    def save_model(self, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        torch.save(self.network.state_dict(), save_path)
        print(f"SoftQRetriever model saved to: {save_path}")

    def load_model(self, load_path):
        state_dict = torch.load(load_path, map_location=self.device)

        self.network.load_state_dict(state_dict)
        self.target_network.load_state_dict(state_dict)

        print(f"SoftQRetriever model loaded from: {load_path}")
