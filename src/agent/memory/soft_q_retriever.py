import os
import random
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Experience:
    query_info: Dict[str, Any]
    case_info: Dict[str, Any]
    reward: int  # 0 or 1


class SoftQRetriever:
    """Case retriever based on soft Q-learning"""

    def __init__(self, memory_manager=None):
        self.device = "cpu"

        # Network parameters
        self.embedding_dim = 133  # Updated to 133 dimensions
        self.hidden_dim = 512
        self.output_dim = 1
        self.temperature = 1.0

        # Experience pool parameters
        self.batch_size = 2
        self.learning_rate = 1e-4

        # Store memory_manager reference
        self.memory_manager = memory_manager

        # Initialize network
        self._init_networks()

        # Experience pool (unlimited size, as experiences are persisted to file)
        self.experience_pool = deque()

        # Optimizer
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)

        # Statistics
        self.total_experiences = 0
        self.update_count = 0

    def _init_networks(self):
        # Main network: two-layer MLP + Sigmoid (outputs probability between 0-1)
        self.network = nn.Sequential(
            nn.Linear(self.embedding_dim, self.hidden_dim),  # query_emb(32) + case_emb(64+32) + root_causes(4) + approve_time(1) = 133
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim // 2, self.output_dim),
            nn.Sigmoid(),  # Outputs probability between 0-1
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
            nn.Sigmoid(),  # Outputs probability between 0-1
        ).to(self.device)

        self.target_network.load_state_dict(self.network.state_dict())

    def _build_query_embedding(self, query_info) -> torch.Tensor:
        from utils.types import QueryInfo

        query_obj = QueryInfo(**query_info)
        embedding_list = self.memory_manager.build_sql_embedding(query_obj)

        embedding = torch.tensor(embedding_list, dtype=torch.float32)

        return embedding.to(self.device)

    def _build_case_embedding(self, case_info) -> torch.Tensor:
        from utils.types import QueryInfo

        case_query_info = case_info.get("query_info")
        case_query_obj = QueryInfo(**case_query_info)
        query_embedding = self.memory_manager.build_sql_embedding(case_query_obj)

        fix_action = case_info.get("fix_action", "")
        action_embedding = self.memory_manager.build_action_embedding(fix_action)

        combined_embedding = query_embedding + action_embedding

        embedding = torch.tensor(combined_embedding, dtype=torch.float32)

        return embedding.to(self.device)

    def _encode_root_causes(self, root_causes) -> torch.Tensor:
        root_cause_types = [
            "missing indexes",
            "inappropriate query knobs",
            "suboptimal plan optimizer",
            "poorly written queries",
        ]

        root_cause_vector = torch.zeros(4)
        for i, root_cause_type in enumerate(root_cause_types):
            if root_cause_type in root_causes:
                root_cause_vector[i] = 1.0

        return root_cause_vector.to(self.device)

    def _compute_features(self, query_info, case_info, root_causes):
        query_embedding = self._build_query_embedding(query_info)
        case_embedding = self._build_case_embedding(case_info)

        root_cause_vector = self._encode_root_causes(root_causes)

        approve_time = case_info.get("approve_time", 0.0)
        approve_time_tensor = torch.tensor([approve_time], dtype=torch.float32).to(self.device)

        features = torch.cat([query_embedding, case_embedding, root_cause_vector, approve_time_tensor])

        return features

    def predict_case_value(self, query_info, case_info, root_causes):
        with torch.no_grad():
            features = self._compute_features(query_info, case_info, root_causes)
            features = features.unsqueeze(0)

            probabilities = self.network(features)
            value = probabilities.item()

            return value

    def select_cases(self, query_info, candidate_cases, root_causes, top_k=4):
        """Select TopK cases"""
        if not candidate_cases:
            return []

        # Calculate value for each case
        case_values = []
        for case_info in candidate_cases:
            value = self.predict_case_value(query_info, case_info, root_causes)
            case_values.append((case_info, value))

        # Sort by value
        case_values.sort(key=lambda x: x[1], reverse=True)

        # Select TopK
        selected_cases = case_values[:top_k]

        # print the selected cases
        # print(f"Selected cases: {selected_cases}")

        # Return cases and probabilities (network already outputs softmax probability)
        result = []
        for case_info, probability in selected_cases:
            # if probability > 0.5:
            result.append((case_info, probability))

        return result

    def add_experience(self, query_info, case_info, reward):
        experience = Experience(query_info=query_info, case_info=case_info, reward=reward)

        self.experience_pool.append(experience)
        self.total_experiences += 1

    def update_network(self):
        """Update network parameters"""
        if len(self.experience_pool) < self.batch_size:
            return

        # Randomly sample experiences
        batch_experiences = random.sample(self.experience_pool, self.batch_size)

        # Prepare training data
        features_list = []
        rewards_list = []

        for experience in batch_experiences:
            # Compute features
            features = self._compute_features(
                experience.query_info,
                experience.case_info,
                experience.case_info.get("root_causes"),
            )
            features_list.append(features)
            rewards_list.append(experience.reward)

        # Convert to tensors
        features_tensor = torch.stack(features_list)
        rewards_tensor = torch.tensor(rewards_list, dtype=torch.float32).to(self.device)

        # Forward pass
        predictions = self.network(features_tensor).squeeze()

        # Compute loss
        loss = F.mse_loss(predictions, rewards_tensor)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1

        # Periodically update target network
        if self.update_count % 10 == 0:
            self.target_network.load_state_dict(self.network.state_dict())

    def save_model(self, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        torch.save(self.network.state_dict(), save_path)
        print(f"SoftQRetriever model saved to: {save_path}")

    def load_model(self, load_path):
        state_dict = torch.load(load_path, map_location=self.device)

        self.network.load_state_dict(state_dict)
        self.target_network.load_state_dict(state_dict)

        print(f"SoftQRetriever model loaded from: {load_path}")
