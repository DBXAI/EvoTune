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
    reward: int  # 0 或 1


class SoftQRetriever:
    """基于软Q学习的案例检索器"""

    def __init__(self, memory_manager=None):
        self.device = "cpu"

        # 网络参数
        self.embedding_dim = 133  # 更新为133维
        self.hidden_dim = 512
        self.output_dim = 1
        self.temperature = 1.0

        # 经验池参数
        self.experience_pool_size = 10000
        self.batch_size = 2
        self.learning_rate = 1e-4

        # 存储memory_manager引用
        self.memory_manager = memory_manager

        # 初始化网络
        self._init_networks()

        # 经验池
        self.experience_pool = deque(maxlen=self.experience_pool_size)

        # 优化器
        self.optimizer = torch.optim.Adam(
            self.network.parameters(), lr=self.learning_rate
        )

        # 统计信息
        self.total_experiences = 0
        self.update_count = 0

    def _init_networks(self):
        # 主网络：两层MLP + Sigmoid（输出0-1之间的概率）
        self.network = nn.Sequential(
            nn.Linear(
                self.embedding_dim, self.hidden_dim
            ),  # query_emb(32) + case_emb(64+32) + root_causes(4) + approve_time(1) = 133
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim // 2, self.output_dim),
            nn.Sigmoid(),  # 输出0-1之间的概率
        ).to(self.device)

        # 目标网络（用于稳定训练）
        self.target_network = nn.Sequential(
            nn.Linear(self.embedding_dim, self.hidden_dim),  # 133维输入
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim // 2, self.output_dim),
            nn.Sigmoid(),  # 输出0-1之间的概率
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
        approve_time_tensor = torch.tensor([approve_time], dtype=torch.float32).to(
            self.device
        )

        features = torch.cat(
            [query_embedding, case_embedding, root_cause_vector, approve_time_tensor]
        )

        return features

    def predict_case_value(self, query_info, case_info, root_causes):
        with torch.no_grad():
            features = self._compute_features(query_info, case_info, root_causes)
            features = features.unsqueeze(0)

            probabilities = self.network(features)
            value = probabilities.item()

            return value

    def select_cases(self, query_info, candidate_cases, root_causes, top_k=4):
        """选择TopK个案例"""
        if not candidate_cases:
            return []

        # 计算每个案例的价值
        case_values = []
        for case_info in candidate_cases:
            value = self.predict_case_value(query_info, case_info, root_causes)
            case_values.append((case_info, value))

        # 按价值排序
        case_values.sort(key=lambda x: x[1], reverse=True)

        # 选择TopK个
        selected_cases = case_values[:top_k]

        # 返回案例和概率（网络已经输出softmax概率）
        result = []
        for case_info, probability in selected_cases:
            # if probability > 0.5:
            result.append((case_info, probability))

        return result

    def add_experience(self, query_info, case_info, reward):
        experience = Experience(
            query_info=query_info, case_info=case_info, reward=reward
        )

        self.experience_pool.append(experience)
        self.total_experiences += 1

    def update_network(self):
        """更新网络参数"""
        if len(self.experience_pool) < self.batch_size:
            return

        # 随机采样经验
        batch_experiences = random.sample(self.experience_pool, self.batch_size)

        # 准备训练数据
        features_list = []
        rewards_list = []

        for experience in batch_experiences:
            # 计算特征
            features = self._compute_features(
                experience.query_info,
                experience.case_info,
                experience.case_info.get("root_causes"),
            )
            features_list.append(features)
            rewards_list.append(experience.reward)

        # 转换为张量
        features_tensor = torch.stack(features_list)
        rewards_tensor = torch.tensor(rewards_list, dtype=torch.float32).to(self.device)

        # 前向传播
        predictions = self.network(features_tensor).squeeze()

        # 计算损失
        loss = F.mse_loss(predictions, rewards_tensor)

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1

        # 定期更新目标网络
        if self.update_count % 10 == 0:
            self.target_network.load_state_dict(self.network.state_dict())
