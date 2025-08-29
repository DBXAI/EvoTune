import json
import logging
import os
import random
import time
import itertools
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Set, Tuple
from datetime import datetime
from utils.types import QueryInfo, CaseInfo

logger = logging.getLogger(__name__)

# 确保logger继承根logger的配置
logger.setLevel(logging.INFO)
logger.propagate = True

@dataclass
class MemoryCase:
    """表示知识库中的一个案例"""
    id: str
    case_info: CaseInfo
    timestamp: float = field(default_factory=time.time)
    metrics: Dict[str, float] = field(default_factory=dict)  # 如性能提升、置信度等
    sql_embedding: Optional[List[float]] = None
    action_embedding: Optional[List[float]] = None
    performance_bin: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryCase":
        return cls(**data)

class MemoryManager:
    def __init__(self, db_path: str = "memory.db", feature_bins: int = 10, archive_size: int = 20, root_cause_types: List[str] = None):
        self.db_path = db_path
        self.feature_bins = feature_bins
        self.archive_size = archive_size
        self.cases: Dict[str, MemoryCase] = {}
        self.feature_map: Dict[str, str] = {}
        self.archive: Set[str] = set()
        self.best_case_id: Optional[str] = None
        self.last_id: int = 0

        # root cause岛屿模型
        self.root_cause_types = root_cause_types or [
            "missing indexes", "suboptimal plan optimizer", "poorly written queries", "inappropriate query knobs"
        ]
        # 假设有15种root cause，支持多根因组合
        self.island_keys = self._generate_island_keys(self.root_cause_types)
        self.islands: Dict[frozenset, Set[str]] = {key: set() for key in self.island_keys}
        self.current_island: frozenset = next(iter(self.islands))
        self.island_generations: Dict[frozenset, int] = {key: 0 for key in self.island_keys}
        self.migration_interval = 10
        self.migration_rate = 0.1
        self.last_migration_generation = 0
        self._init_database()
        self.load()

    def _init_database(self):
        if not os.path.exists(self.db_path):
            with open(self.db_path, 'w') as f:
                json.dump({}, f)

    def _generate_island_keys(self, root_cause_types: List[str]) -> List[frozenset]:
        keys = []
        for i in range(1, len(root_cause_types) + 1):
            keys.extend([frozenset(pair) for pair in itertools.combinations(root_cause_types, i)])
        return keys

    def add_case(self, case: MemoryCase) -> str:
        self.cases[case.id] = case
        self._enforce_archive_limit()
        feature_key = self._feature_coords_to_key(self._calculate_feature_coords(case))
        if feature_key not in self.feature_map or self._is_better(case, self.cases[self.feature_map[feature_key]]):
            self.feature_map[feature_key] = case.id
        self._update_archive(case)
        self._update_best_case(case)
        # 分配到island
        if isinstance(case.case_info.get("root_causes"), str):
            root_causes = frozenset([case.case_info.get("root_causes")])
        else:
            root_causes = frozenset(case.case_info.get("root_causes"))
        
        if root_causes in self.islands:
            self.islands[root_causes].add(case.id)
        else:
            # 新组合自动建island
            self.islands[root_causes] = set([case.id])
            self.island_generations[root_causes] = 0
        self.save_to_memory()
        logger.debug(f"Added case {case.id} to island {root_causes}")
        if self.should_migrate():
            self.migrate_cases()
        return case.id
    
    def save_to_memory(self):
        with open(self.db_path, 'w') as f:
            json.dump({cid: case.to_dict() for cid, case in self.cases.items()}, f)

    def save_case(self, query_info, fix_action, root_causes, old_time, new_time, label):
        # 计算embedding
        sql_embedding = self.build_sql_embedding(query_info)
        action_embedding = self.build_action_embedding(str(fix_action))
        score = old_time - new_time

        # 组装case
        case_info = {
            "query_info": query_info,
            "fix_action": fix_action,
            "root_causes": root_causes,
            "old_time": old_time,
            "new_time": new_time,
            "label": label,
            "result_metrics": {"performance": new_time, "approve_time": score}
        }
        case = MemoryCase(
            id=len(self.cases) + 1,
            case_info=case_info,
            metrics=case_info.get("result_metrics", {}),
            sql_embedding=sql_embedding,
            action_embedding=action_embedding,
            performance_bin=int(score * self.feature_bins),
        )
        self.add_case(case)

    def get_case(self, case_id: str) -> Optional[MemoryCase]:
        return self.cases.get(case_id)

    def sample_case(self, exploration_ratio=0.4, exploitation_ratio=0.4) -> MemoryCase:
        rand_val = random.random()
        if rand_val < exploration_ratio:
            # 探索：随机采样
            return random.choice(list(self.cases.values()))
        elif rand_val < exploration_ratio + exploitation_ratio and self.archive:
            # 利用：从精英档案采样
            return self.cases[random.choice(list(self.archive))]
        else:
            # 随机采样
            return random.choice(list(self.cases.values()))

    def get_top_cases(self, n: int = 5, metric: str = "performance") -> List[MemoryCase]:
        sorted_cases = sorted(self.cases.values(), key=lambda c: c.metrics.get(metric, 0.0), reverse=True)
        return sorted_cases[:n]

    def load(self):
        if os.path.exists(self.db_path):
            with open(self.db_path, 'r') as f:
                data = json.load(f)
                for cid, cdict in data.items():
                    self.cases[cid] = MemoryCase.from_dict(cdict)

    def _calculate_feature_coords(self, case: MemoryCase) -> List[int]:
        # 以SQL embedding、action embedding、score为特征分箱
        # 假设embedding已为一维均值降维，score为第三维（要修改）
        sql_emb = case.sql_embedding or [0.0]
        action_emb = case.action_embedding or [0.0]
        # 降维：取均值
        sql_val = sum(sql_emb) / len(sql_emb)
        action_val = sum(action_emb) / len(action_emb)
        # score
        score = case.metrics.get("performance", 0.0)
        # 分箱
        sql_bin = int(sql_val * self.feature_bins) % self.feature_bins
        action_bin = int(action_val * self.feature_bins) % self.feature_bins
        # score分箱（假设性能提升最大为10s，最小为-10s，可自定义）
        score_norm = max(min((score + 10) / 20, 1.0), 0.0)  # 归一化到0-1
        score_bin = int(score_norm * self.feature_bins)
        return [sql_bin, action_bin, score_bin]

    def _feature_coords_to_key(self, coords: List[int]) -> str:
        return "-".join(str(c) for c in coords)

    def _is_better(self, case1: MemoryCase, case2: MemoryCase) -> bool:
        # 以performance为主
        return case1.metrics.get("performance", 0.0) > case2.metrics.get("performance", 0.0)

    def _update_archive(self, case: MemoryCase):
        if len(self.archive) < self.archive_size:
            self.archive.add(case.id)
            return
        archive_cases = [self.cases[aid] for aid in self.archive]
        worst_case = min(archive_cases, key=lambda c: c.metrics.get("performance", 0.0))
        if self._is_better(case, worst_case):
            self.archive.remove(worst_case.id)
            self.archive.add(case.id)

    def _update_best_case(self, case: MemoryCase):
        if self.best_case_id is None or self._is_better(case, self.cases[self.best_case_id]):
            self.best_case_id = case.id
            logger.info(f"New best case: {case.id}")

    def _enforce_archive_limit(self):
        # 保证archive和cases数量不超过上限
        if len(self.cases) > self.archive_size * 5:
            sorted_cases = sorted(self.cases.values(), key=lambda c: c.metrics.get("performance", 0.0))
            to_remove = sorted_cases[:len(self.cases) - self.archive_size * 5]
            for case in to_remove:
                if case.id in self.archive:
                    self.archive.remove(case.id)
                del self.cases[case.id]

    def set_current_island(self, island_key: frozenset):
        self.current_island = island_key
        logger.debug(f"Switched to island {island_key}")

    def next_island(self) -> frozenset:
        keys = list(self.islands.keys())
        idx = keys.index(self.current_island)
        self.current_island = keys[(idx + 1) % len(keys)]
        logger.debug(f"Advanced to island {self.current_island}")
        return self.current_island

    def increment_island_generation(self, island_key: Optional[frozenset] = None):
        key = island_key if island_key is not None else self.current_island
        self.island_generations[key] += 1
        logger.debug(f"Island {key} generation incremented to {self.island_generations[key]}")

    def should_migrate(self) -> bool:
        max_gen = max(self.island_generations.values())
        return (max_gen - self.last_migration_generation) >= self.migration_interval

    def migrate_cases(self):
        # 迁移高分case到相邻island（环状拓扑）
        keys = list(self.islands.keys())
        for i, key in enumerate(keys):
            island = self.islands[key]
            if not island:
                continue
            island_cases = [self.cases[pid] for pid in island if pid in self.cases]
            if not island_cases:
                continue
            island_cases.sort(key=lambda c: c.metrics.get("performance", 0.0), reverse=True)
            num_to_migrate = max(1, int(len(island_cases) * self.migration_rate))
            migrants = island_cases[:num_to_migrate]
            # 环状相邻island
            target_islands = [keys[(i + 1) % len(keys)], keys[(i - 1) % len(keys)]]
            for migrant in migrants:
                for target in target_islands:
                    migrant_copy = MemoryCase(
                        id=f"{migrant.id}_migrant_{'_'.join(sorted(target))}",
                        case_info=migrant.case_info,
                        metrics=migrant.metrics.copy(),
                        sql_embedding=migrant.sql_embedding,
                        action_embedding=migrant.action_embedding,
                        performance_bin=migrant.performance_bin,
                    )
                    self.cases[migrant_copy.id] = migrant_copy
                    self.islands[target].add(migrant_copy.id)
                    self.island_generations[target] += 1
                    logger.debug(f"Migrated case {migrant.id} from island {key} to island {target}")
        self.last_migration_generation = max(self.island_generations.values())
        logger.info(f"Migration completed at generation {self.last_migration_generation}")

    def retrieve_cases(self, query_info: QueryInfo, n: int = 5, exploration_ratio=0.4) -> Tuple[Optional[MemoryCase], List[MemoryCase]]:
        """
        采样本SQL的修复历史（parent），以及相似SQL的修复历史（inspirations）。
        inspirations优先顺序：最优、提升最大、随机。
        """
        query_text = query_info.query.lower().strip()
        
        # 1. 本SQL修复历史
        same_sql_cases = []
        for case in self.cases.values():
            stored_query_info = case.case_info["query_info"]
            # 处理stored_query_info可能是字典或QueryInfo对象的情况
            if isinstance(stored_query_info, dict):
                stored_query = stored_query_info["query"]
            else:
                stored_query = stored_query_info.query
            if stored_query.lower().strip() == query_text:
                same_sql_cases.append(case)
        
        history_cases = []
        if same_sql_cases:
            rand_val = random.random()
            if rand_val < exploration_ratio:
                # 随机采样本SQL的修复历史
                history_cases.append(random.choice(same_sql_cases))
            else:
                # 采样本SQL最优的修复历史
                history_cases.append(max(same_sql_cases, key=lambda c: c.metrics.get("performance", 0.0)))
        
        # 2. 相似SQL修复历史
        inspirations = []
        similar_cases = self.retrieve_embedding(query_info, n)

        if similar_cases:
            # 最优的相似SQL修复历史
            inspirations.append(similar_cases[0])

            # 提升最大的历史（approve_time最大）
            best_approve = max(similar_cases, key=lambda c: c.metrics.get("approve_time", 0.0))
            inspirations.append(best_approve)

        # 2.3 随机补足
        remaining = n - len(inspirations)
        if remaining > 0:
            random_pool = [c for c in self.cases.values() if c.id not in [x.id for x in inspirations]]
            if random_pool:
                random_insp = random.sample(random_pool, min(remaining, len(random_pool)))
                inspirations.extend(random_insp)

        return history_cases, inspirations[:n]

    def retrieve_embedding(self, query_info, top_n=5, metric='cosine'):
        """
        基于embedding相似度，返回与query_info最相近的top-N历史case。
        支持欧氏距离和余弦相似度。
        """
        import numpy as np
        from numpy.linalg import norm

        # 1. 获取当前query的embedding
        query_emb = np.array(self.build_sql_embedding(query_info))
        if norm(query_emb) == 0:
            return []

        # 2. 计算所有历史case的embedding相似度
        case_sims = []
        for case in self.cases.values():
            if case.sql_embedding is None:
                continue
            case_emb = np.array(case.sql_embedding)
            if metric == 'cosine':
                sim = np.dot(query_emb, case_emb) / (norm(query_emb) * norm(case_emb) + 1e-8)
            elif metric == 'euclidean':
                sim = -norm(query_emb - case_emb)  # 距离越小越相似
            else:
                raise ValueError('Unknown metric')
            case_sims.append((sim, case))
        # 3. 排序并返回top-N
        case_sims.sort(reverse=True, key=lambda x: x[0])
        return [c for _, c in case_sims[:top_n]]

    def build_sql_embedding(self, query_info: QueryInfo):
        """
        使用RCRank异常检测模型的get_embedding接口，根据QueryInfo对象生成高质量embedding。
        """
        import torch
        import numpy as np
        # 延迟加载RCRankPredictor，避免重复初始化
        if not hasattr(self, '_rcrank_predictor'):
            from agent.plan.online_predict import RCRankPredictor
            self._rcrank_predictor = RCRankPredictor(device='cpu')  # 可根据实际情况设为cuda
        predictor = self._rcrank_predictor

        from agent.plan.RCRank.utils.plan_encoding import PlanEncoder
        from agent.plan.RCRank.model.modules.QueryFormer.utils import Encoding
        import pandas as pd
        encoding = Encoding(None, {'NA': 0})

        query = query_info.query if hasattr(query_info, 'query') else ""

        query_plan = query_info.plan_json
        df = pd.DataFrame([{
            'plan_json': json.dumps({'Plan': query_plan}) if isinstance(query_plan, dict) else query_plan
        }])
        plan_encoder = PlanEncoder(df=df, encoding=encoding)
        encoded_df = plan_encoder.df
        plan_tensor = encoded_df['json_plan_tensor'].iloc[0]
        
        # 处理SQL查询（BERT tokenizer 输出为映射）
        sql = predictor.tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=512).to(predictor.device)
        
        # 按照online_predict中的方式准备plan张量（不要额外增加batch维度）
        plan = {
            'x': plan_tensor['x'].to(predictor.device).to(torch.float32),
            'attn_bias': plan_tensor['attn_bias'].to(predictor.device),
            'rel_pos': plan_tensor['rel_pos'].to(predictor.device),
            'heights': plan_tensor['heights'].to(predictor.device)
        }
        
        # 准备时间序列和日志数据，形状与标准化方式对齐online_predict
        timeseries = torch.tensor(query_info.external_metrics)
        if len(timeseries.shape) == 2:
            timeseries = timeseries.unsqueeze(0)
        timeseries = (timeseries - predictor.timeseries_train_mean.unsqueeze(1)) / (predictor.timeseries_train_std.unsqueeze(1) + 1e-6)
        timeseries = timeseries.to(predictor.device)
        
        log = torch.tensor(query_info.internal_metrics)
        if len(log.shape) == 1:
            log = log.unsqueeze(0)
        log = (log - predictor.logs_train_mean) / (predictor.logs_train_std + 1e-6)
        log = log.to(predictor.device)
        
        # 6. 获取embedding
        emb = predictor.model.get_embedding(sql, plan, timeseries, log)
        return emb.squeeze(0).cpu().numpy().tolist()
    
    def build_action_embedding(self, action: str):
        """
        使用BERT对action字符串进行embedding，返回[CLS]向量。
        """
        import torch
        from transformers import BertTokenizer, BertModel
        # 延迟加载BERT模型，避免重复初始化
        if not hasattr(self, '_bert_tokenizer'):
            bert_path = 'agent/plan/RCRank/bert-base-uncased'
            self._bert_tokenizer = BertTokenizer.from_pretrained(bert_path)
            self._bert_model = BertModel.from_pretrained(bert_path)
            self._bert_model.eval()
        tokenizer = self._bert_tokenizer
        model = self._bert_model
        # 编码action
        inputs = tokenizer(action, return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
            cls_emb = outputs.last_hidden_state[:, 0, :]  # 取[CLS]向量
        return cls_emb.squeeze(0).cpu().numpy().tolist()