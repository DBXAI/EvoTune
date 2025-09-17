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
import numpy as np
from numpy.linalg import norm

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

    # def to_dict(self) -> Dict[str, Any]:
    #     return asdict(self)

    # @classmethod
    # def from_dict(cls, data: Dict[str, Any]) -> "MemoryCase":
    #     return cls(**data)

class MemoryManager:
    def __init__(self, db_path: str = "memory.db", archive_size: int = 20, root_cause_types: List[str] = None):
        self.db_path = db_path
        self.archive_size = archive_size
        self.cases: Dict[str, MemoryCase] = {}
        self.archive: Set[str] = set()
        self.best_case_id: Optional[str] = None

        # root cause岛屿模型
        self.root_cause_types = root_cause_types or [
            "missing indexes", "suboptimal plan optimizer", "poorly written queries", "inappropriate query knobs"
        ]
        # 假设有15种root cause，支持多根因组合
        self.island_keys = self._generate_island_keys(self.root_cause_types)
        self.islands: Dict[frozenset, Set[str]] = {key: set() for key in self.island_keys}

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
        # 保存前校验根因与动作/SQL是否匹配，并路由到正确岛屿
        self._validate_and_route_root_causes(case)
        # 存入cases并维护结构
        self.cases[case.id] = case
        self._enforce_archive_limit()

        self._update_archive(case)
        self._update_best_case(case)
        # 将案例加入对应island
        root_causes_key = self._normalize_root_causes(case.case_info.get("root_causes"))
        if root_causes_key in self.islands:
            self.islands[root_causes_key].add(case.id)
        else:
            self.islands[root_causes_key] = set([case.id])

        self.save_to_memory(case)
        logger.debug(f"Added case {case.id} to island {root_causes_key}")
        return case.id
    
    def save_to_memory(self, new_case: MemoryCase = None):
        if new_case is not None:
            # 增量保存：只添加新案例
            try:
                # 尝试读取现有数据
                with open(self.db_path, 'r') as f:
                    data_to_save = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                # 文件不存在或格式错误，创建新的数据结构
                data_to_save = {}
            
            # 添加新案例
            data_to_save[new_case.id] = new_case.case_info
            
            # 写入文件
            with open(self.db_path, 'w') as f:
                json.dump(data_to_save, f)
        else:
            # 全量保存：保存所有案例（用于兼容性）
            with open(self.db_path, 'w') as f:
                data_to_save = {}
                for cid, case in self.cases.items():
                    data_to_save[cid] = case.case_info
                json.dump(data_to_save, f)

    def save_case(self, query_info, fix_action, root_causes, old_time, new_time, label):
        # 计算 approve_time
        approve_time = old_time - new_time

        # 组装case_info，只存储必要信息
        case_info = {
            "query_info": query_info.__dict__,
            "fix_action": fix_action,
            "root_causes": root_causes,
            "old_time": old_time,
            "new_time": new_time,
            "label": label,
            "approve_time": approve_time
        }
        
        # 创建临时MemoryCase用于计算和存储
        case = MemoryCase(
            id=str(len(self.cases) + 1),
            case_info=case_info,
            metrics={},  # 将在load时重新计算
            sql_embedding=None,  # 将在load时重新计算
            action_embedding=None,  # 将在load时重新计算
        )
        self.add_case(case)

        return case.id

    def get_case(self, case_id: str) -> Optional[MemoryCase]:
        return self.cases.get(case_id)

    # 将 id 转为 MemoryCase
    def ids_to_cases(self, ids: List[str]) -> List[MemoryCase]:
        return [self.cases[cid] for cid in ids]

    # def sample_case(self, exploration_ratio=0.4, exploitation_ratio=0.4) -> MemoryCase:
    #     rand_val = random.random()
    #     if rand_val < exploration_ratio:
    #         # 探索：随机采样
    #         return random.choice(list(self.cases.values()))
    #     elif rand_val < exploration_ratio + exploitation_ratio and self.archive:
    #         # 利用：从精英档案采样
    #         return self.cases[random.choice(list(self.archive))]
    #     else:
    #         # 随机采样
    #         return random.choice(list(self.cases.values()))

    # def get_top_cases(self, n: int = 5, metric: str = "approve_time") -> List[MemoryCase]:
    #     sorted_cases = sorted(self.cases.values(), key=lambda c: c.metrics.get(metric, 0.0), reverse=True)
    #     return sorted_cases[:n]

    def load(self):
        if os.path.exists(self.db_path):
            with open(self.db_path, 'r') as f:
                data = json.load(f)
                for cid, case_info in data.items():
                    # 从case_info重新构建MemoryCase
                    case = self._build_memory_case_from_info(cid, case_info)
                    self.cases[cid] = case
                    
                    # 将case添加到对应的island中
                    root_causes = case.case_info.get("root_causes")
                    if root_causes:
                        island_key = self._normalize_root_causes(root_causes)
                        self.islands[island_key].add(cid)
                # 依据正样例与approve_time重建archive（取前archive_size个）
                positive_cases = [c for c in self.cases.values() if c.case_info.get("label") == "positive"]
                positive_cases.sort(key=lambda c: c.metrics.get("approve_time", 0.0), reverse=True)
                self.archive = set([c.id for c in positive_cases[: self.archive_size]])

    def _build_memory_case_from_info(self, case_id: str, case_info: dict) -> MemoryCase:
        """从case_info重新构建完整的MemoryCase"""
        # 计算embedding
        raw_info = case_info["query_info"]
        # 确保is_rewrite字段存在，默认为False
        if "is_rewrite" not in raw_info:
            raw_info["is_rewrite"] = False
        query_info = QueryInfo(**raw_info)
        sql_embedding = self.build_sql_embedding(query_info)
        action_embedding = self.build_action_embedding(str(case_info["fix_action"]))
        
        # 计算metrics
        approve_time = case_info.get("approve_time", 0.0)
        new_time = case_info.get("new_time", 0.0)
        metrics = {
            "performance": new_time,
            "approve_time": approve_time
        }
        
        return MemoryCase(
            id=case_id,
            case_info=case_info,
            metrics=metrics,
            sql_embedding=sql_embedding,
            action_embedding=action_embedding,
        )

    # def _calculate_feature_coords(self, case: MemoryCase) -> List[int]:
    #     # 以SQL embedding、action embedding、score为特征分箱
    #     # 假设embedding已为一维均值降维，score为第三维（要修改）
    #     sql_emb = case.sql_embedding or [0.0]
    #     action_emb = case.action_embedding or [0.0]
    #     # 降维：取均值
    #     sql_val = sum(sql_emb) / len(sql_emb)
    #     action_val = sum(action_emb) / len(action_emb)
    #     # score - 使用approve_time
    #     score = case.metrics.get("approve_time", 0.0)
    #     # 分箱
    #     sql_bin = int(sql_val * self.feature_bins) % self.feature_bins
    #     action_bin = int(action_val * self.feature_bins) % self.feature_bins
    #     # score分箱（假设性能提升最大为10s，最小为-10s，可自定义）
    #     score_norm = max(min((score + 10) / 20, 1.0), 0.0)  # 归一化到0-1
    #     score_bin = int(score_norm * self.feature_bins)
    #     return [sql_bin, action_bin, score_bin]

    def _feature_coords_to_key(self, coords: List[int]) -> str:
        return "-".join(str(c) for c in coords)

    def _is_better(self, case1: MemoryCase, case2: MemoryCase) -> bool:
        # 以approve_time为主（性能提升越大越好）
        return case1.metrics.get("approve_time", 0.0) > case2.metrics.get("approve_time", 0.0)

    # 分别选出同 SQL 与不同 SQL 的提升最大的1个
    def best_by_approve(self, cases: List[MemoryCase]) -> Optional[MemoryCase]:
        if not cases:
            return None
        return max(cases, key=lambda c: c.case_info.get("approve_time") or c.metrics.get("approve_time", 0.0))

    def _update_archive(self, case: MemoryCase):
        # 仅正样例才进入archive
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

    def _update_best_case(self, case: MemoryCase):
        # 仅正样例才允许成为best_case
        if case.case_info.get("label") != "positive":
            return

        if self.best_case_id is None or self._is_better(case, self.cases[self.best_case_id]):
            self.best_case_id = case.id

    def _enforce_archive_limit(self):
        if len(self.archive) > self.archive_size:
            # 获取archive中的案例，按性能排序
            archive_cases = [self.cases[case_id] for case_id in self.archive if case_id in self.cases]
            sorted_archive_cases = sorted(archive_cases, key=lambda c: c.metrics.get("approve_time"))
            
            # 从archive中移除性能最差的案例
            to_remove_count = len(self.archive) - self.archive_size
            to_remove_cases = sorted_archive_cases[:to_remove_count]
            
            for case in to_remove_cases:
                if case.id in self.archive:
                    self.archive.remove(case.id)

    # def set_current_island(self, island_key: frozenset):
    #     self.current_island = island_key
    #     logger.debug(f"Switched to island {island_key}")

    # def next_island(self) -> frozenset:
    #     keys = list(self.islands.keys())
    #     idx = keys.index(self.current_island)
    #     self.current_island = keys[(idx + 1) % len(keys)]
    #     logger.debug(f"Advanced to island {self.current_island}")
    #     return self.current_island

    # def increment_island_generation(self, island_key: Optional[frozenset] = None):
    #     key = island_key if island_key is not None else self.current_island
    #     self.island_generations[key] += 1
    #     logger.debug(f"Island {key} generation incremented to {self.island_generations[key]}")

    # def should_migrate(self) -> bool:
    #     max_gen = max(self.island_generations.values())
    #     return (max_gen - self.last_migration_generation) >= self.migration_interval

    # def migrate_cases(self):
    #     # 迁移高分case到相邻island（环状拓扑）
    #     keys = list(self.islands.keys())
    #     for i, key in enumerate(keys):
    #         island = self.islands[key]
    #         if not island:
    #             continue
    #         island_cases = [self.cases[pid] for pid in island if pid in self.cases]
    #         if not island_cases:
    #             continue
    #         island_cases.sort(key=lambda c: c.metrics.get("approve_time", 0.0), reverse=True)
    #         num_to_migrate = max(1, int(len(island_cases) * self.migration_rate))
    #         migrants = island_cases[:num_to_migrate]
    #         # 环状相邻island
    #         target_islands = [keys[(i + 1) % len(keys)], keys[(i - 1) % len(keys)]]
    #         for migrant in migrants:
    #             for target in target_islands:
    #                 migrant_copy = MemoryCase(
    #                     id=f"{migrant.id}_migrant_{'_'.join(sorted(target))}",
    #                     case_info=migrant.case_info,
    #                     metrics=migrant.metrics.copy(),
    #                     sql_embedding=migrant.sql_embedding,
    #                     action_embedding=migrant.action_embedding,
    #                     performance_bin=migrant.performance_bin,
    #                 )
    #                 self.cases[migrant_copy.id] = migrant_copy
    #                 self.islands[target].add(migrant_copy.id)
    #                 self.island_generations[target] += 1
    #                 logger.debug(f"Migrated case {migrant.id} from island {key} to island {target}")
    #     self.last_migration_generation = max(self.island_generations.values())
    #     logger.info(f"Migration completed at generation {self.last_migration_generation}")

    def retrieve_cases(self, query_info: QueryInfo, root_causes: Any, pos_n: int = 2, neg_n: int = 2, mode: str = "exploit") -> Dict[str, Any]:
        """
        使用 islands 构造样本池并按标签输出：
        - exploitation pool: 目标根因对应的岛
        - exploration pool: 其他所有岛合并
        Positive/Negative 划分依据 case_info.label（缺失时回退到 approve_time>0 视为正例）。
        Positive 返回规则：至少包含同 SQL 与不同 SQL 的 approve_time 最大各 1，其余按提升降序填充至 pos_n。
        Negative 返回规则：从负例池随机采样最多 neg_n 个。
        返回：{"mode": str, "positive": [case_info], "negative": [case_info]}
        """
        query_text = query_info.query.lower().strip()
        target_root = self._normalize_root_causes(root_causes)

        # 从 islands 获取池
        target_island_ids = list(self.islands.get(target_root, set()))
        other_island_ids = []
        for island_key, ids in self.islands.items():
            if island_key != target_root:
                other_island_ids.extend(list(ids))

        # exploitation 池：同岛全部（内部同/异 SQL 信息仅用于后续挑选）
        exploit_pool = self.ids_to_cases(target_island_ids)
        explore_pool = self.ids_to_cases(other_island_ids)

        # ε-greedy 选择模式
        chosen_pool = explore_pool if mode == "explore" else exploit_pool

        # 在 exploit 池内优先同 SQL 的案例
        def split_by_sql(cases: List[MemoryCase]) -> Tuple[List[MemoryCase], List[MemoryCase]]:
            same_sql_list = []
            diff_sql_list = []
            for c in cases:
                stored_query = c.case_info.get("query_info").get("query")
                if stored_query.lower().strip() == query_text:
                    same_sql_list.append(c)
                else:
                    diff_sql_list.append(c)
            return same_sql_list, diff_sql_list

        # 基于 label/approve_time 划分正负并组装结果
        same_sql_pool, diff_sql_pool = split_by_sql(chosen_pool)

        # 正例池与负例池
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

        # 其余正例按 approve_time 降序补齐
        remaining_pos = [c for c in (pos_same + pos_diff) if c not in selected_pos]
        remaining_pos_sorted = sorted(
            remaining_pos,
            key=lambda c: c.case_info.get("approve_time") or c.metrics.get("approve_time", 0.0),
            reverse=True,
        )
        selected_pos.extend(remaining_pos_sorted)
        selected_pos = selected_pos[:max(1, pos_n)]

        # 负例随机采样
        if len(neg_all) > neg_n:
            selected_neg = random.sample(neg_all, neg_n)
        else:
            selected_neg = neg_all

        # 导出 case_info
        pos_infos = [c.case_info for c in selected_pos]
        neg_infos = [c.case_info for c in selected_neg]

        return {"mode": mode, "positive": pos_infos, "negative": neg_infos}

    def retrieve_embedding(self, query_info, top_n=5, metric='cosine', case_ids=None):
        """
        基于embedding相似度，返回与query_info最相近的top-N历史case。
        可选：仅在给定的 case_ids 候选集合中检索（例如仅在 archive 内）。
        支持欧氏距离和余弦相似度。
        """
        # 1. 获取当前query的embedding
        query_emb = np.array(self.build_sql_embedding(query_info))
        if norm(query_emb) == 0:
            return []

        # 2. 计算候选集合的embedding相似度（若不提供则遍历所有cases）
        case_sims = []
        if case_ids is not None:
            candidates = [self.get_case(cid) for cid in case_ids]
        else:
            candidates = list(self.cases.values())
        for case in candidates:
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
        # 延迟加载RCRankPredictor，避免重复初始化
        if not hasattr(self, '_rcrank_predictor'):
            from agent.plan.online_predict import RCRankPredictor
            self._rcrank_predictor = RCRankPredictor(device='cpu')  # 可根据实际情况设为cuda
        predictor = self._rcrank_predictor

        from agent.plan.RCRank.utils.plan_encoding import PlanEncoder
        from agent.plan.RCRank.model.modules.QueryFormer.utils import Encoding
        import pandas as pd
        encoding = Encoding(None, {'NA': 0})

        query = query_info.query
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

    def _normalize_root_causes(self, root_causes: Any) -> frozenset:
        if root_causes is None:
            return frozenset()
        if isinstance(root_causes, str):
            return frozenset([root_causes])
        try:
            return frozenset(root_causes)
        except Exception:
            return frozenset([str(root_causes)])

    def _infer_root_causes_from_action(self, fix_action, query_info) -> frozenset:
        """基于动作文本的简单启发式推断根因类别集合。"""
        fix_action = str(fix_action).lower()
        query_text = query_info.query.lower().strip()
        inferred: Set[str] = set()
        if "index" in fix_action:
            inferred.add("missing indexes")
        if ("set" in fix_action) and ("=" in fix_action):
            inferred.add("inappropriate query knobs")
        if ("/*+" in query_text) or ("*/" in query_text):
            inferred.add("suboptimal plan optimizer")
        if query_info.is_rewrite:
            inferred.add("poorly written queries")
        return frozenset(inferred) if inferred else frozenset()

    def _validate_and_route_root_causes(self, case: MemoryCase) -> None:
        """在写入前校验动作/SQL与根因匹配，不匹配则修正根因并路由至对应岛屿。"""
        declared = self._normalize_root_causes(case.case_info.get("root_causes"))
        raw_info = case.case_info.get("query_info")
        query_info = QueryInfo(**raw_info)
        inferred = self._infer_root_causes_from_action(case.case_info.get("fix_action"), query_info)
        # 若声明为空而推断有值，或两者不一致，则以推断为准（非空优先）
        if inferred and (inferred != declared):
            case.case_info["root_causes"] = list(inferred)
        # 确保对应岛存在
        key = self._normalize_root_causes(case.case_info.get("root_causes"))
        if key not in self.islands:
            self.islands[key] = set()
            self.island_generations[key] = 0

    def explore_root_from_archive(self, query_info: QueryInfo, exclude_roots: set()) -> Optional[Any]:
        """
        从 archive 中检索最相似 SQL 的根因，用于根因切换。
        返回最相似 SQL 案例的根因，如果没有找到则返回 None。
        
        Args:
            query_info: 查询信息
            exclude_roots: 要排除的根因集合（已失败的根因），元素为字符串化后的根因表示
        """
        if not self.archive:
            return None
        
        query_text = query_info.query.lower().strip()
        exclude_roots = exclude_roots
        
        # 从 archive 中获取案例
        archive_cases = [self.cases[aid] for aid in self.archive]
        if not archive_cases:
            return None
        
        # 计算相似度并排序：仅在 archive 候选集合中检索
        # 要修改
        similar_cases = self.retrieve_embedding(query_info, top_n=len(archive_cases), case_ids=list(self.archive))
        
        # 找到最相似且未被排除的案例
        for case in similar_cases:
            # 使用规范化后的根因集合，并以字符串形式与 exclude_roots 对齐
            case_root_str = str(self._normalize_root_causes(case.case_info.get("root_causes")))
            if case_root_str not in exclude_roots:
                return case.case_info.get("root_causes")
        
        return None