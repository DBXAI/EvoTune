import itertools
import json
import logging
import os
import random
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.linalg import norm

# 导入软Q检索器
from agent.memory.soft_q_retriever import SoftQRetriever
from agent.plan.RCRank.utils.load_data import (
    load_dataset_valid as load_dataset_tensor_valid,
)
from utils.types import CaseInfo, QueryInfo

logger = logging.getLogger(__name__)

# 确保logger继承根logger的配置
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


class MemoryManager:
    def __init__(self, config, predictor=None):
        self.db_path = config.get("db_path")
        self.archive_size = config.get("archive_size")
        self.cases = {}
        self.archive = set()
        self.best_case_id = None

        self.enable_finetune = config.get("enable_finetune")
        self.finetune_threshold = config.get("finetune_threshold")
        self.finetune_epochs = config.get("finetune_epochs")
        self.finetune_batch_size = config.get("finetune_batch_size")
        self.finetune_device = config.get("finetune_device")
        self.model_output_dir = config.get("model_output_dir")

        self.bce_weight = config.get("bce_weight")
        self.contrastive_weight = config.get("contrastive_weight")

        self.bert_model_path = config.get("bert_model_path")

        # 检索模式配置：static（静态）或 dynamic（动态）
        self.retrieval_mode = config.get("retrieval_mode", "dynamic")

        # 样本保存配置
        self.enable_save_samples = config.get("enable_save_samples", False)
        self.samples_save_path = config.get("samples_save_path")

        # Retriever 模型保存/加载路径
        self.retriever_model_path = config.get("retriever_model_path")

        self.root_cause_types = [
            "missing indexes",
            "inappropriate query knobs",
            "suboptimal plan optimizer",
            "poorly written queries",
        ]
        # 假设有15种root cause，支持多根因组合
        self.island_keys = self._generate_island_keys(self.root_cause_types)
        self.islands = {key: set() for key in self.island_keys}
        
        # 为 "normal" 根因也创建 island
        normal_island_key = self._normalize_root_causes(["normal"])
        self.islands[normal_island_key] = set()

        self._rcrank_predictor = predictor

        # 初始化软Q检索器
        self.retriever = SoftQRetriever(memory_manager=self)

        # load the saved retriever model
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

            # 统一为字符串化的计划JSON
            if isinstance(plan_json, dict):
                plan_json = json.dumps({"Plan": plan_json})
            external_metrics = raw.get("external_metrics", [])
            internal_metrics = raw.get("internal_metrics", [])

            roots = info.get("root_causes") or []
            roots = roots if isinstance(roots, list) else [roots]
            label_vec = [1 if r in roots else 0 for r in label_space]

            duration = info.get("new_time", 0.0)
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
        bce_weight=0.7,
        contrastive_weight=0.3,
    ):
        # use the hybrid loss (BCE + supervised contrastive learning) to finetune the RCRank model
        os.makedirs(output_dir, exist_ok=True)

        train_dataloader, _, _, train_len, _, _, _ = load_dataset_tensor_valid(csv_path, batch_size=batch_size, device=device)

        if not hasattr(self, "_rcrank_predictor") or self._rcrank_predictor is None:
            from agent.plan.online_predict import RCRankPredictor

            self._rcrank_predictor = RCRankPredictor(device=device, train_data_path=csv_path)
        predictor = self._rcrank_predictor
        model = predictor.model.to(device)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # 重要参数 尝试 1e-4 或 3e-4
        temperature = 0.07

        # BCE loss function
        bce_loss_fn = nn.BCELoss(reduction="mean")

        def enhanced_supcon_loss(emb, labels, case_labels, temperature=0.07):
            """
            增强的监督对比学习损失：
            - 拉近：具有正确根因的样本（positive cases）
            - 推远：具有错误根因的样本（negative cases）

            Args:
                emb: [B, D] - 样本的embedding
                labels: [B, K] - 多标签one-hot向量（根因标签）
                case_labels: [B] - 案例标签（"positive"或"negative"）
                temperature: 温度参数
            """
            emb = F.normalize(emb, dim=1)
            logits = torch.div(torch.matmul(emb, emb.T), temperature)  # [B, B]

            # 掩码：排除自身
            self_mask = torch.eye(logits.size(0), dtype=torch.bool, device=logits.device)

            # 构建正负样本掩码
            batch_size = emb.size(0)
            pos_mask = torch.zeros_like(logits, dtype=torch.bool)
            neg_mask = torch.zeros_like(logits, dtype=torch.bool)

            for i in range(batch_size):
                for j in range(batch_size):
                    if i == j:
                        continue

                    # 正样本：两个都是positive cases且共享根因
                    if case_labels[i] == "positive" and case_labels[j] == "positive" and (labels[i] @ labels[j]) > 0:
                        pos_mask[i, j] = True

                    # 负样本：一个是positive一个是negative，或者都是negative
                    elif (case_labels[i] == "positive" and case_labels[j] == "negative") or (case_labels[i] == "negative" and case_labels[j] == "positive") or (case_labels[i] == "negative" and case_labels[j] == "negative"):
                        neg_mask[i, j] = True

            # 分母掩码：排除自身
            logits_mask = ~self_mask

            # 稳定化
            logits_max, _ = torch.max(logits.masked_fill(~logits_mask, float("-inf")), dim=1, keepdim=True)
            logits = logits - logits_max.detach()
            exp_logits = torch.exp(logits) * logits_mask.float()

            # 计算正样本损失（拉近正确根因）
            pos_loss = 0.0
            pos_count = 0
            for i in range(batch_size):
                if pos_mask[i].any():
                    pos_logits = logits[i][pos_mask[i]]
                    pos_exp_logits = exp_logits[i][pos_mask[i]]
                    pos_loss += -torch.log(pos_exp_logits.sum() / exp_logits[i].sum() + 1e-12)
                    pos_count += 1

            # 计算负样本损失（推远错误根因）
            neg_loss = 0.0
            neg_count = 0
            for i in range(batch_size):
                if neg_mask[i].any():
                    neg_logits = logits[i][neg_mask[i]]
                    neg_exp_logits = exp_logits[i][neg_mask[i]]
                    # 负样本损失：希望负样本的相似度尽可能小
                    neg_loss += torch.log(neg_exp_logits.sum() / exp_logits[i].sum() + 1e-12)
                    neg_count += 1

            # 组合损失
            total_loss = 0.0
            if pos_count > 0:
                total_loss += pos_loss / pos_count
            if neg_count > 0:
                total_loss += neg_loss / neg_count

            return total_loss if total_loss > 0 else torch.tensor(0.0, device=emb.device, requires_grad=True)

        def hybrid_loss(
            emb,
            labels,
            pred_label,
            case_labels,
            bce_weight=0.7,
            contrastive_weight=0.3,
            temperature=0.07,
        ):
            """
            混合损失：L_total = λ₁ * L_BCE + λ₂ * L_Enhanced_Contrastive
            """
            # 1. BCE分类损失（只对positive cases计算，negative cases标签是错误的）
            positive_mask = torch.tensor([label == "positive" for label in case_labels], device=pred_label.device)
            if positive_mask.any():
                # 只对positive cases计算BCE损失
                positive_pred = pred_label[positive_mask]
                positive_labels = labels[positive_mask]
                bce_loss = bce_loss_fn(positive_pred, positive_labels)
            else:
                # 如果没有positive cases，BCE损失为0
                bce_loss = torch.tensor(0.0, device=pred_label.device, requires_grad=True)

            # 2. 增强监督对比学习损失（拉近正确根因，推远错误根因）
            contrastive_loss = enhanced_supcon_loss(emb, labels, case_labels, temperature)

            # 3. 加权组合
            total_loss = bce_weight * bce_loss + contrastive_weight * contrastive_loss

            return total_loss, bce_loss, contrastive_loss

        # 3) 训练循环
        print("start finetune")
        for epoch_idx in range(epochs):
            epoch_total_loss = 0.0
            epoch_bce_loss = 0.0
            epoch_contrastive_loss = 0.0

            print(f"finetune epoch {epoch_idx}")
            for batch in train_dataloader:
                # batch keys: query(list[str]), plan(dict of tensors), log[tensor], timeseries[tensor], multilabel[tensor]
                sql_texts = batch["query"]
                plan = batch["plan"]
                logs = batch["log"].to(device)
                timeseries = batch["timeseries"].to(device)
                labels = batch["multilabel"].to(device).float()

                # 获取case_labels（positive/negative标签）
                # 从batch中获取case_label信息
                if "case_label" in batch:
                    case_labels = batch["case_label"]
                else:
                    # 如果没有case_label列，使用默认值
                    case_labels = ["positive"] * len(sql_texts)

                # 文本编码
                sql = predictor.tokenizer(
                    sql_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                ).to(device)
                # 计划张量准备
                plan = {
                    "x": plan["x"].squeeze(1).to(device).to(torch.float32),
                    "attn_bias": plan["attn_bias"].squeeze(1).to(device),
                    "rel_pos": plan["rel_pos"].squeeze(1).to(device),
                    "heights": plan["heights"].squeeze(1).to(device),
                }

                optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    # 始终使用混合损失
                    emb = model.get_embedding(sql, plan, timeseries, logs)  # [B, D]
                    pred_label, _ = model(sql, plan, timeseries, logs)  # [B, K]
                    total_loss, bce_loss, contrastive_loss = hybrid_loss(
                        emb,
                        labels,
                        pred_label,
                        case_labels,
                        bce_weight,
                        contrastive_weight,
                        temperature,
                    )
                    epoch_bce_loss += bce_loss.item()
                    epoch_contrastive_loss += contrastive_loss.item()

                    total_loss.backward()
                    optimizer.step()
                    epoch_total_loss += total_loss.item()

            # 记录训练日志
            avg_total_loss = epoch_total_loss / max(1, len(train_dataloader))
            avg_bce_loss = epoch_bce_loss / max(1, len(train_dataloader))
            avg_contrastive_loss = epoch_contrastive_loss / max(1, len(train_dataloader))

            print(f"Hybrid finetune epoch {epoch_idx}: total_loss={avg_total_loss:.6f}, " f"bce_loss={avg_bce_loss:.6f}, contrastive_loss={avg_contrastive_loss:.6f}")

        # 4) 保存权重
        best_path = os.path.join(output_dir, "best_model.pt")
        torch.save(model.state_dict(), best_path)
        return best_path

    def _reload_predictor(self, model_path, stats_data_path, device):
        if not hasattr(self, "_rcrank_predictor") or self._rcrank_predictor is None:
            from agent.plan.online_predict import RCRankPredictor

            self._rcrank_predictor = RCRankPredictor(model_path=model_path, train_data_path=stats_data_path, device=device)
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
            self.bce_weight,
            self.contrastive_weight,
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
        logger.debug(f"Added case {case.id} to island {root_causes_key}")

        try:
            self._maybe_finetune()
        except Exception as e:
            print(f"Online finetune failed: {e}")
        return case.id

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

    def save_case(self, query_info, fix_action, root_causes, old_time, new_time, label):
        approve_time = old_time - new_time

        case_info = {
            "query_info": query_info.__dict__,
            "fix_action": fix_action,
            "root_causes": root_causes,
            "old_time": old_time,
            "new_time": new_time,
            "label": label,
            "approve_time": approve_time,
        }

        case = MemoryCase(
            id=str(len(self.cases) + 1),
            case_info=case_info,
            metrics={},
            sql_embedding=None,
            action_embedding=None,
        )
        self.add_case(case)

        return case.id

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
            # self._last_finetune_count = 0
            # self._maybe_finetune()
        else:
            self._last_finetune_count = 0

    def _build_memory_case_from_info(self, case_id, case_info):
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
        统一的案例检索接口，根据配置的 retrieval_mode 选择使用静态或动态检索
        """
        if self.retrieval_mode == "static":
            return self.retrieve_static_cases(query_info, root_causes, pos_n, neg_n, mode)
        else:
            return self.retrieve_dynamic_cases(query_info, root_causes, pos_n, neg_n, mode)

    def retrieve_dynamic_cases(self, query_info, root_causes, pos_n=2, neg_n=2, mode="exploit"):
        # 获取所有候选案例
        all_cases = list(self.cases.values())
        if not all_cases:
            return {"mode": mode, "positive": [], "negative": []}

        # 根据mode过滤候选案例
        if mode == "exploit":
            # exploit模式：选择同根因的案例
            candidate_cases = []
            for case in all_cases:
                case_root_causes = case.case_info.get("root_causes")
                if any(rc in case_root_causes for rc in root_causes):
                    candidate_cases.append(case.case_info)
        elif mode == "explore":
            # exploit模式：选择不同根因的案例
            candidate_cases = []
            for case in all_cases:
                case_root_causes = case.case_info.get("root_causes")
                if not any(rc in case_root_causes for rc in root_causes):
                    candidate_cases.append(case.case_info)

        # 如果没有候选案例，返回空结果
        if not candidate_cases:
            return {"mode": mode, "positive": [], "negative": []}

        # 使用Soft Q Learning检索器选择案例
        selected_cases = self.retriever.select_cases(query_info.__dict__, candidate_cases, root_causes, top_k=pos_n + neg_n)

        # 按标签分类
        positive_cases = []
        negative_cases = []

        for case_info, probability in selected_cases:
            if case_info.get("label") == "positive":
                positive_cases.append(case_info)
            else:
                negative_cases.append(case_info)

        # 限制数量
        positive_cases = positive_cases[:pos_n]
        negative_cases = negative_cases[:neg_n]

        return {
            "mode": mode,
            "positive": positive_cases,
            "negative": negative_cases,
        }

    def retrieve_static_cases(self, query_info, root_causes, pos_n=2, neg_n=2, mode="exploit"):
        """
        使用传统的 islands 构造样本池并按标签输出：
        - exploitation pool: 目标根因对应的岛
        - exploration pool: 其他所有岛合并
        Positive/Negative 划分依据 case_info.label（缺失时回退到 approve_time>0 视为正例）。
        Positive 返回规则：至少包含同 SQL 与不同 SQL 的 approve_time 最大各 1，其余按提升降序填充至 pos_n。
        Negative 返回规则：从负例池随机采样最多 neg_n 个。
        返回：{"mode": str, "positive": [case_info], "negative": [case_info]}
        """
        query_text = query_info.query.lower().strip()
        target_root = self._normalize_root_causes(root_causes)

        target_island_ids = list(self.islands.get(target_root, set()))
        other_island_ids = []
        for island_key, ids in self.islands.items():
            if island_key != target_root:
                other_island_ids.extend(list(ids))

        # exploitation pool
        exploit_pool = self.ids_to_cases(target_island_ids)
        explore_pool = self.ids_to_cases(other_island_ids)

        # ε-greedy mode
        chosen_pool = explore_pool if mode == "explore" else exploit_pool

        # in exploit pool, prioritize same SQL cases
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

        # split by label/approve_time
        same_sql_pool, diff_sql_pool = split_by_sql(chosen_pool)

        # positive pool and negative pool
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

        # remaining positive cases are sorted by approve_time
        remaining_pos = [c for c in (pos_same + pos_diff) if c not in selected_pos]
        remaining_pos_sorted = sorted(
            remaining_pos,
            key=lambda c: c.case_info.get("approve_time") or c.metrics.get("approve_time", 0.0),
            reverse=True,
        )
        selected_pos.extend(remaining_pos_sorted)
        selected_pos = selected_pos[: max(1, pos_n)]

        # negative cases are sampled randomly
        if len(neg_all) > neg_n:
            selected_neg = random.sample(neg_all, neg_n)
        else:
            selected_neg = neg_all

        # export case_info
        pos_infos = [c.case_info for c in selected_pos]
        neg_infos = [c.case_info for c in selected_neg]

        return {"mode": mode, "positive": pos_infos, "negative": neg_infos}

    def retrieve_embedding(self, query_info, top_n=5, metric="cosine", case_ids=None):
        query_emb = np.array(self.build_sql_embedding(query_info))
        if norm(query_emb) == 0:
            return []

        # the embedding similarity
        case_sims = []
        if case_ids is not None:
            candidates = [self.get_case(cid) for cid in case_ids]
        else:
            candidates = list(self.cases.values())
        for case in candidates:
            if case.sql_embedding is None:
                continue
            case_emb = np.array(case.sql_embedding)
            if metric == "cosine":
                sim = np.dot(query_emb, case_emb) / (norm(query_emb) * norm(case_emb) + 1e-8)
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

            self._rcrank_predictor = RCRankPredictor(device="cpu")
        predictor = self._rcrank_predictor

        import pandas as pd

        from agent.plan.RCRank.model.modules.QueryFormer.utils import Encoding
        from agent.plan.RCRank.utils.plan_encoding import PlanEncoder

        encoding = Encoding(None, {"NA": 0})

        query = query_info.query
        query_plan = query_info.plan_json

        df = pd.DataFrame([{"plan_json": (json.dumps({"Plan": query_plan}) if isinstance(query_plan, dict) else query_plan)}])
        plan_encoder = PlanEncoder(df=df, encoding=encoding)
        encoded_df = plan_encoder.df
        plan_tensor = encoded_df["json_plan_tensor"].iloc[0]

        sql = predictor.tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=512).to(predictor.device)

        plan = {
            "x": plan_tensor["x"].to(predictor.device).to(torch.float32),
            "attn_bias": plan_tensor["attn_bias"].to(predictor.device),
            "rel_pos": plan_tensor["rel_pos"].to(predictor.device),
            "heights": plan_tensor["heights"].to(predictor.device),
        }

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
            cls_emb = outputs.last_hidden_state[:, 0, :]  # [CLS] vector, 768维
            # 直接截断到64维（取前64个维度）
            truncated_emb = cls_emb[:, :64]  # 768 -> 64
        return truncated_emb.squeeze(0).cpu().numpy().tolist()

    def _normalize_root_causes(self, root_causes):
        if root_causes is None:
            return frozenset()
        if isinstance(root_causes, str):
            return frozenset([root_causes])
        try:
            return frozenset(root_causes)
        except Exception:
            return frozenset([str(root_causes)])

    def _infer_root_causes_from_action(self, fix_action, query_info):
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

    def _validate_and_route_root_causes(self, case):
        # validate and route the root causes
        declared = self._normalize_root_causes(case.case_info.get("root_causes"))
        raw_info = case.case_info.get("query_info")
        query_info = QueryInfo(**raw_info)
        inferred = self._infer_root_causes_from_action(case.case_info.get("fix_action"), query_info)

        if inferred and (inferred != declared):
            case.case_info["root_causes"] = list(inferred)

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

        query_text = query_info.query.lower().strip()
        exclude_roots = exclude_roots

        archive_cases = [self.cases[aid] for aid in self.archive]
        if not archive_cases:
            return None

        similar_cases = self.retrieve_embedding(query_info, top_n=len(archive_cases), case_ids=list(self.archive))

        for case in similar_cases:
            case_root_str = str(self._normalize_root_causes(case.case_info.get("root_causes")))
            if case_root_str not in exclude_roots:
                return case.case_info.get("root_causes")

        return None

    def save_experience(self, query_info, case_info, reward):
        """保存单条经验到文件（JSONL格式，追加模式）"""

        if not self.samples_save_path:
            return

        os.makedirs(os.path.dirname(self.samples_save_path), exist_ok=True)

        exp_dict = {
            "query_info": query_info,
            "case_info": case_info,
            "reward": reward,
        }

        with open(self.samples_save_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(exp_dict, ensure_ascii=False) + "\n")
