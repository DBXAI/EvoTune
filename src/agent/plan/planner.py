import os
import sys
import re
import time
import threading
from sqlalchemy import create_engine, text

# Add RCRank to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
rcrank_path = os.path.join(current_dir, "RCRank")
if rcrank_path not in sys.path:
    sys.path.append(rcrank_path)


import numpy as np
import pandas as pd
import json
# import ast
# import torch
from typing import Dict, List, Any, Tuple
from agent.memory.memory_manager import MemoryManager
from utils.types import QueryInfo, AnomalyInfo
from agent.plan.online_predict import RCRankPredictor
from model.modules.QueryFormer.utils import Encoding
from RCRank.utils.plan_encoding import PlanEncoder
from agents import Agent, Runner
from config.base_config import PLANNER_CONFIG
from agent.prompt import build_root_cause_diagnosis_prompt, get_root_cause_llm_instructions

class Planner:
    def __init__(self, memory_manager: MemoryManager, predictor: RCRankPredictor):
        self.memory_manager = memory_manager
        self.predictor = predictor
        self.device = predictor.device
        
        # Initialize encoding components
        self.encoding = Encoding(None, {'NA': 0})
    
    def encode_plan(self, query_plan: Dict) -> Dict:
        """将查询执行计划编码为模型所需的格式"""            
        # 创建一个只包含当前计划的DataFrame
        df = pd.DataFrame([{
            'plan_json': json.dumps({'Plan': query_plan}) if isinstance(query_plan, dict) else query_plan
        }])
        
        # 使用PlanEncoder处理执行计划
        plan_encoder = PlanEncoder(df=df, encoding=self.encoding)
        encoded_df = plan_encoder.df
        
        # 获取编码后的计划
        return encoded_df['json_plan_tensor'].iloc[0]
    
    def model_predict(self, query_info: QueryInfo) -> Dict[str, Any]:
        """分析慢查询和异常信息，预测可能的根因"""
        
        # 构建预测所需的输入数据
        query, plan, timeseries, log = self._build_prediction_input(query_info)
        
        # 使用RCRankPredictor进行预测
        pred_label_opt, pred_label = self.predictor.predict(
            query=query,
            plan=plan,
            timeseries=timeseries,
            log=log
        )
        
        # 解析预测结果
        result = self._extract_top_root_cause(pred_label, pred_label_opt)
        # 附加小模型的原始概率向量，供上层决策与LLM提示
        result["pred_label_opt"] = pred_label_opt.cpu().numpy()
        return result

    def diagnosis_agent(self, prompt) -> str:
        return Runner.run_sync(starting_agent=Agent(
        name="RootCauseLLM",
        model="gpt-4.1",
        instructions=get_root_cause_llm_instructions()
        ), input=prompt).final_output.strip()

    def llm_predict(self, query_info, root_causes, state_confidence, tuning_history=None):
        """Use LLM to diagnose the root cause of slow SQL with confidence-aware guidance.

        state_confidence: 历史运行态置信度映射 {label: confidence in [0,1]}
        tuning_history: 每个根因的成功/失败计数列表
        """
        # 使用新的 ε-greedy 检索结构：{"positive", "negative"}
        retrieval = self.memory_manager.retrieve_cases(query_info, root_causes)
        positives = retrieval.get('positive', [])
        negatives = retrieval.get('negative', [])

        # 合并置信度指导到统一的严格JSON输出prompt
        prompt = build_root_cause_diagnosis_prompt(
            query_info,
            root_causes,
            positives,
            negatives,
            state_confidence=state_confidence,
            tuning_history=tuning_history
        )
        print("prompt: ", prompt)

        output = self.diagnosis_agent(prompt)
        print("output: ", output)
        parsed = self.parse_and_validate(output)
        labels, conf_map = (parsed if parsed is not None else (None, None))
        # 若解析失败或不一致，要求大模型重新输出一次
        retry = 0
        while labels is None and retry < 5:
            output = self.diagnosis_agent(prompt)
            parsed = self.parse_and_validate(output)
            labels, conf_map = (parsed if parsed is not None else (None, None))
            retry += 1

        if labels is None:
            labels = ["normal"]
            conf_map = {"normal": 1.0}

        return {"root_causes": labels, "confidence": conf_map}


    def parse_and_validate(self, json_text: str):
        try:
            data = json.loads(json_text)
            # 读取 LLM 报告的预测根因与全量置信度
            reported_pred = [str(x).lower().strip() for x in data.get("predicted_root_causes", [])]
            items = data.get("root_causes", [])

            # 根据置信度>0.5 计算应当的预测集合
            derived_pred = []
            conf_map = {}
            for it in items:
                label = str(it.get("label", "")).lower().strip()
                conf = float(it.get("confidence", 0.0))
                if conf > 0.5:
                    derived_pred.append(label)
                conf_map[label] = conf

            # 一致性校验：必须完全相等（集合意义）
            if set(reported_pred) != set(derived_pred):
                return None
            
            if len(derived_pred) == 0:
                derived_pred = ["normal"]
                conf_map = {"normal": 1.0}

            return derived_pred, conf_map
        except Exception:
            return None

    def _build_prediction_input(self, query_info: QueryInfo) -> Tuple[str, Dict, List, List]:
        """构建RCRankPredictor所需的输入数据"""

        query = query_info.query if hasattr(query_info, 'query') else ""
        plan = self.encode_plan(query_info.plan_json)
        # timeseries = json.loads(query_info.external_metrics)
        # log = json.loads(query_info.internal_metrics)
        timeseries = query_info.external_metrics
        log = query_info.internal_metrics
        
        return query, plan, timeseries, log
    
    def _extract_top_root_cause(self, pred_label, pred_label_opt) -> Dict[str, Any]:
        """从预测结果中提取最主要的根因"""
        # 将预测结果转换到CPU并转为numpy数组
        pred_label = pred_label.cpu().numpy()
        pred_label_opt = pred_label_opt.cpu().numpy()
        
        # 问题类型映射
        issue_types = {
            0: "missing indexes",
            1: "inappropriate query knobs",
            2: "suboptimal plan optimizer",
            3: "poorly written queries",
            4: "normal"
        }
        
        print("pred_label: ", pred_label)
        print("pred_label_opt: ", pred_label_opt)
        
        if pred_label.sum() == 0:
            return {"root_causes": "normal"}
        
        # 找到预测为1的类别索引
        predicted_classes = np.where(pred_label[0] == 1)[0]
        if len(predicted_classes) == 0:
            return {"root_causes": "normal"}
            # return {"root_cause": "normal", "confidence": pred_label_opt}
        
        # 获取所有预测的根因
        root_causes = []
        root_causes_confidence = []
        for class_idx in predicted_classes:
            root_cause = issue_types.get(class_idx)
            if root_cause not in root_causes:
                root_causes.append(root_cause)
                root_causes_confidence.append(pred_label_opt[0][class_idx])
        
        return {"root_causes": root_causes}
        # root_cause = {
        #     "root_cause": root_causes,
        #     "confidence": root_causes_confidence
        # }