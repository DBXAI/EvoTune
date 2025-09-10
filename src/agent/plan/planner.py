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

class Planner:
    def __init__(self, memory_manager: MemoryManager):
        self.memory_manager = memory_manager
        self.device = PLANNER_CONFIG['device']
        model_path = PLANNER_CONFIG['model_path']
        train_data_path = PLANNER_CONFIG['train_data_path']

        # Initialize RCRankPredictor
        self.predictor = RCRankPredictor(
            model_path=model_path,
            train_data_path=train_data_path,
            device=self.device,
            opt_threshold=PLANNER_CONFIG['opt_threshold'],
            num_classes=PLANNER_CONFIG['num_classes']
        )
        
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
        return self._extract_top_root_cause(pred_label, pred_label_opt)

    def llm_predict(self, query_info, root_causes):
        """Use LLM to diagnose the root cause of slow SQL."""
        
        # 获取历史案例和灵感，就像action_manager.py中的step函数一样
        historical_cases, inspirations = self.memory_manager.retrieve_cases(query_info)
        
        # 构建历史案例信息
        historical_info = ""
        if historical_cases:
            historical_info = f"""
        3. Historical Cases (Same SQL):
        • Previous Root Cause: {historical_cases.get('root_causes')}
        • Previous Actions: {historical_cases.get('fix_action')}
        • Performance Improvement: {historical_cases.get('approve_time')}
        """
        
        # 构建灵感案例信息
        inspirations_info = ""
        if inspirations:
            inspirations_info = "\n        4. Similar Cases (Inspirations):"
            for i, inspiration in enumerate(inspirations, 1):  # 限制显示前3个案例
                inspirations_info += f"""
        • Case {i}:
          - Query_info: {inspiration.get('query_info', {})}
          - Root Cause: {inspiration.get('root_causes')}
          - Actions: {inspiration.get('fix_action')}
          - Performance Improvement: {inspiration.get('approve_time')}
        """

        prompt = f"""
        As a database performance diagnosis expert, your task is to identify the **most likely root cause** of the following slow SQL query.

        ## Context
        1. Query Information:
        • SQL: {query_info.query}
        • Execution Plan: {query_info.plan_json}
        • Execution Runtime: {query_info.execution_time} s
        • Internal Metrics: Tuples Returned:{query_info.internal_metrics[0]}, Blocks Hit:{query_info.internal_metrics[1]}, Blocks Read:{query_info.internal_metrics[2]}, Tuples Fetched:{query_info.internal_metrics[3]}, Index Tuples Fetched:{query_info.internal_metrics[4]}, Sequential Tuples Read:{query_info.internal_metrics[5]}, Sequential Scans:{query_info.internal_metrics[6]}, Index Scans:{query_info.internal_metrics[7]}, Heap Blocks Hit:{query_info.internal_metrics[8]}, Heap Blocks Read:{query_info.internal_metrics[9]}, Index Blocks Hit:{query_info.internal_metrics[10]}, Index Blocks Read:{query_info.internal_metrics[11]}
        • CPU: {query_info.external_metrics[0]}
        • Read I/O: {query_info.external_metrics[1]}
        • Write I/O: {query_info.external_metrics[2]}
        • Virtual Memory: {query_info.external_metrics[3]}
        • Physical Memory: {query_info.external_metrics[4]}
        • Net Received: {query_info.external_metrics[5]}
        • Net Sent: {query_info.external_metrics[6]}

        2. Already Eliminated Causes:
        The following causes have been confirmed **not** to be the root cause: {root_causes}{historical_info}{inspirations_info}

        ## Constraints
        - You must select **only** from the following allowed labels:
        missing indexes, inappropriate query knobs, suboptimal plan optimizer, poorly written queries
        - Consider the historical cases and similar cases when making your diagnosis. Avoid root causes with little or no past benefit and prioritize those with substantial gains.
        - Output must be **exactly one or more** of the above labels, separated by commas if multiple.
        - Do **not** output explanations, reasoning, or any other text.

        ## Output Format
        <root cause label(s) here — exactly as listed above>
        """

        print("prompt: ", prompt)

        result = Runner.run_sync(starting_agent=Agent(
            name="RootCauseLLM",
            model="gpt-4.1",
            instructions="""
            You are a database slow SQL root cause diagnosis expert.
            Only output the allowed root cause labels, nothing else.
            """
        ), input=prompt)
        
        new_root_causes = result.final_output.strip()
        return {"root_causes": new_root_causes}


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
        
        if len(root_causes) == 1:
            return {"root_causes": root_causes[0]}
            # root_cause = {
            #     "root_cause": root_causes[0],
            #     "confidence": root_causes_confidence[0]
            # }
        else:
            return {"root_causes": root_causes}
            # root_cause = {
            #     "root_cause": root_causes,
            #     "confidence": root_causes_confidence
            # }