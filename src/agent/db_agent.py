import sys
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
import logging
import re
import os
import time
import threading
import pandas as pd
from sqlalchemy import create_engine, text
import asyncio
import signal
import atexit
import time

# 添加项目根目录到Python路径
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from agent.plan.planner import Planner
from agent.action.action_manager import ActionManager
from agent.memory.memory_manager import MemoryManager
from utils.types import QueryInfo, AnomalyInfo
from src.database.pg_env import PostgresDB

logger = logging.getLogger(__name__)

# 确保logger继承根logger的配置
logger.setLevel(logging.INFO)
logger.propagate = True

class DBAgent:
    def __init__(self, configs=None, memory_path: str = "memory_results", max_attempts: int = 10):
        self.configs = configs
        self.db = PostgresDB(
            user=configs['POSTGRES_CONFIG']['user'],
            password=configs['POSTGRES_CONFIG']['password'],
            host=configs['POSTGRES_CONFIG']['host'],
            port=configs['POSTGRES_CONFIG']['port'],
            dbname=configs['POSTGRES_CONFIG']['dbname'],
            workload_type=configs['POSTGRES_CONFIG']['workload_type'],
            postgres_path=configs['POSTGRES_CONFIG']['postgres_path'],
            postgres_data=configs['POSTGRES_CONFIG']['postgres_data'],
            log_path=configs['POSTGRES_CONFIG']['log_path']
        )

        self.memory_manager = MemoryManager(memory_path)
        self.planner = Planner(self.memory_manager)
        self.action_manager = ActionManager(memory_manager=self.memory_manager, db=self.db, planner=self.planner)
        self.max_attempts = max_attempts
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口，确保资源清理"""
        await self.cleanup()
        
    async def cleanup(self):
        """清理资源"""
        try:
            # 清理action_manager
            if self.action_manager:
                try:
                    # 防止取消传播到当前任务
                    await asyncio.shield(self.action_manager.cleanup())
                except Exception as e:
                    logger.warning(f"清理action_manager时出现异常: {e}")
            
            # 清理数据库连接
            if self.db:
                try:
                    if hasattr(self.db, 'cleanup') and callable(getattr(self.db, 'cleanup')):
                        await self.db.cleanup()
                    elif hasattr(self.db, 'close'):
                        self.db.close()
                except Exception as e:
                    logger.warning(f"关闭数据库连接时出现异常: {e}")
            
            logger.info("DBAgent资源清理完成")
            
        except Exception as e:
            logger.error(f"清理资源时出现异常: {e}")
        finally:
            # 不再主动取消事件循环中的其他任务，避免传播 CancelledError
            pass

    def parse_action_result(self, diagnosis: str):
        """从diagnosis文本中提取Explanation、Fix_Action和Query_Rewrite部分"""
        
        # 提取Explanation
        explanation_match = re.search(r'Explanation:\s*([\s\S]+?)(?=Fix_Action:|$)', diagnosis, re.IGNORECASE)
        explanation = explanation_match.group(1).strip() if explanation_match else ""
        
        # 提取FIX_ACTION
        fix_match = re.search(r'Fix_Action:\s*([\s\S]+?)Query_Rewrite:', diagnosis, re.IGNORECASE)
        fix_action = fix_match.group(1).strip() if fix_match else ""
        
        # 提取Query_Rewrite
        is_rewrite_match = re.search(r'Query_Rewrite:\s*(yes|no)', diagnosis, re.IGNORECASE)
        is_rewrite = is_rewrite_match.group(1).strip() if is_rewrite_match else ""
        
        return explanation, fix_action, is_rewrite
    
    def parse_query_rewrite(self, fix_action: str):
        """
        提取修复动作（所有CREATE/SET等）和主SQL（从第一个WITH/SELECT/INSERT/UPDATE/DELETE开始的所有内容）。
        返回二元组（修复动作, 主SQL）。
        """
        # 按行分割
        lines = fix_action.strip().splitlines()
        sql_start = None
        
        for idx, line in enumerate(lines):
            l = line.lstrip().upper()
            # 检查是否是SQL语句开始（包括WITH子句）
            if (l.startswith("WITH") or l.startswith("SELECT") or l.startswith("INSERT") or 
                l.startswith("UPDATE") or l.startswith("DELETE")):
                sql_start = idx
                break
        
        if sql_start is not None:
            fix_part = "\n".join([line for line in lines[:sql_start] if line.strip()])
            sql_part = "\n".join(lines[sql_start:]).strip()
            return fix_part, sql_part
        else:
            # 如果没有找到SQL语句，整个内容作为修复动作
            return fix_action.strip(), ""
        
    def extract_fix_action(self, action_result: str):
        explanation, fix_action, is_rewrite = self.parse_action_result(action_result)
        print("diagnosis output:")
        print("explanation:", explanation)
        print("fix_action:", fix_action)
        print("is_rewrite:", is_rewrite)
        if is_rewrite == "yes":
            fix_action, rewrite_sql = self.parse_query_rewrite(fix_action)
            return fix_action, rewrite_sql
        else:
            return fix_action, ""


    def read_queries(self, slow_query_path, order):
        queries_order = []
        with open(os.path.join(slow_query_path, order), "r", encoding="utf-8") as f:
            for fname in f:
                fname = fname.strip()
                if fname.endswith('.sql'):
                    queries_order.append(os.path.join(slow_query_path, fname))
        return queries_order
    
    def clean_query(self, sql: str) -> str:
        # 去除单行注释
        sql = re.sub(r'--.*?(\n|$)', '', sql)
        # 去除开头和结尾的空白字符
        sql = sql.strip()
        # 将多个空白字符替换为单个空格
        sql = re.sub(r'\s+', ' ', sql)
        return sql

    async def run(self, slow_query_path, order, max_iterations):
        # 读取负载
        queries_order = self.read_queries(slow_query_path, order)

        best_sql_actions = {}
        best_sql_times = {}
        workload_best_time = None
        workload_best_actions = None

        for round_idx in range(max_iterations):
            logger.info(f"=== 第 {round_idx+1} 轮诊断 ===")
            round_start = time.time()
            round_sql_times = {}
            round_sql_actions = {}
            for i, query_path in enumerate(queries_order):
                logger.info(f"处理第 {i+1}/{len(queries_order)} 条SQL")
                print(f"处理第 {i+1}/{len(queries_order)} 条SQL")

                # 读取慢SQL
                query = open(query_path, "r", encoding="utf-8").read()
                query_id = query_path.split("/")[-1].replace(".sql", "")
                query = self.clean_query(query)

                # 压测获取相关数据
                result = self.db.run_sql_and_collect_all(query, duration=300)

                query_info = QueryInfo(
                    query_id=query_id,
                    query=result["query"],
                    plan_json=result["plan_json"],
                    internal_metrics=result["internal_metrics"],
                    external_metrics=result["external_metrics"],
                    execution_time=result["duration"]
                )

                # 诊断根因
                analysis_result = self.planner.model_predict(query_info)
                root_causes = analysis_result.get('root_causes', None)
                logger.info(f"小模型诊断根因: {root_causes}")

                # if query_id == "01":
                #     root_causes = "inappropriate query knobs"
                # elif query_id == "02":
                #     root_causes = "suboptimal plan optimizer"
                # elif query_id == "03":
                #     root_causes = ["inappropriate query knobs", "missing indexes"]
                # elif query_id == "04":
                #     root_causes = ["inappropriate query knobs", "missing indexes"]
                # elif query_id == "05":
                #     root_causes = ["inappropriate query knobs", "suboptimal plan optimizer", "missing indexes"]
                # elif query_id == "06":
                #     root_causes = "inappropriate query knobs"
                # elif query_id == "07":
                #     root_causes = "inappropriate query knobs"
                # elif query_id == "08":
                #     root_causes = ["inappropriate query knobs", "missing indexes"]
                # elif query_id == "09":
                #     root_causes = "inappropriate query knobs"
                # elif query_id == "10":
                #     root_causes = "inappropriate query knobs"
                # elif query_id == "11":
                #     root_causes = ["inappropriate query knobs", "missing indexes"]
                # elif query_id == "12":
                #     root_causes = "inappropriate query knobs"
                # elif query_id == "13":
                #     root_causes = "inappropriate query knobs"
                # elif query_id == "14":
                #     root_causes = "inappropriate query knobs"
                # elif query_id == "15":
                #     root_causes = "inappropriate query knobs"
                # elif query_id == "16":
                #     root_causes = "inappropriate query knobs"
                # elif query_id == "17":
                #     root_causes = ["inappropriate query knobs", "missing indexes"]
                # elif query_id == "18":
                #     root_causes = ["inappropriate query knobs", "missing indexes", "poorly written queries"]
                # elif query_id == "19":
                #     root_causes = "inappropriate query knobs"
                # elif query_id == "20":
                #     root_causes = "missing indexes"
                # elif query_id == "21":
                #     root_causes = ["inappropriate query knobs", "missing indexes"]
                # elif query_id == "22":
                #     root_causes = "missing indexes"
                    
                # logger.info(f"指定SQL {query_id} 根因: {root_causes}")

                # 调用action_manager.step处理
                action_result = await self.action_manager.step(query_info, root_causes)
                # 提取fix action
                fix_action, rewrite_sql = self.extract_fix_action(action_result)
                # 进行action验证
                evaluation_result = await self.action_manager.evaluate_action(fix_action, rewrite_sql, query_info)

                # 初始化变量
                new_root_causes = None
                new_time = None
                old_time = query_info.execution_time

                if evaluation_result.get("status") == -1:
                    print(evaluation_result.get("msg"))
                    logger.info(f"SQL {query_id} 修复失败，执行下一条SQL语句")
                    continue
                elif evaluation_result.get("status") == 1:
                    logger.info(f"SQL {query_id} 修复成功")
                    fix_action = evaluation_result.get("fix_action")
                    rewrite_sql = evaluation_result.get("rewrite_sql")
                    if rewrite_sql != "":
                        query_info.query = rewrite_sql                   
                    new_time = evaluation_result.get("new_time")
                    self.memory_manager.save_case(query_info, fix_action, root_causes, old_time, new_time, "positive")
                else:
                    logger.info(f"SQL {query_id} 修复无效, 调用大模型辅助诊断")
                    # 修复无效，调用大模型辅助诊断
                    new_analysis_result = self.planner.llm_predict(query_info, root_causes)
                    new_root_causes = new_analysis_result.get("root_causes", None)
                    logger.info(f"大模型诊断根因: {new_root_causes}")
                    new_action_result = await self.action_manager.step(query_info, new_root_causes)
                    new_fix_action, new_rewrite_sql = self.extract_fix_action(new_action_result)
                    new_evaluation_result = await self.action_manager.evaluate_action(new_fix_action, new_rewrite_sql, query_info)
                    if new_evaluation_result.get("status") == -1:
                        print(new_evaluation_result.get("msg"))
                        logger.info(f"SQL {query_id} 修复失败，执行下一条SQL语句")
                        continue
                    elif new_evaluation_result.get("status") == 1:
                        logger.info(f"SQL {query_id} 修复成功")
                        fix_action = new_evaluation_result.get("fix_action")
                        rewrite_sql = new_evaluation_result.get("rewrite_sql")
                        if rewrite_sql != "":
                            query_info.query = rewrite_sql
                        new_time = new_evaluation_result.get("new_time")
                        self.memory_manager.save_case(query_info, fix_action, new_root_causes, old_time, new_time, "positive")
                    elif new_evaluation_result.get("status") == 0:
                        #这里有点奇怪，是不是没有提升需要将根因设置成unknown
                        logger.info(f"SQL {query_id} 修复无效")
                        fix_action = new_evaluation_result.get("fix_action") + evaluation_result.get("fix_action", "")
                        rewrite_sql = new_evaluation_result.get("rewrite_sql")
                        if rewrite_sql != "":
                            query_info.query = rewrite_sql
                        new_time = new_evaluation_result.get("new_time") 
                        self.memory_manager.save_case(query_info, fix_action, new_root_causes, old_time, new_time, "negative")

                # 记录当前SQL执行前后的时间，记录是不是更短的执行时间
                if new_time is not None and (query_id not in best_sql_times or new_time < best_sql_times[query_id]):
                    best_sql_times[query_id] = new_time
                    best_sql_actions[query_id] = fix_action
                round_sql_times[query_id] = new_time if new_time is not None else old_time
                round_sql_actions[query_id] = fix_action
                # 使用最终的根因（如果有新的根因则使用新的，否则使用原始的）
                final_root_causes = new_root_causes if new_root_causes is not None else root_causes #这里有点奇怪，是不是没有提升需要将根因设置成unknown
                logger.info(f"SQL {query_id} 原先执行时间: {old_time}s")
                logger.info(f"SQL {query_id} 本轮执行时间: {new_time if new_time is not None else old_time}s")
                logger.info(f"SQL {query_id} 本轮修复SQL: {rewrite_sql}")
                logger.info(f"SQL {query_id} 本轮修复actions: {fix_action}")
                logger.info(f"SQL {query_id} 本轮修复根因: {final_root_causes}")

            round_end = time.time()
            logger.info(f"第 {round_idx+1} 轮调优时间: {round_end - round_start:.4f}s")
            round_total_time = sum([t for t in round_sql_times.values() if t is not None])
            logger.info(f"第 {round_idx+1} 轮workload总执行时间: {round_total_time:.4f}s")
            print(f"第 {round_idx+1} 轮workload总执行时间: {round_total_time:.4f}s")
            print(f"第 {round_idx+1} 轮调优时间: {round_end - round_start:.4f}s")
            if workload_best_time is None or round_total_time < workload_best_time:
                workload_best_time = round_total_time
                workload_best_actions = round_sql_actions.copy()

        logger.info(f"最优workload执行时间: {workload_best_time:.4f}s")
        logger.info(f"最优SQL修复actions: {workload_best_actions}")
        logger.info(f"各SQL最优执行时间: {best_sql_times}")
        logger.info(f"各SQL最优修复actions: {best_sql_actions}")
        logger.info(f"诊断流程结束")