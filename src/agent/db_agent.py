import sys
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
import logging
import re
import os
import time
import random
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

    def _remove_hints_from_sql(self, sql: str) -> str:
        """从 SQL 中移除所有 hint 注释块 /*+ ... */"""
        import re
        # 移除所有 /*+ ... */ 格式的 hint
        sql_without_hints = re.sub(r'/\*\+[\s\S]*?\*/', '', sql)
        # 清理多余的空格
        sql_without_hints = ' '.join(sql_without_hints.split())
        return sql_without_hints.strip()

    async def run(self, slow_query_path, order, max_iterations, no_improvement_threshold=3, epsilon=0.1):
        """
        执行数据库调优流程
        
        Args:
            slow_query_path: 慢查询文件路径
            order: 查询顺序文件
            max_iterations: 最大迭代轮数
            no_improvement_threshold: 无提升次数阈值，达到后切换根因
        """
        # 读取负载
        queries_order = self.read_queries(slow_query_path, order)

        best_sql_actions = {}
        best_sql_times = {}
        workload_best_time = None
        workload_best_actions = None
        
        # 跟踪每个SQL的简化状态：来源、失败计数、archive已尝试根因、当前根因
        # {query_id: {"source": 'model'|'llm'|'archive', "fail_count": int, "root_tried": set[str], "current_root": Any}}
        sql_root_cause_states = {}

        for round_idx in range(max_iterations):
            logger.info(f"=== 第 {round_idx+1} 轮诊断 ===")
            round_start = time.time()
            round_sql_times = {}
            round_sql_actions = {}
            for i, query_path in enumerate(queries_order):
                logger.info(f"处理第 {i+1}/{len(queries_order)} 条SQL")

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
                    execution_time=result["duration"],
                    is_rewrite=False
                )

                # 根因选择器（本轮只决定一次）：model -> llm -> archive
                if query_id not in sql_root_cause_states:
                    sql_root_cause_states[query_id] = {"source": 'model', "fail_count": 0, "root_tried": set(), "current_root": None, "mode": "exploit"}
                    # {frozenset(["missing indexes", "inappropriate query knobs"])}
                    
                state = sql_root_cause_states[query_id]
                if state["source"] == 'model':
                    if state["current_root"] is None:
                        logger.info(f"SQL {query_id} 使用小模型进行诊断")
                        state["current_root"] = self.planner.model_predict(query_info).get('root_causes', None)
                elif state["source"] == 'llm':
                    if state["current_root"] is None:
                        logger.info(f"SQL {query_id} 使用大模型进行诊断")
                        # 将 frozenset 转换为扁平的字符串列表，供 LLM 使用
                        tried_roots = []
                        for root_set in state["root_tried"]:
                            tried_roots.extend(list(root_set))
                        state["current_root"] = self.planner.llm_predict(query_info, tried_roots).get('root_causes', None)
                elif state["source"] == 'archive':
                    if state["current_root"] is None:
                        logger.info(f"SQL {query_id} 使用相似根因进行诊断")
                        exclude = {str(self.memory_manager._normalize_root_causes(k)) for k in state["root_tried"]}
                        next_root = self.memory_manager.explore_root_from_archive(query_info, exclude_roots=exclude)
                        if not next_root:
                            all_roots = [list(r) for r in self.memory_manager.island_keys]  # List[List[str]]
                            candidates = [r for r in all_roots if str(self.memory_manager._normalize_root_causes(r)) not in exclude]
                            if candidates == []:
                                logger.info(f"SQL {query_id} 所有根因都已经探索过，结束探索")
                                continue
                            else:
                                next_root = random.choice(candidates)
                        state["current_root"] = next_root
                
                root_causes = state["current_root"]
                if not root_causes:
                    logger.info(f"SQL {query_id} 本轮无可用根因，跳过至下一条SQL")
                    continue

                logger.info(f"SQL {query_id} 当前诊断根因: {root_causes}")

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
                action_result = await self.action_manager.step(query_info, root_causes, mode=state["mode"])
                # 提取fix action
                fix_action, rewrite_sql = self.extract_fix_action(action_result)
                # 进行action验证
                evaluation_result = await self.action_manager.evaluate_action(fix_action, rewrite_sql, query_info)

                # 获取修复内容
                fix_action = evaluation_result.get("fix_action")
                rewrite_sql = evaluation_result.get("rewrite_sql")
                if rewrite_sql != "":
                    # 去除 hint 后比较 SQL 是否真的被重写了
                    original_sql = query_info.query.strip()
                    sql_without_hint = self._remove_hints_from_sql(rewrite_sql.strip())
                    
                    # 只有当去除 hint 后的 SQL 与原始 SQL 不同时，才标记为重写
                    if sql_without_hint != original_sql:
                        query_info.query = rewrite_sql
                        query_info.is_rewrite = True
                    else:
                        # 如果只是添加了 hint，不标记为重写
                        query_info.query = rewrite_sql
                        query_info.is_rewrite = False

                # 初始化变量
                new_time = None
                old_time = query_info.execution_time

                # 处理评估结果并更新根因状态
                if evaluation_result.get("status") == -1:
                    print(evaluation_result.get("msg"))
                    logger.info(f"SQL {query_id} 修复失败，执行下一条SQL语句")
                    new_time = 300
                    # 保存案例并获取修正后的根因
                    case_id = self.memory_manager.save_case(query_info, fix_action, root_causes, old_time, new_time, "negative")
                    # 获取修正后的根因
                    case = self.memory_manager.get_case(case_id)
                    root_causes = case.case_info.get("root_causes")
                    state["fail_count"] += 1
                    state["mode"] = "exploit"
                elif evaluation_result.get("status") == 1:
                    logger.info(f"SQL {query_id} 修复成功")
                    new_time = evaluation_result.get("new_time")
                    # 保存案例并获取修正后的根因
                    case_id = self.memory_manager.save_case(query_info, fix_action, root_causes, old_time, new_time, "positive")
                    # 获取修正后的根因
                    case = self.memory_manager.get_case(case_id)
                    root_causes = case.case_info.get("root_causes")
                    
                    # 成功：重置连续失败计数，保留来源与当前根因
                    if (query_id not in best_sql_times) or (best_sql_times[query_id] - new_time) / best_sql_times[query_id] > 0.1:
                        state['fail_count'] = 0
                    else:
                        state["fail_count"] += 1
                    if random.random() < epsilon: 
                        # ε-greedy探索：以epsilon概率探索新根因
                        logger.info(f"ε-greedy决定探索新根因")
                        state["source"] = 'archive'
                        state["current_root"] = None
                        state["mode"] = "explore"
                    else:
                        # ε-greedy利用：以(1-epsilon)概率继续利用当前根因
                        logger.info(f"ε-greedy利用当前根因")
                        state["mode"] = "exploit"
                else:
                    # 修复无效：记负样例并推进来源（下轮再尝试）
                    logger.info(f"SQL {query_id} 修复无效，记录负样例")
                    new_time = evaluation_result.get("new_time", old_time)
                    # 保存案例并获取修正后的根因
                    case_id = self.memory_manager.save_case(query_info, fix_action, root_causes, old_time, new_time, "negative")
                    # 获取修正后的根因
                    case = self.memory_manager.get_case(case_id)
                    root_causes = case.case_info.get("root_causes")
                    state["fail_count"] += 1
                    state["mode"] = "exploit"
                    
                if state["fail_count"] >= no_improvement_threshold:
                    state["fail_count"] = 0
                    if state["source"] == 'model': 
                        # model不行换大模型
                        state["source"] = 'llm'
                    elif state["source"] == 'llm':
                        # 大模型不行换相似根因
                        state["source"] = 'archive'
                    # 记录已尝试根因（frozenset）并在下轮选新根因
                    state["root_tried"].add(self.memory_manager._normalize_root_causes(root_causes))
                    state["current_root"] = None

                # 记录当前SQL执行前后的时间，记录是不是更短的执行时间
                if (query_id not in best_sql_times) or (new_time < best_sql_times[query_id]):
                    best_sql_times[query_id] = new_time
                    best_sql_actions[query_id] = fix_action
                round_sql_times[query_id] = new_time
                round_sql_actions[query_id] = fix_action
                sql_root_cause_states[query_id] = state

                logger.info(f"SQL {query_id} 原先执行时间: {old_time}s")
                logger.info(f"SQL {query_id} 本轮执行时间: {new_time}s")
                logger.info(f"SQL {query_id} 本轮修复SQL: {rewrite_sql}")
                logger.info(f"SQL {query_id} 本轮修复actions: {fix_action}")
                logger.info(f"SQL {query_id} 本轮修复根因: {root_causes}")

            round_end = time.time()
            logger.info(f"第 {round_idx+1} 轮调优时间: {round_end - round_start:.4f}s")
            round_total_time = sum([t for t in round_sql_times.values()])
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