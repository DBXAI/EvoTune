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
import numpy as np
from scipy.stats import beta

# 添加项目根目录到Python路径
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from agent.plan.planner import Planner
from agent.action.action_manager import ActionManager
from agent.memory.memory_manager import MemoryManager
from utils.types import QueryInfo, AnomalyInfo
from src.database.pg_env import PostgresDB
from src.database.tidb_env import TiDB
from agent.plan.online_predict import RCRankPredictor

logger = logging.getLogger(__name__)

# 确保logger继承根logger的配置
logger.setLevel(logging.INFO)
logger.propagate = True

class DBAgent:
    def __init__(self, configs=None):
        self.configs = configs
        self.agent_config = configs.get('AGENT_CONFIG')
        
        db_type = configs.get('DATABASE_TYPE', 'postgres')
        if db_type.lower() == 'postgres':
            db_config = configs.get('POSTGRES_CONFIG')
            self.db = PostgresDB(db_config)
            logger.info("使用PostgreSQL数据库")
        elif db_type.lower() == 'tidb':
            db_config = configs.get('TiDB_CONFIG')
            self.db = TiDB(db_config)
            logger.info("使用TiDB数据库")

        rcrank_config = configs.get('RCRANK_PREDICTOR_CONFIG')
        predictor = RCRankPredictor(rcrank_config)

        memory_config = configs.get('MEMORY_MANAGER_CONFIG')
        self.memory_manager = MemoryManager(memory_config, predictor=predictor)
        
        self.planner = Planner(self.memory_manager, predictor=predictor, agent_config=self.agent_config)
        self.action_manager = ActionManager(memory_manager=self.memory_manager, db=self.db, planner=self.planner, configs=configs)
        
        self.alpha_0 = 1.0  # Beta prior alpha
        self.beta_0 = 1.0   # Beta prior beta
        self.p_crit = 0.5   # Critical probability threshold
        self.epsilon_min = 0.1  
        self.epsilon_max = 0.8  
        self.kappa = 0.5    
        self.N_th = 3      # Failure count threshold
    
    def calculate_dynamic_epsilon(self, root_cause_key, state):
        # Get success and failure counts for this root cause
        s_r = state.get("successes").get(root_cause_key, 0)  # successes
        n_r = state.get("attempts").get(root_cause_key, 0)   # total attempts
        
        # Calculate reliability score P_r = 1 - F_Beta(p_crit; alpha_0 + s_r, beta_0 + n_r - s_r)
        if n_r == 0:
            # No attempts yet, use prior
            alpha = self.alpha_0
            beta_param = self.beta_0
        else:
            alpha = self.alpha_0 + s_r
            beta_param = self.beta_0 + n_r - s_r
        
        P_r = 1 - beta.cdf(self.p_crit, alpha, beta_param)
        U_r = 1 - P_r

        G_nr = 1 / (1 + np.exp(-self.kappa * (n_r - self.N_th)))

        epsilon_r = self.epsilon_min + (self.epsilon_max - self.epsilon_min) * G_nr * U_r
        
        return epsilon_r
    
    def decide_exploration_mode(self, query_info, predicted_root, state):
        """
        Decide between exploration and exploitation based on dynamic epsilon-greedy
        
        Args:
            query_info: Current query information
            predicted_root: Predicted root cause
            state: Current SQL state
            
        Returns:
            mode: 'explore' or 'exploit'
        """
        predicted_root_key = self.memory_manager._normalize_root_causes(predicted_root)
        
        # Calculate dynamic epsilon for the predicted root cause
        epsilon_r = self.calculate_dynamic_epsilon(predicted_root_key, state)
        print(f"epsilon_r: {epsilon_r}")
        
        # Sample p ~ U(0,1) and decide
        p = random.random()
        mode = "explore" if p < epsilon_r else "exploit"
        
        return mode
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.cleanup()
        
    async def cleanup(self):
        try:
            if self.action_manager:
                try:
                    await asyncio.shield(self.action_manager.cleanup())
                except Exception as e:
                    logger.warning(f"清理action_manager时出现异常: {e}")
            
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
            pass

    def read_queries(self, slow_query_path, order):
        queries_order = []
        with open(os.path.join(slow_query_path, order), "r", encoding="utf-8") as f:
            for fname in f:
                fname = fname.strip()
                if fname.endswith('.sql'):
                    queries_order.append(os.path.join(slow_query_path, fname))
        return queries_order
    
    def clean_query(self, sql):
        sql = re.sub(r'--.*?(\n|$)', '', sql)
        sql = sql.strip()
        sql = re.sub(r'\s+', ' ', sql)
        return sql

    def _remove_hints_from_sql(self, sql):
        import re
        sql_without_hints = re.sub(r'/\*\+[\s\S]*?\*/', '', sql)
        sql_without_hints = ' '.join(sql_without_hints.split())
        return sql_without_hints.strip()

    async def run(self, slow_query_path, order, duration, no_improvement_threshold=3, epsilon=0.1):
        """
        执行数据库调优流程
        
        Args:
            slow_query_path: 慢查询文件路径
            order: 查询顺序文件
            duration: 超时时间（小时），在超时前会持续迭代优化
            no_improvement_threshold: 无提升次数阈值，达到后切换根因
        """
        # 读取负载
        queries_order = self.read_queries(slow_query_path, order)

        best_sql_actions = {}
        best_sql_times = {}
        best_sql_texts = {}
        workload_best_time = None
        workload_best_actions = None
        
        # 跟踪每个SQL的简化状态
        # {query_id: {"root_tried": set[str], "current_root": Any, "mode": 'exploit'|'explore', "confidence": Dict[str,float], "attempts": Dict[str,int], "successes": Dict[str,int]}}
        sql_root_cause_states = {}

        # 记录开始时间
        start_time = time.time()
        round_idx = 0
        
        while True:
            current_time = time.time()
            elapsed_time = current_time - start_time
            
            # 检查是否超时
            if elapsed_time >= duration*3600:
                logger.info(f"达到超时时间 {duration}s，停止优化")
                break
                
            round_idx += 1
            logger.info(f"=== 第 {round_idx} 轮诊断 ===")
            round_start = time.time()
            round_sql_times = {}
            round_sql_actions = {}
            for i, query_path in enumerate(queries_order):    
                logger.info(f"处理第 {i+1}/{len(queries_order)} 条SQL")

                # 读取慢SQL
                query = open(query_path, "r", encoding="utf-8").read()
                query_id = query_path.split("/")[-1].replace(".sql", "")
                query = self.clean_query(query)

                # 选择本轮用于诊断的数据来源SQL：优先使用历史最佳SQL（已包含rewrite/hint），否则使用原始SQL
                original_sql = query
                collected_sql = best_sql_texts.get(query_id, original_sql)

                # 仅当存在历史最佳DDL/SET修复动作时，先应用动作后收集一次；否则直接收集一次
                pre_index_names = []
                pre_knob_names = []
                pre_applied = False
                if query_id in best_sql_actions and best_sql_actions[query_id]:
                    pre_fix_action = best_sql_actions[query_id]
                    success, _, _, _ = self.db.execute(pre_fix_action)
                    if success:
                        pre_index_names = self.action_manager.extract_index_names(pre_fix_action)
                        pre_knob_names = self.action_manager.extract_knob_names(pre_fix_action)
                        pre_applied = True

                result = self.db.run_sql_and_collect_all(collected_sql)

                # 构建 QueryInfo；若 collected_sql 与 original_sql 不同，视为 rewrite
                query_info = QueryInfo(
                    query_id=query_id,
                    query=collected_sql,
                    plan_json=result["plan_json"],
                    internal_metrics=result["internal_metrics"],
                    external_metrics=result["external_metrics"],
                    execution_time=result["duration"],
                    is_rewrite=(collected_sql != original_sql)
                )

                # 若首次遇到该 SQL，先用本轮默认执行结果初始化“最佳”记录
                if query_id not in best_sql_times:
                    best_sql_times[query_id] = query_info.execution_time
                    best_sql_actions[query_id] = ""
                    best_sql_texts[query_id] = collected_sql

                # 初始化状态
                if query_id not in sql_root_cause_states:
                    sql_root_cause_states[query_id] = {"root_tried": set(), "current_root": None, "mode": "exploit", "confidence": {}, "attempts": {}, "successes": {}}
                    
                state = sql_root_cause_states[query_id]
                
                # 预测根因
                logger.info(f"SQL {query_id} 开始预测根因")
                predicted_root, state = self.planner.predict(query_info, state, self.memory_manager)
                
                # 检查预测的根因是否已经在root_tried中
                predicted_root_key = self.memory_manager._normalize_root_causes(predicted_root)
                if predicted_root_key in state["root_tried"]:
                    logger.info(f"SQL {query_id} 预测根因已在root_tried中，强制进入探索模式")
                    state["mode"] = "explore"
                else:
                    # Dynamic epsilon-greedy 决策
                    state["mode"] = self.decide_exploration_mode(query_info, predicted_root, state)
                    logger.info(f"SQL {query_id} 动态epsilon决策")

                if state["mode"] == "explore":
                    logger.info(f"SQL {query_id} 进入探索模式（archive 相似根因）")
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
                else:
                    logger.info(f"SQL {query_id} 进入利用模式（使用预测的根因）")
                    state["current_root"] = predicted_root
                
                root_causes = state["current_root"]
                if not root_causes:
                    logger.info(f"SQL {query_id} 本轮无可用根因，跳过至下一条SQL")
                    continue

                logger.info(f"SQL {query_id} 当前诊断根因: {root_causes}")

                # 调用action_manager.step处理（包含evaluate_action和经验池更新）
                evaluation_result = await self.action_manager.step(query_info, root_causes, mode=state["mode"])

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
                    new_time = self.agent_config.get("sql_execution_timeout")
                    # 保存案例并获取修正后的根因
                    case_id = self.memory_manager.save_case(query_info, fix_action, root_causes, old_time, new_time, "negative")
                    # 获取修正后的根因
                    case = self.memory_manager.get_case(case_id)
                    root_causes = case.case_info.get("root_causes")
                    # 记录该根因一次无效尝试
                    key = self.memory_manager._normalize_root_causes(root_causes)
                    state["attempts"][key] = state["attempts"].get(key, 0) + 1
                elif evaluation_result.get("status") == 1:
                    logger.info(f"SQL {query_id} 修复成功")
                    new_time = evaluation_result.get("new_time")
                    # 保存案例并获取修正后的根因
                    case_id = self.memory_manager.save_case(query_info, fix_action, root_causes, old_time, new_time, "positive")
                    # 获取修正后的根因
                    case = self.memory_manager.get_case(case_id)
                    root_causes = case.case_info.get("root_causes")
                    
                    # 成功：将该根因尝试计数清零并累计成功次数
                    key = self.memory_manager._normalize_root_causes(root_causes)
                    state["attempts"][key] = 0
                    state["successes"][key] = state["successes"].get(key, 0) + 1
                    # 成功后置空 current_root，下次允许小模型重新诊断（刷新根因）
                    state["current_root"] = None
                    # state["confidence"] = {}
                else:
                    # 修复无效：记负样例并推进来源（下轮再尝试）
                    logger.info(f"SQL {query_id} 修复无效，记录负样例")
                    new_time = evaluation_result.get("new_time", old_time)
                    # 保存案例并获取修正后的根因
                    case_id = self.memory_manager.save_case(query_info, fix_action, root_causes, old_time, new_time, "negative")
                    # 获取修正后的根因
                    case = self.memory_manager.get_case(case_id)
                    root_causes = case.case_info.get("root_causes")
                    # 记录该根因一次无效尝试
                    key = self.memory_manager._normalize_root_causes(root_causes)
                    state["attempts"][key] = state["attempts"].get(key, 0) + 1
                
                # 依据无效尝试计数达到阈值时，标记该根因为已尝试并置空 current_root
                key = self.memory_manager._normalize_root_causes(root_causes)
                if state.get("attempts", {}).get(key, 0) >= no_improvement_threshold:
                    state["root_tried"].add(self.memory_manager._normalize_root_causes(root_causes))

                # 仅在修复成功且更优时更新“最佳”记录
                if new_time < best_sql_times[query_id]:
                    best_sql_times[query_id] = new_time
                    best_sql_actions[query_id] = fix_action
                    best_sql_texts[query_id] = query_info.query
                round_sql_times[query_id] = new_time
                round_sql_actions[query_id] = fix_action
                sql_root_cause_states[query_id] = state

                logger.info(f"SQL {query_id} 原先执行时间: {old_time}s")
                logger.info(f"SQL {query_id} 本轮执行时间: {new_time}s")
                logger.info(f"SQL {query_id} 本轮修复SQL: {rewrite_sql}")
                logger.info(f"SQL {query_id} 本轮修复根因: {root_causes}")
                logger.info(f"SQL {query_id} 本轮修复actions: {fix_action}")

                # 在新动作回滚完成后，再回滚历史动作
                if pre_applied:
                    self.action_manager.rollback_action(pre_index_names, pre_knob_names)


            round_end = time.time()         
            logger.info(f"第 {round_idx} 轮调优时间: {round_end - round_start:.4f}s")
            round_total_time = sum([t for t in round_sql_times.values()])
            logger.info(f"第 {round_idx} 轮workload总执行时间: {round_total_time:.4f}s")
            logger.info(f"第 {round_idx} 轮各SQL执行时间: {round_sql_times}")
            
            # if workload_best_time is None or round_total_time < workload_best_time:
            #     workload_best_time = round_total_time
            #     workload_best_actions = round_sql_actions.copy()

            logger.info(f"迭代至第 {round_idx} 轮，各SQL最优执行时间: {best_sql_times}")
            logger.info(f"迭代至第 {round_idx} 轮，workload最优总执行时间: {sum(best_sql_times.values())}")
            logger.info(f"迭代至第 {round_idx} 轮，各SQL最优修复actions: {best_sql_actions}")

        # 输出最终结果
        final_time = time.time()
        total_elapsed = final_time - start_time
        logger.info(f"=== 优化完成 ===")
        logger.info(f"总用时: {total_elapsed:.2f}s（超时限制: {duration}s）")
        logger.info(f"完成轮数: {round_idx} 轮")
        logger.info(f"各SQL最优执行时间: {best_sql_times}")
        logger.info(f"workload最优总执行时间: {sum(best_sql_times.values()):.4f}s")
        logger.info(f"各SQL最优修复actions: {best_sql_actions}")
        logger.info(f"诊断流程结束")