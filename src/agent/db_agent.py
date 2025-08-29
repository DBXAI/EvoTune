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
    
    async def diagnose_and_repair(self, 
                                query_info: Dict[str, Any], 
                                anomaly_info: Dict[str, Any]) -> Dict[str, Any]:
        # 可以删了
        """
        主要诊断和修复流程，采用循环方式直到问题解决或达到最大尝试次数
        """
        
        # 构造输入对象进行类型检查
        query = QueryInfo(
            query_id=query_info['query_id'],
            query=query_info['query'],
            query_plan=query_info['query_plan'],
            query_kpis=query_info['query_kpis'],
            log_all=query_info['log_all'],
            execution_time=query_info['execution_time']
        )

        anomaly = AnomalyInfo(
            kpis=anomaly_info['kpis'],
            kpi_descriptions=anomaly_info['kpi_descriptions'],
            log_info=anomaly_info['log_info']
        )

        attempts = 0
        all_results = []
        is_resolved = False
        # 初始化已解决根因列表，避免重复诊断
        resolved_root_causes = []
        
        while attempts < self.max_attempts and not is_resolved:
            attempts += 1
            logger.info(f"开始第 {attempts} 次诊断修复尝试")
            
            # 从内存模块获取历史信息
            historical_cases, inspirations = self.memory_manager.retrieve_cases(query, n=5, exploration_ratio=0.4)

            # 分析阶段
            analysis_result = self.planner.model_predict(query, anomaly, resolved_root_causes)
                        
            if analysis_result['root_cause'] == "no issue":
                logger.info("未发现新的问题根因，终止诊断")
                break
            elif analysis_result['root_cause'] in resolved_root_causes:
                logger.info(f"根因 '{analysis_result['root_cause']}' 已经解决过，跳过此次诊断")
                continue
            else:
                # logger.info(f"第 {attempts} 次分析结果: 目前针对该慢查询，最主要的根因是：{analysis_result['root_cause']}, 置信度为：{analysis_result['confidence']}")
                logger.info(f"第 {attempts} 次分析结果: 目前针对该慢查询，最主要的根因是：{analysis_result['root_cause']}")
            
            # 尝试多次生成 action_results，直到提取到 SQL 修复语句
            max_action_attempts = 5
            action_attempts = 0
            sql_fixes = []
            while action_attempts < max_action_attempts and not sql_fixes:
                action_attempts += 1
                action_results = await self.action_manager.action_step(
                    analysis_result['root_cause'],
                    query,
                    historical_cases,
                    inspirations
                )
                # logger.info(f"第 {action_attempts} 次执行结果: {action_results}")
                diagnosis = action_results.get('diagnosis', '')
                sql_fixes = self.extract_sql_fix(diagnosis)
            
            if not sql_fixes:
                logger.info(f"未能提取到SQL修复语句，重新生成SQL修复语句")
                continue
            
            # 评估修复效果
            evaluation = await self.planner.evaluate_action(
                sql_fixes,
                analysis_result['root_cause'],
                query,
                simulation_mode=True
            )
            logger.info(f"第 {attempts} 次评估结果: {evaluation[1]}")
            
            # 存储本次尝试结果
            attempt_result = {
                "attempt": attempts,
                "analysis": analysis_result,
                "actions": action_results,
                "evaluation": evaluation
            }
            all_results.append(attempt_result)
            
            # 判断是否解决问题（性能是否提升）
            current_root_cause_resolved = evaluation[0]  # True表示性能提升，False表示无提升
            
            if current_root_cause_resolved == 1:
                # 性能提升，说明根因预测准确
                logger.info(f"第 {attempts} 次修复有效，根因 '{analysis_result['root_cause']}' 已解决")
                # 将已解决的根因加入列表
                resolved_root_causes.append(analysis_result['root_cause'])
                
                # 更新查询信息
                if analysis_result['root_cause'] == "repeatedly executing subqueries":
                    # 对于重复子查询类型的问题，直接使用优化后的SQL替换原SQL
                    new_query = ""
                    for i in range(len(sql_fixes)):
                        new_query += sql_fixes[i]
                    query.query = new_query
                    logger.info(f"第 {attempts} 次修复后的SQL: {query.query}")
                
                # 更新异常信息用于下一轮诊断
                # query = self._get_updated_query_info(query)
                
                # 检查是否所有问题都已解决
                if len(resolved_root_causes) >= self.planner.predictor.num_classes - 1:  # 减1是因为有一个"unknown"类别
                    logger.info("所有可能的根因都已诊断完毕，终止诊断")
                    is_resolved = True
                    break
            elif current_root_cause_resolved == 2:
                # 性能无提升，说明根因预测有误或修复方案无效
                logger.info(f"第 {attempts} 次修复无效，根因 '{analysis_result['root_cause']}' 可能不是真正问题")
                # 我们仍将此根因标记为"已处理"，避免重复尝试
                resolved_root_causes.append(analysis_result['root_cause'])
                # 直接进入下一次诊断尝试
            else:
                logger.info(f"第 {attempts} 次修复动作有误，重新生成SQL修复语句")
        
        # # 存储最终案例
        # final_result = all_results[-1] if all_results else None
        # if final_result:
        #     case_info = CaseInfo(
        #         query_info=query,
        #         anomaly_info=anomaly,
        #         root_causes=final_result['analysis']['root_causes'],
        #         actions_taken=final_result['actions'],
        #         result_metrics=final_result['evaluation'][1],
        #         confidence=final_result['analysis']['confidence'],
        #         status="completed" if is_resolved else "partial",
        #         attempts=attempts
        #     )
        #     # self.memory_manager.store_case(case_info.__dict__)
            
        is_resolved = True
        
        return {
            "is_resolved": is_resolved,
            "attempts": attempts,
            "all_results": all_results,
            "case_id": query.query_id
        }
        
    @staticmethod
    def metric_sampler(collecting_ref, metrics_ref, SAMPLE_INTERVAL, SAMPLE_COUNT):
        #可以删了
        import psutil
        collecting = collecting_ref[0]
        metrics = metrics_ref[0]
        prev_disk = psutil.disk_io_counters()
        prev_net = psutil.net_io_counters()
        samples = 0
        while collecting_ref[0] and samples < SAMPLE_COUNT:
            cpu = psutil.cpu_percent(interval=None)
            cur_disk = psutil.disk_io_counters()
            read_b = cur_disk.read_bytes - prev_disk.read_bytes
            write_b = cur_disk.write_bytes - prev_disk.write_bytes
            prev_disk = cur_disk
            cur_net = psutil.net_io_counters()
            recv_b = cur_net.bytes_recv - prev_net.bytes_recv
            sent_b = cur_net.bytes_sent - prev_net.bytes_sent
            prev_net = cur_net
            mem = psutil.virtual_memory()
            used = mem.total - mem.available
            total = mem.total
            metrics[0].append(cpu)
            metrics[1].append(read_b / SAMPLE_INTERVAL)
            metrics[2].append(write_b / SAMPLE_INTERVAL)
            metrics[3].append(recv_b / SAMPLE_INTERVAL)
            metrics[4].append(sent_b / SAMPLE_INTERVAL)
            metrics[5].append(used)
            metrics[6].append(total)
            samples += 1
            time.sleep(SAMPLE_INTERVAL)


    def _get_updated_query_info(self, query_info: QueryInfo) -> QueryInfo:
        # 可以删了
        """
        执行SQL，采集执行结果、耗时、系统资源、执行计划、统计信息。
        返回更新的QueryInfo对象
        """
        SAMPLE_INTERVAL = 0.01
        SAMPLE_COUNT = 9
        metrics = [[] for _ in range(7)]
        collecting = [True]  # 用list包装以便线程内可变
        metrics_ref = [metrics]
        # 启动采样线程
        sampler = threading.Thread(target=DBAgent.metric_sampler, args=(collecting, metrics_ref, SAMPLE_INTERVAL, SAMPLE_COUNT))
        sampler.start()
        start = time.perf_counter()
        engine = create_engine(self.db_url)
        try:
            with engine.connect() as conn:
                try:
                    result = conn.execute(text(query_info.query)).fetchall()
                except Exception as e:
                    logger.error(f"执行查询失败: {e}")
                    result = None
        except Exception as e:
            logger.error(f"连接数据库失败: {e}")
            result = None
        end = time.perf_counter()
        collecting[0] = False
        sampler.join()
        duration = end - start
        # 获取执行计划和queryid
        plan_json = None
        queryid = None
        try:
            with engine.connect() as conn:
                print("query_info.query: ", query_info.query)
                plan_res = conn.execute(text(f"EXPLAIN (ANALYZE, VERBOSE, FORMAT JSON) {query_info.query}"))
                plan_raw = plan_res.fetchone()
                print("plan_raw:", plan_raw)
                if plan_raw and len(plan_raw) > 0:
                    plan_json = plan_raw[0]
                    if isinstance(plan_json, list) and len(plan_json) > 0:
                        plan_json = plan_json[0]
                    if isinstance(plan_json, dict):
                        print("plan_json type: ", type(plan_json))
                        queryid = plan_json.get("Query Identifier", None)
                else:
                    logger.warning("执行计划返回为空")
        except Exception as e:
            logger.error(f"获取执行计划失败: {e}")
            pass
        # 获取统计信息
        log_all = None
        if queryid is not None:
            try:
                with engine.connect() as conn:
                    sql = '''\
                    SELECT total_exec_time, rows,
                        shared_blks_hit, shared_blks_read, shared_blks_dirtied, shared_blks_written,
                        local_blks_hit, local_blks_read, local_blks_dirtied, local_blks_written,
                        temp_blks_read, temp_blks_written, blk_read_time
                    FROM pg_stat_statements
                    WHERE queryid = :qid
                    '''
                    res = conn.execute(text(sql), {'qid': queryid}).fetchone()
                    log_all = list(res) if res else [None]*13
            except Exception as e:
                logger.error(f"获取统计信息失败: {e}")
                log_all = [None]*13
        else:
            log_all = [None]*13
            
        return QueryInfo(
            query_id=query_info.query_id,
            query=query_info.query,
            query_plan=plan_json,
            query_kpis=metrics,
            log_all=log_all,
            execution_time=duration
        )
   
    async def cleanup(self):
        """清理资源"""
        await self.action_manager.cleanup() 

    def parse_action_result(self, diagnosis: str):
        """从diagnosis文本中提取Explanation、FIX_ACTION和Query_Rewrite部分"""
        
        # 提取Explanation
        explanation_match = re.search(r'Explanation:\s*([\s\S]+?)(?=FIX_ACTION:|$)', diagnosis, re.IGNORECASE)
        explanation = explanation_match.group(1).strip() if explanation_match else ""
        
        # 提取FIX_ACTION
        fix_match = re.search(r'FIX_ACTION:\s*([\s\S]+?)Query_Rewrite:', diagnosis, re.IGNORECASE)
        fix_action = fix_match.group(1).strip() if fix_match else ""
        
        # 提取Query_Rewrite
        is_rewrite_match = re.search(r'Query_Rewrite:\s*(yes|no)', diagnosis, re.IGNORECASE)
        is_rewrite = is_rewrite_match.group(1).strip().lower() if is_rewrite_match else ""
        
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
            if l.startswith("WITH") or l.startswith("SELECT") or l.startswith("INSERT") or l.startswith("UPDATE") or l.startswith("DELETE"):
                sql_start = idx
                break
        if sql_start is not None:
            fix_part = "\n".join([line for line in lines[:sql_start] if line.strip()])
            sql_part = "\n".join(lines[sql_start:]).strip()
            return fix_part, sql_part
        else:
            return fix_action.strip(), ""    
        
    def extract_fix_action(self, action_result: str):
        explanation, fix_action, is_rewrite = self.parse_action_result(action_result)
        print("diagnosis output:", explanation)
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
                logger.info(f"诊断根因: {root_causes}")
                print(f"小模型诊断根因: {root_causes}")
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
                    logger.info(f"SQL {query_id} 修复失败，执行下一条SQL语句")
                    print(f"SQL {query_id} 修复失败，执行下一条SQL语句")
                    continue
                elif evaluation_result.get("status") == 1:
                    print(f"SQL {query_id} 修复成功")
                    fix_action = evaluation_result.get("fix_action")
                    rewrite_sql = evaluation_result.get("rewrite_sql")
                    if rewrite_sql != "":
                        query_info.query = rewrite_sql                   
                    new_time = evaluation_result.get("new_time")
                    self.memory_manager.save_case(query_info, fix_action, root_causes, old_time, new_time, "positive")
                else:
                    print(f"SQL {query_id} 修复无效, 调用大模型辅助诊断")
                    # 修复无效，调用大模型辅助诊断
                    new_analysis_result = self.planner.llm_predict(query_info, root_causes)
                    new_root_causes = new_analysis_result.get("root_causes", None)
                    print(f"大模型诊断根因: {new_root_causes}")
                    new_action_result = await self.action_manager.step(query_info, new_root_causes)
                    new_fix_action, new_rewrite_sql = self.extract_fix_action(new_action_result)
                    new_evaluation_result = await self.action_manager.evaluate_action(new_fix_action, new_rewrite_sql, query_info)
                    if new_evaluation_result.get("status") == -1:
                        logger.info(f"SQL {query_id} 修复失败，执行下一条SQL语句")
                        continue
                    elif new_evaluation_result.get("status") == 1:
                        print(f"SQL {query_id} 修复成功")
                        fix_action = new_evaluation_result.get("fix_action")
                        rewrite_sql = new_evaluation_result.get("rewrite_sql")
                        if rewrite_sql != "":
                            query_info.query = rewrite_sql
                        new_time = new_evaluation_result.get("new_time")
                        self.memory_manager.save_case(query_info, fix_action, new_root_causes, old_time, new_time, "positive")
                    elif new_evaluation_result.get("status") == 0:
                        #这里有点奇怪，是不是没有提升需要将根因设置成unknown
                        print(f"SQL {query_id} 修复无效")
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
                logger.info(f"SQL {query_id} 本轮执行时间: {new_time if new_time is not None else old_time}s")
                logger.info(f"SQL {query_id} 本轮修复actions: {fix_action}")
                logger.info(f"SQL {query_id} 本轮修复根因: {final_root_causes}")
                print(f"SQL {query_id} 原先执行时间: {old_time}s")
                print(f"SQL {query_id} 本轮执行时间: {new_time if new_time is not None else old_time}s")
                print(f"SQL {query_id} 本轮修复根因: {final_root_causes}")

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