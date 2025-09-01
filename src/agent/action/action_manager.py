import os
import sys
import asyncio
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from contextlib import AsyncExitStack, asynccontextmanager
from agents import Agent, Runner
from agents.mcp import MCPServerStdio, MCPServerSse
from agents._config import set_default_openai_api
from agent.action import diagnose_tools
from config.base_config import POSTGRES_CONFIG
import nest_asyncio
import re

# 添加项目根目录到Python路径
project_root = str(Path(__file__).parent.parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from agent.memory.memory_manager import MemoryManager
from src.utils.types import QueryInfo, AnomalyInfo
from config.base_config import API_SETTINGS
from src.database.pg_env import PostgresDB

nest_asyncio.apply()

def delete_all_loggers():
    """清理和重置所有日志记录器"""
    import logging
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for logger in loggers:
        handlers = logger.handlers[:]
        for handler in handlers:
            logger.removeHandler(handler)
        logger.propagate = True
        logger.setLevel(logging.CRITICAL)

# 清理所有日志记录器
delete_all_loggers()

# 从配置文件加载API设置
os.environ['OPENAI_API_KEY'] = API_SETTINGS['openai']['api_key']
os.environ['OPENAI_BASE_URL'] = API_SETTINGS['openai']['base_url']

# 设置使用 chat_completions API
set_default_openai_api("chat_completions")

class ServerConfig:
    def __init__(self, name, server_type, params):
        self.name = name
        self.server_type = server_type
        self.params = params

    @classmethod
    def from_dict(cls, config):
        server_type = MCPServerStdio if config['server_type'] == 'stdio' else MCPServerSse
        if 'cwd' in config['params'] and config['params']['cwd'] == '.':
            config['params']['cwd'] = os.path.dirname(os.path.abspath(__file__))
        return cls(
            name=config['name'],
            server_type=server_type,
            params=config['params']
        )

class MultiServerManager:
    def __init__(self, configs):
        self.configs = configs
        self.servers = {}

    @asynccontextmanager
    async def manage_servers(self):
        try:
            for config in self.configs:
                server = await config.server_type(
                    name=config.name,
                    params=config.params
                ).__aenter__()
                self.servers[config.name] = server
            yield list(self.servers.values())
        except Exception as e:
            # 如果初始化过程中出现异常，清理已创建的服务器
            for server in self.servers.values():
                try:
                    await server.__aexit__(type(e), e, e.__traceback__)
                except Exception:
                    pass
            raise
        finally:
            # 确保所有服务器都被正确清理
            for server in self.servers.values():
                try:
                    await server.__aexit__(None, None, None)
                except asyncio.CancelledError:
                    # 忽略取消错误
                    pass
                except Exception as e:
                    print(f"Warning: Error cleaning up server: {e}")
            self.servers.clear()

class ActionManager:
    def __init__(self, config_path = None, memory_manager = None, db = None, planner = None):
        self.db = db
        self.memory_manager = memory_manager
        self.planner = planner

        self.config_path = config_path or os.path.join(os.path.dirname(__file__), 'action_server_config.yaml')
        self.agent = None
        self.server_manager = None
        self.exit_stack = AsyncExitStack()
        self._servers = []
        self.fixAgent = Agent(
            name="FixAgent",
            # model="gpt-4.1",
            model="gpt-4.1",
            instructions=self._get_fix_agent_instructions(),
        )

    async def __aenter__(self):
        """异步上下文管理器的进入方法"""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器的退出方法"""
        
        # 仅释放agent引用，实际资源交由 exit_stack 管理
        if self.agent:
            self.agent = None

        # 最后清理 exit_stack
        try:
            await self.exit_stack.aclose()
        except asyncio.CancelledError:
            # 忽略取消错误
            pass
        except Exception as e:
            print(f"Warning: Error during exit stack cleanup: {e}")

        # 清理引用
        self._servers = []
        self.server_manager = None

    async def initialize(self):
        """初始化 MCP 服务器和 Agent"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        
        configs = [ServerConfig.from_dict(service_config) for service_config in config_data['services']]
        self.server_manager = MultiServerManager(configs)
            
        self._servers = await self.exit_stack.enter_async_context(self.server_manager.manage_servers())
        
        self.agent = Agent(
            name="DBDiagnosticAgent",
            # model="gpt-4o-mini",
            # model="o3-mini",
            # model="o4-mini",
            # model="gpt-4.1",
            model="gpt-4.1",
            instructions=self._get_agent_instructions(),
            mcp_servers=self._servers,
        )
        return self.agent

    async def cleanup(self):
        """清理资源"""
        try:
            await self.__aexit__(None, None, None)
        except Exception as e:
            print(f"Warning: Error during cleanup: {e}")

    def _get_fix_agent_instructions(self) -> str:
        return """
            You are a SQL Precision Repair Agent with two core responsibilities:
            1. Executable Repair: When provided with a raw, non-executable SQL statement and its exact database error message,
            produce a corrected SQL statement (or the minimal sequence of statements) that will execute successfully in the target SQL dialect.
            2. Output Alignment: When given two SQL statements whose result sets do not match,
            rewrite the SQL so that its output exactly matches the output of the original SQL.
            """

    def _get_agent_instructions(self) -> str:
        return """You are an expert in database performance diagnosis and SQL optimization.
            Your role is to:
            1. Understand detailed database and query execution context.
            2. Identify how the root cause leads to performance degradation.
            3. Output exactly three labeled sections: Explanation, Fix_Action and Query_Rewrite.
            4. Always make the Fix_Action executable directly in the target database.
            5. Never include extra text, comments, or sections beyond the required format.
            6. Always apply the given MCP APIs when producing fixes, if applicable.
            7. Assume the provided execution plan, metrics, and environment info are accurate.
            8. Keep the Explanation concise and technical, avoiding generic descriptions."""    

    def _build_prompt(self, root_cause, base_info, action_space_apis, historical_cases, inspirations):
        """
        Format a prompt that feeds the LLM with all necessary context and enforces
        a consistent two-part output: Explanation + SQL Fix.
        """

        query_info = base_info.get('query_info')
        db_info = base_info.get('database_info', {})
        historical_cases = ""
        inspirations = ""
        
        return f"""
            As an SQL optimization expert, your task is to diagnose and fix the following slow SQL query.
            You have the context of slow SQL and the root cause of the performance degradation as below:

            ## Context
            1. Query Information:
            • SQL: {query_info.query}
            • Execution Plan: {query_info.plan_json}
            • Execution Runtime: {query_info.execution_time} s
            • Internal Metrics: Tuples Returned:{query_info.internal_metrics[0]}, Blocks Hit:{query_info.internal_metrics[1]}, Blocks Read:{query_info.internal_metrics[2]}, Tuples Fetched:{query_info.internal_metrics[3]}, Index Tuples Fetched:{query_info.internal_metrics[4]}, Sequential Tuples Read:{query_info.internal_metrics[5]}, Sequential Scans:{query_info.internal_metrics[6]}, Index Scans:{query_info.internal_metrics[7]}, Heap Blocks Hit:{query_info.internal_metrics[8]}, Heap Blocks Read:{query_info.internal_metrics[9]}, Index Blocks Hit:{query_info.internal_metrics[10]}, Index Blocks Read:{query_info.internal_metrics[11]}
            
            2. Database Environment Information:
            • Database Type: {db_info.get('db_type')}
            • Workload Type: {db_info.get('workload_type')}
            • Database Size: {db_info.get('size')}
            • Database Schema: {db_info.get('schema')}
            • CPU: {query_info.external_metrics[0]}
            • Read I/O: {query_info.external_metrics[1]}
            • Write I/O: {query_info.external_metrics[2]}
            • Virtual Memory: {query_info.external_metrics[3]}
            • Physical Memory: {query_info.external_metrics[4]}
            • Net Received: {query_info.external_metrics[5]}
            • Net Sent: {query_info.external_metrics[6]}

            3. **Identified Root Cause**:
            {root_cause}

            4. Historical Cases:
            {historical_cases}

            5. Inspirations:
            {inspirations}

            ## Action Space  
            - If the root cause is "inappropriate query knobs", you MUST perform actions within Knob Space using SET knob = value.
            - If the root cause is "suboptimal plan optimizer", you MUST perform actions within Plan Optimizer Space using PostgreSQL comment-style hints (/*+ Set(knob_name value) Set(knob_name value) ... */).
            - If the root cause is "missing indexes", you MUST perform actions within Index Space using CREATE INDEX IF NOT EXISTS or DROP INDEX IF EXISTS.
            - If the root cause is "poorly written queries", you MUST perform actions within Query Rewrite Space using rewrite SQL.
            - **If multiple root causes are identified, you MUST consider all FIX_ACTION outputs jointly and determine the optimal solution that addresses the combined root causes, while strictly limiting actions to the relevant action spaces of the identified root causes. Do not perform any actions outside of these specified root causes.**

            1. Knob Space:  
            - Knob Space is strictly limited to the following knobs: 
            constraint_exclusion: type: enum, min: null, max: null, default: partition
            cpu_index_tuple_cost: type: float, min: 0.0, max: 10, default: 0.005
            cpu_operator_cost: type: float, min: 0.0, max: 1, default: 0.0025
            cpu_tuple_cost: type: float, min: 0.0, max: 10, default: 0.01
            cursor_tuple_fraction: type: float, min: 0.0, max: 1.0, default: 0.1
            default_statistics_target: type: integer, min: 1, max: 5000, default: 100
            effective_cache_size: type: integer, min: 1, max: 2147483647, default: 524288
            from_collapse_limit: type: integer, min: 1, max: 24, default: 8
            geqo: type: boolean, min: 0, max: 1, default: 0
            hash_mem_multiplier: type: float, min: 1.0, max: 1000.0, default: 2.0
            jit: type: boolean, min: 0, max: 1, default: 1
            jit_above_cost: type: float, min: -1.0, max: 500000, default: 100000.0
            jit_expressions: type: boolean, min: 0, max: 1, default: 1
            jit_inline_above_cost: type: float, min: -1.0, max: 5000000, default: 500000.0
            jit_optimize_above_cost: type: float, min: -1.0, max: 5000000, default: 500000.0
            jit_tuple_deforming: type: boolean, min: 0, max: 1, default: 1
            join_collapse_limit: type: integer, min: 1, max: 24, default: 8
            max_parallel_workers: type: integer, min: 0, max: 1024, default: 8
            max_parallel_workers_per_gather: type: integer, min: 0, max: 1024, default: 2
            min_parallel_index_scan_size: type: integer, min: 0, max: 715827882, default: 64
            min_parallel_table_scan_size: type: integer, min: 0, max: 715827882, default: 1024
            parallel_setup_cost: type: float, min: 0.0, max: 50000.0, default: 1000.0
            parallel_tuple_cost: type: float, min: 0.0, max: 10, default: 0.1
            plan_cache_mode: type: enum, min: null, max: null, default: auto
            random_page_cost: type: float, min: 0.0, max: 100, default: 4.0
            seq_page_cost: type: float, min: 0.0, max: 100, default: 1.0
            work_mem: type: integer, min: 64, max: 2147483647, default: 4096

            2. Plan Optimizer Space: 
            - Plan Optimizer Space is strictly limited to the following knobs: 
            enable_async_append: type: boolean, min: 0, max: 1, default: 1
            enable_bitmapscan: type: boolean, min: 0, max: 1, default: 1
            enable_gathermerge: type: boolean, min: 0, max: 1, default: 1
            enable_hashagg: type: boolean, min: 0, max: 1, default: 1
            enable_hashjoin: type: boolean, min: 0, max: 1, default: 1
            enable_incremental_sort: type: boolean, min: 0, max: 1, default: 1
            enable_indexonlyscan: type: boolean, min: 0, max: 1, default: 1
            enable_indexscan: type: boolean, min: 0, max: 1, default: 1
            enable_material: type: boolean, min: 0, max: 1, default: 1
            enable_memoize: type: boolean, min: 0, max: 1, default: 1
            enable_mergejoin: type: boolean, min: 0, max: 1, default: 1
            enable_nestloop: type: boolean, min: 0, max: 1, default: 1
            enable_parallel_append: type: boolean, min: 0, max: 1, default: 1
            enable_parallel_hash: type: boolean, min: 0, max: 1, default: 1
            enable_partition_pruning: type: boolean, min: 0, max: 1, default: 1
            enable_partitionwise_aggregate: type: boolean, min: 0, max: 1, default: 0
            enable_partitionwise_join: type: boolean, min: 0, max: 1, default: 0
            enable_seqscan: type: boolean, min: 0, max: 1, default: 1
            enable_sort: type: boolean, min: 0, max: 1, default: 1

            3. Index Space: 
            - The naming convention for newly created indexes must follow the pattern: (table_name)_(col1)_(col2)_idx.
            - Index creation should be strictly guided by the slow_sql queries.
            - Avoid generating duplicate or subsumed indexes, and do not drop primary keys, unique indexes, or any indexes that enforce constraints.
            - Existing indexes in the database: 

            4. Query Rewrite Space: 
            - The output must always be a semantically equivalent SELECT statement that produces exactly the same result set as the original query.
            - The following rules are provided in the format ["rule name": "rule description"], consider the rules when rewriting the SQL:
            "name": "SORT_UNION_TRANSPOSE_MATCH_NULL_FETCH", "rewrite_rules_structured": ["conditions": "1. A SQL query performs a `UNION ALL` operation on multiple select statements. 2. Following the `UNION ALL`, there's an `ORDER BY` clause intending to sort the results, with no `OFFSET` clause present in the query.3. Each select statement involved in the `UNION ALL` could potentially benefit from pre-application of the order defined by the `ORDER BY` clause, without violating the intended sort order across the combined result set.", "transformations": "1. Identify the `ORDER BY` clause following a `UNION ALL` that combines multiple select statements and verify there's no associated `OFFSET` directive. 2. For each select statement involved in the `UNION ALL`:   - Apply the `ORDER BY` clause directly to the select statement if doing so does not change the intended order of the final result set. This effectively means prepending an `ORDER BY` to each select statement based on the `ORDER BY` clause initially following the `UNION ALL` operation. 3. Retain the original `ORDER BY` clause following the `UNION ALL`. This ensures that the final result set is ordered as intended, taking into account that individual select statements are now pre-sorted. 4. Implement this transformation only if it's determined that sorting individual select statements before the `UNION ALL` would maintain the overall desired sort order and is expected to either improve performance by reducing sorting overhead or have no negative impact."]

            ## Constraints
            - You **must** use the following MCP APIs when producing the fix:
            {action_space_apis}
            - Allowed optimization methods: change online knobs, add indexes, add query hints, rewrite SQL, or other safe optimization actions.
            - Fix must be executable directly in {db_info.get('db_type')}.
            - The fix should be safe and should not drop or delete data unless explicitly required by the root cause.

            ## Output Format
            You must output ONLY in the following format, with explanation and fix action:

            Explanation:
            <concise technical reason why the root cause is causing the slowdown now>

            Fix_Action:
            <your fix SQL or optimization action here>

            Query_Rewrite:
            <yes or no>

            ## Output Requirements
            - "Explanation" must be a short, precise, technical summary — no generic statements.
            - If the fix rewrites the SQL, "Query_Rewrite" must be "yes", else "no".
            - If rewriting, provide the optimized SQL under Fix_Action.
            - If adding indexes or changing knobs, provide the exact SQL or command.
            - Do not output any extra sections, comments, or steps.
            """

    def _build_fix_prompt(self, fix_action, err):
        return f"""
        You have been given a non-executable SQL statement and the exact database error:
        • Non-executable SQL: {fix_action}
        • Database Error Message: {err}
        Your sole objective is to produce a corrected, fully-formed SQL statement (or an ordered sequence of SQL statements) that will run without error.
        Strict rules — follow exactly:
        1) OUTPUT NOTHING BUT THE SQL STATEMENT(S). No explanations, no comments, no markdown, no surrounding quotes.
        2) If a single statement suffices, output exactly that statement, terminated by a single semicolon and a newline.
        3) If multiple statements are required, output them in execution order, each terminated by a semicolon and a newline.
        -- Output (strict) --
        <paste only the corrected fix action statement(s) here, exactly as SQL, nothing else>
        """
    
    def _build_rewrite_prompt(self, origin_sql, rewrite_sql):
        return f"""
        You are given two SQL statements:
        • Reference SQL (correct output): {origin_sql}
        • Divergent SQL (incorrect output): {rewrite_sql}
        Your task is to rewrite the Divergent SQL so that its result set matches the Reference SQL exactly.

        Strict output rules:
        1. Output only the corrected SQL statement.
        2. Do not include explanations, comments, markdown, or any extra text.
        3. The SQL must be fully executable and self-contained.
        4. Preserve the original intent and structure of the Divergent SQL as much as possible while ensuring the output matches exactly.
        5. If you cannot rewrite the SQL, output the original SQL.

        — Output Format —
        <paste only the revised SQL here, exactly as SQL, nothing else>
        """

    async def step(self, query_info, root_cause):
        """分析慢查询并生成修复操作的工作流"""
        if not self.agent:
            await self.initialize()
        
        # workflow:
        # 1. information collect
        # 2. match_history
        # 3. action_define
        # 4. action_generate
        # 5. evaluate_action
        base_info = diagnose_tools.base_information_collect(query_info, POSTGRES_CONFIG)
        historical_cases, inspirations = self.memory_manager.retrieve_cases(query_info)
        action_space_apis= await self.action_space_collect(base_info, historical_cases, root_cause, POSTGRES_CONFIG)
        action_result = await self.action_generate(root_cause, base_info, action_space_apis, historical_cases, inspirations)

        return action_result

    @staticmethod
    def extract_index_names(sql):
        """从多条CREATE INDEX语句中提取所有索引名"""
        return re.findall(r'CREATE\s+INDEX\s+(?:IF\s+NOT\s+EXISTS\s+)?([\w_]+)', sql, re.IGNORECASE)
        
    @staticmethod
    def extract_knob_names(sql):
        """从多条SET语句中提取所有knob名"""
        return re.findall(r'SET\s+([a-zA-Z0-9_]+)\s*=', sql, re.IGNORECASE)

    def rollback_action(self, index_names, knob_names):
        """撤销修复动作"""
        # 撤销所有索引
        for index_name in index_names:
            try:  
                with self.db.connection.cursor() as cur:
                    cur.execute(f"DROP INDEX IF EXISTS {index_name}")
                self.db.connection.commit()
                print(f"已撤销索引: {index_name}")
            except Exception as e:
                self.db.connection.rollback()
                print(f"撤销索引失败: {e}")

        # 撤销所有knob设置
        for knob_name in knob_names:
            try:
                with self.db.connection.cursor() as cur:
                    cur.execute(f"RESET {knob_name};")
                self.db.connection.commit()
                print(f"{knob_name} 已恢复默认值")
            except Exception as e:
                self.db.connection.rollback()
                print(f"恢复{knob_name}失败: {e}")

    async def evaluate_action(self, fix_action, rewrite_sql, query_info):
        """
        1. 检测修复操作能否执行：不能执行，返回让大模型重新生成
        2. 检测修复操作前后结果是否一致：不一致，返回让大模型重新生成
        3. 检测修复操作是否带来性能提升
        """
        # 1. 检查修复SQL能否执行
        if fix_action != "":
            success, _, _, err = self.db.execute(fix_action)
        else:
            success = True
            err = ""
        retry_count = 0
        max_retry = 5
        while not success and retry_count < max_retry:
            prompt = self._build_fix_prompt(fix_action, err)
            result = await Runner.run(starting_agent=self.fixAgent, input=prompt)
            new_fix_action = result.final_output
            print(f"new_fix_action: {new_fix_action}")
            success, _, _, err = self.db.execute(new_fix_action)
            fix_action = new_fix_action
            retry_count += 1
        if not success:
            return {"status": -1, "msg": f"SQL修复失败: {fix_action}, 错误: {err}", "action": fix_action, "new_time": -1}

        if rewrite_sql != "":
            # 2. 原始慢SQL的执行结果
            retry_count = 0
            max_retry = 5
            origin_sql = query_info.query
            match = self.db.compare_sql_results(origin_sql, rewrite_sql)
            print(f"match: {match}")
            while not match and retry_count < max_retry:
                prompt = self._build_rewrite_prompt(origin_sql, rewrite_sql)
                result = await Runner.run(starting_agent=self.fixAgent, input=prompt)
                new_rewrite_sql = result.final_output
                print(f"new_rewrite_sql: {new_rewrite_sql}")
                success, _, _, err = self.db.execute(new_rewrite_sql, timeout=300)
                rewrite_sql = new_rewrite_sql
                retry_count += 1
                match = self.db.compare_sql_results(origin_sql, rewrite_sql)
            if not match:
                return {"status": -1, "msg": "修复后结果仍不一致", "action": rewrite_sql, "new_time": -1}
            if not success:
                return {"status": -1, "msg": f"修复后SQL执行失败: {err}", "action": rewrite_sql, "new_time": -1}

        if rewrite_sql != "":
            success, _, new_time, err = self.db.execute(rewrite_sql, timeout=300)
        else:
            success, _, new_time, err = self.db.execute(query_info.query, timeout=300)

        old_time = query_info.execution_time

        if fix_action != "":
            index_names = self.extract_index_names(fix_action)
            knob_names = self.extract_knob_names(fix_action)
            self.rollback_action(index_names, knob_names)

        # 4. 性能对比与后续处理
        if (old_time - new_time) / old_time > 0.1:
            return {"status": 1, "msg": f"修复成功，性能提升，耗时由{old_time:.4f}s降至{new_time:.4f}s", "fix_action": fix_action, "rewrite_sql": rewrite_sql, "new_time": new_time}
        else:
            return {"status": 0, "msg": f"修复无效，性能未提升或下降，原耗时{old_time:.4f}s，新耗时{new_time:.4f}s", "fix_action": fix_action, "rewrite_sql": rewrite_sql, "new_time": new_time}

    async def action_step(self, root_cause, query_info, historical_cases, inspirations):
        """
        1. 小模型根因诊断+大模型修复；
        2. 如果没有效果，尝试大模型根因诊断+大模型修复；
        3. 如果没有效果，存入到memory中，继续下一轮迭代
        """
        action_result = await self.action_workflow(root_cause, query_info, historical_cases, inspirations)
        if action_result.get("status") == -1:
            return action_result
        if action_result.get("status") == 1:
            sql_fixes = action_result.get("action")
            old_time = query_info.execution_time
            new_time = action_result.get("new_time")
            self.memory_manager.save(query_info, sql_fixes, root_cause, old_time, new_time, "positive")
            return action_result
        else:
            new_root_cause = self.planner.llm_predict(query_info, None, [], [root_cause])
            new_action_result = await self.action_workflow(new_root_cause, query_info, historical_cases, inspirations)
            if new_action_result.get("status") == -1:
                return new_action_result
            if new_action_result.get("status") == 1:
                sql_fixes = new_action_result.get("action")
                old_time = query_info.execution_time
                new_time = new_action_result.get("new_time")
                self.memory_manager.save(query_info, sql_fixes, new_root_cause, old_time, new_time, "unknown")
                return new_action_result
            if new_action_result.get("status") == 0:
                sql_fixes = new_action_result.get("action") + action_result.get("action")
                old_time = query_info.execution_time
                new_time = new_action_result.get("new_time")
                self.memory_manager.save(query_info, sql_fixes, root_cause, old_time, new_time, "negative")
                return new_action_result

    async def action_space_collect(self, base_info, historical_cases, root_causes, postgres_config):
        """基于根因，调用MCP工具从数据库收集更多相关信息，供后续大模型决策。
        返回严格的JSON字符串，包含各信息块与可用操作空间建议。
        """
        # 根因
        root_causes = root_causes if isinstance(root_causes, list) else ([root_causes] if root_causes else [])

        # 根据根因选择要调用的MCP工具
        tool_apis = []
        # 索引相关
        if "missing indexes" in root_causes:
            tool_apis += [
                "indexes_info_collect(postgres_config)",
                "indexes_action_space(query)"
            ]
        # Knob相关
        if "inappropriate query knobs" in root_causes:
            tool_apis += [
                "system_knobs_info_collect(postgres_config)",
                "system_knobs_action_space(postgres_config)",
            ]
        # 执行计划/优化器相关
        if "suboptimal plan optimizer" in root_causes:
            tool_apis += [
                "plan_info_collect(query, postgres_config)",
                "plan_action_space(query)"
            ]
        # SQL 可读性/重写相关
        if "poorly written queries" in root_causes:
            tool_apis += [
                "query_info_collect(query, postgres_config)",
                "rewrite_action_space(query, postgres_config)"
            ]

        if not tool_apis:
            tool_apis = [
                "plan_info_collect(query, postgres_config)",
                "query_info_collect(query, postgres_config)",
                "system_knobs_info_collect(postgres_config)",
                "indexes_info_collect(postgres_config)"
            ]
        
        return tool_apis
        # result = await Runner.run(starting_agent=self.agent, input=prompt)
        # return result.final_output

    async def action_generate(self, root_cause, base_info, action_space, historical_cases, inspirations):
        prompt = self._build_prompt(root_cause, base_info, action_space, historical_cases, inspirations)
        print(f"prompt已生成")
        result = await Runner.run(starting_agent=self.agent, input=prompt)
        return result.final_output