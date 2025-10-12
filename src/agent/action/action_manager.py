import os
import sys
import asyncio
import yaml
from pathlib import Path
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from contextlib import AsyncExitStack, asynccontextmanager
from agents import Agent, Runner
from agents.mcp import MCPServerStdio, MCPServerSse
from agents._config import set_default_openai_api
from agent.action import diagnose_tools
import nest_asyncio
import re

# 添加项目根目录到Python路径
project_root = str(Path(__file__).parent.parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from agent.memory.memory_manager import MemoryManager
from src.utils.types import QueryInfo, AnomalyInfo
from src.database.pg_env import PostgresDB
from agent.prompt import *

nest_asyncio.apply()

# 避免LLM API调用时打印retry日志
def delete_all_loggers():
    import logging
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for logger in loggers:
        handlers = logger.handlers[:]
        for handler in handlers:
            logger.removeHandler(handler)
        logger.propagate = True
        logger.setLevel(logging.CRITICAL)

delete_all_loggers()


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
            for server in self.servers.values():
                try:
                    await server.__aexit__(type(e), e, e.__traceback__)
                except Exception:
                    pass
            raise
        finally:
            for server in self.servers.values():
                try:
                    await server.__aexit__(None, None, None)
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    print(f"Warning: Error cleaning up server: {e}")
            self.servers.clear()

class ActionManager:
    def __init__(self, memory_manager, db, planner, configs):
        self.db = db
        self.memory_manager = memory_manager
        self.planner = planner
        self.agent_config = configs.get('AGENT_CONFIG')
        self.configs = configs

        
        self.agent = None
        self.server_manager = None
        self.exit_stack = AsyncExitStack()
        self._servers = []
        
        self.fix_model = self.agent_config.get('fix_agent_model')
        self.diagnostic_model = self.agent_config.get('diagnostic_agent_model')
        self.sql_timeout = self.agent_config.get('sql_execution_timeout')
        self.config_path = self.agent_config.get('action_server_config')
        self.fix_timeout = self.agent_config.get('fix_timeout')

        self.api_settings = self.configs.get('API_SETTINGS')
        
        if self.api_settings:
            openai_config = self.api_settings.get('openai')
            os.environ['OPENAI_API_KEY'] = openai_config.get('api_key')
            os.environ['OPENAI_BASE_URL'] = openai_config.get('base_url')
            set_default_openai_api("chat_completions")

        self.fixAgent = Agent(
            name="FixAgent",
            model=self.fix_model,
            instructions=get_fix_agent_instructions(),
        )

    async def __aenter__(self):
        """异步上下文管理器的进入方法"""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器的退出方法"""
        if self.agent:
            self.agent = None

        try:
            await self.exit_stack.aclose()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"Warning: Error during exit stack cleanup: {e}")

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
            model=self.diagnostic_model,
            instructions=get_diagnostic_agent_instructions(),
            mcp_servers=self._servers,
        )
        return self.agent

    async def cleanup(self):
        try:
            await self.__aexit__(None, None, None)
        except Exception as e:
            print(f"Warning: Error during cleanup: {e}")

    async def step(self, query_info, root_cause, mode="exploit"):
        """分析慢查询并生成修复操作的工作流"""
        if not self.agent:
            await self.initialize()
        
        # workflow:
        # 1. information collect
        # 2. match_history
        # 3. action_define
        # 4. action_generate
        # 5. evaluate_action
        postgres_config = self.configs.get('POSTGRES_CONFIG')
        base_info = diagnose_tools.base_information_collect(query_info, postgres_config)
        
        retrieval = self.memory_manager.retrieve_cases(query_info, root_cause, mode=mode)
        positives = retrieval.get('positive', [])
        negatives = retrieval.get('negative', [])
        
        action_space = await self.action_space_collect(root_cause, query_info)
        action_result = await self.action_generate(root_cause, base_info, action_space, mode, positives, negatives)

        return action_result

    @staticmethod
    def extract_index_names(sql):
        return re.findall(r'CREATE\s+INDEX\s+(?:IF\s+NOT\s+EXISTS\s+)?([\w_]+)', sql, re.IGNORECASE)
        
    @staticmethod
    def extract_knob_names(sql):
        return re.findall(r'SET\s+([a-zA-Z0-9_]+)\s*=', sql, re.IGNORECASE)

    def rollback_action(self, index_names, knob_names):
        # undo all index settings
        for index_name in index_names:
            try:  
                with self.db.connection.cursor() as cur:
                    cur.execute(f"SET statement_timeout = 0")
                    cur.execute(f"DROP INDEX IF EXISTS {index_name}")
                self.db.connection.commit()
                print(f"已撤销索引: {index_name}")
            except Exception as e:
                self.db.connection.rollback()
                print(f"撤销索引失败: {e}")

        # undo all knob settings
        for knob_name in knob_names:
            try:
                with self.db.connection.cursor() as cur:
                    cur.execute(f"SET statement_timeout = 0")
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
        # 在执行前对 fix_action / rewrite_sql 进行预处理，得到最终 fix_action 与初始 fix_sql
        fix_action, fix_sql = await self.derive_fix_action_and_sql(fix_action, rewrite_sql, query_info.query)

        # 1. 检查修复SQL能否执行（如有 DDL/SET 等修复，需先执行）
        if fix_action != "":
            success, _, _, err = self.db.execute(fix_action, timeout=self.fix_timeout)
        else:
            success = True
            err = ""
        retry_count = 0
        max_retry = 5
        while not success and retry_count < max_retry:
            prompt = build_fix_prompt(fix_action, err)
            result = await Runner.run(starting_agent=self.fixAgent, input=prompt)
            new_fix_action = result.final_output
            print(f"new_fix_action: {new_fix_action}")
            new_fix_action, new_fix_sql = await self.derive_fix_action_and_sql(new_fix_action, rewrite_sql, query_info.query)
            success, _, _, err = self.db.execute(new_fix_action, timeout=self.fix_timeout)
            fix_action = new_fix_action
            fix_sql = new_fix_sql
            retry_count += 1
        if not success:
            return {"status": -1, "msg": f"SQL修复失败: {fix_action}, 错误: {err}", "fix_action": fix_action, "rewrite_sql": (fix_sql if fix_sql != query_info.query else ""), "new_time": 20}

        old_time = query_info.execution_time

        # 执行性能测试
        success, _, new_time, err = self.db.execute(fix_sql, timeout=self.sql_timeout)

        if fix_action != "":
            index_names = self.extract_index_names(fix_action)
            knob_names = self.extract_knob_names(fix_action)
            self.rollback_action(index_names, knob_names)

        if not success:
            if err and isinstance(err, str) and ("timeout" in err.lower() or "querycanceled" in err):
                return {"status": 0, "msg": f"修复无效，性能未提升或下降，原耗时{old_time:.4f}s，新耗时{new_time:.4f}s", "fix_action": fix_action, "rewrite_sql": (fix_sql if fix_sql != query_info.query else ""), "new_time": new_time}
            return {"status": -1, "msg": f"性能测试SQL执行失败: {err}", "fix_action": fix_action, "rewrite_sql": (fix_sql if fix_sql != query_info.query else ""), "new_time": 20}

        # 4. 性能对比与后续处理
        if (old_time - new_time) / old_time > 0.1:
            return {"status": 1, "msg": f"修复成功，性能提升，耗时由{old_time:.4f}s降至{new_time:.4f}s", "fix_action": fix_action, "rewrite_sql": (fix_sql if fix_sql != query_info.query else ""), "new_time": new_time}
        else:
            return {"status": 0, "msg": f"修复无效，性能未提升或下降，原耗时{old_time:.4f}s，新耗时{new_time:.4f}s", "fix_action": fix_action, "rewrite_sql": (fix_sql if fix_sql != query_info.query else ""), "new_time": new_time}

    async def derive_fix_action_and_sql(self, fix_action, rewrite_sql, origin_sql):
        # extract hint
        match = re.search(r"/\*\+[\s\S]*?\*/", fix_action)
        if match:
            hint = match.group(0).strip()
            # remove the hint from fix_action
            clean_fix_action = (fix_action[:match.start()] + fix_action[match.end():]).strip()
            if rewrite_sql != "":
                await self.check_sql_rewrite(origin_sql, rewrite_sql)
                target_sql = rewrite_sql
            else:
                target_sql = origin_sql
            fix_sql = self._inject_hint_into_sql(hint, target_sql)
            return clean_fix_action, fix_sql
        else:
            # no hint
            clean_fix_action = fix_action
            if rewrite_sql != "":
                await self.check_sql_rewrite(origin_sql, rewrite_sql)
                fix_sql = rewrite_sql
            else:
                fix_sql = origin_sql
            return clean_fix_action, fix_sql

    def _inject_hint_into_sql(self, hint, sql):
        pattern = r'\b(WITH|SELECT|EXPLAIN)\b'
        
        def replace_match(match):
            # check if in parentheses
            start_pos = match.start()
            keyword = match.group(1).upper()
            
            # count the number of left and right parentheses
            left_count = sql[:start_pos].count('(')
            right_count = sql[:start_pos].count(')')
            
            if left_count > right_count:
                return match.group(0)
            
            # special handling: if SELECT and in view definition (previous "as" keyword)
            if keyword == 'SELECT':
                # simple check: if previous "as" keyword, not inject hint
                before_text = sql[:start_pos].lower().strip()
                if before_text.endswith('as'):
                    return match.group(0)
            
            # special handling: if WITH, not inject hint (only inject before main SELECT)
            if keyword == 'WITH':
                return match.group(0)
            
            # if previous hint, not add again
            if start_pos > 0:
                prefix = sql[:start_pos].strip()
                if prefix.endswith(hint.strip()):
                    return match.group(0)
            
            return hint + ' ' + match.group(0)
        
        result = re.sub(pattern, replace_match, sql, flags=re.IGNORECASE)
        return result

    async def check_sql_rewrite(self, origin_sql, rewrite_sql):
        retry_count = 0
        max_retry = 5
        match = self.db.compare_sql_results(origin_sql, rewrite_sql)
        print(f"match: {match}")
        while not match and retry_count < max_retry:
            prompt = build_rewrite_prompt(origin_sql, rewrite_sql)
            result = await Runner.run(starting_agent=self.fixAgent, input=prompt)
            new_rewrite_sql = result.final_output
            print(f"new_rewrite_sql: {new_rewrite_sql}")
            rewrite_sql = new_rewrite_sql
            retry_count += 1
            match = self.db.compare_sql_results(origin_sql, rewrite_sql)
            print(f"match: {match}")

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

    async def action_space_collect(self, root_causes, query_info=None):
        knob_config = self.configs.get('KNOB_CONFIG')
        current_knob_values = self.db.get_current_knob_values(knob_config)
        return build_action_space_prompt(root_causes, self.db, knob_config, current_knob_values, query_info)

    async def action_generate(self, root_cause, base_info, action_space, mode, positives, negatives):
        prompt = build_action_prompt(root_cause, base_info, action_space, mode, positives, negatives)
        print(f"prompt: {prompt}")
        print(f"prompt已生成")
        result = await Runner.run(starting_agent=self.agent, input=prompt)
        return result.final_output