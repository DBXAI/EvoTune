import asyncio
import os
import re
import sys
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path

import nest_asyncio
import numpy as np
import yaml
from agents import Agent, Runner
from agents._config import set_default_openai_api
from agents.mcp import MCPServerSse, MCPServerStdio
from scipy.stats import spearmanr

from agent.action import diagnose_tools

# 添加项目根目录到Python路径
project_root = str(Path(__file__).parent.parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

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
        server_type = MCPServerStdio if config["server_type"] == "stdio" else MCPServerSse
        if "cwd" in config["params"] and config["params"]["cwd"] == ".":
            config["params"]["cwd"] = os.path.dirname(os.path.abspath(__file__))
        return cls(name=config["name"], server_type=server_type, params=config["params"])


class MultiServerManager:
    def __init__(self, configs):
        self.configs = configs
        self.servers = {}

    @asynccontextmanager
    async def manage_servers(self):
        try:
            for config in self.configs:
                server = await config.server_type(name=config.name, params=config.params).__aenter__()
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
        self.agent_config = configs.get("AGENT_CONFIG")
        self.configs = configs

        self.agent = None
        self.server_manager = None
        self.exit_stack = AsyncExitStack()
        self._servers = []

        self.fix_model = self.agent_config.get("fix_agent_model")
        self.diagnostic_model = self.agent_config.get("diagnostic_agent_model")
        self.sql_timeout = self.agent_config.get("sql_execution_timeout")
        self.config_path = self.agent_config.get("action_server_config")
        self.fix_timeout = self.agent_config.get("fix_timeout")

        self.api_settings = self.configs.get("API_SETTINGS")

        if self.api_settings:
            openai_config = self.api_settings.get("openai")
            os.environ["OPENAI_API_KEY"] = openai_config.get("api_key")
            os.environ["OPENAI_BASE_URL"] = openai_config.get("base_url")
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
        with open(self.config_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)

        configs = [ServerConfig.from_dict(service_config) for service_config in config_data["services"]]
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

        # 动态workflow:
        # 1. information collect
        # 2. match_history
        # 3. action_define
        # 4. action_generate
        # 5. evaluate_action

        postgres_config = self.configs.get("POSTGRES_CONFIG")
        base_info = diagnose_tools.base_information_collect(query_info, postgres_config)

        retrieval = self.memory_manager.retrieve_cases(query_info, root_cause, mode=mode)
        positives = retrieval.get("positive", [])
        negatives = retrieval.get("negative", [])
        action_space = await self.action_space_collect(root_cause, query_info)

        action_result = await self.action_generate(root_cause, base_info, action_space, mode, positives, negatives)

        # 提取fix action和rewrite sql
        fix_action, rewrite_sql = self.extract_fix_action(action_result)

        # 进行action验证
        evaluation_result = await self.evaluate_action(fix_action, rewrite_sql, query_info)

        reward = evaluation_result.get("status", 0)
        case = positives + negatives

        # 保存经验（与检索模式无关，仅根据enable_save_samples判断）
        if self.memory_manager.enable_save_samples and case:
            for case_info in case:
                self.memory_manager.save_experience(query_info.__dict__, case_info, reward)

        # 只在动态检索模式下更新经验池和网络
        if self.memory_manager.retrieval_mode == "dynamic" and case:
            for case_info in case:
                self.memory_manager.retriever.add_experience(query_info.__dict__, case_info, reward)
            self.memory_manager.retriever.update_network()

        return evaluation_result

    @staticmethod
    def extract_index_names(sql):
        return re.findall(r"CREATE\s+INDEX\s+(?:IF\s+NOT\s+EXISTS\s+)?([\w_]+)", sql, re.IGNORECASE)

    @staticmethod
    def extract_knob_names(sql):
        return re.findall(r"SET\s+([a-zA-Z0-9_]+)\s*=", sql, re.IGNORECASE)

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

        print(f"execute fix_action")

        # 1. 检查修复SQL能否执行（如有 DDL/SET 等修复，需先执行）
        if fix_action != "":
            success, _, _, err = self.db.execute(fix_action, timeout=self.fix_timeout)
        else:
            success = True
            err = ""
        print(f"execute fix_action done")

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
            return {
                "status": -1,
                "msg": f"SQL修复失败: {fix_action}, 错误: {err}",
                "fix_action": fix_action,
                "rewrite_sql": (fix_sql if fix_sql != query_info.query else ""),
                "new_time": 20,
            }

        old_time = query_info.execution_time

        print(f"execute performance test")
        # 执行性能测试
        success, _, new_time, err = self.db.execute(fix_sql, timeout=self.sql_timeout)
        print(f"execute performance test done")

        if fix_action != "":
            index_names = self.extract_index_names(fix_action)
            knob_names = self.extract_knob_names(fix_action)
            self.rollback_action(index_names, knob_names)

        if not success:
            if err and isinstance(err, str) and ("timeout" in err.lower() or "querycanceled" in err):
                return {
                    "status": 0,
                    "msg": f"修复无效，性能未提升或下降，原耗时{old_time:.4f}s，新耗时{new_time:.4f}s",
                    "fix_action": fix_action,
                    "rewrite_sql": (fix_sql if fix_sql != query_info.query else ""),
                    "new_time": new_time,
                }
            return {
                "status": -1,
                "msg": f"性能测试SQL执行失败: {err}",
                "fix_action": fix_action,
                "rewrite_sql": (fix_sql if fix_sql != query_info.query else ""),
                "new_time": 20,
            }

        # 4. 性能对比与后续处理
        if (old_time - new_time) / old_time > 0.1:
            return {
                "status": 1,
                "msg": f"修复成功，性能提升，耗时由{old_time:.4f}s降至{new_time:.4f}s",
                "fix_action": fix_action,
                "rewrite_sql": (fix_sql if fix_sql != query_info.query else ""),
                "new_time": new_time,
            }
        else:
            return {
                "status": 0,
                "msg": f"修复无效，性能未提升或下降，原耗时{old_time:.4f}s，新耗时{new_time:.4f}s",
                "fix_action": fix_action,
                "rewrite_sql": (fix_sql if fix_sql != query_info.query else ""),
                "new_time": new_time,
            }

    async def derive_fix_action_and_sql(self, fix_action, rewrite_sql, origin_sql):
        # extract hint
        match = re.search(r"/\*\+[\s\S]*?\*/", fix_action)
        if match:
            hint = match.group(0).strip()
            # remove the hint from fix_action
            clean_fix_action = (fix_action[: match.start()] + fix_action[match.end() :]).strip()
            if rewrite_sql != "":
                rewrite_sql = await self.check_sql_rewrite(origin_sql, rewrite_sql)
                target_sql = rewrite_sql
            else:
                target_sql = origin_sql
            fix_sql = self._inject_hint_into_sql(hint, target_sql)
            return clean_fix_action, fix_sql
        else:
            # no hint
            clean_fix_action = fix_action
            if rewrite_sql != "":
                rewrite_sql = await self.check_sql_rewrite(origin_sql, rewrite_sql)
                fix_sql = rewrite_sql
            else:
                fix_sql = origin_sql
            return clean_fix_action, fix_sql

    def _inject_hint_into_sql(self, hint, sql):
        pattern = r"\b(WITH|SELECT|EXPLAIN)\b"

        def replace_match(match):
            # check if in parentheses
            start_pos = match.start()
            keyword = match.group(1).upper()

            # count the number of left and right parentheses
            left_count = sql[:start_pos].count("(")
            right_count = sql[:start_pos].count(")")

            if left_count > right_count:
                return match.group(0)

            # special handling: if SELECT and in view definition (previous "as" keyword)
            if keyword == "SELECT":
                # simple check: if previous "as" keyword, not inject hint
                before_text = sql[:start_pos].lower().strip()
                if before_text.endswith("as"):
                    return match.group(0)

            # special handling: if WITH, not inject hint (only inject before main SELECT)
            if keyword == "WITH":
                return match.group(0)

            # if previous hint, not add again
            if start_pos > 0:
                prefix = sql[:start_pos].strip()
                if prefix.endswith(hint.strip()):
                    return match.group(0)

            return hint + " " + match.group(0)

        result = re.sub(pattern, replace_match, sql, flags=re.IGNORECASE)
        return result

    async def check_sql_rewrite(self, origin_sql, rewrite_sql):
        print(f"start QED check")
        qed_match = self.db.check_sql_equivalence(origin_sql, rewrite_sql)
        print(f"QED equivalence check: {qed_match}")

        if qed_match:
            return rewrite_sql

        retry_count = 0
        max_retry = 5
        match = self.db.compare_sql_results(origin_sql, rewrite_sql)
        print(f"Result set comparison match: {match}")

        while not match and retry_count < max_retry:
            prompt = build_rewrite_prompt(origin_sql, rewrite_sql)
            result = await Runner.run(starting_agent=self.fixAgent, input=prompt)
            new_rewrite_sql = result.final_output
            print(f"new_rewrite_sql: {new_rewrite_sql}")
            rewrite_sql = new_rewrite_sql
            retry_count += 1
            match = self.db.compare_sql_results(origin_sql, rewrite_sql)
            print(f"Result set comparison match: {match}")

        if not match:
            print("Failed to rewrite SQL, return original SQL")
            return origin_sql

        return rewrite_sql

    async def action_space_collect(self, root_causes, query_info=None):
        """
        Dynamically collect action space, perform pruning and refinement based on historical cases
        """
        knob_config = self.configs.get("KNOB_CONFIG")
        current_knob_values = self.db.get_current_knob_values(knob_config)

        base_action_space = build_action_space_prompt(root_causes, self.db, knob_config, current_knob_values, query_info)

        # Action Pruning: 基于历史同根因案例避免无效操作
        pruning_guidance = self.action_pruning(root_causes, query_info)

        # Action Refinement: 基于历史同根因案例指导有效搜索方向
        refinement_guidance = self.action_refinement(root_causes, query_info)

        # 整合pruning和refinement指导到action space
        enhanced_action_space = self._integrate_pruning_and_refinement(base_action_space, pruning_guidance, refinement_guidance)

        return enhanced_action_space

    def action_pruning(self, root_causes, query_info):
        has_index_root = "missing indexes" in root_causes
        has_query_root = "poorly written queries" in root_causes

        if not has_index_root and not has_query_root:
            return ""

        pruning_guidance = []
        pruning_guidance.append("• Action Pruning: Based on historical failure cases with the same root cause, the following discrete operations have been proven ineffective, please avoid using them:")

        if has_index_root:
            failed_index_actions = self._get_failed_index_actions(root_causes, query_info)
            if failed_index_actions:
                pruning_guidance.append("The ineffective indexes:")
                for index_name in sorted(failed_index_actions):
                    pruning_guidance.append(f"{index_name}")

        if has_query_root:
            applicable_rules = self._get_applicable_rewrite_rules(query_info)
            if applicable_rules:
                pruning_guidance.append(f"The ineffective query rewrite rules:")
                for rule in applicable_rules:
                    pruning_guidance.append(f"{rule}")

        return "\n".join(pruning_guidance)

    def _get_failed_index_actions(self, root_causes, query_info):
        """
        Retrieve historical fix_actions with the same root cause from memory_manager, extract failed index names
        """
        # 获取同根因的历史案例
        target_root = self.memory_manager._normalize_root_causes(root_causes)
        target_island_ids = list(self.memory_manager.islands.get(target_root, set()))

        if not target_island_ids:
            return set()

        # 获取同根因的案例
        same_root_cases = self.memory_manager.ids_to_cases(target_island_ids)

        # 进一步筛选同SQL的案例
        query_id = query_info.query_id
        same_sql_cases = []
        for case in same_root_cases:
            if case.case_info.get("query_info").get("query_id") == query_id:
                same_sql_cases.append(case)

        # 如果没有同SQL的案例，使用所有同根因的案例
        if not same_sql_cases:
            return set()

        # 提取失败的索引名称
        failed_index_actions = set()
        for case in same_sql_cases:
            # 只处理negative cases或性能没有提升的案例
            case_label = case.case_info.get("label")
            if case_label == "negative":
                fix_action = case.case_info.get("fix_action")
                if fix_action:
                    index_names = self.extract_index_names(str(fix_action))
                    for index_name in index_names:
                        failed_index_actions.add(index_name)

        return failed_index_actions

    def _get_applicable_rewrite_rules(self, query_info):
        target_root = self.memory_manager._normalize_root_causes(["poorly written queries"])
        target_island_ids = list(self.memory_manager.islands.get(target_root, set()))

        if not target_island_ids:
            return []

        same_root_cases = self.memory_manager.ids_to_cases(target_island_ids)

        query_id = query_info.query_id
        same_sql_cases = []
        for case in same_root_cases:
            if case.case_info.get("query_info").get("query_id") == query_id:
                same_sql_cases.append(case)

        if not same_sql_cases:
            return []

        failed_rules = set()
        # 使用当前查询作为原始查询（同SQL案例的原始查询应该是相同的）
        original_query = query_info.query

        for case in same_sql_cases:
            case_label = case.case_info.get("label")
            if case_label == "negative":
                # 获取历史案例中的改写后查询
                rewritten_query = case.case_info.get("query_info").get("query")

                from agent.action.diagnose_tools import SQLRuleMatcher

                rule_matcher = SQLRuleMatcher()

                applicable_rules = rule_matcher.match_rules(original_query)
                for rule in applicable_rules:
                    if rule_matcher._rule_used_in_rewrite(rule, original_query, rewritten_query):
                        failed_rules.add(rule["name"])

        failed_rule_info = []
        for rule_name in failed_rules:
            from agent.action.diagnose_tools import SQLRuleMatcher

            rule_matcher = SQLRuleMatcher()
            rule_info = rule_matcher.get_rule_info(rule_name)
            if rule_info:
                failed_rule_info.append(rule_info)

        return failed_rule_info

    def action_refinement(self, root_causes, query_info):
        has_knob_root = "inappropriate query knobs" in root_causes
        has_hint_root = "suboptimal plan optimizer" in root_causes

        if not has_knob_root and not has_hint_root:
            return ""

        refinement_guidance = []
        refinement_guidance.append("• Action Refinement: Based on historical success cases with the same root cause, the following continuous operations have been proven effective, suggest prioritizing exploration:")

        if has_knob_root:
            knob_guidance = self._get_knob_refinement_guidance(root_causes, query_info)
            if knob_guidance:
                refinement_guidance.append(knob_guidance)

        if has_hint_root:
            hint_guidance = self._get_hint_refinement_guidance(root_causes, query_info)
            if hint_guidance:
                refinement_guidance.append(hint_guidance)

        return "\n".join(refinement_guidance)

    def _get_knob_refinement_guidance(self, root_causes, query_info, min_threshold=10):
        target_root = self.memory_manager._normalize_root_causes(root_causes)
        target_island_ids = list(self.memory_manager.islands.get(target_root))

        if not target_island_ids:
            return ""

        same_root_cases = self.memory_manager.ids_to_cases(target_island_ids)

        # 使用embedding相似度检索相似SQL案例
        similar_cases = self.memory_manager.retrieve_embedding(query_info, top_n=50)
        # 过滤出同根因的相似案例
        similar_same_root_cases = []
        for case in similar_cases:
            case_root = self.memory_manager._normalize_root_causes(case.case_info.get("root_causes"))
            if case_root == target_root:
                similar_same_root_cases.append(case)

        if not similar_same_root_cases:
            return ""

        # 过滤出approve_time > 0的案例
        positive_cases = []
        for case in similar_same_root_cases:
            approve_time = case.case_info.get("approve_time")
            if approve_time and approve_time > 0:
                positive_cases.append(case)

        # 需要cases大于指定数量才提供建议
        if len(positive_cases) < min_threshold:
            return ""

        knob_data = self._extract_knob_data(positive_cases)

        if not knob_data:
            return ""

        guidance = []
        guidance.append("\n• Knob Tuning Space Refinement: Based on historical successful cases, the following knob value ranges are recommended for exploration:")

        for knob_name in knob_data.keys():
            values = knob_data[knob_name]["values"]

            if len(values) < min_threshold:
                continue

            # 找到最大值和最小值
            min_value = min(values)
            max_value = max(values)

            guidance.append(f"- Knob '{knob_name}': Recommended range [{min_value}, {max_value}],  Based on {len(values)} successful historical cases")

        if len(guidance) <= 1:  # 只有标题，没有具体建议
            return ""
        else:
            return "\n".join(guidance)

    def _get_hint_refinement_guidance(self, root_causes, query_info):
        target_root = self.memory_manager._normalize_root_causes(root_causes)
        target_island_ids = list(self.memory_manager.islands.get(target_root))

        if not target_island_ids:
            return ""

        same_root_cases = self.memory_manager.ids_to_cases(target_island_ids)

        query_id = query_info.query_id
        same_sql_cases = []
        for case in same_root_cases:
            if case.case_info.get("query_info").get("query_id") == query_id:
                same_sql_cases.append(case)

        if not same_sql_cases:
            return ""

        hint_data = self._extract_hint_usage_data(same_sql_cases)

        if not hint_data:
            return ""

        guidance = []

        positive_hints = []
        for hint_key, hint_stats in hint_data.items():
            if hint_stats["avg_improvement"] > 0:
                positive_hints.append((hint_key, hint_stats))

        if positive_hints:
            guidance.append("\n• Query Hint Refinement: Based on historical hint usage data, the following hints have shown positive performance impact:")
            positive_hints.sort(key=lambda x: x[1]["avg_improvement"], reverse=True)

            for hint_key, hint_stats in positive_hints:
                guidance.append(f"- Hint '{hint_key}': Average improvement {hint_stats['avg_improvement']:.1f}%")

        return "\n".join(guidance)

    def _extract_knob_data(self, cases):
        knob_data = {}

        for case in cases:
            fix_action = case.case_info.get("fix_action")
            knob_settings = self._extract_knob_settings(str(fix_action))
            approve_time = case.case_info.get("approve_time")

            for knob_name, knob_value in knob_settings.items():
                if knob_name not in knob_data:
                    knob_data[knob_name] = {
                        "values": [],
                        "performance": [],
                        "best_value": None,
                        "best_performance": -float("inf"),
                    }

                knob_data[knob_name]["values"].append(knob_value)
                knob_data[knob_name]["performance"].append(approve_time)

                if approve_time > knob_data[knob_name]["best_performance"]:
                    knob_data[knob_name]["best_performance"] = approve_time
                    knob_data[knob_name]["best_value"] = knob_value

        return knob_data

    def _extract_hint_usage_data(self, cases):
        hint_data = {}

        for case in cases:
            query = case.case_info.get("query_info").get("query")

            hints = self._extract_hints_from_action(str(query))
            if not hints:
                continue

            performance_improvement = case.case_info.get("approve_time")

            for hint in hints:
                hint_key = hint["full_hint"]  # Use full hint as key
                if hint_key not in hint_data:
                    hint_data[hint_key] = {
                        "knob": hint["knob"],
                        "value": hint["value"],
                        "improvements": [],
                        "usage_count": 0,
                        "avg_improvement": 0,
                    }

                hint_data[hint_key]["improvements"].append(performance_improvement)
                hint_data[hint_key]["usage_count"] += 1
                hint_data[hint_key]["avg_improvement"] = sum(hint_data[hint_key]["improvements"]) / len(hint_data[hint_key]["improvements"])

        return hint_data

    def _extract_knob_settings(self, action_str):
        knob_settings = {}

        set_pattern = r"set\s+(\w+)\s*=\s*([^\s;]+)"
        matches = re.findall(set_pattern, action_str.lower())

        for knob_name, knob_value in matches:
            knob_settings[knob_name] = knob_value

        return knob_settings

    def _extract_hints_from_action(self, action_str):
        hints = []

        # Pattern to match /*+ ... */
        hint_pattern = r"/\*\+([^*]+)\*/"
        matches = re.findall(hint_pattern, action_str)

        for match in matches:
            set_pattern = r"Set\(([^)]+)\)"
            set_matches = re.findall(set_pattern, match)

            for set_match in set_matches:
                parts = set_match.strip().split()
                if len(parts) >= 2:
                    knob = parts[0]
                    value = " ".join(parts[1:])
                    hints.append(
                        {
                            "knob": knob,
                            "value": value,
                            "full_hint": f"Set({knob} {value})",
                        }
                    )

        return hints

    def _integrate_pruning_and_refinement(self, base_action_space, pruning_guidance, refinement_guidance):
        """
        将pruning和refinement指导整合到action space中
        """
        enhanced_space = base_action_space

        if pruning_guidance:
            enhanced_space += f"\n\n{pruning_guidance}"

        if refinement_guidance:
            enhanced_space += f"\n\n{refinement_guidance}"

        # if pruning_guidance or refinement_guidance:
        #     enhanced_space += "\n\n## Comprehensive Optimization Strategy"
        #     enhanced_space += "\n- Prioritize operation types and strategies from historical success cases"
        #     enhanced_space += "\n- Avoid operation types and strategies from historical failure cases"
        #     enhanced_space += "\n- Refine and innovate within successful areas, rather than completely copying"
        #     enhanced_space += "\n- Adjust historical success strategies based on current query characteristics"

        return enhanced_space

    def _analyze_knob_pattern(self, x, y):
        x, y = np.array(x), np.array(y)
        if len(x) < 3:
            return "insufficient data", None

        # Calculate Spearman correlation
        corr, _ = spearmanr(x, y)
        if corr < -0.5:
            return "monotonic decreasing", corr
        elif corr > 0.5:
            return "monotonic increasing", corr

        # Find local extrema points
        try:
            minima = argrelextrema(y, np.less)[0]
            maxima = argrelextrema(y, np.greater)[0]

            if len(minima) > 0:
                best_idx = minima[np.argmin(y[minima])]
                return f"local minimum around {x[best_idx]}", (x[best_idx], y[best_idx])
            elif len(maxima) > 0:
                worst_idx = maxima[np.argmax(y[maxima])]
                return f"local maximum around {x[worst_idx]}", (
                    x[worst_idx],
                    y[worst_idx],
                )
            else:
                return "no clear pattern", None
        except:
            return "insufficient data", None

    async def action_generate(self, root_cause, base_info, action_space, mode, positives, negatives):
        prompt = build_action_prompt(root_cause, base_info, action_space, mode, positives, negatives)
        # print(f"prompt: {prompt}")
        # print(f"prompt已生成")
        result = await Runner.run(starting_agent=self.agent, input=prompt)
        return result.final_output

    def extract_fix_action(self, action_result):
        """从action_result中提取fix_action和rewrite_sql"""
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

    def parse_action_result(self, diagnosis):
        # extract Explanation
        explanation_match = re.search(r"Explanation:\s*([\s\S]+?)(?=Fix_Action:|$)", diagnosis, re.IGNORECASE)
        explanation = explanation_match.group(1).strip() if explanation_match else ""

        # extract FIX_ACTION
        fix_match = re.search(r"Fix_Action:\s*([\s\S]+?)Query_Rewrite:", diagnosis, re.IGNORECASE)
        fix_action = fix_match.group(1).strip() if fix_match else ""

        # extract Query_Rewrite
        is_rewrite_match = re.search(r"Query_Rewrite:\s*(yes|no)", diagnosis, re.IGNORECASE)
        is_rewrite = is_rewrite_match.group(1).strip() if is_rewrite_match else ""

        return explanation, fix_action, is_rewrite

    def parse_query_rewrite(self, fix_action):
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
