import asyncio
import logging
import time

import nest_asyncio
from agents import Agent, Runner

from agent.action.action_evaluation import evaluate_action
from agent.action.action_space import action_space_collect
from agent.action.action_utils import (
    base_information_collect,
    extract_index_names,
    extract_knob_names,
    parse_query_rewrite,
)
from agent.contracts import parse_tuning_agent_output
from agent.prompt import *
from utils.runtime_metrics import (
    extract_run_usage_metrics,
    format_runtime_metrics,
    merge_runtime_metrics,
)
from utils.openai_agents import configure_openai_agents

nest_asyncio.apply()


# Avoid noisy third-party retry logs during LLM API calls without muting project logs.
def quiet_external_loggers():
    project_prefixes = ("agent", "database", "utils", "__main__")

    for name, logger_obj in logging.root.manager.loggerDict.items():
        if not isinstance(logger_obj, logging.Logger):
            continue
        if name.startswith(project_prefixes):
            continue

        handlers = logger_obj.handlers[:]
        for handler in handlers:
            logger_obj.removeHandler(handler)
        logger_obj.propagate = True
        logger_obj.setLevel(logging.ERROR)


quiet_external_loggers()


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = True


class ActionManager:
    def __init__(self, memory_manager, db, planner, configs):
        self.db = db
        self.memory_manager = memory_manager
        self.planner = planner
        self.agent_config = configs.get("AGENT_CONFIG")
        self.configs = configs

        self.agent = None

        self.fix_model = self.agent_config.get("fix_agent_model")
        self.diagnostic_model = self.agent_config.get("diagnostic_agent_model")
        # Get sql_timeout from DATABASE_CONFIG.query_timeout, fallback to AGENT_CONFIG.sql_execution_timeout
        database_config = configs.get("DATABASE_CONFIG", {})
        self.sql_timeout = database_config.get("query_timeout")
        self.fix_timeout = self.agent_config.get("fix_timeout")

        # Ablation experiment parameters
        self.enable_retrieval = self.agent_config.get("enable_retrieval", True)
        self.retrieval_content = self._normalize_retrieval_content(
            self.agent_config.get("retrieval_content", "all")
        )
        self.enable_action_define = self.agent_config.get("enable_action_define", True)
        self.enable_action_pruning = self.agent_config.get("enable_action_pruning", True)

        self.api_settings = self.configs.get("API_SETTINGS")

        if self.api_settings:
            configure_openai_agents(self.api_settings)

        self.fixAgent = Agent(
            name="FixAgent",
            model=self.fix_model,
            instructions=get_fix_agent_instructions(),
        )

    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.agent:
            self.agent = None

    async def initialize(self):
        """Initialize Agent"""
        self.agent = Agent(
            name="DBDiagnosticAgent",
            model=self.diagnostic_model,
            instructions=get_diagnostic_agent_instructions(),
        )
        return self.agent

    async def cleanup(self):
        try:
            await self.__aexit__(None, None, None)
        except Exception as e:
            print(f"Warning: Error during cleanup: {e}")

    def extract_index_names(self, sql):
        return extract_index_names(sql)

    def extract_knob_names(self, sql):
        return extract_knob_names(sql)

    def _normalize_retrieval_content(self, value):
        normalized = str(value or "all").strip().lower()
        alias_map = {
            "all": "all",
            "full": "all",
            "both": "all",
            "case_only": "case_only",
            "cases_only": "case_only",
            "case": "case_only",
            "cases": "case_only",
            "experience_only": "experience_only",
            "experiences_only": "experience_only",
            "experience": "experience_only",
            "experiences": "experience_only",
            "none": "none",
            "off": "none",
            "disabled": "none",
        }
        return alias_map.get(normalized, "all")

    def _build_prompt_retrieval_context(self, retrieval):
        raw_positives = retrieval.get("positive", [])
        raw_negatives = retrieval.get("negative", [])
        raw_experiences = retrieval.get("experiences", "None.")

        if self.retrieval_content == "none":
            return [], [], "None."
        if self.retrieval_content == "experience_only":
            return [], [], raw_experiences
        if self.retrieval_content == "case_only":
            return raw_positives, raw_negatives, "None."
        return raw_positives, raw_negatives, raw_experiences

    async def step(self, query_info, root_cause, mode="exploit"):
        """Analyze slow query and generate fix action workflow"""
        if not self.agent:
            await self.initialize()

        runtime_metrics = {}

        # Dynamic workflow:
        # 1. information collect
        # 2. match_history
        # 3. action_define
        # 4. action_generate
        # 5. evaluate_action

        database_config = self.configs.get("DATABASE_CONFIG")
        base_info = base_information_collect(query_info, database_config)

        # Retrieve cases based on enable_retrieval flag
        retrieval_start = time.perf_counter()
        if self.enable_retrieval:
            retrieval = self.memory_manager.retrieve_cases(query_info, root_cause, mode=mode)
        else:
            retrieval = {"positive": [], "negative": [], "experiences": "None."}
        positives, negatives, experiences = self._build_prompt_retrieval_context(retrieval)
        retrieved_cases = retrieval.get("positive", []) + retrieval.get("negative", [])
        runtime_metrics["T_retrieval_infer"] = time.perf_counter() - retrieval_start

        action_space_start = time.perf_counter()
        action_space = await action_space_collect(
            root_cause,
            query_info,
            enable_action_define=self.enable_action_define,
            enable_action_pruning=self.enable_action_pruning,
            configs=self.configs,
            db=self.db,
            memory_manager=self.memory_manager,
        )
        runtime_metrics["T_action_space"] = time.perf_counter() - action_space_start

        action_gen_start = time.perf_counter()
        action_run_result = await self.action_generate(root_cause, base_info, action_space, mode, positives, negatives, experiences)
        runtime_metrics["T_action_gen"] = time.perf_counter() - action_gen_start
        runtime_metrics = merge_runtime_metrics(
            runtime_metrics,
            extract_run_usage_metrics(action_run_result, "action_gen"),
        )
        action_result = str(action_run_result.final_output)

        # Extract fix action and rewrite sql
        fix_action, rewrite_sql = self.extract_fix_action(action_result)

        # Validate action
        action_eval_start = time.perf_counter()
        evaluation_result = await evaluate_action(
            fix_action,
            rewrite_sql,
            query_info,
            db=self.db,
            fix_timeout=self.fix_timeout,
            sql_timeout=self.sql_timeout,
            fix_agent=self.fixAgent,
        )
        if evaluation_result.get("fix_action"):
            index_names = extract_index_names(evaluation_result.get("fix_action"))
            knob_names = extract_knob_names(evaluation_result.get("fix_action"))
            self.rollback_action(index_names, knob_names)
        runtime_metrics["T_action_eval"] = time.perf_counter() - action_eval_start

        # Use approve_time as reward (performance improvement: old_time - new_time)
        # approve_time represents the time saved, which is the actual reward signal
        reward = evaluation_result.get("approve_time", 0.0)
        case = retrieved_cases

        # Save experience (independent of retrieval mode, only based on enable_save_samples)
        if self.memory_manager.enable_save_samples and case:
            # Generate next_query_info_id: next round, same query_id
            next_query_info_id = self.memory_manager.generate_next_query_info_id(query_info.query_id)
            
            for case_info in case:
                # root_causes should be extracted from case_info or use the root_cause parameter
                case_root_causes = case_info.get("root_causes", [])
                if not case_root_causes and root_cause:
                    case_root_causes = root_cause if isinstance(root_cause, list) else [root_cause]
                self.memory_manager.save_experience(
                    query_info, 
                    case_info, 
                    case_root_causes,
                    reward,
                    next_query_info_id=next_query_info_id,
                    done=False
                )

        # Only update experience pool and network in dynamic retrieval mode
        retriever_update_start = time.perf_counter()
        if self.memory_manager.retrieval_mode == "dynamic" and case:
            # Generate next_query_info_id: next round, same query_id
            next_query_info_id = self.memory_manager.generate_next_query_info_id(query_info.query_id)
            
            for case_info in case:
                case_root_causes = case_info.get("root_causes", [])
                if not case_root_causes and root_cause:
                    case_root_causes = root_cause if isinstance(root_cause, list) else [root_cause]
                self.memory_manager.retriever.add_experience(
                    query_info,
                    case_info,
                    case_root_causes,
                    reward,
                    next_query_info_id=next_query_info_id,
                    done=False
                )
            self.memory_manager.retriever.update_network()
        runtime_metrics["T_retriever_update"] = time.perf_counter() - retriever_update_start

        evaluation_result["runtime_metrics"] = merge_runtime_metrics(
            evaluation_result.get("runtime_metrics"),
            runtime_metrics,
        )
        evaluation_result["used_cases"] = {
            "positive": retrieval.get("positive", []),
            "negative": retrieval.get("negative", []),
        }
        logger.info(
            "[Runtime][SQL %s] Action summary: %s",
            getattr(query_info, "query_id", "unknown"),
            format_runtime_metrics(evaluation_result["runtime_metrics"]),
        )

        return evaluation_result

    def rollback_action(self, index_names, knob_names):
        # undo all index settings
        for index_name in index_names:
            try:
                with self.db.connection.cursor() as cur:
                    cur.execute("SET statement_timeout = 0")
                    cur.execute(f"DROP INDEX IF EXISTS {index_name}")
                self.db.connection.commit()
                print(f"Rolled back index: {index_name}")
            except Exception as e:
                self.db.connection.rollback()
                print(f"Failed to rollback index: {e}")

        # undo all knob settings
        for knob_name in knob_names:
            try:
                with self.db.connection.cursor() as cur:
                    cur.execute("SET statement_timeout = 0")
                    cur.execute(f"RESET {knob_name};")
                self.db.connection.commit()
                print(f"{knob_name} restored to default value")
            except Exception as e:
                self.db.connection.rollback()
                print(f"Failed to restore {knob_name}: {e}")

    async def action_generate(self, root_cause, base_info, action_space, mode, positives, negatives, experiences):
        prompt = build_action_prompt(root_cause, base_info, action_space, mode, positives, negatives, experiences)
        query_info = base_info.get("query_info")
        # logger.info(
        #     "[Prompt][Tuning][SQL %s]\n%s",
        #     getattr(query_info, "query_id", "unknown"),
        #     prompt,
        # )
        return await Runner.run(starting_agent=self.agent, input=prompt)

    def extract_fix_action(self, action_result):
        """Extract fix_action and rewrite_sql from action_result"""
        explanation, fix_action, is_rewrite = self.parse_action_result(action_result)
        # print("Diagnosis output:")
        # print("Explanation:", explanation)
        print("Fix action:", fix_action)
        print("Is rewrite:", is_rewrite)
        if is_rewrite == "yes":
            fix_action, rewrite_sql = parse_query_rewrite(fix_action)
            return fix_action, rewrite_sql
        else:
            return fix_action, ""

    def parse_action_result(self, diagnosis):
        return parse_tuning_agent_output(diagnosis)
