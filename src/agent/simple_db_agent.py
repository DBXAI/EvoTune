import asyncio
import logging
import os
import re
import time
from agents import Agent, Runner
from agent.action.action_manager import ActionManager
from agent.action.action_evaluation import evaluate_action
from agent.action.action_utils import base_information_collect, parse_query_rewrite
from agent.contracts import parse_tuning_agent_output
from agent.memory.memory_manager import MemoryManager
from agent.plan.online_predict import RCRankPredictor
from agent.prompt import (
    build_simple_retrieval_prompt,
    get_simple_retrieval_agent_instructions,
)
from database.pg_env import PostgresDB
from database.tidb_env import TiDB
from utils.openai_agents import configure_openai_agents
from utils.types import QueryInfo

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = True

class SimpleDBAgent:
    def __init__(self, configs=None):
        self.configs = configs
        self.agent_config = configs.get("AGENT_CONFIG")
        self.memory_manager = None

        db_type = configs.get("DATABASE_TYPE", "postgres")
        db_config = None
        if db_type.lower() == "postgres":
            db_config = configs.get("DATABASE_CONFIG")
            self.db = PostgresDB(db_config)
        elif db_type.lower() == "tidb":
            db_config = configs.get("TiDB_CONFIG")
            self.db = TiDB(db_config)
        else:
            raise ValueError(f"Unsupported database type: {db_type}")

        self.db_config = db_config
        self.query_best_time_workload = configs.get("WORKLOAD_QUERY_BEST_TIME_NAME")
        self.query_best_times = self._normalize_query_best_times(
            configs.get("WORKLOAD_QUERY_BEST_TIMES")
        )

        self.diagnostic_model = self.agent_config.get("diagnostic_agent_model")
        self.api_settings = self.configs.get("API_SETTINGS")
        self.enable_action_define = self.agent_config.get("enable_action_define", True)
        self.enable_action_pruning = self.agent_config.get("enable_action_pruning", True)
        self.simple_case_pos_n = self.agent_config.get("simple_case_pos_n", 2)
        self.simple_case_neg_n = self.agent_config.get("simple_case_neg_n", 2)

        if self.api_settings:
            configure_openai_agents(self.api_settings)

        planner_config = configs.get("PLANNER_CONFIG")
        memory_config = configs.get("MEMORY_MANAGER_CONFIG")
        if memory_config:
            predictor = RCRankPredictor(planner_config) if planner_config else None
            self.memory_manager = MemoryManager(
                memory_config,
                predictor=predictor,
                planner_config=planner_config,
                agent_config=self.agent_config,
                api_settings=configs.get("API_SETTINGS"),
            )
            self.memory_manager.retrieval_mode = "static"

        self.diagnostic_agent = Agent(
            name="SimpleRetrievalAgent",
            model=self.diagnostic_model,
            instructions=get_simple_retrieval_agent_instructions(),
        )

        self.action_manager = ActionManager(
            memory_manager=self.memory_manager,
            db=self.db,
            planner=None,
            configs=configs,
        )

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.cleanup()

    async def cleanup(self):
        try:
            if self.action_manager:
                try:
                    await asyncio.shield(self.action_manager.cleanup())
                except Exception as e:
                    logger.warning(f"Error cleaning up action_manager: {e}")

            if self.db:
                try:
                    if hasattr(self.db, "cleanup") and callable(getattr(self.db, "cleanup")):
                        await self.db.cleanup()
                    elif hasattr(self.db, "close"):
                        self.db.close()
                except Exception as e:
                    logger.warning(f"Error closing database connection: {e}")

            logger.info("SimpleDBAgent resource cleanup completed")

        except Exception as e:
            logger.error(f"Error during resource cleanup: {e}")
        finally:
            pass

    def parse_action_result(self, diagnosis):
        return parse_tuning_agent_output(diagnosis)

    def extract_fix_action(self, action_result):
        explanation, fix_action, is_rewrite = self.parse_action_result(action_result)
        print("diagnosis output:")
        print("explanation:", explanation)
        print("fix_action:", fix_action)
        print("is_rewrite:", is_rewrite)
        if is_rewrite == "yes":
            fix_action, rewrite_sql = parse_query_rewrite(fix_action)
            return fix_action, rewrite_sql
        else:
            return fix_action, ""

    def read_queries(self, slow_query_path, order):
        queries_order = []
        with open(os.path.join(slow_query_path, order), "r", encoding="utf-8") as f:
            for fname in f:
                fname = fname.strip()
                if fname.endswith(".sql"):
                    queries_order.append(os.path.join(slow_query_path, fname))
        return queries_order

    def clean_query(self, sql):
        sql = re.sub(r"--.*?(\n|$)", "", sql)
        sql = sql.strip()
        sql = re.sub(r"\s+", " ", sql)
        return sql

    def _normalize_query_id(self, query_id):
        normalized = str(query_id).strip().upper()
        if normalized.startswith("Q"):
            normalized = normalized[1:]

        match = re.match(r"^0*(\d+)([A-Z]*)$", normalized)
        if match:
            number = str(int(match.group(1)))
            suffix = match.group(2)
            return f"Q{number}{suffix}"

        normalized = normalized.lstrip("0") or normalized
        return f"Q{normalized}"

    def _normalize_query_best_times(self, query_best_times):
        normalized = {}
        for query_id, best_time in (query_best_times or {}).items():
            try:
                normalized[self._normalize_query_id(query_id)] = float(best_time)
            except (TypeError, ValueError):
                logger.warning("Skip invalid per-query best time: %s=%s", query_id, best_time)
        return normalized

    def get_query_best_time(self, query_id):
        return self.query_best_times.get(self._normalize_query_id(query_id))

    def build_workload_target_times(self, queries_order):
        if not self.query_best_times:
            return {}

        resolved = {}
        for idx, query_path in enumerate(queries_order, start=1):
            query_id = os.path.splitext(os.path.basename(query_path))[0]
            direct_key = self._normalize_query_id(query_id)
            order_key = self._normalize_query_id(idx)

            if direct_key in self.query_best_times:
                resolved[query_id] = self.query_best_times[direct_key]
            elif order_key in self.query_best_times:
                resolved[query_id] = self.query_best_times[order_key]

        return resolved

    def _remove_hints_from_sql(self, sql):
        import re

        sql_without_hints = re.sub(r"/\*\+[\s\S]*?\*/", "", sql)
        sql_without_hints = " ".join(sql_without_hints.split())
        return sql_without_hints.strip()

    def _build_base_info(self, query_info):
        if self.configs.get("DATABASE_TYPE", "postgres").lower() == "postgres":
            return base_information_collect(query_info, self.db_config)

        database_name = getattr(self.db, "database", None) or self.db_config.get("dbname") or "unknown"
        size = "unknown"
        if hasattr(self.db, "get_size") and callable(getattr(self.db, "get_size")):
            try:
                size = self.db.get_size()
            except Exception:
                size = "unknown"

        return {
            "query_info": query_info,
            "database_info": {
                "db_type": getattr(self.db, "db_type", self.db_config.get("db_type", "unknown")),
                "workload_type": getattr(self.db, "workload_type", self.db_config.get("workload_type", "unknown")),
                "dbname": database_name,
                "size": size,
                "schema": database_name,
            },
        }

    def _infer_case_root_causes(self, fix_action, query_info):
        if not self.memory_manager:
            return []
        inferred = self.memory_manager._infer_root_causes_from_action(fix_action, query_info)
        normalized = self.memory_manager._normalize_root_causes_list(inferred)
        return normalized

    def retrieve_static_cases(self, query_info):
        if not self.memory_manager:
            return {"mode": "static_simple", "positive": [], "negative": []}
        retrieval = self.memory_manager.retrieve_similar_cases(
            query_info,
            pos_n=self.simple_case_pos_n,
            neg_n=self.simple_case_neg_n,
        )
        logger.info(
            "SQL %s static retrieval selected %s positive and %s negative cases",
            getattr(query_info, "query_id", "unknown"),
            len(retrieval.get("positive", [])),
            len(retrieval.get("negative", [])),
        )
        return retrieval

    async def simple_generate_action(self, query_info):
        """Simple LLM generation without diagnosis, with static case retrieval."""
        base_info = self._build_base_info(query_info)
        retrieval = self.retrieve_static_cases(query_info)
        prompt = build_simple_retrieval_prompt(
            query_info,
            base_info,
            retrieval.get("positive", []),
            retrieval.get("negative", []),
        )
        logger.info(
            "[Prompt][SimpleRetrieval][SQL %s]\n%s",
            getattr(query_info, "query_id", "unknown"),
            prompt,
        )

        result = await Runner.run(starting_agent=self.diagnostic_agent, input=prompt)
        return result.final_output, retrieval

    async def run(self, slow_query_path, order, duration, no_improvement_threshold=3):
        """
        Execute simplified database optimization process using direct LLM action generation
        with static case retrieval and no diagnosis stage.

        Args:
            slow_query_path: Path to slow query files
            order: Query order file
            duration: Timeout in hours, will continue optimization until timeout
            no_improvement_threshold: Threshold for no improvement attempts
        """
        # Read workload
        queries_order = self.read_queries(slow_query_path, order)
        workload_target_times = self.build_workload_target_times(queries_order)
        if self.query_best_times and len(workload_target_times) != len(queries_order):
            logger.warning(
                "Resolved per-query best times for %s/%s queries in workload %s",
                len(workload_target_times),
                len(queries_order),
                self.query_best_time_workload or "unknown",
            )

        best_sql_actions = {}
        best_sql_times = {}
        best_sql_texts = {}

        # Record start time
        start_time = time.time()
        round_idx = 0

        while True:
            current_time = time.time()
            elapsed_time = current_time - start_time

            # Check timeout
            if elapsed_time >= duration * 3600:
                logger.info(f"Reached timeout {duration}h, stopping optimization")
                break

            round_idx += 1
            logger.info(f"=== Round {round_idx} Diagnosis ===")
            if self.memory_manager:
                self.memory_manager.set_current_round(round_idx)
            round_start = time.time()
            round_sql_times = {}
            round_sql_actions = {}

            for i, query_path in enumerate(queries_order):
                logger.info(f"Processing {i+1}/{len(queries_order)} SQL")
                query_id = query_path.split("/")[-1].replace(".sql", "")
                target_time = self.get_query_best_time(query_id)
                if target_time is not None:
                    workload_target_times[query_id] = target_time
                # target_time = workload_target_times.get(query_id)
                timeout_time = self.db_config.get("query_timeout") or 0.0
                fallback_time = timeout_time
                pre_index_names = []
                pre_knob_names = []
                pre_applied = False
                try:
                    # Read slow SQL
                    query = open(query_path, "r", encoding="utf-8").read()
                    query = self.clean_query(query)

                    # Use best SQL from history if available, otherwise use original
                    original_sql = query
                    collected_sql = best_sql_texts.get(query_id, original_sql)

                    # Apply previous best actions if they exist
                    if query_id in best_sql_actions and best_sql_actions[query_id]:
                        pre_fix_action = best_sql_actions[query_id]
                        success, _, _, _ = self.db.execute(pre_fix_action)
                        if success:
                            pre_index_names = self.action_manager.extract_index_names(pre_fix_action)
                            pre_knob_names = self.action_manager.extract_knob_names(pre_fix_action)
                            pre_applied = True

                    # Collect query execution information
                    result = self.db.run_sql_and_collect_all(collected_sql)

                    # Build QueryInfo
                    query_info = QueryInfo(
                        query_id=query_id,
                        query=collected_sql,
                        plan_json=result["plan_json"],
                        internal_metrics=result["internal_metrics"],
                        external_metrics=result["external_metrics"],
                        execution_time=result["duration"],
                        is_rewrite=(collected_sql != original_sql),
                    )
                    fallback_time = query_info.execution_time

                    # Initialize best records for new SQL
                    if query_id not in best_sql_times:
                        best_sql_times[query_id] = query_info.execution_time
                        best_sql_actions[query_id] = ""
                        best_sql_texts[query_id] = collected_sql

                    skip_threshold = target_time if target_time is not None else 1.0
                    if query_info.execution_time <= skip_threshold:
                        round_sql_times[query_id] = query_info.execution_time
                        round_sql_actions[query_id] = ""
                        if target_time is not None:
                            logger.info(
                                "SQL %s execution time %ss is already at or below workload target %ss (%s), skipping optimization",
                                query_id,
                                query_info.execution_time,
                                target_time,
                                self.query_best_time_workload or "unknown workload",
                            )
                        else:
                            logger.info(
                                "SQL %s execution time %ss is already below 1.0s, skipping optimization",
                                query_id,
                                query_info.execution_time,
                            )
                        continue

                    logger.info(f"SQL {query_id} starting simple LLM generation with static retrieval")

                    # Direct LLM action generation with static retrieval
                    action_result, retrieval = await self.simple_generate_action(query_info)

                    # Extract fix action
                    fix_action, rewrite_sql = self.extract_fix_action(action_result)

                    # Validate action
                    evaluation_result = await evaluate_action(
                        fix_action,
                        rewrite_sql,
                        query_info,
                        db=self.db,
                        fix_timeout=self.action_manager.fix_timeout,
                        sql_timeout=self.action_manager.sql_timeout,
                        fix_agent=self.action_manager.fixAgent,
                    )
                    if evaluation_result.get("fix_action"):
                        index_names = self.action_manager.extract_index_names(evaluation_result.get("fix_action"))
                        knob_names = self.action_manager.extract_knob_names(evaluation_result.get("fix_action"))
                        self.action_manager.rollback_action(index_names, knob_names)

                    # Get final fix content
                    fix_action = evaluation_result.get("fix_action")
                    rewrite_sql = evaluation_result.get("rewrite_sql")
                    if rewrite_sql != "":
                        # Check if SQL was actually rewritten (excluding hints)
                        original_sql = query_info.query.strip()
                        sql_without_hint = self._remove_hints_from_sql(rewrite_sql.strip())

                        if sql_without_hint != original_sql:
                            query_info.query = rewrite_sql
                            query_info.is_rewrite = True
                        else:
                            query_info.query = rewrite_sql
                            query_info.is_rewrite = False

                    # Initialize variables
                    new_time = None
                    old_time = query_info.execution_time

                    # Process evaluation results
                    if evaluation_result.get("status") == -1:
                        print(evaluation_result.get("msg"))
                        logger.info(f"SQL {query_id} fix failed, moving to next SQL")
                        new_time = timeout_time or old_time
                    elif evaluation_result.get("status") == 1:
                        logger.info(f"SQL {query_id} fix successful")
                        new_time = evaluation_result.get("new_time", old_time)
                    else:
                        logger.info(f"SQL {query_id} fix ineffective, recording negative case")
                        new_time = evaluation_result.get("new_time", old_time)

                    if self.memory_manager:
                        inferred_roots = self._infer_case_root_causes(fix_action, query_info)
                        case_label = "positive" if evaluation_result.get("status") == 1 else "negative"
                        self.memory_manager.save_case(
                            query_info,
                            fix_action,
                            inferred_roots,
                            old_time,
                            new_time,
                            case_label,
                            used_cases=retrieval,
                        )

                    # Update best records only if improvement achieved
                    if new_time < best_sql_times[query_id]:
                        best_sql_times[query_id] = new_time
                        best_sql_actions[query_id] = fix_action
                        best_sql_texts[query_id] = query_info.query
                    if target_time is not None and new_time <= target_time:
                        logger.info(
                            "SQL %s reached workload target time %ss with current execution time %ss",
                            query_id,
                            target_time,
                            new_time,
                        )

                    round_sql_times[query_id] = new_time
                    round_sql_actions[query_id] = fix_action

                    logger.info(f"SQL {query_id} original execution time: {old_time}s")
                    logger.info(f"SQL {query_id} current execution time: {new_time}s")
                    logger.info(f"SQL {query_id} current fix SQL: {rewrite_sql}")
                    logger.info(f"SQL {query_id} current fix actions: {fix_action}")
                except Exception as e:
                    logger.error(f"SQL {query_id} error during simple optimization: {e}", exc_info=True)
                    round_sql_times[query_id] = fallback_time
                    round_sql_actions[query_id] = ""
                    logger.info(f"SQL {query_id} skipping current SQL, continuing to next")
                    continue
                finally:
                    # Rollback previous actions after current SQL processing
                    if pre_applied:
                        self.action_manager.rollback_action(pre_index_names, pre_knob_names)

            round_end = time.time()
            logger.info(f"Round {round_idx} optimization time: {round_end - round_start:.4f}s")
            round_total_time = sum([t for t in round_sql_times.values()])
            logger.info(f"Round {round_idx} workload total execution time: {round_total_time:.4f}s")
            logger.info(f"Round {round_idx} SQL execution times: {round_sql_times}")

            logger.info(f"Up to round {round_idx}, best SQL execution times: {best_sql_times}")
            logger.info(f"Up to round {round_idx}, workload best total execution time: {sum(best_sql_times.values())}")
            if workload_target_times:
                logger.info(f"Workload target SQL execution times: {workload_target_times}")
                logger.info(f"Workload target total execution time: {sum(workload_target_times.values())}")
            logger.info(f"Up to round {round_idx}, best SQL fix actions: {best_sql_actions}")

        # Output final results
        final_time = time.time()
        total_elapsed = final_time - start_time
        logger.info("=== Optimization Complete ===")
        logger.info(f"Total time: {total_elapsed:.2f}s (timeout limit: {duration}h)")
        logger.info(f"Completed rounds: {round_idx}")
        logger.info(f"Best SQL execution times: {best_sql_times}")
        logger.info(f"Workload best total execution time: {sum(best_sql_times.values()):.4f}s")
        if workload_target_times:
            logger.info(f"Workload target SQL execution times: {workload_target_times}")
            logger.info(f"Workload target total execution time: {sum(workload_target_times.values()):.4f}s")
        logger.info(f"Best SQL fix actions: {best_sql_actions}")
        logger.info("Diagnosis process ended")
