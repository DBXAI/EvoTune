import asyncio
import copy
import json
import logging
import os
import re
import time
from agent.action.action_manager import ActionManager
from agent.memory.memory_manager import MemoryManager
from agent.plan.online_predict import RCRankPredictor
from agent.plan.planner import Planner
from database.pg_env import PostgresDB
from database.tidb_env import TiDB
from utils.runtime_metrics import format_runtime_metrics, merge_runtime_metrics
from utils.types import QueryInfo

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = True


class DBAgent:
    def __init__(self, configs=None):
        self.configs = configs
        self.agent_config = configs.get("AGENT_CONFIG")
        self.skip_below_seconds = float(
            self.agent_config.get("skip_below_seconds", 1.0) or 1.0
        )

        db_type = configs.get("DATABASE_TYPE", "postgres")
        if db_type.lower() == "postgres":
            db_config = configs.get("DATABASE_CONFIG")
            self.db = PostgresDB(db_config)
        elif db_type.lower() == "tidb":
            db_config = dict(configs.get("TiDB_CONFIG") or {})
            # TiDB sessions must know the configured baseline defaults; do
            # not inherit mutable global values from a previous experiment.
            db_config["KNOB_CONFIG"] = configs.get("KNOB_CONFIG", {})
            self.db = TiDB(db_config)

        planner_config = configs.get("PLANNER_CONFIG")
        predictor = RCRankPredictor(planner_config)

        memory_config = configs.get("MEMORY_MANAGER_CONFIG")
        self.memory_manager = MemoryManager(
            memory_config,
            predictor=predictor,
            planner_config=planner_config,
            agent_config=self.agent_config,
            api_settings=configs.get("API_SETTINGS"),
        )

        self.planner = Planner(
            self.memory_manager,
            predictor=predictor,
            agent_config=self.agent_config,
            db=self.db,
            database_config=db_config,
        )
        self.action_manager = ActionManager(
            memory_manager=self.memory_manager,
            db=self.db,
            planner=self.planner,
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

            logger.info("DBAgent resource cleanup completed")

        except Exception as e:
            logger.error(f"Error during resource cleanup: {e}")
        finally:
            pass

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

    def _remove_hints_from_sql(self, sql):
        import re

        sql_without_hints = re.sub(r"/\*\+[\s\S]*?\*/", "", sql)
        sql_without_hints = " ".join(sql_without_hints.split())
        return sql_without_hints.strip()

    @staticmethod
    def _checkpoint_safe(value):
        if isinstance(value, dict):
            return {str(k): DBAgent._checkpoint_safe(v) for k, v in value.items()}
        if isinstance(value, (set, frozenset, tuple, list)):
            return [DBAgent._checkpoint_safe(v) for v in value]
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                pass
        return value

    def _save_checkpoint(
        self,
        path,
        best_sql_actions,
        best_sql_times,
        best_sql_texts,
        sql_root_cause_states,
        next_round,
        consecutive_no_improvement=None,
        finished_queries=None,
    ):
        if not path:
            return
        payload = {
            "best_sql_actions": best_sql_actions,
            "best_sql_times": best_sql_times,
            "best_sql_texts": best_sql_texts,
            "sql_root_cause_states": sql_root_cause_states,
            "consecutive_no_improvement": consecutive_no_improvement or {},
            "finished_queries": sorted(finished_queries or []),
            "next_round": next_round,
        }
        safe_payload = self._checkpoint_safe(payload)
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        temporary = f"{path}.tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(safe_payload, handle, ensure_ascii=False)
        os.replace(temporary, path)

    async def run(
        self,
        slow_query_path,
        order,
        duration,
        no_improvement_threshold=3,
        resume_state_path=None,
    ):
        """
        Execute database optimization process

        Args:
            slow_query_path: Path to slow query files
            order: Query order file
            duration: Timeout in hours, will continue optimization until timeout
            no_improvement_threshold: Threshold for no improvement attempts, switch root cause when reached
        """
        # Read workload
        queries_order = self.read_queries(slow_query_path, order)

        best_sql_actions = {}
        best_sql_times = {}
        best_sql_texts = {}

        # Track query-local localization state and feedback.
        sql_root_cause_states = {}
        consecutive_no_improvement = {}
        finished_queries = set()

        # A continuation run can preserve the best per-query result from a
        # previous process.  The original tuning loop kept these structures
        # only in memory, so without this checkpoint a restart would silently
        # discard the previously discovered best actions.
        resume_round = 0
        if resume_state_path:
            try:
                with open(resume_state_path, "r", encoding="utf-8") as handle:
                    resume_state = json.load(handle)
                best_sql_actions.update(resume_state.get("best_sql_actions", {}))
                best_sql_times.update(
                    {str(k): float(v) for k, v in resume_state.get("best_sql_times", {}).items()}
                )
                best_sql_texts.update(resume_state.get("best_sql_texts", {}))
                sql_root_cause_states.update(resume_state.get("sql_root_cause_states", {}))
                consecutive_no_improvement.update(
                    {
                        str(k): int(v)
                        for k, v in resume_state.get("consecutive_no_improvement", {}).items()
                    }
                )
                finished_queries.update(resume_state.get("finished_queries", []))
                resume_round = int(resume_state.get("next_round", 0)) - 1
                if resume_round > 0:
                    logger.info(
                        "Loaded DREAM continuation state: next round %s, %d best query times",
                        resume_round + 1,
                        len(best_sql_times),
                    )
            except Exception as exc:
                logger.warning("Unable to load DREAM continuation state %s: %s", resume_state_path, exc)

        # Record start time
        start_time = time.time()
        round_idx = max(0, resume_round)

        while True:
            current_time = time.time()
            elapsed_time = current_time - start_time

            # Check timeout
            if elapsed_time >= duration * 3600:
                logger.info(f"Reached timeout {duration}h, stopping optimization")
                break

            round_idx += 1
            logger.info(f"=== Round {round_idx} Diagnosis ===")
            # Set current round in memory_manager for generating next_query_info_id
            self.memory_manager.set_current_round(round_idx)
            round_start = time.time()
            round_sql_times = {}
            round_sql_actions = {}
            for i, query_path in enumerate(queries_order):
                logger.info(f"Processing {i+1}/{len(queries_order)} SQL")

                # Read slow SQL
                query = open(query_path, "r", encoding="utf-8").read()
                query_id = query_path.split("/")[-1].replace(".sql", "")
                query = self.clean_query(query)

                # Select SQL for diagnosis: prefer best SQL from history (with rewrite/hint), otherwise use original
                original_sql = query
                collected_sql = best_sql_texts.get(query_id, original_sql)

                # Apply previous best actions if they exist
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

                # Collect query execution information
                try:
                    result = self.db.run_sql_and_collect_all(collected_sql)
                except Exception as e:
                    logger.error(f"SQL {query_id} error collecting SQL execution data: {e}", exc_info=True)
                    database_config = self.configs.get(
                        "DATABASE_CONFIG", self.configs.get("TiDB_CONFIG", {})
                    )
                    timeout_time = database_config.get("query_timeout")
                    round_sql_times[query_id] = timeout_time
                    round_sql_actions[query_id] = ""
                    logger.info(f"SQL {query_id} skipping current SQL, continuing to next")
                    continue

                logger.info(f"SQL {query_id} collected SQL execution time: {result['duration']}s")

                # Build QueryInfo; if collected_sql differs from original_sql, treat as rewrite
                query_info = QueryInfo(
                    query_id=query_id,
                    query=collected_sql,
                    plan_json=result["plan_json"],
                    internal_metrics=result["internal_metrics"],
                    external_metrics=result["external_metrics"],
                    execution_time=result["duration"],
                    is_rewrite=(collected_sql != original_sql),
                )

                # Initialize best records for new SQL
                if query_id not in best_sql_times:
                    best_sql_times[query_id] = query_info.execution_time
                    best_sql_actions[query_id] = ""
                    best_sql_texts[query_id] = collected_sql

                # Keep the ablation workload gate explicit and configurable.
                # This is intentionally checked after the real baseline
                # execution and before root-cause prediction/action generation.
                if (
                    self.skip_below_seconds > 0
                    and query_info.execution_time < self.skip_below_seconds
                ):
                    logger.info(
                        "SQL %s execution time %.3fs is below %.3fs; skipping optimization",
                        query_id,
                        query_info.execution_time,
                        self.skip_below_seconds,
                    )
                    round_sql_times[query_id] = query_info.execution_time
                    round_sql_actions[query_id] = ""
                    if pre_applied:
                        self.action_manager.rollback_action(pre_index_names, pre_knob_names)
                    continue

                # Initialize state
                if query_id not in sql_root_cause_states:
                    sql_root_cause_states[query_id] = {
                        "root_tried": set(),
                        "current_root": None,
                        "mode": "exploit",
                        "confidence": {},
                        "attempts": {},
                        "successes": {},
                        "component_attempts": {},
                        "component_successes": {},
                        "component_failures": {},
                    }

                state = sql_root_cause_states[query_id]
                # Expose the query-level stagnation counter to the diagnosis
                # prompt.  This lets the LLM switch to randomized component
                # exploration after repeated non-improving attempts.
                state["query_consecutive_no_improvement"] = int(
                    consecutive_no_improvement.get(query_id, 0)
                )

                # Predict root cause
                logger.info(f"SQL {query_id} starting root cause prediction")
                # Save state backup for exception recovery
                state_backup = copy.deepcopy(state)
                try:
                    predicted_root, state = self.planner.predict(query_info, state, self.memory_manager)
                except Exception as e:
                    logger.error(f"SQL {query_id} error during root cause prediction: {e}", exc_info=True)
                    sql_root_cause_states[query_id] = state_backup
                    round_sql_times[query_id] = query_info.execution_time
                    round_sql_actions[query_id] = ""
                    logger.info(f"SQL {query_id} skipping current SQL, continuing to next")
                    continue

                predicted_root_key = self.memory_manager._normalize_root_causes(predicted_root)
                if predicted_root_key in state["root_tried"]:
                    logger.info(
                        f"SQL {query_id} predicted root cause already in root_tried; resetting active localization"
                    )
                    self.planner.localization_selector.reset_active(state)
                    state["current_root"] = None
                    continue

                state["mode"] = "exploit"
                logger.info(
                    "SQL %s localized by %s in %s mode",
                    query_id,
                    state.get("localization_expert"),
                    state["mode"],
                )

                root_causes = state["current_root"]
                if not root_causes:
                    logger.info(f"SQL {query_id} no available root cause this round, skipping to next SQL")
                    continue

                logger.info(f"SQL {query_id} current diagnosis root cause: {root_causes}")

                # Preserve the pre-action state for the observation memory.  The
                # current query_info may later be replaced by a rewritten SQL.
                observation_query_info = copy.deepcopy(query_info)

                # Call action_manager.step to process (includes evaluate_action and experience pool update)
                try:
                    evaluation_result = await self.action_manager.step(
                        query_info,
                        root_causes,
                        mode=state["mode"],
                        tuning_state={
                            "success_count": sum(state.get("component_successes", {}).values()),
                            "failure_count": sum(state.get("component_failures", {}).values()),
                        },
                    )
                except Exception as e:
                    logger.error(f"SQL {query_id} error during step execution: {e}", exc_info=True)
                    new_time = query_info.execution_time
                    round_sql_times[query_id] = new_time
                    round_sql_actions[query_id] = ""
                    sql_root_cause_states[query_id] = state
                    logger.info(f"SQL {query_id} skipping current SQL, continuing to next")
                    continue

                sql_runtime_metrics = merge_runtime_metrics(
                    state.get("last_runtime_metrics"),
                    evaluation_result.get("runtime_metrics"),
                )
                logger.info(
                    "[Runtime][SQL %s] Overhead summary: %s",
                    query_id,
                    format_runtime_metrics(sql_runtime_metrics),
                )

                # Get fix content
                fix_action = evaluation_result.get("fix_action")
                rewrite_sql = evaluation_result.get("rewrite_sql")
                modified_components = self.memory_manager.infer_modified_components(
                    fix_action,
                    observation_query_info,
                    rewrite_sql,
                )
                if rewrite_sql != "":
                    # Check if SQL was actually rewritten (excluding hints)
                    original_sql = query_info.query.strip()
                    sql_without_hint = self._remove_hints_from_sql(rewrite_sql.strip())

                    # Only mark as rewrite if SQL without hints differs from original SQL
                    if sql_without_hint != original_sql:
                        query_info.query = rewrite_sql
                        query_info.is_rewrite = True
                    else:
                        # If only hints were added, don't mark as rewrite
                        query_info.query = rewrite_sql
                        query_info.is_rewrite = False

                # Initialize variables
                new_time = None
                old_time = query_info.execution_time

                # Process evaluation results and update root cause state
                if evaluation_result.get("status") == -1:
                    error_msg = evaluation_result.get("msg", "Unknown error")
                    print(f"Error: {error_msg}")
                    logger.info(f"SQL {query_id} fix failed, moving to next SQL")
                    database_config = self.configs.get(
                        "DATABASE_CONFIG", self.configs.get("TiDB_CONFIG", {})
                    )
                    new_time = database_config.get("query_timeout") or old_time

                    root_causes_list = modified_components
                    for rc in root_causes_list:
                        state["component_attempts"][rc] = state["component_attempts"].get(rc, 0) + 1
                        state["component_failures"][rc] = state["component_failures"].get(rc, 0) + 1

                    case_id = self.memory_manager.save_case(
                        observation_query_info,
                        fix_action,
                        root_causes,
                        old_time,
                        new_time,
                        "negative",
                        tuning_attempts=state["component_attempts"].copy(),
                        used_cases=evaluation_result.get("used_cases"),
                        modified_components=modified_components,
                    )
                    # Get corrected root cause
                    case = self.memory_manager.get_case(case_id)
                    root_causes = case.case_info.get("root_causes")
                    # Record one invalid attempt for this root cause
                    key = self.memory_manager._normalize_root_causes(root_causes)
                    state["attempts"][key] = state["attempts"].get(key, 0) + 1
                elif evaluation_result.get("status") == 1:
                    logger.info(f"SQL {query_id} fix successful")
                    new_time = evaluation_result.get("new_time")

                    root_causes_list = modified_components
                    for rc in root_causes_list:
                        state["component_attempts"][rc] = state["component_attempts"].get(rc, 0) + 1
                        state["component_successes"][rc] = state["component_successes"].get(rc, 0) + 1

                    case_id = self.memory_manager.save_case(
                        observation_query_info,
                        fix_action,
                        root_causes,
                        old_time,
                        new_time,
                        "positive",
                        tuning_attempts=state["component_attempts"].copy(),
                        used_cases=evaluation_result.get("used_cases"),
                        modified_components=modified_components,
                    )
                    # Get corrected root cause
                    case = self.memory_manager.get_case(case_id)
                    root_causes = case.case_info.get("root_causes")

                    # Success: accumulate success count
                    key = self.memory_manager._normalize_root_causes(root_causes)
                    state["attempts"][key] = state["attempts"].get(key, 0) + 1
                    state["successes"][key] = state["successes"].get(key, 0) + 1
                else:
                    # Fix ineffective: record negative case and advance source (try again next round)
                    logger.info(f"SQL {query_id} fix ineffective, recording negative case")
                    new_time = evaluation_result.get("new_time", old_time)

                    root_causes_list = modified_components
                    for rc in root_causes_list:
                        state["component_attempts"][rc] = state["component_attempts"].get(rc, 0) + 1
                        state["component_failures"][rc] = state["component_failures"].get(rc, 0) + 1

                    case_id = self.memory_manager.save_case(
                        observation_query_info,
                        fix_action,
                        root_causes,
                        old_time,
                        new_time,
                        "negative",
                        tuning_attempts=state["component_attempts"].copy(),
                        used_cases=evaluation_result.get("used_cases"),
                        modified_components=modified_components,
                    )
                    # Get corrected root cause
                    case = self.memory_manager.get_case(case_id)
                    root_causes = case.case_info.get("root_causes")
                    # Record one invalid attempt for this root cause
                    key = self.memory_manager._normalize_root_causes(root_causes)
                    state["attempts"][key] = state["attempts"].get(key, 0) + 1

                # Contextual-bandit feedback is the normalized performance
                # change.  Positive feedback is immediately reliable; a
                # negative reward becomes reliable only after eta attempts.
                try:
                    reward = (old_time - new_time) / max(abs(old_time), 1e-8)
                except (TypeError, ValueError, ZeroDivisionError):
                    reward = 0.0
                localization_attempts = int(state.get("localization_attempts", 0)) + 1
                state["localization_attempts"] = localization_attempts
                delta = float(self.agent_config.get("localization_min_improvement", 0.1))
                eta = max(1, int(self.agent_config.get("localization_negative_eta", 3)))
                reliable = reward >= delta or localization_attempts >= eta
                self.planner.update_localization_feedback(
                    state,
                    reward=reward,
                    reliable=reliable,
                )

                # When invalid attempt count reaches threshold, mark root cause as tried and clear current_root
                if state.get("attempts", {}).get(key, 0) >= no_improvement_threshold and not state.get(
                    "successes", {}
                ).get(key, 0):
                    state["root_tried"].add(key)

                # Update best records only if the candidate is better.
                improved = new_time < best_sql_times[query_id]
                if improved:
                    best_sql_times[query_id] = new_time
                    best_sql_actions[query_id] = fix_action
                    best_sql_texts[query_id] = query_info.query
                    consecutive_no_improvement[query_id] = 0
                else:
                    consecutive_no_improvement[query_id] = (
                        consecutive_no_improvement.get(query_id, 0) + 1
                    )
                # A sub-one-second result only skips optimization for this
                # round.  The query remains eligible for measurement in the
                # next round so repeated executions can reveal a lower stable
                # runtime.  No query-level early-stop state is recorded here.
                round_sql_times[query_id] = new_time
                round_sql_actions[query_id] = fix_action
                sql_root_cause_states[query_id] = state

                logger.info(f"SQL {query_id} original execution time: {old_time}s")
                logger.info(f"SQL {query_id} current execution time: {new_time}s")
                logger.info(f"SQL {query_id} current fix SQL: {rewrite_sql}")
                logger.info(f"SQL {query_id} current diagnosis root cause: {root_causes}")
                logger.info(f"SQL {query_id} current fix actions: {fix_action}")

                # Rollback previous actions after new action evaluation
                if pre_applied:
                    self.action_manager.rollback_action(pre_index_names, pre_knob_names)

                self._save_checkpoint(
                    resume_state_path,
                    best_sql_actions,
                    best_sql_times,
                    best_sql_texts,
                    sql_root_cause_states,
                    round_idx,
                    consecutive_no_improvement,
                    finished_queries,
                )

            round_end = time.time()
            logger.info(f"Round {round_idx} optimization time: {round_end - round_start:.4f}s")
            round_total_time = sum([t for t in round_sql_times.values()])
            logger.info(f"Round {round_idx} workload total execution time: {round_total_time:.4f}s")
            logger.info(f"Round {round_idx} SQL execution times: {round_sql_times}")

            logger.info(f"Up to round {round_idx}, best SQL execution times: {best_sql_times}")
            logger.info(f"Up to round {round_idx}, workload best total execution time: {sum(best_sql_times.values())}")
            logger.info(f"Up to round {round_idx}, best SQL fix actions: {best_sql_actions}")
            self._save_checkpoint(
                resume_state_path,
                best_sql_actions,
                best_sql_times,
                best_sql_texts,
                sql_root_cause_states,
                round_idx + 1,
                consecutive_no_improvement,
                finished_queries,
            )

        # Output final results
        final_time = time.time()
        total_elapsed = final_time - start_time
        logger.info("=== Optimization Complete ===")
        logger.info(f"Total time: {total_elapsed:.2f}s (timeout limit: {duration}h)")
        logger.info(f"Completed rounds: {round_idx}")
        logger.info(f"Best SQL execution times: {best_sql_times}")
        logger.info(f"Workload best total execution time: {sum(best_sql_times.values()):.4f}s")
        logger.info(f"Best SQL fix actions: {best_sql_actions}")
        logger.info("Diagnosis process ended")
