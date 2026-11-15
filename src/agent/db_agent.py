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
    def _compose_actions(previous, candidate):
        """Keep accepted configuration changes in execution order."""
        previous = str(previous or "").strip()
        candidate = str(candidate or "").strip()
        if not candidate:
            return previous
        if not previous:
            return candidate
        separator = "\n" if previous.endswith(";") else ";\n"
        return previous + separator + candidate

    @classmethod
    def _update_configuration(
        cls, query_id, fix_action, query_text, old_time, trial_time, status,
        pre_applied, current_actions, current_texts, best_actions, best_texts,
        best_times,
    ):
        """Accept an improvement while preserving the best observed result."""
        if status == -1 or trial_time >= old_time:
            return False
        previous = current_actions.get(query_id, "") if pre_applied else ""
        current_actions[query_id] = cls._compose_actions(previous, fix_action)
        current_texts[query_id] = query_text
        if trial_time < best_times[query_id]:
            best_times[query_id] = trial_time
            best_actions[query_id] = current_actions[query_id]
            best_texts[query_id] = query_text
        return True

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
        next_query_index=0,
        current_sql_actions=None,
        current_sql_texts=None,
        round_sql_times=None,
        round_sql_actions=None,
    ):
        if not path:
            return
        checkpoint_states = copy.deepcopy(sql_root_cause_states)
        for state in checkpoint_states.values():
            state.get("localization", {}).pop("experts", None)
        payload = {
            "best_sql_actions": best_sql_actions,
            "best_sql_times": best_sql_times,
            "best_sql_texts": best_sql_texts,
            "current_sql_actions": current_sql_actions or {},
            "current_sql_texts": current_sql_texts or {},
            "sql_root_cause_states": checkpoint_states,
            "consecutive_no_improvement": consecutive_no_improvement or {},
            "finished_queries": sorted(finished_queries or []),
            "next_round": next_round,
            "next_query_index": next_query_index,
            "round_sql_times": round_sql_times or {},
            "round_sql_actions": round_sql_actions or {},
            "localization_expert_posteriors": self.planner.localization_selector.export_global_state(),
        }
        safe_payload = self._checkpoint_safe(payload)
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        temporary = f"{path}.tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(safe_payload, handle, ensure_ascii=False)
        os.replace(temporary, path)
        model_path = self.memory_manager.retriever_model_path
        if self.memory_manager.retrieval_mode == "dynamic" and model_path:
            model_temporary = f"{model_path}.tmp"
            self.memory_manager.retriever.save_model(model_temporary)
            os.replace(model_temporary, model_path)

    async def run(
        self,
        slow_query_path,
        order,
        duration,
        no_improvement_threshold=3,
        resume_state_path=None,
        max_rounds=None,
    ):
        """
        Execute database optimization process

        Args:
            slow_query_path: Path to slow query files
            order: Query order file
            duration: Timeout in hours, will continue optimization until timeout
            max_rounds: Optional number of complete round-robin passes
            no_improvement_threshold: Threshold for no improvement attempts, switch root cause when reached
        """
        # Read workload
        queries_order = self.read_queries(slow_query_path, order)

        best_sql_actions = {}
        best_sql_times = {}
        best_sql_texts = {}
        resume_round_sql_times = {}
        resume_round_sql_actions = {}
        current_sql_actions = {}
        current_sql_texts = {}

        # Track query-local localization state and feedback.
        sql_root_cause_states = {}
        consecutive_no_improvement = {}
        finished_queries = set()

        # A continuation run can preserve the best per-query result from a
        # previous process.  The original tuning loop kept these structures
        # only in memory, so without this checkpoint a restart would silently
        # discard the previously discovered best actions.
        resume_round = 0
        resume_query_index = 0
        if resume_state_path:
            try:
                with open(resume_state_path, "r", encoding="utf-8") as handle:
                    resume_state = json.load(handle)
                best_sql_actions.update(resume_state.get("best_sql_actions", {}))
                best_sql_times.update(
                    {str(k): float(v) for k, v in resume_state.get("best_sql_times", {}).items()}
                )
                best_sql_texts.update(resume_state.get("best_sql_texts", {}))
                resume_round_sql_times.update(
                    {str(k): float(v) for k, v in resume_state.get("round_sql_times", {}).items()}
                )
                resume_round_sql_actions.update(resume_state.get("round_sql_actions", {}))
                current_sql_actions.update(
                    resume_state.get("current_sql_actions", best_sql_actions)
                )
                current_sql_texts.update(
                    resume_state.get("current_sql_texts", best_sql_texts)
                )
                sql_root_cause_states.update(resume_state.get("sql_root_cause_states", {}))
                for state in sql_root_cause_states.values():
                    state["root_tried"] = {
                        frozenset(self.memory_manager._normalize_root_causes_list(root))
                        for root in state.get("root_tried", [])
                    }
                self.planner.localization_selector.load_global_state(
                    resume_state.get("localization_expert_posteriors")
                )
                consecutive_no_improvement.update(
                    {
                        str(k): int(v)
                        for k, v in resume_state.get("consecutive_no_improvement", {}).items()
                    }
                )
                finished_queries.update(resume_state.get("finished_queries", []))
                resume_round = int(resume_state.get("next_round", 0)) - 1
                resume_query_index = int(resume_state.get("next_query_index", 0))
                if "next_query_index" not in resume_state and resume_round == 0:
                    # Older checkpoints saved the first round after every
                    # completed query without an explicit position.
                    for query_path in queries_order:
                        query_id = os.path.basename(query_path).removesuffix(".sql")
                        if query_id not in best_sql_times:
                            break
                        resume_query_index += 1
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
        completed_rounds = 0
        early_stop_patience = max(
            1, int(self.agent_config.get("query_early_stop_patience", 10))
        )
        query_ids = {
            os.path.basename(path).removesuffix(".sql") for path in queries_order
        }

        while True:
            if max_rounds is not None and completed_rounds >= max_rounds:
                logger.info("Reached the requested maximum of %d rounds", max_rounds)
                break
            if query_ids and query_ids.issubset(finished_queries):
                logger.info("All queries met an early-stopping criterion")
                break
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
            continuing_partial_round = round_idx == resume_round + 1 and resume_query_index > 0
            round_sql_times = dict(resume_round_sql_times) if continuing_partial_round else {}
            round_sql_actions = dict(resume_round_sql_actions) if continuing_partial_round else {}
            round_interrupted = False
            next_query_index = 0
            for i, query_path in enumerate(queries_order):
                if round_idx == resume_round + 1 and i < resume_query_index:
                    continue
                if time.time() - start_time >= duration * 3600:
                    logger.info(
                        "Reached timeout %sh before SQL %s/%s; ending the current round",
                        duration,
                        i + 1,
                        len(queries_order),
                    )
                    round_interrupted = True
                    next_query_index = i
                    break
                logger.info(f"Processing {i+1}/{len(queries_order)} SQL")

                # Read slow SQL
                query = open(query_path, "r", encoding="utf-8").read()
                query_id = query_path.split("/")[-1].replace(".sql", "")
                if query_id in finished_queries:
                    continue
                query = self.clean_query(query)

                # Reconstruct the currently accepted state for this query.
                original_sql = query
                collected_sql = current_sql_texts.get(query_id, original_sql)

                # Apply previous best actions if they exist
                pre_index_names = []
                pre_knob_names = []
                pre_applied = False
                had_pre_action = bool(current_sql_actions.get(query_id))
                pre_apply_start = time.perf_counter()
                if had_pre_action:
                    pre_fix_action = current_sql_actions[query_id]
                    success, _, _, _ = self.db.execute(pre_fix_action)
                    if success:
                        pre_index_names = self.action_manager.extract_index_names(pre_fix_action)
                        pre_knob_names = self.action_manager.extract_knob_names(pre_fix_action)
                        pre_applied = True
                    else:
                        logger.warning("Unable to restore accepted actions for SQL %s", query_id)
                        current_sql_actions[query_id] = ""
                pre_apply_time = time.perf_counter() - pre_apply_start if had_pre_action else 0.0

                # Collect query execution information
                try:
                    baseline_start = time.perf_counter()
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

                baseline_metrics = merge_runtime_metrics(
                    {
                        "T_pre_apply": pre_apply_time,
                        "T_baseline_collect": time.perf_counter() - baseline_start,
                    },
                    result.get("runtime_metrics"),
                )
                logger.info(
                    "[Runtime][SQL %s] Baseline summary: %s",
                    query_id,
                    format_runtime_metrics(baseline_metrics),
                )
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
                current_sql_actions.setdefault(query_id, "")
                current_sql_texts.setdefault(query_id, collected_sql)
                if query_info.execution_time < best_sql_times[query_id]:
                    best_sql_times[query_id] = query_info.execution_time
                    best_sql_actions[query_id] = current_sql_actions[query_id]
                    best_sql_texts[query_id] = current_sql_texts[query_id]

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
                    finished_queries.add(query_id)
                    if self.memory_manager.retrieval_mode == "dynamic":
                        self.memory_manager.retriever.finalize_pending_experiences(
                            self.memory_manager.generate_next_query_info_id(
                                query_id, round_num=round_idx
                            )
                        )
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
                        "component_state_attempts": {},
                        "localization_failure_rewards": {},
                        "query_success_count": 0,
                        "query_failure_count": 0,
                        "initial_latency": query_info.execution_time,
                    }

                state = sql_root_cause_states[query_id]
                state.setdefault("initial_latency", query_info.execution_time)
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
                            "success_count": state.get("query_success_count", 0),
                            "failure_count": state.get("query_failure_count", 0),
                        },
                        initial_latency=state["initial_latency"],
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
                state_attempts = state.setdefault("component_state_attempts", {})
                for component in modified_components:
                    state_attempts[component] = state_attempts.get(component, 0) + 1
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
                database_config = self.configs.get(
                    "DATABASE_CONFIG", self.configs.get("TiDB_CONFIG", {})
                )
                timeout_time = float(database_config.get("query_timeout") or old_time)
                try:
                    trial_time = float(evaluation_result.get("new_time"))
                except (TypeError, ValueError):
                    trial_time = timeout_time

                # Process evaluation results and update root cause state
                if evaluation_result.get("status") == -1:
                    error_msg = evaluation_result.get("msg", "Unknown error")
                    print(f"Error: {error_msg}")
                    logger.info(f"SQL {query_id} fix failed, moving to next SQL")
                    new_time = timeout_time

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
                        tuning_attempts=state_attempts.copy(),
                        used_cases=evaluation_result.get("used_cases"),
                        modified_components=modified_components,
                        validation_status=evaluation_result.get("status"),
                        retrieval_state_counts=evaluation_result.get("retrieval_state_counts"),
                    )
                    # Get corrected root cause
                    case = self.memory_manager.get_case(case_id)
                    root_causes = case.case_info.get("root_causes")
                    # Record one invalid attempt for this root cause
                    key = self.memory_manager._normalize_root_causes(root_causes)
                    attempt_key = ",".join(sorted(key))
                    state["attempts"][attempt_key] = state["attempts"].get(attempt_key, 0) + 1
                elif evaluation_result.get("status") == 1:
                    logger.info(f"SQL {query_id} fix successful")
                    new_time = trial_time

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
                        tuning_attempts=state_attempts.copy(),
                        used_cases=evaluation_result.get("used_cases"),
                        modified_components=modified_components,
                        validation_status=evaluation_result.get("status"),
                        retrieval_state_counts=evaluation_result.get("retrieval_state_counts"),
                    )
                    # Get corrected root cause
                    case = self.memory_manager.get_case(case_id)
                    root_causes = case.case_info.get("root_causes")

                    # Success: accumulate success count
                    key = self.memory_manager._normalize_root_causes(root_causes)
                    attempt_key = ",".join(sorted(key))
                    state["attempts"][attempt_key] = state["attempts"].get(attempt_key, 0) + 1
                    state["successes"][attempt_key] = state["successes"].get(attempt_key, 0) + 1
                else:
                    # An improvement below delta is accepted for the query
                    # trajectory but remains a negative RAWS label.
                    logger.info(f"SQL {query_id} fix below localization threshold, recording negative case")
                    new_time = trial_time
                    small_improvement = new_time < old_time

                    root_causes_list = modified_components
                    for rc in root_causes_list:
                        state["component_attempts"][rc] = state["component_attempts"].get(rc, 0) + 1
                        counter = "component_successes" if small_improvement else "component_failures"
                        state[counter][rc] = state[counter].get(rc, 0) + 1

                    case_id = self.memory_manager.save_case(
                        observation_query_info,
                        fix_action,
                        root_causes,
                        old_time,
                        new_time,
                        "negative",
                        tuning_attempts=state_attempts.copy(),
                        used_cases=evaluation_result.get("used_cases"),
                        modified_components=modified_components,
                        validation_status=evaluation_result.get("status"),
                        retrieval_state_counts=evaluation_result.get("retrieval_state_counts"),
                    )
                    # Get corrected root cause
                    case = self.memory_manager.get_case(case_id)
                    root_causes = case.case_info.get("root_causes")
                    # Record one invalid attempt for this root cause
                    key = self.memory_manager._normalize_root_causes(root_causes)
                    attempt_key = ",".join(sorted(key))
                    state["attempts"][attempt_key] = state["attempts"].get(attempt_key, 0) + 1
                    if small_improvement:
                        state["successes"][attempt_key] = state["successes"].get(attempt_key, 0) + 1

                # Contextual-bandit feedback is the normalized performance
                # change.  Positive feedback is immediately reliable; a
                # negative reward becomes reliable only after eta attempts.
                try:
                    reward = (old_time - new_time) / max(abs(old_time), 1e-8)
                except (TypeError, ValueError, ZeroDivisionError):
                    reward = 0.0
                delta = float(self.agent_config.get("localization_min_improvement", 0.1))
                eta = max(1, int(self.agent_config.get("localization_negative_eta", 3)))
                bucket_key = ",".join(predicted_root)
                failures = state.setdefault("localization_failure_rewards", {})
                if reward >= delta:
                    reliable = True
                    failures.pop(bucket_key, None)
                else:
                    bucket = failures.setdefault(bucket_key, [])
                    bucket.append(reward)
                    reliable = len(bucket) >= eta
                    if reliable:
                        reward = sum(bucket) / len(bucket)
                        failures[bucket_key] = []
                self.planner.update_localization_feedback(
                    state,
                    reward=reward,
                    reliable=reliable,
                )
                if evaluation_result.get("status") != -1 and new_time < old_time:
                    state["query_success_count"] = state.get("query_success_count", 0) + 1
                    # A retained action changes Q_t; exploration counts for
                    # the previous execution state no longer apply.
                    state["component_state_attempts"] = {}
                    state["localization_failure_rewards"] = {}
                else:
                    state["query_failure_count"] = state.get("query_failure_count", 0) + 1

                # When invalid attempt count reaches threshold, mark root cause as tried and clear current_root
                if state.get("attempts", {}).get(attempt_key, 0) >= no_improvement_threshold and not state.get(
                    "successes", {}
                ).get(attempt_key, 0):
                    state["root_tried"].add(key)

                # Retain a valid action when it improves the current state.
                # Keep the best observed configuration separately for reports.
                improved = self._update_configuration(
                    query_id, fix_action, query_info.query, old_time, new_time,
                    evaluation_result.get("status"), pre_applied,
                    current_sql_actions, current_sql_texts,
                    best_sql_actions, best_sql_texts, best_sql_times,
                )
                if improved:
                    consecutive_no_improvement[query_id] = 0
                else:
                    consecutive_no_improvement[query_id] = (
                        consecutive_no_improvement.get(query_id, 0) + 1
                    )
                if consecutive_no_improvement[query_id] >= early_stop_patience:
                    finished_queries.add(query_id)
                    logger.info(
                        "SQL %s stopped after %d consecutive non-improving attempts",
                        query_id, early_stop_patience,
                    )
                if self.skip_below_seconds > 0 and new_time < self.skip_below_seconds:
                    finished_queries.add(query_id)
                if query_id in finished_queries and self.memory_manager.retrieval_mode == "dynamic":
                    self.memory_manager.retriever.finalize_pending_experiences(
                        self.memory_manager.generate_next_query_info_id(query_id)
                    )
                round_sql_times[query_id] = new_time if improved else old_time
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
                    next_query_index=i + 1,
                    current_sql_actions=current_sql_actions,
                    current_sql_texts=current_sql_texts,
                    round_sql_times=round_sql_times,
                    round_sql_actions=round_sql_actions,
                )

            round_end = time.time()
            logger.info(f"Round {round_idx} optimization time: {round_end - round_start:.4f}s")
            evaluated_total_time = sum(round_sql_times.values())
            logger.info(
                "Round %d evaluated SQL total execution time: %.4fs",
                round_idx, evaluated_total_time,
            )
            estimated_workload_times = {
                query_id: round_sql_times.get(query_id, best_sql_times.get(query_id))
                for query_id in query_ids
            }
            estimated_workload_times = {
                query_id: latency for query_id, latency in estimated_workload_times.items()
                if latency is not None
            }
            logger.info(
                "Round %d workload estimated total execution time: %.4fs (%d prior-best values reused)",
                round_idx, sum(estimated_workload_times.values()),
                len(estimated_workload_times) - len(round_sql_times),
            )
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
                round_idx if round_interrupted else round_idx + 1,
                consecutive_no_improvement,
                finished_queries,
                next_query_index=next_query_index if round_interrupted else 0,
                current_sql_actions=current_sql_actions,
                current_sql_texts=current_sql_texts,
                round_sql_times=round_sql_times if round_interrupted else {},
                round_sql_actions=round_sql_actions if round_interrupted else {},
            )
            if not round_interrupted:
                completed_rounds += 1

        # A run ending at the time budget has no next query step. Close those
        # retrieval transitions as terminal before saving the final model.
        if self.memory_manager.retrieval_mode == "dynamic":
            retriever = self.memory_manager.retriever
            for next_id in list(retriever.pending_experiences):
                retriever.finalize_pending_experiences(next_id)
            model_path = self.memory_manager.retriever_model_path
            if model_path:
                temporary = f"{model_path}.tmp"
                retriever.save_model(temporary)
                os.replace(temporary, model_path)

        # Output final results
        final_time = time.time()
        total_elapsed = final_time - start_time
        logger.info("=== Optimization Complete ===")
        logger.info(f"Total time: {total_elapsed:.2f}s (timeout limit: {duration}h)")
        logger.info(f"Completed rounds: {round_idx}")
        logger.info(f"Best SQL execution times: {best_sql_times}")
        logger.info(f"Workload best total execution time: {sum(best_sql_times.values()):.4f}s")
        logger.info(f"Best SQL fix actions: {best_sql_actions}")
        logger.info(f"Final accepted SQL actions: {current_sql_actions}")
        logger.info(f"Final accepted SQL texts: {current_sql_texts}")
        logger.info("Diagnosis process ended")
