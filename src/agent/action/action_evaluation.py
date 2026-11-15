import logging
import re
import time

from agents import Runner

from agent.prompt import build_fix_prompt, build_rewrite_prompt


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = True


class InvalidRewriteError(ValueError):
    """The candidate rewrite could not be validated as equivalent."""


class ValidationAgent:
    """Validate, repair, and trial an LLM-generated tuning action."""

    def __init__(self, db, fix_agent, fix_timeout, sql_timeout, max_retries=3, min_improvement=0.1):
        self.db = db
        self.fix_agent = fix_agent
        self.fix_timeout = fix_timeout
        self.sql_timeout = sql_timeout
        self.max_retries = max(0, int(max_retries))
        self.min_improvement = float(min_improvement)

    async def evaluate(self, fix_action, rewrite_sql, query_info):
        return await evaluate_action(
            fix_action, rewrite_sql, query_info,
            db=self.db, fix_timeout=self.fix_timeout,
            sql_timeout=self.sql_timeout, fix_agent=self.fix_agent,
            max_retries=self.max_retries,
            min_improvement=self.min_improvement,
        )


def _round_time_value(value):
    if isinstance(value, float):
        return round(value, 3)
    return value


async def evaluate_action(fix_action, rewrite_sql, query_info, *, db, fix_timeout, sql_timeout, fix_agent, max_retries=3, min_improvement=0.1):
    validation_start = time.perf_counter()
    action_apply_time = 0.0
    validation_metrics = {
        "T_qed": 0.0,
        "T_result_compare": 0.0,
        "qed_calls": 0,
        "qed_cache_hits": 0,
        "result_compare_calls": 0,
    }

    def execute_action(action):
        # Count every attempt, including actions that fail and are repaired.
        nonlocal action_apply_time
        start = time.perf_counter()
        try:
            return db.execute(action, timeout=fix_timeout)
        finally:
            action_apply_time += time.perf_counter() - start

    def with_runtime_metrics(result):
        elapsed = time.perf_counter() - validation_start
        result["runtime_metrics"] = {
            **validation_metrics,
            "T_action_apply": action_apply_time,
            # The remaining ValidationAgent time includes rewrite checks,
            # FixAgent repairs, candidate SQL execution, and the decision.
            "T_validation": max(0.0, elapsed - action_apply_time),
        }
        return result

    try:
        fix_action, fix_sql = await derive_fix_action_and_sql(
            fix_action, rewrite_sql, query_info.query, db=db, fix_agent=fix_agent,
            max_retries=max_retries, runtime_metrics=validation_metrics,
        )
    except InvalidRewriteError as exc:
        return with_runtime_metrics({
            "status": -1,
            "msg": str(exc),
            "fix_action": "",
            "rewrite_sql": "",
            "new_time": _round_time_value(sql_timeout),
            "old_time": _round_time_value(query_info.execution_time),
            "approve_time": 0.0,
        })

    print("Executing fix_action")

    if fix_action != "":
        success, _, _, err = execute_action(fix_action)
    else:
        success = True
        err = ""
    print("Fix_action execution completed")

    retry_count = 0
    max_retry = max(0, int(max_retries))
    while not success and retry_count < max_retry:
        prompt = build_fix_prompt(fix_action, err)
        # logger.info(
        #     "[Prompt][Fix][SQL %s][Retry %s]\n%s",
        #     getattr(query_info, "query_id", "unknown"),
        #     retry_count,
        #     prompt,
        # )
        result = await Runner.run(starting_agent=fix_agent, input=prompt)
        new_fix_action = result.final_output
        print(f"new_fix_action: {new_fix_action}")
        try:
            new_fix_action, new_fix_sql = await derive_fix_action_and_sql(
                new_fix_action, rewrite_sql, query_info.query, db=db, fix_agent=fix_agent,
                max_retries=max_retries, runtime_metrics=validation_metrics,
            )
        except InvalidRewriteError as exc:
            return with_runtime_metrics({
                "status": -1,
                "msg": str(exc),
                "fix_action": "",
                "rewrite_sql": "",
                "new_time": _round_time_value(sql_timeout),
                "old_time": _round_time_value(query_info.execution_time),
                "approve_time": 0.0,
            })
        success, _, _, err = execute_action(new_fix_action)
        fix_action = new_fix_action
        fix_sql = new_fix_sql
        retry_count += 1
    if not success:
        old_time = _round_time_value(query_info.execution_time)
        approve_time = 0.0
        return with_runtime_metrics({
            "status": -1,
            "msg": f"SQL fix failed: {fix_action}, error: {err}",
            "fix_action": fix_action,
            "rewrite_sql": (fix_sql if fix_sql != query_info.query else ""),
            "new_time": _round_time_value(sql_timeout),
            "old_time": old_time,
            "approve_time": approve_time,
        })

    old_time = _round_time_value(query_info.execution_time)

    print("Executing performance test")
    success, _, new_time, err = db.execute(fix_sql, timeout=sql_timeout)
    print("Performance test completed")
    new_time = _round_time_value(new_time)

    # If we get here, ActionManager is responsible for rollback when needed.
    approve_time = _round_time_value(old_time - new_time) if success else 0.0

    if not success:
        if err and isinstance(err, str) and ("timeout" in err.lower() or "querycanceled" in err):
            return with_runtime_metrics({
                "status": 0,
                "msg": f"Fix ineffective, performance not improved or degraded, original time {old_time:.4f}s, new time {new_time:.4f}s",
                "fix_action": fix_action,
                "rewrite_sql": (fix_sql if fix_sql != query_info.query else ""),
                "new_time": new_time,
                "old_time": old_time,
                "approve_time": approve_time,
            })
        return with_runtime_metrics({
            "status": -1,
            "msg": f"Performance test SQL execution failed: {err}",
            "fix_action": fix_action,
            "rewrite_sql": (fix_sql if fix_sql != query_info.query else ""),
            "new_time": sql_timeout,
            "old_time": old_time,
            "approve_time": approve_time,
        })

    if (old_time - new_time) / max(abs(old_time), 1e-8) >= min_improvement:
        return with_runtime_metrics({
            "status": 1,
            "msg": f"Fix successful, performance improved, time reduced from {old_time:.4f}s to {new_time:.4f}s",
            "fix_action": fix_action,
            "rewrite_sql": (fix_sql if fix_sql != query_info.query else ""),
            "new_time": new_time,
            "old_time": old_time,
            "approve_time": approve_time,
        })
    return with_runtime_metrics({
        "status": 0,
        "msg": f"Fix ineffective, performance not improved or degraded, original time {old_time:.4f}s, new time {new_time:.4f}s",
        "fix_action": fix_action,
        "rewrite_sql": (fix_sql if fix_sql != query_info.query else ""),
        "new_time": new_time,
        "old_time": old_time,
        "approve_time": approve_time,
    })


async def derive_fix_action_and_sql(fix_action, rewrite_sql, origin_sql, *, db, fix_agent, max_retries=3, runtime_metrics=None):
    match = re.search(r"/\*\+[\s\S]*?\*/", fix_action)
    if match:
        hint = match.group(0).strip()
        clean_fix_action = (fix_action[: match.start()] + fix_action[match.end() :]).strip()
        if rewrite_sql != "":
            rewrite_sql = await check_sql_rewrite(
                origin_sql, rewrite_sql, db=db, fix_agent=fix_agent, max_retries=max_retries,
                runtime_metrics=runtime_metrics,
            )
            if rewrite_sql is None:
                raise InvalidRewriteError("Rewritten SQL failed semantic validation")
            target_sql = rewrite_sql
        else:
            target_sql = origin_sql
        fix_sql = _inject_hint_into_sql(hint, target_sql)
        return clean_fix_action, fix_sql

    clean_fix_action = fix_action
    if rewrite_sql != "":
        rewrite_sql = await check_sql_rewrite(
            origin_sql, rewrite_sql, db=db, fix_agent=fix_agent, max_retries=max_retries,
            runtime_metrics=runtime_metrics,
        )
        if rewrite_sql is None:
            raise InvalidRewriteError("Rewritten SQL failed semantic validation")
        fix_sql = rewrite_sql
    else:
        fix_sql = origin_sql
    return clean_fix_action, fix_sql


def _inject_hint_into_sql(hint, sql):
    pattern = r"\b(WITH|SELECT|EXPLAIN)\b"

    def replace_match(match):
        start_pos = match.start()
        keyword = match.group(1).upper()

        left_count = sql[:start_pos].count("(")
        right_count = sql[:start_pos].count(")")

        if left_count > right_count:
            return match.group(0)

        if keyword == "SELECT":
            before_text = sql[:start_pos].lower().strip()
            if before_text.endswith("as"):
                return match.group(0)

        if keyword == "WITH":
            return match.group(0)

        if start_pos > 0:
            prefix = sql[:start_pos].strip()
            if prefix.endswith(hint.strip()):
                return match.group(0)

        return hint + " " + match.group(0)

    result = re.sub(pattern, replace_match, sql, flags=re.IGNORECASE)
    return result


async def check_sql_rewrite(origin_sql, rewrite_sql, *, db, fix_agent, max_retries=3, runtime_metrics=None):
    # QED is PostgreSQL-specific in this implementation. Other DBMSs must
    # still compare query results before accepting a rewrite.
    if rewrite_sql.strip().rstrip(";") == origin_sql.strip().rstrip(";"):
        return None
    if getattr(db, "skip_equivalence_validation", False):
        qed_match = None
    else:
        print("start QED check")
        qed_start = time.perf_counter()
        try:
            qed_match = db.check_sql_equivalence(origin_sql, rewrite_sql)
        finally:
            if runtime_metrics is not None:
                runtime_metrics["T_qed"] = runtime_metrics.get("T_qed", 0.0) + time.perf_counter() - qed_start
                runtime_metrics["qed_calls"] = runtime_metrics.get("qed_calls", 0) + 1
                if getattr(db, "last_qed_cache_hit", False):
                    runtime_metrics["qed_cache_hits"] = runtime_metrics.get("qed_cache_hits", 0) + 1
        print(f"QED equivalence check: {qed_match}")

    if qed_match:
        return rewrite_sql

    retry_count = 0
    max_retry = max(0, int(max_retries))
    def compare_results(candidate):
        started = time.perf_counter()
        try:
            return db.compare_sql_results(origin_sql, candidate)
        finally:
            if runtime_metrics is not None:
                runtime_metrics["T_result_compare"] = (
                    runtime_metrics.get("T_result_compare", 0.0) + time.perf_counter() - started
                )
                runtime_metrics["result_compare_calls"] = runtime_metrics.get("result_compare_calls", 0) + 1

    match = compare_results(rewrite_sql)
    print(f"Result set comparison match: {match}")

    while not match and retry_count < max_retry:
        prompt = build_rewrite_prompt(origin_sql, rewrite_sql)
        # logger.info(
        #     "[Prompt][Rewrite][Retry %s]\n%s",
        #     retry_count,
        #     prompt,
        # )
        result = await Runner.run(starting_agent=fix_agent, input=prompt)
        new_rewrite_sql = result.final_output
        print(f"new_rewrite_sql: {new_rewrite_sql}")
        rewrite_sql = new_rewrite_sql
        retry_count += 1
        match = compare_results(rewrite_sql)
        print(f"Result set comparison match: {match}")

    if not match or rewrite_sql.strip().rstrip(";") == origin_sql.strip().rstrip(";"):
        print("Failed to validate rewritten SQL")
        return None

    return rewrite_sql
