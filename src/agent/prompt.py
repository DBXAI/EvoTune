"""
Prompt generation functions for database optimization system.
This module contains all prompt generation functions extracted from action_manager.py and planner.py.
"""

from agent.contracts import (
    CANONICAL_COMPONENTS,
    COMPONENT_ACTION_SPACE_NAMES,
    format_component_label,
    normalize_components,
    normalize_tuning_history,
)


best_sql_times = {
    "TPCH": {
        "Q1": 9.883060,
        "Q2": 4.435821,
        "Q3": 5.071305,
        "Q4": 0.790032,
        "Q5": 5.788042,
        "Q6": 1.391722,
        "Q7": 2.450940,
        "Q8": 1.729923,
        "Q9": 8.097767,
        "Q10": 4.430317,
        "Q11": 0.979684,
        "Q12": 2.103801,
        "Q13": 11.883281,
        "Q14": 1.976638,
        "Q15": 5.546087,
        "Q16": 2.079047,
        "Q17": 0.083923,
        "Q18": 23.506269,
        "Q19": 0.117512,
        "Q20": 1.461456,
        "Q21": 3.018528,
        "Q22": 7.072805,
    },
    "TPCDS": {
        "Q1": 300.000000,
        "Q2": 20.674408,
        "Q3": 0.238711,
        "Q4": 300.000000,
        "Q5": 70.429734,
        "Q6": 300.000000,
        "Q7": 5.471828,
        "Q8": 4.977158,
        "Q9": 79.473810,
        "Q10": 300.000000,
        "Q11": 300.000000,
        "Q12": 1.603219,
        "Q13": 7.215901,
        "Q14": 300.000000,
        "Q15": 3.649739,
        "Q16": 98.854028,
        "Q17": 16.998348,
        "Q18": 5.859661,
        "Q19": 2.899983,
        "Q20": 1.184857,
        "Q21": 2.102600,
        "Q22": 27.152599,
        "Q23": 300.000000,
        "Q24": 23.721941,
        "Q25": 18.937374,
        "Q26": 9.230094,
        "Q27": 7.336706,
        "Q28": 34.023110,
        "Q29": 2.003364,
        "Q30": 300.000000,
        "Q31": 146.832702,
        "Q32": 12.171474,
        "Q33": 10.072693,
        "Q34": 3.335732,
        "Q35": 300.000000,
        "Q36": 16.377175,
        "Q37": 0.034528,
        "Q38": 145.982584,
        "Q39": 12.611345,
        "Q40": 0.882909,
        "Q41": 18.646022,
        "Q42": 2.161797,
        "Q43": 4.224569,
        "Q44": 25.043756,
        "Q45": 2.739354,
        "Q46": 7.213401,
        "Q47": 300.000000,
        "Q48": 8.925575,
        "Q49": 3.201867,
        "Q50": 1.649846,
        "Q51": 56.011620,
        "Q52": 2.910233,
        "Q53": 4.363119,
        "Q54": 123.027910,
        "Q55": 2.135851,
        "Q56": 15.530809,
        "Q57": 31.879963,
        "Q58": 13.000547,
        "Q59": 40.885755,
        "Q60": 19.523776,
        "Q61": 0.017021,
        "Q62": 4.580046,
        "Q63": 7.258341,
        "Q64": 25.360461,
        "Q65": 59.328525,
        "Q66": 6.304007,
        "Q67": 186.245771,
        "Q68": 3.833772,
        "Q69": 4.347820,
        "Q70": 48.076645,
        "Q71": 4.300104,
        "Q72": 8.691721,
        "Q73": 2.951616,
        "Q74": 300.000000,
        "Q75": 126.500465,
        "Q76": 7.839012,
        "Q77": 7.635641,
        "Q78": 250.786035,
        "Q79": 4.210714,
        "Q80": 8.035321,
        "Q81": 300.000000,
        "Q82": 0.135833,
        "Q83": 3.357301,
        "Q84": 1.629378,
        "Q85": 1.643650,
        "Q86": 9.255396,
        "Q87": 138.634698,
        "Q88": 60.482861,
        "Q89": 9.558431,
        "Q90": 3.513042,
        "Q91": 0.429883,
        "Q92": 9.157462,
        "Q93": 2.100484,
        "Q94": 31.477500,
        "Q95": 68.868398,
        "Q96": 3.340921,
        "Q97": 44.934617,
        "Q98": 3.966343,
        "Q99": 10.075517,
    },
    "JOB": {
        "Q1": 0.207289,
        "Q2": 0.174250,
        "Q3": 0.143039,
        "Q4": 0.109824,
        "Q5": 0.536570,
        "Q6": 0.285439,
        "Q7": 0.179154,
        "Q8": 0.439616,
        "Q9": 0.209086,
        "Q10": 0.175065,
        "Q11": 0.248149,
        "Q12": 0.304847,
        "Q13": 0.188929,
        "Q14": 0.371559,
        "Q15": 0.058961,
        "Q16": 0.068868,
        "Q17": 0.133608,
        "Q18": 0.353786,
        "Q19": 0.396548,
        "Q20": 0.319278,
        "Q21": 0.962659,
        "Q22": 0.352000,
        "Q23": 2.090212,
        "Q24": 1.205844,
        "Q25": 0.780641,
        "Q26": 2.522518,
        "Q27": 0.499087,
        "Q28": 0.374647,
        "Q29": 6.064952,
        "Q30": 5.792696,
        "Q31": 1.291820,
        "Q32": 0.848086,
        "Q33": 1.351708,
        "Q34": 3.434997,
        "Q35": 0.000052,
        "Q36": 0.000088,
        "Q37": 1.306864,
        "Q38": 0.334343,
        "Q39": 0.000069,
        "Q40": 0.094141,
        "Q41": 0.186353,
        "Q42": 0.191693,
        "Q43": 0.123724,
        "Q44": 0.632144,
        "Q45": 0.000064,
        "Q46": 0.000069,
        "Q47": 0.000057,
        "Q48": 0.000059,
        "Q49": 0.000046,
        "Q50": 0.000048,
        "Q51": 0.539783,
        "Q52": 0.279526,
        "Q53": 0.134393,
        "Q54": 0.615610,
        "Q55": 1.010046,
        "Q56": 0.587199,
        "Q57": 4.990829,
        "Q58": 1.289380,
        "Q59": 1.174189,
        "Q60": 4.091684,
        "Q61": 1.926888,
        "Q62": 3.080776,
        "Q63": 1.356690,
        "Q64": 4.041575,
        "Q65": 4.276380,
        "Q66": 1.394408,
        "Q67": 0.223519,
        "Q68": 1.672122,
        "Q69": 1.277519,
        "Q70": 0.480913,
        "Q71": 1.452588,
        "Q72": 1.172114,
        "Q73": 0.000101,
        "Q74": 0.000082,
        "Q75": 0.000080,
        "Q76": 0.818028,
        "Q77": 0.762241,
        "Q78": 0.372250,
        "Q79": 0.759741,
        "Q80": 0.769927,
        "Q81": 0.926769,
        "Q82": 0.494359,
        "Q83": 0.000096,
        "Q84": 0.000074,
        "Q85": 0.284890,
        "Q86": 0.670342,
        "Q87": 0.652387,
        "Q88": 1.079833,
        "Q89": 0.747400,
        "Q90": 1.254347,
        "Q91": 0.000136,
        "Q92": 0.000109,
        "Q93": 0.000086,
        "Q94": 1.459996,
        "Q95": 0.330279,
        "Q96": 0.000096,
        "Q97": 1.223700,
        "Q98": 0.354450,
        "Q99": 0.000126,
        "Q100": 0.000161,
        "Q101": 0.000110,
        "Q102": 0.000146,
        "Q103": 1.314771,
        "Q104": 0.434778,
        "Q105": 0.010450,
        "Q106": 1.020073,
        "Q107": 0.679441,
        "Q108": 2.728070,
        "Q109": 0.159751,
        "Q110": 0.179895,
        "Q111": 0.283601,
        "Q112": 0.208859,
        "Q113": 0.285074,
    },
}


def normalize_workload_name(value):
    normalized = str(value or "").strip().upper()
    if "OSSINSIGHT" in normalized:
        return "OSSINSIGHT"
    if normalized in {"TPC-DS", "TPCDS"}:
        return "TPCDS"
    if normalized in {"TPC-H", "TPCH"}:
        return "TPCH"
    if normalized == "JOB":
        return "JOB"
    return None


def get_workload_best_sql_times(workload, booster=False):
    workload_name = normalize_workload_name(workload)
    if not booster or not workload_name:
        return {}
    return dict(best_sql_times.get(workload_name, {}))


def fmt_cases(cases):
    if not cases:
        return "None."

    lines = []
    for i, case in enumerate(cases, 1):
        query_info = case.get("query_info", {})
        lines.append(
            f"\n    Case {i}:"
            f"\n    SQL: {query_info.get('query')}"
            f"\n    Execution Plan: {query_info.get('plan_json')}"
            f"\n    Origin Execution Runtime: {query_info.get('execution_time')}s"
            f"\n    Components: {format_component_label(case.get('root_causes'))}"
            f"\n    Tuning_Action: {case.get('fix_action')}"
            f"\n    Improve Time: {case.get('approve_time')}s"
        )
    return "\n".join(lines)


def build_action_prompt(root_cause, base_info, action_space, mode, positives, negatives, experiences):
    """
    Format a prompt that feeds the LLM with all necessary context and enforces
    a consistent Appendix A Tuning Agent output contract.
    """
    query_info = base_info.get("query_info")
    db_info = base_info.get("database_info", {})
    components = normalize_components(root_cause)
    action_space = action_space or "None"

    positives_text = fmt_cases(positives)
    negatives_text = fmt_cases(negatives)
    experiences_text = experiences or "None."

    # Mode prompt
    if mode == "exploit":
        mode_instruction = """You are now in EXPLOIT mode.
    - Positive Observations come from historical fixes under the same predicted components.
    - Negative Observations record ineffective directions and must be avoided.
    - Treat prior Tuning_Action entries as optimization patterns, not reusable output."""
    else:
        mode_instruction = """You are now in EXPLORE mode.
    - Positive Observations come from historical fixes under different predicted components.
    - Negative Observations capture ineffective directions and must be avoided.
    - Use them only to inspire new actions inside the allowed action space for the current components."""

    return f"""You are an SQL optimization expert. The bottleneck-related tuning components have already been localized. Your job is to recommend executable tuning action(s) only within the localized tuning subspace, using the current query context, retrieved historical observations with experiences, and the refined action space.
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
    • Database Schema: {db_info.get('schema') or db_info.get('dbname') or 'unknown'}
    • CPU: {query_info.external_metrics[0]}
    • Read I/O: {query_info.external_metrics[1]}
    • Write I/O: {query_info.external_metrics[2]}
    • Virtual Memory: {query_info.external_metrics[3]}
    • Physical Memory: {query_info.external_metrics[4]}
    • Net Received: {query_info.external_metrics[5]}
    • Net Sent: {query_info.external_metrics[6]}
    3. Localized Bottleneck Component(s): {components}
    4. Tuning Mode
    {mode_instruction}
    5. Retrieved Historical Observations and Experiences
    • Positive Observations (historically effective): {positives_text}
    • Negative Observations (historically ineffective): {negatives_text}
    • Experiences: {experiences_text}
    6. Reflective Reasoning Guidance
    Before proposing your optimization solution, analyze the historical observations and experiences to understand why certain approaches succeeded or failed. Adapt successful patterns to the current context while avoiding patterns that failed in similar situations.
    7. Refined Tuning Subspace
    {action_space}
    ## Constraints
    • Tuning_Action must be executable directly in {db_info.get('db_type')}.
    • The Tuning_Action must only operate on the component(s) listed in {components}.
    • The Tuning_Action should be safe and must not drop or delete data.
    ## Output Format
    You must output ONLY in the following format:
    Tuning_Action:
    <your tuning action here, do not include any other text, if no tuning action is needed, output an empty string>
    Query_Rewrite:
    <yes or no>
    ## Output Requirements
    • Tuning_Action must contain only executable tuning statements.
    • Set Query_Rewrite to yes if the selected action(s) involve query rewriting, and place the full rewritten SQL in Tuning_Action; otherwise, set Query_Rewrite to no.
    • If multiple executable statements are needed, place all of them in Tuning_Action in executable form."""


def build_fix_prompt(fix_action, err):
    """Build prompt for fixing non-executable SQL statements."""
    return f"""You have been given a non-executable SQL statement and the exact database error:
    • Non-executable SQL: {fix_action}
    • Database Error Message: {err}
    Your sole objective is to produce a corrected, fully-formed SQL statement (or an ordered sequence of SQL statements) that will run without error.
    Strict output rules:
    1. OUTPUT NOTHING BUT THE SQL STATEMENT(S). No explanations, no comments, no markdown, no surrounding quotes.
    2. If a single statement suffices, output exactly that statement, terminated by a single semicolon and a newline.
    3. If multiple statements are required, output them in execution order, each terminated by a semicolon and a newline.
     — Output Format —
    <paste only the corrected fix action statement(s) here, exactly as SQL, nothing else>"""


def build_rewrite_prompt(origin_sql, rewrite_sql):
    """Build prompt for rewriting SQL to match reference output."""
    return f"""You are given two SQL statements:
    • Reference SQL (correct output): {origin_sql}
    • Divergent SQL (incorrect output): {rewrite_sql}
    Your task is to rewrite the Divergent SQL so that its result set matches the Reference SQL exactly.

    Strict output rules:
    1. Output only the revised SQL statement, which must be executable and self-contained. Do not include explanations, comments, markdown, or any extra text.
    2. Preserve the original intent and structure of the Divergent SQL as much as possible while ensuring the output matches exactly.
    3. If you cannot rewrite the SQL, output the original SQL.

    — Output Format —
    <paste only the revised SQL here, exactly as SQL, nothing else>"""


def build_root_cause_diagnosis_prompt(
    query_info,
    tuning_history=None,
    database_info=None,
    timeout=300,
    consecutive_no_improvement=0,
):
    """Build prompt for the Detection Agent in tuning-space localization."""

    tuning_history = normalize_tuning_history(tuning_history)
    database_info = database_info or {}

    # Handle timeout case
    execution_time_text = f"{query_info.execution_time} s"
    if query_info.execution_time == timeout:
        execution_time_text = f"{query_info.execution_time} (timeout)"

    # Format tuning history (success/failure counts)
    if tuning_history:
        lines = []
        for item in tuning_history:
            label = item.get("label")
            s_cnt = item.get("success_count", 0)
            f_cnt = item.get("fail_count", 0)
            lines.append(f"{label}: success_count={s_cnt}, fail_count={f_cnt};")
        history_text = " ".join(lines)
        history_text = "{" + history_text + "}"
    else:
        history_text = "No previous tuning attempts recorded."

    # Make stagnation explicit to the LLM.  The contextual selector still
    # chooses which localization expert to call, but once the trajectory has
    # accumulated repeated failures the LLM must deliberately explore a less
    # tried component instead of repeating the same diagnosis.
    total_successes = sum(int(item.get("success_count", 0)) for item in tuning_history)
    total_failures = sum(int(item.get("fail_count", 0)) for item in tuning_history)
    max_component_failures = max(
        (int(item.get("fail_count", 0)) for item in tuning_history),
        default=0,
    )
    total_attempts = total_successes + total_failures
    trajectory_stalled = (
        (total_attempts >= 3 and total_successes == 0)
        or max_component_failures >= 3
        or (total_attempts >= 6 and total_successes / max(total_attempts, 1) < 0.2)
        or int(consecutive_no_improvement) >= 3
    )
    if trajectory_stalled:
        stagnation_guidance = f"""
    • Trajectory status: STALLED (repeated previous attempts have not produced
      reliable improvement; the current query has {consecutive_no_improvement}
      consecutive non-improving attempts). Perform a randomized exploration
      step now: choose
      at least one low-attempt or previously untried component from [index,
      knob, hint, query], and consider a new component combination. Do not
      simply repeat the component or component combination with the highest
      failure count. Use the plan and runtime evidence to reject implausible
      choices, but keep the selection exploratory. This exploration applies to
      root-cause localization only; the output must still follow the strict
      JSON schema below.
        """
    else:
        stagnation_guidance = """
    • Trajectory status: no persistent stagnation detected. Use the historical
      feedback together with the plan and runtime evidence; retain some
      exploration of components with little or no prior feedback.
        """

    return f"""You are a database performance diagnosis expert. Your task is to identify the most promising tuning component space(s) for the following slow SQL query. To do so, reason directly from the current query context, trajectory-level tuning feedback, and DBA-style knowledge, and determine which component space(s) are most likely to deliver substantial performance improvement.
    ## Candidate Tuning Components
    The allowed components are exactly: [index, knob, hint, query]
    ## Context
    1. Query Information
    • SQL Statement: {query_info.query}
    • Execution Plan: {query_info.plan_json}
    • Execution Runtime: {execution_time_text}
    • Internal Metrics: Tuples Returned:{query_info.internal_metrics[0]}, Blocks Hit:{query_info.internal_metrics[1]}, Blocks Read:{query_info.internal_metrics[2]}, Tuples Fetched:{query_info.internal_metrics[3]}, Index Tuples Fetched:{query_info.internal_metrics[4]}, Sequential Tuples Read:{query_info.internal_metrics[5]}, Sequential Scans:{query_info.internal_metrics[6]}, Index Scans:{query_info.internal_metrics[7]}, Heap Blocks Hit:{query_info.internal_metrics[8]}, Heap Blocks Read:{query_info.internal_metrics[9]}, Index Blocks Hit:{query_info.internal_metrics[10]}, Index Blocks Read:{query_info.internal_metrics[11]}
    • CPU: {query_info.external_metrics[0]}
    • Read I/O: {query_info.external_metrics[1]}
    • Write I/O: {query_info.external_metrics[2]}
    • Virtual Memory: {query_info.external_metrics[3]}
    • Physical Memory: {query_info.external_metrics[4]}
    • Net Received: {query_info.external_metrics[5]}
    • Net Sent: {query_info.external_metrics[6]}
    2. Database Environment Information
    • Database Type: {database_info.get('db_type')}
    • Workload Type: {database_info.get('workload_type')}
    • Database Size: {database_info.get('size')}
    • Database Schema: {database_info.get('schema') or database_info.get('dbname') or 'unknown'}
    ## Reasoning Guidance
    1. Historical Tuning Summary
    {history_text}
    2. Trajectory-Guided Reasoning
    Guide component reasoning according to the following principles:
    • If success_count > 0 and fail_count = 0: prioritize the component since past interventions consistently validate it as a bottleneck.
    • If success_count = 0 and fail_count > 0: deprioritize the component and encourage exploration of alternative components, since previous interventions have not led to improvement.
    • If success_count > 0 and fail_count > 0: keep the component under consideration and make a more nuanced judgment based on DBA knowledge and the available evidence.
    3. Stagnation-Aware Exploration
    {stagnation_guidance}
    4. DBA Knowledge-Guided Reasoning
    Use DBA-style knowledge to reason over the query context to identify target tuning components.
    • Plan suboptimality: A plan bottleneck typically arises from suboptimal access paths, join orders, or join strategies that introduce unnecessary overhead, such as redundant sorting, full table scans, or excessive intermediate results. Typical evidence includes cases where the optimizer prefers sequential scans despite the presence of indexes under high selectivity, incurs extra sorting due to unordered access paths, or exhibits hash join spill-to-disk behavior.
    • Index bottlenecks: Index bottlenecks are characterized by large-range scans on selective predicates. Typical evidence includes a large gap between scanned and returned rows, predicates that cannot be effectively exploited by indexes, and repeated heap/table access due to non-covering indexes.
    • Knob misconfiguration: Knob tuning becomes important when analytical operators spill to disk or lack sufficient memory or parallel resources. Typical evidence includes disk-backed temporary tables, underutilized parallel workers, and high-cost join or aggregation execution under limited resources.
    • Query formulation issues: Query rewriting is prioritized when the SQL structure itself introduces unnecessary work. Typical evidence includes implicit type conversions that block B-Tree access and trigger full scans, and overlong IN lists.
    ## Constraints (STRICT)
    • You MUST select tuning components only from the following labels: {", ".join(CANONICAL_COMPONENTS)}.
    • You MUST output strict JSON only, containing a single field predicted_components. No extra text and no explanation.
    • predicted_components MUST be a non-empty array of the final selected tuning components.
    Output JSON Format:
    {{"predicted_components": ["index", "knob"]}}"""


def build_action_space_prompt(root_causes, db, knob_config, current_knob_values, query_info=None):
    """Build action space prompt based on root causes."""
    # Root causes
    root_causes = normalize_components(root_causes)

    # Build action space prompt based on root causes
    action_space_prompt = ""
    is_tidb = str(getattr(db, "db_type", "")).lower() == "tidb"

    # Index related
    if "index" in root_causes:
        current_indexes = db.get_indexes() if db else []
        index_syntax = (
            "Use TiDB/MySQL syntax CREATE INDEX IF NOT EXISTS <name> ON <table> (<columns>) "
            "or DROP INDEX <name> ON <table>."
            if is_tidb else
            "Use CREATE INDEX IF NOT EXISTS or DROP INDEX IF EXISTS."
        )
        action_space_prompt += f"""• Index Space: 
    - {index_syntax} Name each newly created index using the pattern <table_name>_<col>_idx.
    - Derive query-specific index candidates from the database schema and current execution plan; consider composite indexes when supported by the query structure and execution plan.
    - Given the existing indexes in the DBMS ({current_indexes}), avoid creating duplicate or subsumed indexes, and do not drop primary keys, unique indexes, or any indexes that enforce constraints.
    """

    # Knob related
    if "knob" in root_causes:
        system_knobs_specs = knob_config.get("system_knobs")
        current_values = current_knob_values.get("system_knobs")

        knob_lines = []
        for knob_name, spec in system_knobs_specs.items():
            current_val = current_values.get(knob_name)
            knob_lines.append(
                f"{knob_name}: type: {spec.get('type')}, min: {spec.get('min')}, "
                f"max: {spec.get('max')}, default: {spec.get('default')}, "
                f"current: {current_val}"
            )

        knob_statement = (
            'Use SET SESSION <knob> = <value>; for query-local TiDB changes; do not use SET GLOBAL.'
            if is_tidb else
            'You MUST adjust database configuration parameters using SET knob = value.'
        )
        action_space_prompt += f"""• Knob Space:  
    - {knob_statement}
    - Knob Space is strictly limited to the following knobs, you must only use these knobs in your Tuning_Action, and any knob not explicitly listed here must not be used: 
    {knob_lines}
    - For each relevant knob, consider its allowed value range, default setting, and current configuration when making recommendations.
    """

    # Execution plan/optimizer related
    if "hint" in root_causes:
        query_knobs_specs = knob_config.get("query_knobs")
        current_values = current_knob_values.get("query_knobs")

        knob_lines = []
        for knob_name, spec in query_knobs_specs.items():
            current_val = current_values.get(knob_name)
            knob_lines.append(
                f"{knob_name}: type: {spec.get('type')}, min: {spec.get('min')}, "
                f"max: {spec.get('max')}, default: {spec.get('default')}, "
                f"current: {current_val}"
            )

        hint_statement = (
            "Use only TiDB optimizer hints such as /*+ HASH_JOIN(t1,t2) */, "
            "/*+ INL_JOIN(t1,t2) */, /*+ MERGE_JOIN(t1,t2) */, "
            "/*+ READ_FROM_STORAGE(TIKV[t]) */, or /*+ MAX_EXECUTION_TIME(N) */."
            if is_tidb else
            "Use PostgreSQL hints to guide query execution."
        )
        action_space_prompt += f"""• Hint Space: 
    - {hint_statement}
    - Hint Space is strictly limited to the following knobs, you must only use these knobs in your Tuning_Action, and any knob not explicitly listed here must not be used: 
    {knob_lines}
    - **IMPORTANT**: When using Hint Space, output ONLY the hint block. Do NOT include the original SQL."""

    # SQL readability/rewrite related
    if "query" in root_causes:
        from agent.action.rule_matcher import SQLRuleMatcher

        matcher = SQLRuleMatcher()
        rule_descriptions = matcher.get_rule_descriptions(query_info.query)

        action_space_prompt += f"""• Query Rewrite Space: 
    - Query rewriting must generate a semantically equivalent SELECT statement that preserves the original result set.
    - Consider only the rewrite rules provided below as the initial rewrite space. Apply only rules that are applicable to the current query, and prioritize rewrites that simplify execution, eliminate redundant operations, or improve data access efficiency:
    {rule_descriptions}
    """

    return action_space_prompt


def get_fix_agent_instructions():
    """Get instructions for the fix agent."""
    return """You are a SQL Precision Repair Agent with two core responsibilities:
    1. Executable Repair: When provided with a raw, non-executable SQL statement and its exact database error message,
    produce a corrected SQL statement (or the minimal sequence of statements) that will execute successfully in the target SQL dialect.
    2. Output Alignment: When given two SQL statements whose result sets do not match,
    rewrite the SQL so that its output exactly matches the output of the original SQL."""


def get_diagnostic_agent_instructions():
    """Get instructions for the diagnostic agent."""
    return """You are an expert in database performance diagnosis and SQL optimization.
    Your role is to:
    1. Understand detailed database and query execution context.
    2. Identify how the predicted components lead to performance degradation.
    3. Output exactly two labeled sections: Tuning_Action and Query_Rewrite.
    4. Always make the Tuning_Action executable directly in the target database.
    5. Never include extra text, comments, or sections beyond the required format.
    6. Always apply the given MCP APIs when producing fixes, if applicable.
    7. Assume the provided execution plan, metrics, and environment info are accurate.
    8. Do not include explanations, markdown, comments, or extra sections."""


def get_simple_retrieval_agent_instructions():
    """Get instructions for the simple retrieval agent."""
    return """You are an expert in database performance tuning and SQL optimization.
    This workflow does not have a separate diagnosis stage.
    Your role is to:
    1. Use the current query context and statically retrieved historical cases to produce a direct fix.
    2. Output exactly three labeled sections: Explanation, Tuning_Action and Query_Rewrite.
    3. Always make the Tuning_Action executable directly in the target database.
    4. Never include extra text, comments, or sections beyond the required format.
    5. Treat retrieved cases as references, not templates to copy blindly.
    6. Keep the Explanation concise and technical, avoiding generic descriptions."""


def get_root_cause_llm_instructions():
    """Get instructions for the root cause diagnosis LLM."""
    return """You are a database slow SQL bottleneck component diagnosis expert.
    Only output strict JSON with a single field predicted_components, nothing else."""


def get_summary_agent_instructions():
    """Get instructions for the experience summary agent."""
    return """You are a database tuning experience summarization expert.
    Summarize a cluster of historical tuning cases into a concise reusable experience."""


def get_reflection_agent_instructions():
    """Get instructions for the reflection generation agent."""
    return """You are a database tuning reflection expert.
    Convert a single tuning case into a concise reusable reflection."""


def build_case_reflection_prompt(case_info):
    """Build prompt for generating a reflection from a single case."""
    query_info = case_info.get("query_info", {})
    root_causes = case_info.get("root_causes", [])
    label = case_info.get("label", "negative")
    old_time = case_info.get("old_time", query_info.get("execution_time", 0.0))
    new_time = case_info.get("new_time", query_info.get("execution_time", 0.0))
    improvement = case_info.get("approve_time", 0.0)

    return f"""Generate a reusable reflection for the following tuning case.
    Query Context:
    - SQL: {query_info.get('query')}
    - Execution Plan: {query_info.get('plan_json')}
    - Runtime Before: {old_time}
    - Runtime After: {new_time}
    - Improvement: {improvement}
    Diagnosis Context:
    - Predicted Components: {root_causes}
    - Applied Tuning_Action: {case_info.get('fix_action')}
    - Outcome Label: {label}
    Requirements (Strict):
    - The reflection should explain why the action succeeded or failed under the query context.
    - Write it in a generalized form using: [Positive/Negative reflection:] Under [query conditions], applying [action pattern] tends to [help / fail], because [reason].
    - Keep it concise and reusable, Avoid unnecessary case-specific details.
    - Output only the reflection text."""


def build_experience_summary_prompt(root_causes, cases, previous_summary=None):
    """Build prompt for summarizing a cluster of cases into one experience."""
    lines = []
    for idx, case in enumerate(cases, 1):
        case_info = case.case_info
        lines.append(f"Case {idx}: " f"reflection={case_info.get('reflection', 'None.')}")

    cases_text = "\n".join(lines) if lines else "None."
    return f"""Summarize the following cluster of tuning cases into one reusable cluster-level experience.
    Root Causes: {root_causes}
    Previous Experience Summary: {previous_summary or 'None'}
    Cluster Case Reflections:
    {cases_text}
    The experience should summarize, in a generalized and reusable form, what action patterns tend to succeed or fail under what query conditions, and why.
    When helpful, describe both the successful and unsuccessful action patterns within a single experience statement, rather than as separate case-by-case summaries.
    Requirements:
    - Do not treat success and failure as two mandatory separate items.
    - You may summarize only success patterns, only failure patterns, or both, depending on what is supported by the cluster.
    - Prefer one coherent experience that explains the boundary between effective and ineffective actions under similar conditions.
    - Output only the experience summary text only."""


def build_simple_diagnosis_prompt(query_info, base_info):
    """Build prompt for simple LLM-based diagnosis without root cause prediction."""
    db_info = base_info.get("database_info", {})

    return f"""As an SQL optimization expert, your task is to diagnose and fix the following slow SQL query.
    You need to analyze the query and provide a direct fix without root cause prediction.

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

    ## Available Action Spaces
    You can perform the following types of optimizations:
    
    • Index Space: Create indexes to improve query performance
    - The naming convention for newly created indexes must follow the pattern: (table_name)_(col1)_(col2)_idx
    - You MUST use the CREATE INDEX IF NOT EXISTS or DROP INDEX IF EXISTS to create or drop indexes.
    - Index creation should be strictly guided by the slow_sql queries
    - Consider the query execution plan to identify beneficial indexes
    - Avoid generating duplicate or subsumed indexes
    - Do not drop primary keys, unique indexes, or constraint indexes
    
    • Knob Space: Adjust database configuration parameters
    - You MUST use the SET knob = value to modify database knobs.
    - Focus on parameters that can improve query performance
    - Be conservative with changes to avoid system instability
    
    • Hint Space: Use PostgreSQL hints to guide query execution
    - Output ONLY the hint block like: /*+ Set(knob_name value) Set(knob_name value) ... */
    - Do NOT include the SQL query when using hints
    - Use hints to force specific join methods, scan types, etc.
    
    • Query Rewrite Space: Rewrite the SQL for better performance
    - Generate a semantically equivalent SELECT statement
    - Ensure the rewritten query produces exactly the same result set
    - Focus on reducing computational complexity and optimizing data access patterns
    - Maintain identical result set and semantic equivalence

    ## Constraints
    - Tuning_Action must be executable directly in {db_info.get('db_type')}
    - The action should be safe and should not drop or delete data unless necessary
    - Ensure all proposed actions are syntactically correct and executable

    ## Output Format
    You must output ONLY in the following format:

    Explanation:
    <concise technical reason for the performance issue and proposed solution>

    Tuning_Action:
    <your fix SQL or optimization action here>

    Query_Rewrite:
    <yes or no>

    ## Output Requirements
    - "Explanation" must be a short, precise, technical summary
    - If the fix rewrites the SQL, "Query_Rewrite" must be "yes", else "no"
    - If rewriting, provide the optimized SQL under Tuning_Action
    - If adding indexes or changing knobs, provide the exact SQL or command
    - Do not output any extra sections, comments, or steps
    - Focus on the most impactful optimization for this specific query"""


def build_simple_retrieval_prompt(query_info, base_info, positives, negatives):
    """Build prompt for simple LLM with static case retrieval and no diagnosis stage."""
    db_info = base_info.get("database_info", {})
    positives_text = fmt_cases(positives)
    negatives_text = fmt_cases(negatives)

    return f"""You are an SQL optimization expert. This workflow has no separate diagnosis stage.
    Your task is to directly propose one executable optimization action for the current slow SQL by jointly using:
    - the current query/runtime context,
    - statically retrieved historical tuning cases.

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
    • Database Schema: {db_info.get('schema') or db_info.get('dbname') or 'unknown'}
    • CPU: {query_info.external_metrics[0]}
    • Read I/O: {query_info.external_metrics[1]}
    • Write I/O: {query_info.external_metrics[2]}
    • Virtual Memory: {query_info.external_metrics[3]}
    • Physical Memory: {query_info.external_metrics[4]}
    • Net Received: {query_info.external_metrics[5]}
    • Net Sent: {query_info.external_metrics[6]}

    3. Statically Retrieved Historical Cases:
    • Positive Cases: {positives_text}
    • Negative Cases: {negatives_text}

    ## Guidance
    • Prefer effective patterns from similar positive cases, but adapt them to the current SQL instead of copying them verbatim.
    • Avoid directions shown to be ineffective in negative cases when the query context is similar.
    • Since there is no diagnosis stage, directly choose one safe and executable optimization action.
    • Prioritize the single most impactful executable fix for the current query.

    ## Constraints
    • Tuning_Action must be executable directly in {db_info.get('db_type')}.
    • The action should be safe and should not drop or delete data unless strictly necessary.
    • Ensure all proposed actions are syntactically correct and executable.
    • The naming convention for newly created indexes must follow the pattern: (table_name)_(col1)_(col2)_idx.
    • You MUST adjust database configuration parameters using "SET knob = value" and focus on parameters that can improve query performance.

    ## Output Format
    You must output ONLY in the following format:

    Explanation:
    <concise technical reason for the performance issue and proposed solution>

    Tuning_Action:
    <your fix SQL or optimization action here>

    Query_Rewrite:
    <yes or no>

    ## Output Requirements
    • "Explanation" must be short, precise, and technical.
    • Tuning_Action must contain only executable tuning statements.
    • Set Query_Rewrite to yes if the selected action involves query rewriting, and place the full rewritten SQL in Tuning_Action; otherwise, set Query_Rewrite to no.
    • If multiple executable statements are needed, place all of them in Tuning_Action in executable form."""
