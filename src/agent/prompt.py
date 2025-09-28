"""
Prompt generation functions for database optimization system.
This module contains all prompt generation functions extracted from action_manager.py and planner.py.
"""

from typing import List, Dict, Any

def fmt_cases(cases):
    lines = []
    for i, case in enumerate(cases, 1):
        lines.append(f"\n    Case {i}:\n    SQL: {case.get('query_info').get('query')}\n    Execution Plan: {case.get('query_info').get('plan_json')}\n    Origin Execution Runtime: {case.get('query_info').get('execution_time')}s\n    Root Causes: {case.get('root_causes')}\n    Fix_Actions: {case.get('fix_action')}\n    Improve: {case.get('approve_time')}")
    return "\n".join(lines)
    # try:
    #     return "\n".join([
    #         f"• SQL: {c.get('query_info').get('query')}\n Execution Plan: {c.get('query_info').get('plan_json')}\n Execution Runtime: {c.get('query_info').get('execution_time')}s\n RootCauses: {c.get('root_causes')}\n  Fix_Action: {c.get('fix_action')}\n  Improve: {c.get('approve_time')}s" for c in cases
    #     ])
    # except Exception:
    #     return str(cases)

def build_action_prompt(root_cause, base_info, action_space, mode, positives, negatives):
    """
    Format a prompt that feeds the LLM with all necessary context and enforces
    a consistent two-part output: Explanation + SQL Fix.
    """
    query_info = base_info.get('query_info')
    db_info = base_info.get('database_info', {})

    positives_text = fmt_cases(positives)
    negatives_text = fmt_cases(negatives)

    # 模式提示
    if mode == 'exploit':
        mode_instruction = """You are now in **EXPLOIT mode**. Refer to the following Positive Cases and Negative Cases, which are derived from historical fixes under the same root cause. 
    - Positive Cases represent effective fixes. Treat their Fix_Actions as **examples of optimization patterns**, You must **not copy them directly or merely reorder them**. Instead, propose **new fix_actions** that explore additional possibilities. 
    - Negative Cases represent ineffective fixes. Reflect on which actions led to no improvement or performance degradation, and avoid adopting these fix_actions in your solution.
        """
    else:
        mode_instruction = """You are now in **EXPLORE mode**. Refer to the following Positive Cases and Negative Cases, which are derived from historical fixes under different root causes.
    - Positive Cases represent effective fixes. Treat their Fix_Actions as **examples of optimization patterns**, You must **not copy them directly or merely reorder them**. Instead, propose **new fix_actions** that explore additional possibilities under the new root cause(s).
    - Negative Cases represent ineffective fixes. Reflect on which actions led to no improvement or performance degradation under other root causes, and avoid adopting these fix_actions in your solution.
        """

    return f"""As an SQL optimization expert, your task is to diagnose and fix the following slow SQL query.
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
    • Database Schema: TPC-H
    • CPU: {query_info.external_metrics[0]}
    • Read I/O: {query_info.external_metrics[1]}
    • Write I/O: {query_info.external_metrics[2]}
    • Virtual Memory: {query_info.external_metrics[3]}
    • Physical Memory: {query_info.external_metrics[4]}
    • Net Received: {query_info.external_metrics[5]}
    • Net Sent: {query_info.external_metrics[6]}

    3. **Identified Root Cause**: {root_cause}

    ## Mode Instruction
    {mode_instruction}
    • Positive Cases (effective references): {positives_text}
    • Negative Cases (ineffective references to avoid): {negatives_text}

    ## Action Space
    - If the root cause is "inappropriate query knobs", you MUST perform actions within Knob Space using SET knob = value.
    - If the root cause is "suboptimal plan optimizer", you MUST perform actions within Plan Optimizer Space using PostgreSQL comment-style hints ONLY. Output ONLY the hint block like: /*+ Set(knob_name value) Set(knob_name value) ... */. Do NOT include the SQL query.
    - If the root cause is "missing indexes", you MUST perform actions within Index Space using CREATE INDEX IF NOT EXISTS or DROP INDEX IF EXISTS.
    - If the root cause is "poorly written queries", you MUST perform actions within Query Rewrite Space using rewrite SQL.
    - **If multiple root causes are identified, you MUST consider all FIX_ACTION outputs jointly and determine the optimal solution that addresses the combined root causes, while strictly limiting actions to the relevant action spaces of the identified root causes. Do not perform any actions outside of these specified root causes.**
    - The Action Space defines the operations that allowed to perform, you MUST strictly limit your Fix_Action to these spaces. Do not adjust or introduce any actions beyond what is explicitly defined in the following Action Space.
    {action_space}

    ## Constraints
    - Fix action must be executable directly in {db_info.get('db_type')}.
    - Please ensure that the proposed fix_action strictly addresses the Identified Root Cause and does not deviate from the Action Space.
    - The fix action should be safe and should not drop or delete data unless explicitly required by the root cause.

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
    - Do not output any extra sections, comments, or steps."""

def build_fix_prompt(fix_action, err):
    """Build prompt for fixing non-executable SQL statements."""
    return f"""You have been given a non-executable SQL statement and the exact database error:
    • Non-executable SQL: {fix_action}
    • Database Error Message: {err}
    Your sole objective is to produce a corrected, fully-formed SQL statement (or an ordered sequence of SQL statements) that will run without error.
    Strict rules — follow exactly:
    1) OUTPUT NOTHING BUT THE SQL STATEMENT(S). No explanations, no comments, no markdown, no surrounding quotes.
    2) If a single statement suffices, output exactly that statement, terminated by a single semicolon and a newline.
    3) If multiple statements are required, output them in execution order, each terminated by a semicolon and a newline.
    -- Output (strict) --
    <paste only the corrected fix action statement(s) here, exactly as SQL, nothing else>"""

def build_rewrite_prompt(origin_sql, rewrite_sql):
    """Build prompt for rewriting SQL to match reference output."""
    return f"""You are given two SQL statements:
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
    <paste only the revised SQL here, exactly as SQL, nothing else>"""

def build_root_cause_diagnosis_prompt(query_info, root_causes, positives, negatives, state_confidence, tuning_history=None, timeout=20):
    """Build prompt for LLM-based root cause diagnosis with per-label confidence map.

    tuning_history: Optional list summarizing per-root-cause success/failure counts, e.g.,
      [
        {"label": "missing indexes", "success_count": 2, "fail_count": 3},
        {"label": "inappropriate query knobs", "success_count": 1, "fail_count": 0}
      ]
    """

    # positives_text = fmt_cases(positives)
    # negatives_text = fmt_cases(negatives)
    
    # 处理 timeout 情况
    execution_time_text = f"{query_info.execution_time} s"
    if query_info.execution_time == timeout:
        execution_time_text = f"{query_info.execution_time} (timeout)"

    # 格式化调优历史（成功/失败计数）
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
        
    return f"""As a database performance diagnosis expert, your task is to identify the **most likely root cause(s) and diagnosis confidence** of the following slow SQL query.

    ## Context
    1. Query Information:
    • SQL: {query_info.query}
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

    2. Plan-Aware Exploration:
    - Analyze execution plan and runtime metrics to guide confidence adjustments:
     * missing indexes: If the execution plan shows large sequential scans on big tables with selective filters (e.g., WHERE clauses on non-indexed columns), and CPU is high while I/O is moderate, increase confidence for 'missing indexes'.
     * suboptimal plan optimizer: If the execution plan shows complex joins, nested loops, or inefficient aggregation strategies (e.g., hashed vs. sorted aggregation), especially when the estimated rows differ significantly from actual rows, increase confidence for 'suboptimal plan optimizer'.
     * inappropriate query knobs: High CPU with low I/O, or repeated parallel workers not utilized efficiently, increase confidence for 'inappropriate query knobs'.
     * poorly written queries: Large aggregations, complex expressions, unnecessary computations in SELECT/GROUP BY clauses, increase confidence for 'poorly written queries'.

    3. Initial Suggestion:
    - Initial model suggested root causes: {root_causes}
    - Initial model confidence (0-1): {state_confidence}
    - Historical Tuning Summary (per root cause): {history_text}

    4. Root Cause Diagnosis Guidance:
    Use the **Query Information** and the **Initial Suggestion** to infer the root causes and their confidence, as follows:
    - If a label has success_count = 0 and fail_count > 0, it is likely incorrect. Gradually reduce its confidence below 0.5, while gradually increasing the confidence of other root causes supported by Query Information until they exceed 0.5. If the label remains low after repeated failures, fully shift confidence to other unexplored candidates root causes.
    - If a label has success_count > 0 and fail_count = 0, it indicates consistent potential. CONTINUE exploring this root cause. Gradually increase its confidence as evidence accumulates.  
    - If a label has success_count > 0 and fail_count > 0, it indicates partial benefit but decreasing returns. CONTINUE using this root cause as a reference, and increase the other root causes confidence for exploraiton. Adjust confidence downward if further attempts do not yield improvements.  

    5. Confidence rules Guidance:
    - Confidence adjustment of each root cause **MUST** be restricted to the range of [0.1,0.3].  
    - If a root cause shows no improvement after multiple rounds, reduce its confidence below 0.5 to retire it from active diagnosis. In such cases, select a new candidate root cause for exploration, initializing its confidence slightly above 0.5. 
    - Confidence values should reflect both historical results and current judgment based on Query Information, Ensure the model makes independent judgments about which root causes to explore next, rather than blindly following previous confidence values.  
 
    ## Constraints (STRICT)
    - You MUST output ALL allowed labels WITH a confidence in [0,1]: missing indexes, inappropriate query knobs, suboptimal plan optimizer, poorly written queries
    - You MUST also output a separate array `predicted_root_causes` listing the final predicted root cause labels (derived from confidence and considering historical priors above).
    - `predicted_root_causes` MUST be exactly the set of labels with confidence > 0.5. Labels with confidence ≤ 0.5 MUST NOT appear in `predicted_root_causes`.
    - You MUST output strict JSON only in the format below. No extra text, no explanation.

    ## Output JSON Example
    {{
      "predicted_root_causes": ["missing indexes"],
      "root_causes": [
        {{"label": "missing indexes", "confidence": 0.72}},
        {{"label": "inappropriate query knobs", "confidence": 0.18}},
        {{"label": "suboptimal plan optimizer", "confidence": 0.10}},
        {{"label": "poorly written queries", "confidence": 0.05}}
      ]
    }}"""


def build_action_space_prompt(root_causes, db=None):
    """Build action space prompt based on root causes."""
    # 根因
    root_causes = root_causes if isinstance(root_causes, list) else [root_causes]

    # 根据根因构建操作空间prompt
    action_space_prompt = ""
    
    # 索引相关
    if "missing indexes" in root_causes:
        action_space_prompt += f"""
    • Index Space: 
    - The naming convention for newly created indexes must follow the pattern: (table_name)_(col1)_(col2)_idx.
    - Index creation should be strictly guided by the slow_sql queries.
    - Consider the query execution plan to identify missing indexes that could improve performance.
    - Avoid generating duplicate or subsumed indexes, and do not drop primary keys, unique indexes, or any indexes that enforce constraints.
    - Do not conflict with the existing indexes in the database: {db.get_indexes()}
    """
    
    # Knob相关
    if "inappropriate query knobs" in root_causes:
        action_space_prompt += """
    • Knob Space:  
    - Knob Space is strictly limited to the following knobs, you must only use these knobs in your Fix_Action, and any knob not explicitly listed here must not be used: 
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
    """
    
    # 执行计划/优化器相关
    if "suboptimal plan optimizer" in root_causes:
        action_space_prompt += """
    • Plan Optimizer Space: 
    - Plan Optimizer Space is strictly limited to the following knobs, you must only use these knobs in your Fix_Action, and any knob not explicitly listed here must not be used: 
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
    - **IMPORTANT**: When using Plan Optimizer Space, output ONLY the hint block. Do NOT include the original SQL. Example: /*+ Set(enable_hashjoin on) Set(enable_sort on) */"""

    # SQL 可读性/重写相关
    if "poorly written queries" in root_causes:
        action_space_prompt += """
    • Query Rewrite Space: 
    - The output must always be a semantically equivalent SELECT statement that produces exactly the same result set as the original query.
    - The following rules are provided in the format ["rule name": "rule description"], consider the rules when rewriting the SQL:
    "name": "SORT_UNION_TRANSPOSE_MATCH_NULL_FETCH", "rewrite_rules_structured": ["conditions": "1. A SQL query performs a `UNION ALL` operation on multiple select statements. 2. Following the `UNION ALL`, there's an `ORDER BY` clause intending to sort the results, with no `OFFSET` clause present in the query.3. Each select statement involved in the `UNION ALL` could potentially benefit from pre-application of the order defined by the `ORDER BY` clause, without violating the intended sort order across the combined result set.", "transformations": "1. Identify the `ORDER BY` clause following a `UNION ALL` that combines multiple select statements and verify there's no associated `OFFSET` directive. 2. For each select statement involved in the `UNION ALL`:   - Apply the `ORDER BY` clause directly to the select statement if doing so does not change the intended order of the final result set. This effectively means prepending an `ORDER BY` to each select statement based on the `ORDER BY` clause initially following the `UNION ALL` operation. 3. Retain the original `ORDER BY` clause following the `UNION ALL`. This ensures that the final result set is ordered as intended, taking into account that individual select statements are now pre-sorted. 4. Implement this transformation only if it's determined that sorting individual select statements before the `UNION ALL` would maintain the overall desired sort order and is expected to either improve performance by reducing sorting overhead or have no negative impact."]
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
    2. Identify how the root cause leads to performance degradation.
    3. Output exactly three labeled sections: Explanation, Fix_Action and Query_Rewrite.
    4. Always make the Fix_Action executable directly in the target database.
    5. Never include extra text, comments, or sections beyond the required format.
    6. Always apply the given MCP APIs when producing fixes, if applicable.
    7. Assume the provided execution plan, metrics, and environment info are accurate.
    8. Keep the Explanation concise and technical, avoiding generic descriptions."""


def get_root_cause_llm_instructions():
    """Get instructions for the root cause diagnosis LLM."""
    return """You are a database slow SQL root cause diagnosis expert.
    Only output the allowed root cause label(s) and their confidence in json format, nothing else."""
