"""
Prompt generation functions for database optimization system.
This module contains all prompt generation functions extracted from action_manager.py and planner.py.
"""

from agent.contracts import (
    CANONICAL_COMPONENTS,
    COMPONENT_ACTION_SPACE_NAMES,
    format_component_label,
    normalize_component_confidence_map,
    normalize_components,
    normalize_tuning_history,
)


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
            f"\n    Improve: {case.get('approve_time')}"
        )
    return "\n".join(lines)


def build_action_prompt(root_cause, base_info, action_space, mode, positives, negatives):
    """
    Format a prompt that feeds the LLM with all necessary context and enforces
    a consistent Appendix A Tuning Agent output contract.
    """
    query_info = base_info.get("query_info")
    db_info = base_info.get("database_info", {})
    components = normalize_components(root_cause)
    action_space = (action_space or "None")

    positives_text = fmt_cases(positives)
    negatives_text = fmt_cases(negatives)

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

    return f"""You are an SQL optimization expert. The bottleneck-related tuning components have already been localized by the Detection Agent. Your job is to recommend executable tuning action(s) only within the localized tuning subspace, using the current query context, retrieved historical observations with experiences, and the refined action space.
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
    6. Reflective Reasoning Guidance
    Before proposing your optimization solution, analyze the historical observations and experiences to understand why certain approaches succeeded or failed. Adapt successful patterns to the current context while avoiding patterns that failed in similar situations.
    7. Refined Tuning Subspace
    {action_space}
    ## Constraints
    • Tuning_Action must be executable directly in {db_info.get('db_type')}.
    • The Tuning_Action must only operate on the component(s) listed in {components}.
    • The Tuning_Action should be safe and should not drop or delete data unless explicitly required by the predicted components.
    ## Output Format
    You must output ONLY in the following format:
    Explanation:
    <concise technical reason why the predicted components are causing the slowdown now>
    Tuning_Action:
    <your tuning action here, do not include any other text, if no tuning action is needed, output an empty string>
    Query_Rewrite:
    <yes or no>
    ## Output Requirements
    • "Explanation" must be short, precise, and technical.
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


def build_root_cause_diagnosis_prompt(query_info, root_causes, state_confidence, tuning_history=None, database_info=None, timeout=300):
    """Build prompt for LLM-based component diagnosis with per-label confidence map."""
    
    root_causes = normalize_components(root_causes)
    state_confidence = normalize_component_confidence_map(state_confidence)
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

    return f"""You are a database performance diagnosis expert. Your task is to localize the most promising tuning component space(s) for the following slow SQL query. Rather than diagnosing generic performance causes from scratch, you should refine the lightweight model's component scores using the current query context, trajectory-level tuning feedback, and DBA-style knowledge to identify a high-impact set of target tuning components.
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
    3. Lightweight Model Scores
    These scores are produced by data-driven model and should be treated as the anchor for reasoning.
    • suggested tuning spaces: {root_causes}
    • confidence scores (0-1): {state_confidence}
    ## Reasoning Guidance
    1. Historical Tuning Summary
    {history_text}
    2. Trajectory-Guided Confidence Refinement
    Refine the component scores according to the following principles:
    • If success_count > 0 and fail_count = 0: increase confidence since past interventions validate it as an actionable bottleneck.
    • If success_count = 0 and fail_count > 0: decrease confidence and encourage exploration of alternative components, since previous interventions have not led to improvement.
    • If success_count > 0 and fail_count > 0: retain the confidence and make conservative adjustment by jointly considering the data-driven score and knowledge-driven reasoning.
    3. DBA Knowledge-Guided Reasoning
    Use DBA-style reasoning over the query context to validate or revise the component scores.
    • Index bottlenecks: Index bottlenecks are characterized by large-range scans on selective predicates. Typical evidence includes a large gap between scanned and returned rows, predicates that cannot be effectively exploited by indexes, and repeated heap/table access due to non-covering indexes.
    • Plan suboptimality: A plan bottleneck is usually reflected by expensive operator choices that force unnecessary sorting, temporary materialization, or inefficient joins. Typical evidence includes temporary-table creation and extra sorting due to unordered access paths.
    • Knob misconfiguration: Knob tuning becomes important when analytical operators spill to disk or lack sufficient memory or parallel resources. Typical evidence includes disk-backed temporary tables, underutilized parallel workers, and high-cost join or aggregation execution under limited resources.
    • Query formulation inefficiency: Query rewriting is prioritized when the SQL structure itself introduces unnecessary work. Typical evidence includes join patterns missing valid ON conditions or using always-true predicates, deep pagination with large OFFSET, UNION without ALL, overlong IN lists, and join patterns with Cartesian-product risk.
    4. Confidence Rules
    • Restrict each component's confidence adjustment to [0.1, 0.3].
    • If a tuning component shows no improvement after multiple rounds, reduce its confidence below 0.5 and increase the confidence of a new candidate component slightly above 0.5.
    • Confidence values should reflect both historical tuning feedback and the current query context.
    ## Constraints (STRICT)
    • You MUST output ALL allowed component labels with a confidence score in [0,1]: {", ".join(CANONICAL_COMPONENTS)}.
    • You MUST also output a separate array predicted_components listing the final selected tuning components.
    • predicted_components MUST be exactly the set of labels with confidence > 0.5. Labels with confidence <= 0.5 MUST NOT appear in predicted_components.
    • You MUST output strict JSON only in the format below. No extra text and no explanation.
    Output JSON Format:
    {{
      "predicted_components": ["index"],
      "components": [
        {{"label": "index", "confidence": 0.72}},
        {{"label": "knob", "confidence": 0.18}},
        {{"label": "hint", "confidence": 0.10}},
        {{"label": "query", "confidence": 0.05}}
      ]
    }}"""


def build_action_space_prompt(root_causes, db, knob_config, current_knob_values, query_info=None):
    """Build action space prompt based on root causes."""
    # Root causes
    root_causes = normalize_components(root_causes)

    # Build action space prompt based on root causes
    action_space_prompt = ""

    # Index related
    if "index" in root_causes:
        current_indexes = db.get_indexes() if db else []
        action_space_prompt += f"""• Index Space: 
    - The naming convention for newly created indexes must follow the pattern: (table_name)_(col1)_(col2)_idx.
    - Index creation should be strictly guided by the slow_sql queries.
    - Consider the query execution plan to identify beneficial indexes that could improve performance.
    - Avoid generating duplicate or subsumed indexes, and do not drop primary keys, unique indexes, or any indexes that enforce constraints.
    - Do not conflict with the existing indexes in the database: {current_indexes}
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

        action_space_prompt += f"""• Knob Space:  
    - Knob Space is strictly limited to the following knobs, you must only use these knobs in your Tuning_Action, and any knob not explicitly listed here must not be used: 
    {knob_lines}
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

        action_space_prompt += f"""• Hint Space: 
    - Hint Space is strictly limited to the following knobs, you must only use these knobs in your Tuning_Action, and any knob not explicitly listed here must not be used: 
    {knob_lines}
    - **IMPORTANT**: When using Hint Space, output ONLY the hint block. Do NOT include the original SQL. Example: /*+ Set(enable_hashjoin on) Set(enable_sort on) */"""

    # SQL readability/rewrite related
    if "query" in root_causes:
        from agent.action.rule_matcher import SQLRuleMatcher

        matcher = SQLRuleMatcher()
        rule_descriptions = matcher.get_rule_descriptions(query_info.query)

        action_space_prompt += f"""• Query Rewrite Space: 
    - Query Rewrite need to generate a semantically equivalent SELECT statement that produces exactly the same result set as the original query while improving performance.
    - **Analysis-Based Optimization**: The following rewrite rules have been identified as applicable based on the query structure analysis:
    {rule_descriptions}
    - **Implementation Guidelines**:
      * Apply only the rules that match the specific query structure and are expected to improve performance
      * Ensure the rewritten query maintains identical result set and semantic equivalence
      * Prioritize rules that reduce computational complexity, eliminate redundant operations, or optimize data access patterns
      * Test the rewritten query to verify it produces the same results as the original
    - **Quality Assurance**: The final output must be a complete, executable SQL statement that preserves all original query semantics while demonstrating measurable performance improvements.
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
    3. Output exactly three labeled sections: Explanation, Tuning_Action and Query_Rewrite.
    4. Always make the Tuning_Action executable directly in the target database.
    5. Never include extra text, comments, or sections beyond the required format.
    6. Always apply the given MCP APIs when producing fixes, if applicable.
    7. Assume the provided execution plan, metrics, and environment info are accurate.
    8. Keep the Explanation concise and technical, avoiding generic descriptions."""


def get_root_cause_llm_instructions():
    """Get instructions for the root cause diagnosis LLM."""
    return """You are a database slow SQL bottleneck component diagnosis expert.
    Only output strict JSON using the fields predicted_components and components, nothing else."""


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
