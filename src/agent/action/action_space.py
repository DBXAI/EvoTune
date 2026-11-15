from agent.contracts import normalize_components
from agent.prompt import build_action_space_prompt


async def action_space_collect(
    root_causes,
    query_info=None,
    *,
    enable_action_define,
    configs,
    db,
):
    root_causes = normalize_components(root_causes)
    if enable_action_define:
        knob_config = configs.get("KNOB_CONFIG")
        current_knob_values = db.get_current_knob_values(knob_config)
        return build_action_space_prompt(
            root_causes, db, knob_config, current_knob_values, query_info
        )
    return _build_basic_action_space(root_causes)


def _build_basic_action_space(root_causes):
    root_causes = normalize_components(root_causes)
    action_space_prompt = ""

    if "index" in root_causes:
        action_space_prompt += """
    • Index Space: 
    - Use CREATE INDEX IF NOT EXISTS or DROP INDEX IF EXISTS to create or drop indexes.
    - Name newly created indexes using the pattern <table_name>_<col>_idx, extending it naturally for composite indexes.
    - Derive query-specific candidate indexes from the database schema and execution plan; consider composite indexes when supported by predicates, joins, grouping, or ordering.
    - Avoid duplicate, subsumed, existing, primary-key, unique, or constraint indexes.
    """

    if "knob" in root_causes:
        action_space_prompt += """
    • Knob Space:  
    - You can adjust database configuration parameters using SET knob = value.
    - Focus on parameters that can improve query performance.
    - Be conservative with changes to avoid system instability.
    - Use SET statement to modify database knobs.
    """

    if "hint" in root_causes:
        action_space_prompt += """
    • Hint Space: 
    - Use PostgreSQL comment-style hints to guide query execution.
    - Output ONLY the hint block like: /*+ Set(knob_name value) Set(knob_name value) ... */
    - Do NOT include the SQL query when using hints.
    - Use hints to force specific join methods, scan types, etc.
    """

    if "rewrite" in root_causes:
        action_space_prompt += """
    • Query Rewrite Space: 
    - Generate a semantically equivalent SELECT statement that produces exactly the same result set as the original query.
    - Apply only rewrite rules that are applicable to the current query structure.
    - Prioritize rules that reduce computational complexity, eliminate redundant operations, or optimize data access patterns.
    - The final output must be a complete, executable SQL statement that preserves all original query semantics while demonstrating measurable performance improvements.
    """

    return action_space_prompt
