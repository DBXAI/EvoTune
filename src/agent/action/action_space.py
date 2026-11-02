import re
from collections import defaultdict

from agent.action.action_utils import extract_index_names
from agent.action.rule_matcher import SQLRuleMatcher
from agent.contracts import normalize_components
from agent.prompt import build_action_space_prompt


DEFAULT_IMPROVEMENT_THRESHOLD = 0.0
DEFAULT_REFINEMENT_MIN_CASES = 10


async def action_space_collect(
    root_causes,
    query_info=None,
    *,
    enable_action_define,
    enable_action_pruning,
    configs,
    db,
    memory_manager,
):
    root_causes = normalize_components(root_causes)

    if enable_action_define:
        knob_config = configs.get("KNOB_CONFIG")
        current_knob_values = db.get_current_knob_values(knob_config)
        base_action_space = build_action_space_prompt(root_causes, db, knob_config, current_knob_values, query_info)
    else:
        base_action_space = _build_basic_action_space(root_causes)

    if enable_action_pruning:
        pruning_guidance = action_pruning(root_causes, query_info, memory_manager)
        refinement_guidance = action_refinement(root_causes, query_info, memory_manager)
        enhanced_action_space = _integrate_pruning_and_refinement(
            base_action_space, pruning_guidance, refinement_guidance
        )
    else:
        enhanced_action_space = base_action_space

    return enhanced_action_space


def action_pruning(root_causes, query_info, memory_manager):
    root_causes = normalize_components(root_causes)
    pruning_guidance = []

    if "index" in root_causes:
        failed_index_actions = _get_failed_index_actions(query_info, memory_manager)
        if failed_index_actions:
            pruning_guidance.append(
                "- Index Pruning: Based on historical failure cases with the same tuning component, "
                "the following indexes have been proven ineffective, please avoid using them:"
            )
            for index_name in sorted(failed_index_actions):
                pruning_guidance.append(f"  {index_name}")

    if "query" in root_causes:
        failed_rules = _get_applicable_rewrite_rules(query_info, memory_manager)
        if failed_rules:
            pruning_guidance.append(
                "- Query Pruning: Based on historical failure cases with the same tuning component, "
                "the following query rewrite rules have been proven ineffective, please avoid using them:"
            )
            for rule in failed_rules:
                pruning_guidance.append(f"  {rule['name']}: {rule.get('conditions', '')}")

    return "\n".join(pruning_guidance)


def _get_failed_index_actions(query_info, memory_manager):
    threshold = _get_improvement_threshold(memory_manager)
    cases = _component_cases(memory_manager, "index")
    same_query_cases = _same_query_cases(cases, query_info)
    if same_query_cases:
        cases = same_query_cases

    index_improvements = defaultdict(list)
    for case in cases:
        fix_action = case.case_info.get("fix_action")
        if not fix_action:
            continue
        for index_name in extract_index_names(str(fix_action)):
            index_improvements[index_name].append(_case_improvement(case))

    failed_index_actions = set()
    for index_name, improvements in index_improvements.items():
        if improvements and _average(improvements) <= threshold:
            failed_index_actions.add(index_name)

    return failed_index_actions


def _get_applicable_rewrite_rules(query_info, memory_manager):
    if query_info is None:
        return []

    threshold = _get_improvement_threshold(memory_manager)
    cases = _same_query_cases(_component_cases(memory_manager, "query"), query_info)
    if not cases:
        return []

    rule_matcher = SQLRuleMatcher()
    original_query = getattr(query_info, "query", "") or ""
    applicable_rules = rule_matcher.match_rules(original_query)
    if not applicable_rules:
        return []

    rule_improvements = defaultdict(list)
    for case in cases:
        rewritten_query = _case_query_info(case).get("query") or case.case_info.get("fix_action") or ""
        if not rewritten_query:
            continue
        for rule in applicable_rules:
            if rule_matcher._rule_used_in_rewrite(rule, original_query, str(rewritten_query)):
                rule_improvements[rule["name"]].append(_case_improvement(case))

    failed_rule_info = []
    for rule_name, improvements in rule_improvements.items():
        if improvements and _average(improvements) <= threshold:
            rule_info = rule_matcher.get_rule_info(rule_name)
            if rule_info:
                failed_rule_info.append(rule_info)

    failed_rule_info.sort(key=lambda rule: rule["name"])
    return failed_rule_info


def action_refinement(root_causes, query_info, memory_manager):
    root_causes = normalize_components(root_causes)
    refinement_guidance = []

    if "knob" in root_causes:
        knob_guidance = _get_knob_refinement_guidance(query_info, memory_manager)
        if knob_guidance:
            refinement_guidance.append(knob_guidance)

    if "hint" in root_causes:
        hint_guidance = _get_hint_refinement_guidance(query_info, memory_manager)
        if hint_guidance:
            refinement_guidance.append(hint_guidance)

    return "\n".join(refinement_guidance)


def _get_knob_refinement_guidance(query_info, memory_manager, min_threshold=None):
    threshold = _get_improvement_threshold(memory_manager)
    min_cases = min_threshold or _get_refinement_min_cases(memory_manager)
    cases = _component_cases(memory_manager, "knob")
    similar_cases = _retrieve_similar_component_cases(query_info, memory_manager, "knob")
    if similar_cases:
        cases = similar_cases

    positive_cases = [case for case in cases if _case_improvement(case) > threshold]
    if len(positive_cases) < min_cases:
        return ""

    knob_data = _extract_knob_data(positive_cases)
    if not knob_data:
        return ""

    guidance = [
        "- Knob Refinement: Based on historical success cases with high-quality configurations, "
        "the following knob domains are recommended for focused exploration:"
    ]

    for knob_name in sorted(knob_data.keys()):
        values = knob_data[knob_name]["values"]
        improvements = knob_data[knob_name]["performance"]
        if len(values) < min_cases or _average(improvements) <= threshold:
            continue

        domain = _format_value_domain(values)
        guidance.append(
            f"  <KNOB_NAME={knob_name}, RANGE={domain}, NUM_CASES={len(values)}, "
            f"AVG_IMPROVEMENT={_average(improvements):.3f}>"
        )

    if len(guidance) <= 1:
        return ""
    return "\n".join(guidance)


def _get_hint_refinement_guidance(query_info, memory_manager):
    threshold = _get_improvement_threshold(memory_manager)
    cases = _component_cases(memory_manager, "hint")
    same_query_cases = _same_query_cases(cases, query_info)
    if same_query_cases:
        cases = same_query_cases

    hint_data = _extract_hint_usage_data(cases)
    if not hint_data:
        return ""

    positive_hints = _select_positive_hints(hint_data, threshold)
    if not positive_hints:
        return ""

    guidance = [
        "- Hint Refinement: Based on historical success cases with the same tuning component, "
        "the following optimizer hints are recommended as positive hint choices:"
    ]
    for hint_key, hint_stats in positive_hints:
        guidance.append(
            f"  <POSITIVE_HINT={hint_key}, AVG_IMPROVEMENT={hint_stats['avg_improvement']:.3f}, "
            f"NUM_CASES={hint_stats['usage_count']}>"
        )

    return "\n".join(guidance)


def _component_cases(memory_manager, component):
    cases = list(getattr(memory_manager, "cases", {}).values())
    if cases:
        return [case for case in cases if component in normalize_components(case.case_info.get("root_causes"))]

    if hasattr(memory_manager, "_normalize_root_causes") and hasattr(memory_manager, "ids_to_cases"):
        target_root = memory_manager._normalize_root_causes([component])
        target_island_ids = list(getattr(memory_manager, "islands", {}).get(target_root, set()))
        return memory_manager.ids_to_cases(target_island_ids)

    return []


def _retrieve_similar_component_cases(query_info, memory_manager, component):
    if query_info is None or not hasattr(memory_manager, "retrieve_embedding"):
        return []
    try:
        cases = memory_manager.retrieve_embedding(query_info, top_n=50)
    except Exception:
        return []
    return [case for case in cases if component in normalize_components(case.case_info.get("root_causes"))]


def _same_query_cases(cases, query_info):
    query_id = getattr(query_info, "query_id", None)
    if query_id is None:
        return []
    return [case for case in cases if str(_case_query_info(case).get("query_id")) == str(query_id)]


def _case_query_info(case):
    query_info = case.case_info.get("query_info") or {}
    if isinstance(query_info, dict):
        return query_info
    return getattr(query_info, "__dict__", {}) or {}


def _case_improvement(case):
    value = case.case_info.get("approve_time")
    if value is None:
        value = getattr(case, "metrics", {}).get("approve_time", 0.0)
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _get_improvement_threshold(memory_manager):
    config = getattr(memory_manager, "agent_config", {}) or {}
    value = config.get("refinement_improvement_threshold", config.get("pruning_improvement_threshold"))
    if value is None:
        value = getattr(memory_manager, "improvement_threshold", DEFAULT_IMPROVEMENT_THRESHOLD)
    try:
        return float(value)
    except (TypeError, ValueError):
        return DEFAULT_IMPROVEMENT_THRESHOLD


def _get_refinement_min_cases(memory_manager):
    config = getattr(memory_manager, "agent_config", {}) or {}
    value = config.get("refinement_min_cases", DEFAULT_REFINEMENT_MIN_CASES)
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return DEFAULT_REFINEMENT_MIN_CASES


def _average(values):
    return sum(values) / len(values) if values else 0.0


def _format_value_domain(values):
    numeric_values = []
    for value in values:
        try:
            numeric_values.append(float(str(value).strip("'\"")))
        except (TypeError, ValueError):
            numeric_values = []
            break

    if numeric_values and len(numeric_values) == len(values):
        return f"[{_format_number(min(numeric_values))}, {_format_number(max(numeric_values))}]"

    unique_values = sorted({str(value) for value in values})
    return "{" + ", ".join(unique_values) + "}"


def _format_number(value):
    return str(int(value)) if float(value).is_integer() else f"{value:g}"


def _select_positive_hints(hint_data, threshold):
    by_knob = defaultdict(list)
    for hint_key, hint_stats in hint_data.items():
        by_knob[hint_stats["knob"]].append((hint_key, hint_stats))

    selected = []
    for hints in by_knob.values():
        hints.sort(key=lambda item: item[1]["avg_improvement"], reverse=True)
        if len(hints) == 1:
            if hints[0][1]["avg_improvement"] > threshold:
                selected.append(hints[0])
            continue

        best_avg = hints[0][1]["avg_improvement"]
        worst_avg = hints[-1][1]["avg_improvement"]
        if best_avg - worst_avg > threshold:
            if best_avg > threshold:
                selected.append(hints[0])
        else:
            selected.extend([item for item in hints if item[1]["avg_improvement"] > threshold])

    selected.sort(key=lambda item: item[1]["avg_improvement"], reverse=True)
    return selected


def _extract_knob_data(cases):
    knob_data = {}

    for case in cases:
        fix_action = case.case_info.get("fix_action")
        knob_settings = _extract_knob_settings(str(fix_action))
        approve_time = _case_improvement(case)

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


def _extract_hint_usage_data(cases):
    hint_data = {}

    for case in cases:
        query = _case_query_info(case).get("query") or ""
        fix_action = case.case_info.get("fix_action") or ""
        action_text = f"{fix_action}\n{query}"

        hints = _extract_hints_from_action(str(action_text))
        if not hints:
            continue

        performance_improvement = _case_improvement(case)

        for hint in hints:
            hint_key = hint["full_hint"]
            if hint_key not in hint_data:
                hint_data[hint_key] = {
                    "knob": hint["knob"],
                    "value": hint["value"],
                    "improvements": [],
                    "usage_count": 0,
                    "avg_improvement": 0.0,
                }

            hint_data[hint_key]["improvements"].append(performance_improvement)
            hint_data[hint_key]["usage_count"] += 1
            hint_data[hint_key]["avg_improvement"] = _average(hint_data[hint_key]["improvements"])

    return hint_data


def _extract_knob_settings(action_str):
    knob_settings = {}

    set_pattern = r"set\s+([a-zA-Z0-9_]+)\s*=\s*([^\s;]+)"
    matches = re.findall(set_pattern, action_str, re.IGNORECASE)

    for knob_name, knob_value in matches:
        knob_settings[knob_name.lower()] = knob_value.strip().strip("'").strip('"')

    return knob_settings


def _extract_hints_from_action(action_str):
    hints = []

    hint_pattern = r"/\*\+([^*]+)\*/"
    matches = re.findall(hint_pattern, action_str)

    for match in matches:
        set_pattern = r"Set\(([^)]+)\)"
        set_matches = re.findall(set_pattern, match, re.IGNORECASE)

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

    if "query" in root_causes:
        action_space_prompt += """
    • Query Rewrite Space: 
    - Generate a semantically equivalent SELECT statement that produces exactly the same result set as the original query.
    - Apply only rewrite rules that are applicable to the current query structure.
    - Prioritize rules that reduce computational complexity, eliminate redundant operations, or optimize data access patterns.
    - The final output must be a complete, executable SQL statement that preserves all original query semantics while demonstrating measurable performance improvements.
    """

    return action_space_prompt


def _integrate_pruning_and_refinement(base_action_space, pruning_guidance, refinement_guidance):
    enhanced_space = base_action_space

    if pruning_guidance:
        enhanced_space += f"\n{pruning_guidance}"

    if refinement_guidance:
        enhanced_space += f"\n{refinement_guidance}"

    return enhanced_space
