import json
import re


CANONICAL_COMPONENTS = ("index", "knob", "hint", "query")
SMALL_MODEL_COMPONENTS = ("index", "hint", "knob", "query")

LEGACY_COMPONENT_LABELS = {
    "index": "index",
    "missing indexes": "index",
    "missing index": "index",
    "knob": "knob",
    "inappropriate query knobs": "knob",
    "inappropriate query knob": "knob",
    "hint": "hint",
    "suboptimal plan optimizer": "hint",
    "plan optimizer": "hint",
    "query": "query",
    "poorly written queries": "query",
    "poorly written query": "query",
    "normal": "normal",
}

COMPONENT_ACTION_SPACE_NAMES = {
    "index": "Index Space",
    "knob": "Knob Space",
    "hint": "Hint Space",
    "query": "Query Rewrite Space",
}

_COMPONENT_ORDER = {label: idx for idx, label in enumerate(CANONICAL_COMPONENTS)}
_EXPLANATION_PATTERN = re.compile(
    r"(?:\*\*?\s*)?Explanation(?:\s*\*\*?)?\s*:?\s*([\s\S]+?)(?=(?:\*\*?\s*)?(?:Tuning_Action|Fix_Action|Query_Rewrite)\s*:|$)",
    re.IGNORECASE,
)
_TUNING_ACTION_PATTERN = re.compile(
    r"(?:\*\*?\s*)?(?:Tuning_Action|Fix_Action)(?:\s*\*\*?)?\s*:?\s*([\s\S]+?)(?=(?:\*\*?\s*)?Query_Rewrite\s*:|$)",
    re.IGNORECASE,
)
_QUERY_REWRITE_PATTERN = re.compile(
    r"(?:\*\*?\s*)?Query_Rewrite(?:\s*\*\*?)?\s*:?\s*(yes|no)",
    re.IGNORECASE,
)


def normalize_component_label(label):
    if label is None:
        return ""
    normalized = str(label).strip().lower()
    return LEGACY_COMPONENT_LABELS.get(normalized, normalized)


def normalize_components(labels):
    if labels is None:
        return []
    if isinstance(labels, str):
        candidates = [labels]
    else:
        try:
            candidates = list(labels)
        except Exception:
            candidates = [labels]

    normalized = []
    seen = set()
    for candidate in candidates:
        label = normalize_component_label(candidate)
        if label not in _COMPONENT_ORDER or label in seen:
            continue
        seen.add(label)
        normalized.append(label)

    normalized.sort(key=lambda item: _COMPONENT_ORDER[item])
    return normalized


def format_component_label(label_or_labels):
    labels = normalize_components(label_or_labels)
    if labels:
        return "[" + ", ".join(labels) + "]" if len(labels) > 1 else labels[0]

    fallback = normalize_component_label(label_or_labels)
    return fallback or str(label_or_labels)


def normalize_component_confidence_map(confidence_map):
    if not isinstance(confidence_map, dict):
        return {}

    normalized = {}
    for label, confidence in confidence_map.items():
        component = normalize_component_label(label)
        if component not in _COMPONENT_ORDER:
            continue
        try:
            normalized[component] = float(confidence)
        except (TypeError, ValueError):
            continue
    return normalized


def normalize_tuning_history(tuning_history):
    normalized_history = []
    for item in tuning_history or []:
        normalized_history.append(
            {
                "label": format_component_label(item.get("label")),
                "success_count": int(item.get("success_count", 0)),
                "fail_count": int(item.get("fail_count", 0)),
            }
        )
    return normalized_history


def parse_detection_output(json_text):
    try:
        data = json.loads(json_text)
    except Exception:
        return None

    predicted_raw = data.get("predicted_components")
    if predicted_raw is None:
        predicted_raw = data.get("predicted_root_causes", [])

    components_raw = data.get("components")
    if components_raw is None:
        components_raw = data.get("root_causes", [])

    if not isinstance(components_raw, list):
        return None

    predicted_components = normalize_components(predicted_raw)
    confidence_map = {}

    for item in components_raw:
        if not isinstance(item, dict):
            return None
        label = normalize_component_label(item.get("label"))
        if label not in _COMPONENT_ORDER:
            return None
        if label in confidence_map:
            return None
        try:
            confidence = float(item.get("confidence"))
        except (TypeError, ValueError):
            return None
        if confidence < 0 or confidence > 1:
            return None
        confidence_map[label] = confidence

    if set(confidence_map.keys()) != set(CANONICAL_COMPONENTS):
        return None

    derived_components = [
        label for label in CANONICAL_COMPONENTS if confidence_map[label] > 0.5
    ]

    if set(predicted_components) != set(derived_components):
        return None

    if not derived_components:
        return ["normal"], confidence_map

    return derived_components, confidence_map


def parse_tuning_agent_output(text):
    text = str(text or "").strip()

    explanation_match = _EXPLANATION_PATTERN.search(text)
    action_match = _TUNING_ACTION_PATTERN.search(text)
    rewrite_match = _QUERY_REWRITE_PATTERN.search(text)

    explanation = explanation_match.group(1).strip() if explanation_match else ""
    tuning_action = action_match.group(1).strip() if action_match else ""
    query_rewrite = rewrite_match.group(1).strip().lower() if rewrite_match else ""

    return explanation, tuning_action, query_rewrite
