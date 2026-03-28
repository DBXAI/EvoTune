from typing import Any, Mapping


def merge_runtime_metrics(*metrics_groups: Mapping[str, Any] | None) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for metrics in metrics_groups:
        if metrics:
            merged.update(metrics)
    return merged


def format_runtime_metrics(metrics: Mapping[str, Any] | None) -> str:
    if not metrics:
        return "none"

    parts: list[str] = []
    for key in sorted(metrics):
        value = metrics[key]
        if isinstance(value, float):
            parts.append(f"{key}={value:.4f}s")
        else:
            parts.append(f"{key}={value}")
    return ", ".join(parts)


def extract_run_usage_metrics(run_result: Any, prefix: str) -> dict[str, int]:
    raw_responses = getattr(run_result, "raw_responses", None) or []

    input_tokens = 0
    output_tokens = 0
    total_tokens = 0
    requests = len(raw_responses)

    for response in raw_responses:
        usage = getattr(response, "usage", None)
        if usage is None:
            continue
        input_tokens += getattr(usage, "input_tokens", 0) or 0
        output_tokens += getattr(usage, "output_tokens", 0) or 0
        total_tokens += getattr(usage, "total_tokens", 0) or 0

    return {
        f"{prefix}_requests": requests,
        f"{prefix}_input_tokens": input_tokens,
        f"{prefix}_output_tokens": output_tokens,
        f"{prefix}_total_tokens": total_tokens,
    }
