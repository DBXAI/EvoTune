import ast
import json
from typing import Iterable, Sequence

import torch


def build_component_labels(
    root_causes: Iterable[str],
    case_label: str,
    label_space: Sequence[str],
    modified_components: Iterable[str] = None,
):
    """Build RAWS labels from the components actually modified by an action."""
    selected = set(modified_components if modified_components is not None else (root_causes or []))
    is_positive = str(case_label).lower() == "positive"
    return [1 if is_positive and component in selected else 0 for component in label_space]


def parse_attempt_vector(value, num_components: int):
    if value is None:
        return [0] * num_components
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = ast.literal_eval(value)
        return parse_attempt_vector(parsed, num_components)
    if isinstance(value, dict):
        return [int(value.get(idx, value.get(str(idx), 0))) for idx in range(num_components)]
    try:
        parsed = list(value)
    except TypeError:
        return [0] * num_components
    if len(parsed) < num_components:
        parsed = parsed + [0] * (num_components - len(parsed))
    return [int(item) for item in parsed[:num_components]]


def raws_supervision_weights(
    labels,
    selected_mask,
    tuning_attempts,
    eta,
    modified_counts=None,
    rho=0.8,
):
    """Return the RAWS weights from Eq. 9 for each component label."""
    selected_mask = selected_mask.to(device=labels.device, dtype=torch.bool)
    tuning_attempts = tuning_attempts.to(device=labels.device)

    positive_mask = (labels > 0.5) & selected_mask
    if modified_counts is None:
        modified_counts = selected_mask.sum(dim=1, keepdim=True)
    else:
        modified_counts = modified_counts.to(device=labels.device).reshape(-1, 1)
    # A failed multi-component action cannot identify which component failed.
    verified_negative_mask = (
        (labels <= 0.5) & selected_mask & (modified_counts == 1)
        & (tuning_attempts >= eta)
    )
    weights = (positive_mask | verified_negative_mask).float()
    multi_component = (modified_counts > 1) & positive_mask
    weights = torch.where(multi_component, weights * float(rho), weights)
    return weights


def masked_bce_loss(
    pred_label,
    labels,
    selected_mask,
    tuning_attempts,
    eta,
    modified_counts=None,
    rho=0.8,
):
    """Weighted BCE over reliable RAWS labels; unlabeled components get zero weight."""
    labels = labels.to(device=pred_label.device, dtype=pred_label.dtype)
    weights = raws_supervision_weights(
        labels, selected_mask, tuning_attempts, eta,
        modified_counts=modified_counts, rho=rho,
    )
    if not weights.any():
        return pred_label.sum() * 0.0

    bce_per_element = -(labels * torch.log(pred_label + 1e-12) + (1 - labels) * torch.log(1 - pred_label + 1e-12))
    return (bce_per_element * weights).sum() / weights.sum().clamp_min(1.0)
