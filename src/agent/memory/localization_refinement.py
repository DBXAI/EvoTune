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
    """Build AAWS labels from the components actually modified by an action."""
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


def masked_bce_loss(
    pred_label,
    labels,
    selected_mask,
    tuning_attempts,
    eta,
    modified_counts=None,
    rho=0.5,
):
    """Masked BCE for EvoTune localization refinement.

    Positive labels are always supervised. Negative labels are supervised only for
    selected components after sufficient exploration. Unselected components are
    treated as unlabeled and never force-trained as negatives.
    """
    selected_mask = selected_mask.to(device=pred_label.device, dtype=torch.bool)
    tuning_attempts = tuning_attempts.to(device=pred_label.device)
    labels = labels.to(device=pred_label.device, dtype=pred_label.dtype)

    positive_mask = labels > 0.5
    verified_negative_mask = (labels <= 0.5) & selected_mask & (tuning_attempts >= eta)
    supervision_mask = positive_mask | verified_negative_mask

    if not supervision_mask.any():
        return pred_label.sum() * 0.0

    bce_per_element = -(labels * torch.log(pred_label + 1e-12) + (1 - labels) * torch.log(1 - pred_label + 1e-12))
    weights = supervision_mask.float()
    if modified_counts is not None:
        modified_counts = modified_counts.to(device=pred_label.device).reshape(-1, 1)
        multi_component = (modified_counts > 1) & positive_mask
        weights = torch.where(multi_component, weights * float(rho), weights)
    return (bce_per_element * weights).sum() / weights.sum().clamp_min(1.0)
