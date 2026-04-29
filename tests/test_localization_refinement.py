import torch

from agent.contracts import parse_detection_output, parse_tuning_agent_output
from agent.memory.localization_refinement import build_component_labels, masked_bce_loss
from agent.plan.localization import EXPERT_LIGHTWEIGHT, EXPERT_LLM, EXPERT_RANDOM, LocalizationExpertSelector


def test_masked_bce_ignores_unselected_negative_components():
    predictions = torch.tensor([[0.2, 0.8, 0.7, 0.1]], requires_grad=True)
    labels = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
    selected = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    attempts = torch.tensor([[10.0, 10.0, 10.0, 10.0]])

    loss = masked_bce_loss(predictions, labels, selected, attempts, eta=3)
    expected = -torch.log(1 - predictions[0, 1] + 1e-12)

    assert torch.allclose(loss, expected)


def test_masked_bce_keeps_positive_selected_components_only():
    predictions = torch.tensor([[0.9, 0.4, 0.7, 0.8]], requires_grad=True)
    labels = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    selected = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    attempts = torch.zeros_like(labels)

    loss = masked_bce_loss(predictions, labels, selected, attempts, eta=3)
    expected = -torch.log(predictions[0, 0] + 1e-12)

    assert torch.allclose(loss, expected)


def test_build_component_labels_follows_impact_label_definition():
    label_space = ["index", "knob", "hint", "query"]

    assert build_component_labels(["index", "knob"], "positive", label_space) == [1, 1, 0, 0]
    assert build_component_labels(["index", "knob"], "negative", label_space) == [0, 0, 0, 0]


def test_localization_selector_uses_asymmetric_priors_and_feedback_window():
    selector = LocalizationExpertSelector({"localization_feedback_window": 2})
    state = {}
    localization = selector.ensure_state(state)

    assert localization["experts"][EXPERT_LIGHTWEIGHT] == {"alpha": 8.0, "beta": 2.0}
    assert localization["experts"][EXPERT_LLM] == {"alpha": 2.0, "beta": 2.0}
    assert localization["experts"][EXPERT_RANDOM] == {"alpha": 2.0, "beta": 8.0}

    selector.start_decision(state, EXPERT_LLM, ["index"], {"index": 0.7})
    assert selector.record_feedback(state, success=False) is None
    update = selector.record_feedback(state, success=True)

    assert update["expert"] == EXPERT_LLM
    assert update["impact"] == 1
    assert localization["experts"][EXPERT_LLM] == {"alpha": 3.0, "beta": 2.0}


def test_detection_parser_recovers_nonempty_subspace_from_all_low_scores():
    labels, confidence = parse_detection_output(
        '{"predicted_components": [], "components": ['
        '{"label":"index","confidence":0.2},'
        '{"label":"knob","confidence":0.4},'
        '{"label":"hint","confidence":0.3},'
        '{"label":"query","confidence":0.1}'
        "]}"
    )

    assert labels == ["knob"]
    assert confidence["knob"] == 0.51


def test_detection_parser_accepts_appendix_predicted_components_only():
    labels, confidence = parse_detection_output('{"predicted_components": ["query", "index"]}')

    assert labels == ["index", "query"]
    assert confidence == {"index": 1.0, "knob": 0.0, "hint": 0.0, "query": 1.0}


def test_tuning_parser_accepts_appendix_two_section_output():
    explanation, action, query_rewrite = parse_tuning_agent_output(
        "Tuning_Action:\nSET work_mem = '64MB';\nQuery_Rewrite:\nno"
    )

    assert explanation == ""
    assert action == "SET work_mem = '64MB';"
    assert query_rewrite == "no"
