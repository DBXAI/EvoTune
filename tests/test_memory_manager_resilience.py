import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

from agent.memory.memory_manager import SummaryAgent


def test_reflect_case_network_failure_returns_empty(monkeypatch):
    def raise_connection_error(*args, **kwargs):
        raise RuntimeError("connection error")

    monkeypatch.setattr(
        "agent.memory.memory_manager.Runner.run_sync",
        raise_connection_error,
    )

    agent = SummaryAgent(summary_model="dummy-model")
    result = agent.reflect_case({"query_info": {"query_id": "07"}})

    assert result == ""


def test_summarize_cases_network_failure_preserves_previous_summary(monkeypatch):
    def raise_connection_error(*args, **kwargs):
        raise RuntimeError("connection error")

    monkeypatch.setattr(
        "agent.memory.memory_manager.Runner.run_sync",
        raise_connection_error,
    )

    agent = SummaryAgent(summary_model="dummy-model")
    cases = [
        type(
            "Case",
            (),
            {
                "id": "1-07",
                "case_info": {
                    "query_info": {"query_id": "07", "query": "select 1"},
                    "fix_action": "",
                    "root_causes": ["knob"],
                    "label": "negative",
                    "approve_time": 0.0,
                    "reflection": "",
                },
            },
        )()
    ]
    result = agent.summarize_cases(["knob"], cases, previous_summary="keep me")

    assert result == "keep me"
