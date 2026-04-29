from types import SimpleNamespace

from agent.action.action_space import action_pruning, action_refinement


class FakeMemory:
    def __init__(self, cases, agent_config=None):
        self.cases = {case.id: case for case in cases}
        self.agent_config = agent_config or {}


def make_case(case_id, root_causes, fix_action, approve_time, query_id="Q1", query="SELECT 1"):
    return SimpleNamespace(
        id=case_id,
        case_info={
            "root_causes": root_causes,
            "fix_action": fix_action,
            "approve_time": approve_time,
            "query_info": {"query_id": query_id, "query": query},
        },
        metrics={"approve_time": approve_time},
    )


def test_index_pruning_uses_appendix_heading_and_failed_index_names():
    query_info = SimpleNamespace(query_id="Q1", query="SELECT * FROM lineitem WHERE l_orderkey = 1")
    memory = FakeMemory(
        [
            make_case(
                "c1",
                ["index"],
                "CREATE INDEX IF NOT EXISTS lineitem_l_orderkey_idx ON lineitem(l_orderkey);",
                0.0,
            )
        ]
    )

    guidance = action_pruning(["index"], query_info, memory)

    assert "- Index Pruning:" in guidance
    assert "lineitem_l_orderkey_idx" in guidance


def test_knob_refinement_uses_appendix_heading_and_refined_range():
    query_info = SimpleNamespace(query_id="Q1", query="SELECT 1")
    memory = FakeMemory(
        [
            make_case("c1", ["knob"], "SET work_mem = 64;", 1.0),
            make_case("c2", ["knob"], "SET work_mem = 128;", 2.0),
        ],
        agent_config={"refinement_min_cases": 2},
    )

    guidance = action_refinement(["knob"], query_info, memory)

    assert "- Knob Refinement:" in guidance
    assert "<KNOB_NAME=work_mem, RANGE=[64, 128], NUM_CASES=2" in guidance
