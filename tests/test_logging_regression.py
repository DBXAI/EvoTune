import subprocess
import sys
import textwrap

from agents import trace

from src.utils.openai_agents import configure_openai_agents


def test_project_prompt_loggers_are_not_muted_by_action_manager_import():
    script = textwrap.dedent(
        """
        import io
        import logging
        import sys

        sys.path.insert(0, "/root/DREAM/src")

        stream = io.StringIO()
        root = logging.getLogger()
        root.handlers.clear()
        root.setLevel(logging.INFO)

        handler = logging.StreamHandler(stream)
        handler.setLevel(logging.INFO)
        root.addHandler(handler)

        import agent.action.action_manager

        logging.getLogger("agent.action.action_manager").info("ACTION_MANAGER_MARKER")
        logging.getLogger("agent.action.action_evaluation").info("ACTION_EVAL_MARKER")

        print(stream.getvalue())
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
    )

    assert "ACTION_MANAGER_MARKER" in result.stdout
    assert "ACTION_EVAL_MARKER" in result.stdout


def test_configure_openai_agents_disables_tracing_by_default():
    configure_openai_agents({"openai": {"api_key": "x", "base_url": "http://example.com"}})
    assert type(trace("demo")).__name__ == "NoOpTrace"
