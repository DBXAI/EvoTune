import os

from agents import set_tracing_disabled
from agents._config import set_default_openai_api


def _coerce_bool(value, default):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def configure_openai_agents(api_settings):
    if not api_settings:
        return

    openai_config = api_settings.get("openai", {})
    api_key = openai_config.get("api_key")
    base_url = openai_config.get("base_url")

    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    if base_url:
        os.environ["OPENAI_BASE_URL"] = base_url

    set_default_openai_api("chat_completions")

    disable_tracing = _coerce_bool(
        openai_config.get("disable_tracing", api_settings.get("disable_tracing")),
        default=True,
    )
    set_tracing_disabled(disable_tracing)
    os.environ["OPENAI_AGENTS_DISABLE_TRACING"] = "true" if disable_tracing else "false"
