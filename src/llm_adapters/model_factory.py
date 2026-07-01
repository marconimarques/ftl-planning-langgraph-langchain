"""Compatibility exports for the LangChain-backed model factory."""

from __future__ import annotations

from ..llm.model_factory import (
    MODEL_REGISTRY,
    LLMConfig,
    _ANTHROPIC_FAST_MODEL,
    _OPENAI_FAST_MODEL,
    create_chat_model,
    get_agent_model,
    get_api_key,
    invoke_structured,
    invoke_text,
    make_llm_config,
)


def make_model(provider: str, model_id: str, api_key: str, max_tokens: int, system_prompt: str | None = None) -> LLMConfig:
    """Legacy name kept during cutover; returns an LLMConfig, not an agent model."""
    return make_llm_config(provider, model_id, api_key, max_tokens, system_prompt)


__all__ = [
    "MODEL_REGISTRY",
    "LLMConfig",
    "_ANTHROPIC_FAST_MODEL",
    "_OPENAI_FAST_MODEL",
    "create_chat_model",
    "get_agent_model",
    "get_api_key",
    "invoke_structured",
    "invoke_text",
    "make_llm_config",
    "make_model",
]
