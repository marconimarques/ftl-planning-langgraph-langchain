"""Provider-neutral LangChain model factory."""

from __future__ import annotations

import functools
from dataclasses import dataclass
import os
from typing import Any


MODEL_REGISTRY: dict[str, tuple[str, str]] = {
    "anthropic": ("anthropic", "claude-sonnet-4-6"),
    "openai": ("openai", "gpt-4o"),
    "haiku": ("anthropic", "claude-haiku-4-5-20251001"),
    "sonnet": ("anthropic", "claude-sonnet-4-6"),
    "opus": ("anthropic", "claude-opus-4-8"),
    "gpt-4o-mini": ("openai", "gpt-4o-mini"),
    "gpt-4o": ("openai", "gpt-4o"),
    "o3": ("openai", "o3"),
}

_ANTHROPIC_FAST_MODEL = "claude-haiku-4-5-20251001"
_OPENAI_FAST_MODEL = "gpt-4o-mini"
_ANTHROPIC_REASONING_MODEL = "claude-sonnet-4-6"
_OPENAI_REASONING_MODEL = "gpt-4o"
_OPENAI_AGENTIC_MODEL = "gpt-4o"


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    model_id: str
    api_key: str
    system_prompt: str = ""
    max_tokens: int = 2048


def get_agent_model(provider: str, role: str = "reasoning") -> str:
    if provider == "openai":
        return _OPENAI_AGENTIC_MODEL if role == "agentic" else _OPENAI_REASONING_MODEL
    return _ANTHROPIC_REASONING_MODEL


def get_api_key(provider: str) -> str:
    if provider == "openai":
        return os.environ["OPENAI_API_KEY"]
    return os.environ["ANTHROPIC_API_KEY"]


def make_llm_config(
    provider: str,
    model_id: str,
    api_key: str,
    max_tokens: int,
    system_prompt: str | None = None,
) -> LLMConfig:
    return LLMConfig(
        provider=provider,
        model_id=model_id,
        api_key=api_key,
        system_prompt=system_prompt or "",
        max_tokens=max_tokens,
    )


def create_chat_model(config: LLMConfig):
    """Create a LangChain chat model lazily so tests can run without provider packages."""
    if config.provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise ImportError(
                "langchain-openai is required. Run `pip install -r requirements.txt`."
            ) from exc
        kwargs: dict[str, Any] = {
            "api_key": config.api_key,
            "model": config.model_id,
        }
        if config.model_id.startswith(("o1", "o3", "o4")):
            kwargs["max_completion_tokens"] = config.max_tokens
        else:
            kwargs["max_tokens"] = config.max_tokens
        return ChatOpenAI(**kwargs)

    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError as exc:
        raise ImportError(
            "langchain-anthropic is required. Run `pip install -r requirements.txt`."
        ) from exc
    return ChatAnthropic(
        api_key=config.api_key,
        model=config.model_id,
        max_tokens=config.max_tokens,
    )


@functools.lru_cache(maxsize=16)
def _get_chat_model(config: LLMConfig):
    """Cache chat model instances — LLMConfig is frozen and hashable."""
    return create_chat_model(config)


def _messages(config: LLMConfig, user_prompt: str) -> list[tuple[str, str]]:
    if config.system_prompt:
        return [("system", config.system_prompt), ("human", user_prompt)]
    return [("human", user_prompt)]


def invoke_text(config: LLMConfig, user_prompt: str) -> str:
    response = _get_chat_model(config).invoke(_messages(config, user_prompt))
    return str(getattr(response, "content", response)).strip()


def invoke_structured(config: LLMConfig, user_prompt: str, schema: type):
    chat_model = _get_chat_model(config)
    if config.provider == "openai":
        structured = chat_model.with_structured_output(schema, method="function_calling")
    else:
        structured = chat_model.with_structured_output(schema)
    if hasattr(structured, "with_retry"):
        structured = structured.with_retry(
            stop_after_attempt=2,
            wait_exponential_jitter=False,
        )
    return structured.invoke(_messages(config, user_prompt))
