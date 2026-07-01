"""Adapters around LangChain-backed LLM calls.

These keep workflow nodes independent from provider-specific model clients.
"""

from __future__ import annotations

from ..llm_adapters.intent_classifier import classify_intent
from ..llm_adapters.or_agent import parse_or_agent
from ..domain.data_types import ScenarioParams
from ..domain.loader import NetworkData
from ..domain.workflow_types import IntentDecision
from ..llm.model_factory import LLMConfig


def classify_intent_with_current_agent(
    query: str,
    language: str,
    classifier_agent: LLMConfig,
) -> IntentDecision:
    label = classify_intent(query, language, classifier_agent)
    if label not in {"what_if", "shock_response"}:
        label = "what_if"
    return IntentDecision(label=label)  # type: ignore[arg-type]


def parse_request_with_current_or_agent(
    or_agent: LLMConfig,
    query: str,
    network: NetworkData,
) -> ScenarioParams:
    """Parser adapter kept behind a single workflow boundary."""
    return parse_or_agent(or_agent, query, network)
