"""Lightweight intent classifier — routes queries to the correct pipeline."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel

from ..llm.model_factory import LLMConfig, invoke_structured
from .model_factory import _ANTHROPIC_FAST_MODEL, _OPENAI_FAST_MODEL, get_api_key, make_model

_CLASSIFIER_PROMPT_PT = """\
Você classifica perguntas de um sistema de planejamento de frota de caminhões.

Contrato do agente:
- Papel: roteador de intenção.
- Entrada: uma pergunta do usuário em linguagem natural.
- Limite de decisão: escolher somente entre "what_if" e "shock_response"; não resolver cenários, não estimar números e não explicar resultados.
- Padrão de saída: JSON com um único campo query_type.

Retorne "shock_response" se o usuário está pedindo a melhor estratégia para
compensar uma deterioração operacional: redução de payload, aumento de custo,
queda de disponibilidade, redução de jornada, etc.
Frases típicas: "melhor reação", "como compensar", "o que fazer se",
"como conter", "como mitigar".

Retorne "what_if" para qualquer outra coisa: simulações, cenários hipotéticos,
cálculo de baseline, perguntas sobre a rede.

Retorne JSON com um único campo: query_type.
"""

_CLASSIFIER_PROMPT_EN = """\
You classify queries for a truck fleet planning system.

Agent contract:
- Role: intent router.
- Input: one natural-language user query.
- Decision boundary: choose only between "what_if" and "shock_response"; do not solve scenarios, estimate numbers, or explain results.
- Output standard: JSON with a single query_type field.

Return "shock_response" if the user is asking for the best strategy to offset
an operational deterioration: payload reduction, cost increase, availability
drop, reduced working hours, etc.
Typical phrasings: "best response to", "how to offset", "what to do if",
"how to contain", "how to mitigate".

Return "what_if" for everything else: simulations, what-if scenarios,
baseline calculations, network questions.

Return JSON with a single field: query_type.
"""


class _IntentResult(BaseModel):
    query_type: Literal["what_if", "shock_response"]


def create_classifier_agent(provider: str, api_key: str, language: str) -> LLMConfig:
    """Create a reusable classifier model config. Uses the fast model for the provider."""
    system_prompt = _CLASSIFIER_PROMPT_PT if language == "pt" else _CLASSIFIER_PROMPT_EN
    fast_model = _OPENAI_FAST_MODEL if provider == "openai" else _ANTHROPIC_FAST_MODEL
    return make_model(provider, fast_model, api_key, 512, system_prompt)


def classify_intent(
    query: str, language: str, agent: Optional[LLMConfig] = None
) -> Literal["what_if", "shock_response"]:
    """Classify a user query with a structured LangChain model call."""
    if agent is None:
        agent = create_classifier_agent("anthropic", get_api_key("anthropic"), language)
    structured = invoke_structured(agent, query, _IntentResult)
    return structured.query_type if structured else "what_if"
