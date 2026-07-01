"""Shock-response workflow graph.

The LLM proposes bounded shock and strategy candidates; Python solves, validates,
and ranks them before audit/render.
"""

from __future__ import annotations

import json

from ..llm_adapters.data_expert import classify_session, serialize_scenarios
from ..llm_adapters.model_factory import MODEL_REGISTRY
from ..llm_adapters.shock_response_agent import (
    build_shock_candidate_context,
    create_shock_candidate_agent,
    create_shock_explanation_agent,
    run_shock_candidate_agent,
    run_shock_explanation_agent,
)
from ..domain.data_types import PipelineResult
from ..domain.loader import NetworkData
from ..domain.workflow_types import AuditRecord, ShockWorkflowResult
from .shock_service import (
    _is_improving_strategy,
    _rank_improving_strategies,
    _select_with_coverage,
    parse_requested_strategy_count,
    solve_shock_candidate_plan,
)


def validate_and_rank_shock_output(output, network=None):
    """Validate numeric shock strategy fields and re-rank strategies in Python.

    When `network` is provided, uses effort-weighted scoring and re-applies
    stakeholder coverage guarantee (_select_with_coverage).
    """
    strategies = list(getattr(output, "strategies", []) or [])
    errors: list[str] = []

    for idx, strategy in enumerate(strategies, start=1):
        params_changed = getattr(strategy, "params_changed", {}) or {}
        if "min_coverage_count" in params_changed:
            errors.append(f"Strategy {idx} changes coverage, which is not a shock-response lever.")
        for field in ("trucks", "cost", "cost_recovered"):
            value = getattr(strategy, field, None)
            if not isinstance(value, (int, float)):
                errors.append(f"Strategy {idx} has non-numeric {field}.")

    redistribution = getattr(output, "redistribution_strategy", None)
    if redistribution is not None:
        for field in ("trucks", "cost", "cost_recovered"):
            value = getattr(redistribution, field, None)
            if not isinstance(value, (int, float)):
                errors.append(f"Redistribution strategy has non-numeric {field}.")

    if errors:
        raise ValueError("; ".join(errors))

    shock_delta = (
        getattr(output, "shock_cost", 0.0) - getattr(output, "baseline_cost", 0.0)
    )
    ranked = _rank_improving_strategies(strategies, shock_delta, network)
    if network is not None and ranked:
        ranked = _select_with_coverage(ranked, len(ranked), network)
    try:
        output.strategies = ranked
    except Exception:
        # Pydantic models are mutable by default here, but keep a fallback for test doubles.
        setattr(output, "strategies", ranked)
    return output


def build_shock_graph(
    *,
    network: NetworkData,
    model_alias: str,
    scenario_history: list[PipelineResult],
    shock_strategy_history: list[list[dict]] | None = None,
    run_agent_fn=None,
    solve_candidates_fn=None,
    run_explanation_fn=None,
):
    """Build the shock-response LangGraph.

    `run_agent_fn` is a deterministic test hook with signature
    `(query, session_context, language) -> output`.
    `solve_candidates_fn` can bypass solver reruns in tests.
    `run_explanation_fn` can bypass the final LLM narrative in tests.
    """
    from langgraph.graph import END, START, StateGraph
    from ..domain.workflow_types import FleetPlanningState

    _shock_history: list[list[dict]] = shock_strategy_history or []

    def _build_context(state):
        language = state.get("language", "pt")
        session_context = ""
        if scenario_history:
            profile = classify_session(scenario_history)
            session_context = serialize_scenarios(
                scenario_history,
                network,
                profile,
                language,
            )
        if _shock_history:
            if language == "pt":
                header = "Estratégias shock anteriores nesta sessão (não repita nenhuma destas):"
                round_label = "Análise"
            else:
                header = "Prior shock strategies this session (do not repeat any of these):"
                round_label = "Analysis"
            lines = [header]
            for i, round_strategies in enumerate(_shock_history, start=1):
                lines.append(f"{round_label} {i}:")
                for s in round_strategies:
                    lines.append("  " + json.dumps(s, ensure_ascii=False, sort_keys=True))
            session_context += "\n\n" + "\n".join(lines)
        return {
            "session_context": session_context,
            "graph_path": ["load_session_history", "build_shock_session_context"],
        }

    def _generate_candidates(state):
        query = state["query"]
        language = state.get("language", "pt")
        if run_agent_fn is not None:
            output = run_agent_fn(query, state.get("session_context", ""), language)
        else:
            provider, model_id = MODEL_REGISTRY[model_alias]
            agent = create_shock_candidate_agent(provider, model_id, language)
            network_context = build_shock_candidate_context(network)
            output = run_shock_candidate_agent(
                agent,
                query,
                state.get("session_context", ""),
                language,
                network_context,
            )
        return {"shock_output": output, "graph_path": ["generate_strategy_candidates"]}

    def _solve_candidates(state):
        if solve_candidates_fn is not None:
            shock_output = solve_candidates_fn(state["shock_output"], network)
        else:
            shock_output = solve_shock_candidate_plan(
                state["shock_output"],
                network,
                state.get("language", "pt"),
                parse_requested_strategy_count(state.get("query", "")),
            )
        return {"shock_output": shock_output, "graph_path": ["solve_strategy_candidates"]}

    def _validate_and_rank(state):
        shock_output = validate_and_rank_shock_output(state["shock_output"], network)
        return {
            "shock_output": shock_output,
            "graph_path": ["validate_strategy_candidates", "rank_recovery"],
        }

    def _generate_explanation(state):
        import logging
        _log = logging.getLogger(__name__)
        output = state["shock_output"]
        language = state.get("language", "pt")
        try:
            if run_explanation_fn is not None:
                narrative = run_explanation_fn(output, state["query"], language)
            else:
                provider, model_id = MODEL_REGISTRY[model_alias]
                agent = create_shock_explanation_agent(provider, model_id, language)
                narrative = run_shock_explanation_agent(agent, output, state["query"], language)
            if narrative:
                output.narrative = narrative
        except Exception as exc:
            _log.warning("Shock narrative generation failed: %s", exc)
        return {"shock_output": output, "graph_path": ["generate_shock_narrative"]}

    def _audit(state):
        provider, _ = MODEL_REGISTRY[model_alias]
        full_path = list(state.get("graph_path", [])) + ["render_shock_response"]
        audit_record = AuditRecord.now(
            user_query=state["query"],
            language=state.get("language", "pt"),
            model_aliases={"selected": model_alias, "provider": provider},
            graph_path=full_path,
            final_explanation=getattr(state.get("shock_output"), "narrative", ""),
        )
        return {"audit_record": audit_record, "graph_path": ["render_shock_response"]}

    graph = StateGraph(FleetPlanningState)
    graph.add_node("build_context", _build_context)
    graph.add_node("generate_candidates", _generate_candidates)
    graph.add_node("solve_candidates", _solve_candidates)
    graph.add_node("validate_and_rank", _validate_and_rank)
    graph.add_node("generate_shock_narrative", _generate_explanation)
    graph.add_node("persist_audit_record", _audit)
    graph.add_edge(START, "build_context")
    graph.add_edge("build_context", "generate_candidates")
    graph.add_edge("generate_candidates", "solve_candidates")
    graph.add_edge("solve_candidates", "validate_and_rank")
    graph.add_edge("validate_and_rank", "generate_shock_narrative")
    graph.add_edge("generate_shock_narrative", "persist_audit_record")
    graph.add_edge("persist_audit_record", END)
    return graph.compile()


def run_shock_response_workflow(
    *,
    query: str,
    language: str,
    model_alias: str,
    network: NetworkData,
    scenario_history: list[PipelineResult],
    shock_strategy_history: list[list[dict]] | None = None,
    compiled_graph=None,
) -> ShockWorkflowResult:
    """Run shock response behind a workflow/audit boundary."""
    graph = compiled_graph or build_shock_graph(
        network=network,
        model_alias=model_alias,
        scenario_history=scenario_history,
        shock_strategy_history=shock_strategy_history,
    )
    state = graph.invoke({"query": query, "language": language, "graph_path": []})
    return ShockWorkflowResult(
        output=state["shock_output"],
        audit_record=state["audit_record"],
        should_append_to_history=False,
    )
