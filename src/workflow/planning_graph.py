"""Planning workflow graph and application facade."""

from __future__ import annotations

from typing import Callable, Optional

from ..llm_adapters.or_agent import _parse_directional_coverage_query
from ..domain.consistency_checks import check_result_consistency
from ..domain.data_types import CoverageComparisonResult, MILPResult, PipelineResult, ScenarioParams
from ..domain.fact_packet import build_fact_packet
from ..domain.loader import NetworkData
from ..domain.workflow_types import (
    AuditRecord,
    PlanningWorkflowResult,
    SecondaryModelResults,
    SolverInput,
    WorkflowError,
)
from .solver_service import run_primary_solver
from .pipeline_completion import run_pipeline_from_params
from .llm_adapters import (
    classify_intent_with_current_agent,
    parse_request_with_current_or_agent,
)
from ..llm.model_factory import LLMConfig


def _solver_input_from_query(query: str, scenario_params: ScenarioParams) -> SolverInput:
    transition = _parse_directional_coverage_query(query)
    if transition is None:
        return SolverInput(scenario_params=scenario_params)
    pct_from, pct_to = transition
    return SolverInput(
        scenario_params=scenario_params,
        request_type="coverage_comparison",
        coverage_pct_from=pct_from,
        coverage_pct_to=pct_to,
    )


def build_planning_graph(
    *,
    network: NetworkData,
    classifier_agent: LLMConfig | None = None,
    or_agent: LLMConfig | None = None,
    classify_fn: Optional[Callable[[str, str], object]] = None,
    parse_fn: Optional[Callable[[str, NetworkData], ScenarioParams]] = None,
):
    """Build the normal planning LangGraph.

    Test hooks (`classify_fn`, `parse_fn`) keep deterministic graph-edge tests
    independent from live LLM calls.
    """
    from langgraph.graph import END, START, StateGraph
    from ..domain.workflow_types import FleetPlanningState

    def _classify(state):
        query = state["query"]
        language = state.get("language", "pt")
        if classify_fn is not None:
            raw_intent = classify_fn(query, language)
            if isinstance(raw_intent, str):
                from ..domain.workflow_types import IntentDecision

                intent = IntentDecision(raw_intent)  # type: ignore[arg-type]
            else:
                intent = raw_intent
        else:
            if classifier_agent is None:
                raise ValueError("classifier_agent is required when classify_fn is not supplied.")
            intent = classify_intent_with_current_agent(query, language, classifier_agent)
        return {"intent": intent, "graph_path": ["classify_intent"]}

    def _route_after_intent(state):
        return state["intent"].label

    def _parse(state):
        query = state["query"]
        if parse_fn is not None:
            scenario_params = parse_fn(query, network)
        else:
            if or_agent is None:
                raise ValueError("or_agent is required when parse_fn is not supplied.")
            scenario_params = parse_request_with_current_or_agent(or_agent, query, network)
        solver_input = _solver_input_from_query(query, scenario_params)
        return {
            "scenario_params": scenario_params,  # type: ignore[typeddict-unknown-key]
            "solver_input": solver_input,
            "graph_path": ["parse_request", "build_solver_input"],
        }

    def _solve(state):
        raw = run_primary_solver(network, state["solver_input"])
        scenario_params = state["solver_input"].scenario_params
        if isinstance(raw, CoverageComparisonResult):
            cov = raw
            milp_result = cov.result_b
            scenario_params.coverage_count_a = cov.coverage_count_a
            scenario_params.coverage_count_b = cov.coverage_count_b
            if cov.result_b.served_cps:
                scenario_params.served_cps = list(cov.result_b.served_cps)
            if cov.result_b.assignments:
                scenario_params.milp_assignments = dict(cov.result_b.assignments)
            return {
                "primary_result": milp_result,
                "coverage_comparison": cov,
                "graph_path": ["run_primary_solver"],
            }
        milp_result = raw
        if milp_result.served_cps:
            scenario_params.served_cps = list(milp_result.served_cps)
        if milp_result.assignments:
            scenario_params.milp_assignments = dict(milp_result.assignments)
        return {"primary_result": milp_result, "graph_path": ["run_primary_solver"]}

    def _audit_parse_solve(state):
        audit_record = AuditRecord.now(
            user_query=state["query"],
            language=state.get("language", "pt"),
            model_aliases={},
            graph_path=list(state.get("graph_path", [])),  # type: ignore[arg-type]
            solver_input=state.get("solver_input"),
            primary_result=state.get("primary_result"),
        )
        return {"audit_record": audit_record}

    def _audit_shock_route(state):
        full_path = list(state.get("graph_path", [])) + ["route_shock_response"]
        audit_record = AuditRecord.now(
            user_query=state["query"],
            language=state.get("language", "pt"),
            model_aliases={},
            graph_path=full_path,  # type: ignore[arg-type]
        )
        return {"audit_record": audit_record, "graph_path": ["route_shock_response"]}

    graph = StateGraph(FleetPlanningState)
    graph.add_node("classify_intent", _classify)
    graph.add_node("parse_request", _parse)
    graph.add_node("run_primary_solver", _solve)
    graph.add_node("persist_prepare_audit", _audit_parse_solve)
    graph.add_node("shock_response_route", _audit_shock_route)
    graph.add_edge(START, "classify_intent")
    graph.add_conditional_edges(
        "classify_intent",
        _route_after_intent,
        {
            "what_if": "parse_request",
            "baseline": "parse_request",
            "relocation": "parse_request",
            "shock_response": "shock_response_route",
            "unsupported": "shock_response_route",
            "clarification": "shock_response_route",
            "data_expert": "shock_response_route",
        },
    )
    graph.add_edge("parse_request", "run_primary_solver")
    graph.add_edge("run_primary_solver", "persist_prepare_audit")
    graph.add_edge("persist_prepare_audit", END)
    graph.add_edge("shock_response_route", END)
    return graph.compile()


def prepare_planning_workflow(
    *,
    query: str,
    language: str,
    network: NetworkData,
    classifier_agent: LLMConfig,
    or_agent: LLMConfig,
    prebuilt: Optional[tuple[MILPResult, ScenarioParams]] = None,
    compiled_graph=None,
) -> tuple[str, MILPResult, ScenarioParams, AuditRecord, Optional[CoverageComparisonResult]]:
    """Classify, parse, and solve the primary MILP through the planning graph."""
    graph_path: list[str] = []
    if prebuilt is not None:
        milp_result, scenario_params = prebuilt
        graph_path.extend(["prebuilt_request"])
    else:
        graph_app = compiled_graph or build_planning_graph(
            network=network,
            classifier_agent=classifier_agent,
            or_agent=or_agent,
        )
        state = graph_app.invoke({"query": query, "language": language, "graph_path": []})
        intent = state["intent"]
        if intent.label == "shock_response":
            audit = state["audit_record"]
            return "shock_response", MILPResult(feasible=False), ScenarioParams(), audit, None

        scenario_params = state["solver_input"].scenario_params
        milp_result = state["primary_result"]
        audit = state["audit_record"]
        coverage_comparison = state.get("coverage_comparison")
        return "what_if", milp_result, scenario_params, audit, coverage_comparison

    audit = AuditRecord.now(
        user_query=query,
        language=language,
        model_aliases={},
        graph_path=graph_path,
        primary_result=milp_result,
    )
    return "what_if", milp_result, scenario_params, audit, None


def complete_planning_workflow(
    *,
    query: str,
    language: str,
    model_alias: str,
    network: NetworkData,
    expert_agent: LLMConfig,
    query_number: int,
    milp_result: MILPResult,
    scenario_params: ScenarioParams,
    llm_insights: bool = True,
    baseline_result: Optional[PipelineResult] = None,
    on_phase: Optional[Callable[[str], None]] = None,
    graph_path: Optional[list[str]] = None,
    coverage_comparison: Optional[CoverageComparisonResult] = None,
) -> PlanningWorkflowResult:
    """Run secondary models, explanation, fact packet, consistency, and audit."""
    path = list(graph_path or [])
    path.extend(["run_secondary_models", "generate_explanation"])
    result = run_pipeline_from_params(
        milp_result=milp_result,
        scenario_params=scenario_params,
        network=network,
        expert_agent=expert_agent,
        query_number=query_number,
        language=language,
        llm_insights=llm_insights,
        baseline_result=baseline_result,
        on_phase=on_phase,
        coverage_comparison=coverage_comparison,
    )
    result.query_text = query
    result.coverage_comparison = coverage_comparison

    path.extend(["run_consistency_checks", "build_fact_packet", "persist_audit_record"])
    consistency = check_result_consistency(result, result.scenario_params, network)
    if not consistency.passed:
        return PlanningWorkflowResult(
            pipeline_result=result,
            should_append_to_history=False,
            error=WorkflowError("consistency_error", "; ".join(consistency.failures)),
            audit_record=AuditRecord.now(
                user_query=query,
                language=language,
                model_aliases={"selected": model_alias},
                graph_path=path,
                primary_result=result.milp_result,
                secondary_results=SecondaryModelResults(result.lbl_result, result.wct_result),
                consistency_result=consistency,
                final_explanation=result.insight,
            ),
        )

    fact_packet = build_fact_packet(result, network, baseline_result)
    audit = AuditRecord.now(
        user_query=query,
        language=language,
        model_aliases={"selected": model_alias},
        graph_path=path,
        primary_result=result.milp_result,
        secondary_results=SecondaryModelResults(result.lbl_result, result.wct_result),
        consistency_result=consistency,
        fact_packet=fact_packet,
        final_explanation=result.insight,
    )
    return PlanningWorkflowResult(pipeline_result=result, audit_record=audit)


def run_planning_workflow(
    *,
    query: str,
    language: str,
    model_alias: str,
    network: NetworkData,
    classifier_agent: LLMConfig,
    or_agent: LLMConfig,
    expert_agent: LLMConfig,
    query_number: int,
    llm_insights: bool = True,
    baseline_result: Optional[PipelineResult] = None,
    prebuilt: Optional[tuple[MILPResult, ScenarioParams]] = None,
    on_phase: Optional[Callable[[str], None]] = None,
) -> PlanningWorkflowResult:
    intent, milp_result, scenario_params, prep_audit, coverage_comparison = prepare_planning_workflow(
        query=query,
        language=language,
        network=network,
        classifier_agent=classifier_agent,
        or_agent=or_agent,
        prebuilt=prebuilt,
    )
    if intent == "shock_response":
        return PlanningWorkflowResult(
            audit_record=prep_audit,
            should_append_to_history=False,
            user_message="shock_response",
        )
    return complete_planning_workflow(
        query=query,
        language=language,
        model_alias=model_alias,
        network=network,
        expert_agent=expert_agent,
        query_number=query_number,
        milp_result=milp_result,
        scenario_params=scenario_params,
        llm_insights=llm_insights,
        baseline_result=baseline_result,
        on_phase=on_phase,
        graph_path=prep_audit.graph_path,
        coverage_comparison=coverage_comparison,
    )
