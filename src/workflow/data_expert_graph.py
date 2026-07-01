"""Data Expert workflow graph."""

from __future__ import annotations

from ..llm_adapters.data_expert import (
    classify_session,
    run_data_expert_agent,
    serialize_scenarios,
)
from ..domain.data_types import PipelineResult
from ..domain.loader import NetworkData
from ..domain.workflow_types import AuditRecord, DataExpertWorkflowResult


def build_data_expert_graph(
    *,
    network: NetworkData,
    model_alias: str,
    scenario_history: list[PipelineResult],
    run_agent_fn=None,
):
    """Build the Data Expert LangGraph.

    `run_agent_fn` is a deterministic test hook with signature
    `(serialized, profile, terminal_ids, language, model_alias, history, network) -> output`.
    """
    from langgraph.graph import END, START, StateGraph
    from ..domain.workflow_types import FleetPlanningState

    def _classify_and_serialize(state):
        language = state.get("language", "pt")
        profile = classify_session(scenario_history)
        serialized = serialize_scenarios(scenario_history, network, profile, language)
        return {
            "session_profile": profile,
            "session_context": serialized,
            "graph_path": [
                "load_session_history",
                "classify_session",
                "precompute_table_rows",
                "precompute_verified_facts",
                "build_data_expert_fact_packet",
            ],
        }

    def _run_narrative(state):
        language = state.get("language", "pt")
        if run_agent_fn is not None:
            output = run_agent_fn(
                state["session_context"],
                state["session_profile"],
                network.terminal_ids,
                language,
                model_alias,
                scenario_history,
                network,
            )
        else:
            output = run_data_expert_agent(
                state["session_context"],
                state["session_profile"],
                network.terminal_ids,
                language,
                model_alias,
                scenario_history,
                network,
            )
        return {"data_expert_output": output, "graph_path": ["generate_data_expert_narrative"]}

    def _audit(state):
        full_path = list(state.get("graph_path", [])) + ["render_data_expert_output"]
        audit_record = AuditRecord.now(
            user_query=state.get("query", "/data-expert"),
            language=state.get("language", "pt"),
            model_aliases={"selected": model_alias},
            graph_path=full_path,
            final_explanation=getattr(state.get("data_expert_output"), "narrative", ""),
        )
        return {"audit_record": audit_record, "graph_path": ["render_data_expert_output"]}

    graph = StateGraph(FleetPlanningState)
    graph.add_node("classify_and_serialize", _classify_and_serialize)
    graph.add_node("generate_narrative", _run_narrative)
    graph.add_node("persist_audit_record", _audit)
    graph.add_edge(START, "classify_and_serialize")
    graph.add_edge("classify_and_serialize", "generate_narrative")
    graph.add_edge("generate_narrative", "persist_audit_record")
    graph.add_edge("persist_audit_record", END)
    return graph.compile()


def run_data_expert_workflow(
    *,
    language: str,
    model_alias: str,
    network: NetworkData,
    scenario_history: list[PipelineResult],
    compiled_graph=None,
) -> DataExpertWorkflowResult:
    graph = compiled_graph or build_data_expert_graph(
        network=network,
        model_alias=model_alias,
        scenario_history=scenario_history,
    )
    state = graph.invoke({"query": "/data-expert", "language": language, "graph_path": []})
    return DataExpertWorkflowResult(
        output=state["data_expert_output"],
        profile=state["session_profile"],
        audit_record=state["audit_record"],
    )
