"""Typed contracts for the fleet planning workflow.

These objects make the workflow boundary explicit: LLM-facing draft data is
separate from normalized, validated, solver-ready input.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Annotated, Any, Literal, Optional, TypedDict

if TYPE_CHECKING:
    from ..llm_adapters.shock_response_agent import ShockResponseOutput
    from ..llm_adapters.data_expert import DataExpertOutput, SessionProfile

from .data_types import CoverageComparisonResult, MILPResult, ModelResult, PipelineResult, ScenarioParams


IntentLabel = Literal[
    "what_if",
    "shock_response",
    "baseline",
    "relocation",
    "data_expert",
    "unsupported",
    "clarification",
]


@dataclass(frozen=True)
class IntentDecision:
    label: IntentLabel
    confidence: float = 1.0
    reason: str = ""


@dataclass
class DraftPlanningRequest:
    """Structured request draft produced by an LLM or temporary adapter."""

    scenario_type: str = "what_if"
    objective: Optional[str] = None
    coverage_from_pct: Optional[float] = None
    coverage_to_pct: Optional[float] = None
    budget: Optional[float] = None
    closed_terminals: list[str] = field(default_factory=list)
    terminal_demand_multipliers: dict[str, float] = field(default_factory=dict)
    terminal_volume_caps: dict[str, float] = field(default_factory=dict)
    cost_component_overrides: dict[str, float] = field(default_factory=dict)
    volume_redistribution: Optional[bool] = None
    requires_clarification: bool = False
    clarification_question: str = ""
    parser_notes: str = ""


@dataclass
class NormalizedPlanningRequest:
    """Deterministically normalized request, not yet solver-approved."""

    scenario_type: str
    objective: str
    coverage_from_pct: Optional[float] = None
    coverage_to_pct: Optional[float] = None
    coverage_count_a: Optional[int] = None
    coverage_count_b: Optional[int] = None
    min_coverage_count: Optional[int] = None
    budget: Optional[float] = None
    closed_terminals: list[str] = field(default_factory=list)
    terminal_demand_multipliers: dict[str, float] = field(default_factory=dict)
    terminal_volume_caps: dict[str, float] = field(default_factory=dict)
    cost_component_overrides: dict[str, float] = field(default_factory=dict)
    volume_redistribution: bool = False
    requires_clarification: bool = False
    clarification_question: str = ""


@dataclass
class ValidationResult:
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    needs_clarification: bool = False
    clarification_question: str = ""


@dataclass
class SolverInput:
    """Canonical solver-ready input. This is the only workflow solver input."""

    scenario_params: ScenarioParams
    request_type: str = "single_scenario"
    coverage_pct_from: Optional[float] = None
    coverage_pct_to: Optional[float] = None


@dataclass
class SecondaryModelResults:
    lane_by_lane: Optional[ModelResult] = None
    weighted_cycle_time: Optional[ModelResult] = None


@dataclass
class ReconciliationResult:
    served_cps_synced: bool = True
    volumes_reconstructed: bool = False
    component_totals_recomputed: bool = False
    notes: list[str] = field(default_factory=list)


@dataclass
class ConsistencyCheckResult:
    passed: bool
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ScenarioFactPacket:
    scenario_label: str
    objective: str
    feasible: bool
    active_constraints: tuple[str, ...]
    total_cost: Optional[float]
    total_cost_delta: Optional[float]
    trucks: Optional[int]
    trucks_delta: Optional[int]
    coverage_count: Optional[int]
    coverage_text: str
    terminal_capacity_facts: tuple[str, ...] = ()
    assignment_facts: tuple[str, ...] = ()
    allowed_explanations: tuple[str, ...] = ()
    forbidden_explanations: tuple[str, ...] = ()
    infeasibility_reason: str = ""


@dataclass
class UserExplanation:
    text: str
    source: str = "llm"


@dataclass
class WorkflowError:
    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class AuditRecord:
    timestamp: str
    user_query: str
    language: str
    model_aliases: dict[str, str]
    graph_path: list[str]
    draft_request: Optional[DraftPlanningRequest]
    normalized_request: Optional[NormalizedPlanningRequest]
    validation_result: Optional[ValidationResult]
    solver_input: Optional[SolverInput]
    solver_status: str
    primary_result: Optional[MILPResult]
    secondary_results: Optional[SecondaryModelResults]
    consistency_result: Optional[ConsistencyCheckResult]
    fact_packet: Optional[ScenarioFactPacket]
    final_explanation: str

    @classmethod
    def now(
        cls,
        *,
        user_query: str,
        language: str,
        model_aliases: dict[str, str],
        graph_path: list[str],
        draft_request: Optional[DraftPlanningRequest] = None,
        normalized_request: Optional[NormalizedPlanningRequest] = None,
        validation_result: Optional[ValidationResult] = None,
        solver_input: Optional[SolverInput] = None,
        primary_result: Optional[MILPResult] = None,
        secondary_results: Optional[SecondaryModelResults] = None,
        consistency_result: Optional[ConsistencyCheckResult] = None,
        fact_packet: Optional[ScenarioFactPacket] = None,
        final_explanation: str = "",
    ) -> "AuditRecord":
        if primary_result is None:
            solver_status = "not_run"
        else:
            solver_status = "feasible" if primary_result.feasible else "infeasible"
        return cls(
            timestamp=datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
            user_query=user_query,
            language=language,
            model_aliases=model_aliases,
            graph_path=graph_path,
            draft_request=draft_request,
            normalized_request=normalized_request,
            validation_result=validation_result,
            solver_input=solver_input,
            solver_status=solver_status,
            primary_result=primary_result,
            secondary_results=secondary_results,
            consistency_result=consistency_result,
            fact_packet=fact_packet,
            final_explanation=final_explanation,
        )


@dataclass
class PlanningWorkflowResult:
    pipeline_result: Optional[PipelineResult] = None
    audit_record: Optional[AuditRecord] = None
    user_message: str = ""
    should_append_to_history: bool = True
    requires_clarification: bool = False
    error: Optional[WorkflowError] = None


@dataclass
class ShockWorkflowResult:
    output: "ShockResponseOutput"
    audit_record: Optional[AuditRecord] = None
    should_append_to_history: bool = False
    error: Optional[WorkflowError] = None


@dataclass
class DataExpertWorkflowResult:
    output: "DataExpertOutput"
    profile: "SessionProfile"
    audit_record: Optional[AuditRecord] = None
    error: Optional[WorkflowError] = None


class FleetPlanningState(TypedDict, total=False):
    query: str
    language: str
    model_alias: str
    graph_path: Annotated[list[str], operator.add]
    intent: IntentDecision
    scenario_params: ScenarioParams
    session_context: str
    shock_output: Any
    data_expert_output: Any
    session_profile: Any
    draft_request: DraftPlanningRequest
    normalized_request: NormalizedPlanningRequest
    validation: ValidationResult
    solver_input: SolverInput
    primary_result: MILPResult
    coverage_comparison: CoverageComparisonResult
    secondary_results: SecondaryModelResults
    reconciliation: ReconciliationResult
    consistency: ConsistencyCheckResult
    fact_packet: ScenarioFactPacket
    explanation: UserExplanation
    audit_record: AuditRecord
    error: WorkflowError
