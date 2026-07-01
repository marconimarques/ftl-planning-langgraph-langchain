"""Compatibility wrappers for the planning pipeline.

New code should use ``src.workflow`` entry points. This module remains for
older imports and focused tests that still call the pre-graph pipeline helpers.
"""

from __future__ import annotations

from typing import Callable, Optional

from ..llm_adapters.or_agent import run_or_agent
from ..domain.data_types import PipelineResult, ScenarioParams
from ..domain.loader import NetworkData
from ..llm.model_factory import LLMConfig
from ..models.solver import run_milp_solver
from ..workflow.pipeline_completion import build_baseline_params, run_pipeline_from_params


def run_pipeline(
    query: str,
    network: NetworkData,
    or_agent: LLMConfig,
    expert_agent: LLMConfig,
    query_number: int,
    language: str,
    llm_insights: bool = True,
    baseline_result: Optional[PipelineResult] = None,
    on_phase: Optional[Callable[[str], None]] = None,
) -> PipelineResult:
    """Legacy full pipeline wrapper."""
    milp_result, scenario_params = run_or_agent(or_agent, query)
    return run_pipeline_from_params(
        milp_result=milp_result,
        scenario_params=scenario_params,
        network=network,
        expert_agent=expert_agent,
        query_number=query_number,
        language=language,
        llm_insights=llm_insights,
        baseline_result=baseline_result,
        on_phase=on_phase,
    )


def run_relocation_pipeline(
    network: NetworkData,
    expert_agent: LLMConfig,
    base_params: ScenarioParams,
    query_number: int,
    language: str,
    llm_insights: bool = True,
    baseline_result: Optional[PipelineResult] = None,
    on_phase: Optional[Callable[[str], None]] = None,
) -> PipelineResult:
    """Legacy relocation wrapper using deterministic solver orchestration."""
    params = ScenarioParams(
        payload=base_params.payload,
        speed_loaded=base_params.speed_loaded,
        speed_empty=base_params.speed_empty,
        availability=base_params.availability,
        overtime_hours=base_params.overtime_hours,
        overtime_cost=base_params.overtime_cost,
        variable_cost_per_km=base_params.variable_cost_per_km,
        fixed_cost_per_truck_month=base_params.fixed_cost_per_truck_month,
        working_days=base_params.working_days,
        net_driving_hours=base_params.net_driving_hours,
        terminals_active=dict(base_params.terminals_active),
        variable_cost_components=dict(base_params.variable_cost_components),
        fixed_cost_components=dict(base_params.fixed_cost_components),
        volume_redistribution=True,
        is_baseline=False,
        objective="minimize_cost",
    )
    if on_phase:
        on_phase("phase_milp")
    milp_result = run_milp_solver(network, params)
    if milp_result.feasible and milp_result.assignments:
        params.milp_assignments = dict(milp_result.assignments)
    return run_pipeline_from_params(
        milp_result=milp_result,
        scenario_params=params,
        network=network,
        expert_agent=expert_agent,
        query_number=query_number,
        language=language,
        llm_insights=llm_insights,
        baseline_result=baseline_result,
        on_phase=on_phase,
    )
