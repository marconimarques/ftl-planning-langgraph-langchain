"""Secondary model and explanation orchestration for planning workflows."""

from __future__ import annotations

import concurrent.futures
from typing import Callable, Optional

from ..llm_adapters.transportation_expert import run_expert_agent
from ..domain.data_types import MILPResult, PipelineResult, ScenarioParams
from ..domain.loader import NetworkData
from ..llm.model_factory import LLMConfig
from ..models.lane_by_lane import run_lane_by_lane
from ..models.weighted_cycle_time import run_weighted_cycle_time


# Dedicated thread for the Expert agent so it runs concurrently with LBL+WCT.
_insight_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)


def run_pipeline_from_params(
    milp_result: MILPResult,
    scenario_params: ScenarioParams,
    network: NetworkData,
    expert_agent: LLMConfig,
    query_number: int,
    language: str,
    llm_insights: bool = True,
    baseline_result: Optional[PipelineResult] = None,
    on_phase: Optional[Callable[[str], None]] = None,
    coverage_comparison=None,
) -> PipelineResult:
    """Run LBL + WCT + Expert given a pre-computed MILP result."""
    is_coverage_comparison = scenario_params.coverage_count_a is not None
    needs_solver_run = scenario_params.skip_capacity_constraints or (
        not is_coverage_comparison and not milp_result.feasible
    ) or (
        not is_coverage_comparison
        and milp_result.feasible
        and milp_result.trucks == 0
        and milp_result.coverage_count == 0
        and scenario_params.budget is None
    )
    if needs_solver_run:
        if on_phase:
            on_phase("phase_milp")
        from ..models.solver import run_milp_solver

        milp_result = run_milp_solver(network, scenario_params)
        if milp_result.served_cps:
            scenario_params.served_cps = list(milp_result.served_cps)

    if milp_result.feasible and milp_result.assignments:
        scenario_params.milp_assignments = milp_result.assignments

    baseline_trucks = baseline_result.milp_result.trucks if baseline_result else None
    baseline_cost = baseline_result.milp_result.total_cost if baseline_result else None
    baseline_params = baseline_result.scenario_params if baseline_result else None
    baseline_milp = baseline_result.milp_result if baseline_result else None

    terminal_demand_totals = {
        tid: sum(network.demand.get(cp, {}).get(tid, 0.0) for cp in network.cp_ids)
        for tid in network.terminal_ids
    }
    cp_demands = {cp: sum(network.demand.get(cp, {}).values()) for cp in network.cp_ids}

    expert_future: Optional[concurrent.futures.Future[str]] = None
    if llm_insights:
        expert_future = _insight_executor.submit(
            run_expert_agent,
            expert_agent,
            milp_result,
            scenario_params,
            language,
            len(network.cp_ids),
            baseline_trucks,
            baseline_cost,
            terminal_demand_totals,
            dict(network.terminal_capacities),
            baseline_params,
            baseline_milp,
            list(network.cp_ids),
            dict(network.cp_capacities),
            cp_demands,
            network,
            coverage_comparison,
        )

    if on_phase:
        on_phase("phase_lbl")
    lbl_result = run_lane_by_lane(network, scenario_params)
    if on_phase:
        on_phase("phase_wct")
    wct_result = run_weighted_cycle_time(network, scenario_params)
    if on_phase:
        on_phase("phase_expert")
    insight = expert_future.result() if expert_future is not None else ""

    return PipelineResult(
        scenario_params=scenario_params,
        lbl_result=lbl_result,
        wct_result=wct_result,
        milp_result=milp_result,
        insight=insight,
        query_number=query_number,
    )


def build_baseline_params(network: NetworkData) -> ScenarioParams:
    """Build default baseline solver parameters from network data."""
    return ScenarioParams(
        payload=network.payload,
        speed_loaded=network.speed_loaded,
        speed_empty=network.speed_empty,
        availability=network.availability,
        overtime_hours=network.overtime_hours,
        overtime_cost=network.overtime_cost,
        variable_cost_per_km=network.variable_cost_per_km,
        fixed_cost_per_truck_month=network.fixed_cost_per_truck_month,
        working_days=network.working_days,
        net_driving_hours=network.net_driving_hours,
        terminals_active={t: True for t in network.terminal_ids},
        min_coverage_count=None,
        budget=None,
        objective="minimize_cost",
        volume_redistribution=False,
        is_baseline=True,
        served_cps=list(network.cp_ids),
        variable_cost_components=dict(network.variable_cost_components),
        fixed_cost_components=dict(network.fixed_cost_components),
    )
