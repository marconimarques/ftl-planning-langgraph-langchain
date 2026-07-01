"""Deterministic bridge from workflow contracts to model services."""

from __future__ import annotations

import dataclasses
import math

from ..domain.data_types import CoverageComparisonResult, MILPResult, ScenarioParams
from ..domain.loader import NetworkData
from ..domain.workflow_types import SolverInput
from ..models.solver import run_milp_solver


def run_coverage_comparison(
    network: NetworkData,
    params: ScenarioParams,
    pct_from: float,
    pct_to: float,
) -> CoverageComparisonResult:
    """Single source of truth for two-level coverage comparisons."""
    n_cps = len(network.cp_ids)
    count_from = math.ceil(pct_from / 100 * n_cps)
    count_to = math.ceil(pct_to / 100 * n_cps)

    params_a = dataclasses.replace(
        params,
        objective="minimize_cost",
        coverage_count_a=count_from,
        coverage_count_b=count_to,
        min_coverage_count=count_from,
    )
    params_b = dataclasses.replace(params_a, min_coverage_count=count_to)
    result_a = run_milp_solver(network, params_a)
    result_b = run_milp_solver(network, params_b)

    return CoverageComparisonResult(
        result_a=result_a,
        result_b=result_b,
        coverage_count_a=count_from,
        coverage_count_b=count_to,
    )


def run_primary_solver(
    network: NetworkData, solver_input: SolverInput
) -> "MILPResult | CoverageComparisonResult":
    """Run the primary deterministic solver path for a validated request."""
    params = solver_input.scenario_params
    if solver_input.request_type != "coverage_comparison":
        return run_milp_solver(network, params)

    if solver_input.coverage_pct_from is None or solver_input.coverage_pct_to is None:
        return MILPResult(
            feasible=False,
            infeasibility_reason="Coverage comparison requires source and target levels.",
        )

    return run_coverage_comparison(
        network, params,
        solver_input.coverage_pct_from,
        solver_input.coverage_pct_to,
    )
