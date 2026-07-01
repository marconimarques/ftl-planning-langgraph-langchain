"""Deterministic consistency checks for computed planning outputs."""

from __future__ import annotations

from .data_types import MILPResult, PipelineResult, ScenarioParams
from .loader import NetworkData
from .workflow_types import ConsistencyCheckResult


def check_result_consistency(
    result: PipelineResult | MILPResult,
    params: ScenarioParams,
    network: NetworkData,
    *,
    tolerance: float = 1.0,
) -> ConsistencyCheckResult:
    milp = result.milp_result if isinstance(result, PipelineResult) else result
    failures: list[str] = []
    warnings: list[str] = []

    if not milp.feasible:
        if not milp.infeasibility_reason:
            failures.append("Infeasible result is missing an infeasibility reason.")
        return ConsistencyCheckResult(passed=not failures, failures=failures, warnings=warnings)

    cost_sum = milp.fixed_cost + milp.variable_cost + milp.overtime_cost_total
    if abs(cost_sum - milp.total_cost) > tolerance:
        failures.append("MILP total cost does not match fixed + variable + overtime cost.")

    if milp.coverage_count != len(milp.served_cps):
        failures.append("MILP coverage count does not match served CP count.")

    active_terminals = {terminal for terminal, active in params.terminals_active.items() if active}
    invalid_assignments = sorted(
        terminal for terminal in milp.assignments.values() if terminal not in active_terminals
    )
    if invalid_assignments:
        failures.append(f"Assignments use inactive terminals: {', '.join(invalid_assignments)}")

    unknown_cps = sorted(set(milp.served_cps) - set(network.cp_ids))
    if unknown_cps:
        failures.append(f"Result includes unknown CPs: {', '.join(unknown_cps)}")

    if milp.cost_difference is not None and milp.cost_a is not None and milp.cost_b is not None:
        if abs((milp.cost_b - milp.cost_a) - milp.cost_difference) > tolerance:
            failures.append("Coverage comparison delta is not target cost minus source cost.")

    if params.coverage_count_a is not None and params.coverage_count_b is not None:
        if milp.trucks_a is None or milp.trucks_b is None or milp.cost_difference is None:
            failures.append("Two-level coverage result is missing comparison fields.")

    return ConsistencyCheckResult(passed=not failures, failures=failures, warnings=warnings)
