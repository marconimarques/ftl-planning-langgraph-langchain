"""Build immutable fact packets for explanation and audit."""

from __future__ import annotations

from .data_types import PipelineResult
from .loader import NetworkData
from .workflow_types import ScenarioFactPacket


def build_fact_packet(
    result: PipelineResult,
    network: NetworkData,
    baseline: PipelineResult | None = None,
) -> ScenarioFactPacket:
    params = result.scenario_params
    milp = result.milp_result
    total_cps = len(network.cp_ids)

    if params.is_baseline:
        label = "Baseline"
    elif params.volume_redistribution:
        label = "Relocation"
    elif params.coverage_count_a is not None and params.coverage_count_b is not None:
        label = "Coverage comparison"
    else:
        label = "What-if"

    if not milp.feasible:
        return ScenarioFactPacket(
            scenario_label=label,
            objective=params.objective,
            feasible=False,
            active_constraints=tuple(_active_constraints(params)),
            total_cost=None,
            total_cost_delta=None,
            trucks=None,
            trucks_delta=None,
            coverage_count=None,
            coverage_text="Infeasible",
            infeasibility_reason=milp.infeasibility_reason,
        )

    if params.coverage_count_a is not None and params.coverage_count_b is not None:
        coverage_text = (
            f"{params.coverage_count_a} to {params.coverage_count_b} of {total_cps} CPs; "
            f"delta cost {milp.cost_difference}"
        )
    else:
        coverage_text = f"{milp.coverage_count} of {total_cps} CPs served"

    baseline_cost = baseline.milp_result.total_cost if baseline else None
    baseline_trucks = baseline.milp_result.trucks if baseline else None
    total_cost_delta = None if baseline_cost is None else milp.total_cost - baseline_cost
    trucks_delta = None if baseline_trucks is None else milp.trucks - baseline_trucks

    assignment_facts = tuple(
        f"{cp}->{terminal}" for cp, terminal in sorted(milp.assignments.items())
    )
    terminal_facts = tuple(
        f"{terminal}: {sum(milp.volumes.get(cp, {}).get(terminal, 0.0) for cp in milp.volumes):.2f}"
        for terminal in network.terminal_ids
        if any(terminal in lanes for lanes in milp.volumes.values())
    )

    return ScenarioFactPacket(
        scenario_label=label,
        objective=params.objective,
        feasible=True,
        active_constraints=tuple(_active_constraints(params)),
        total_cost=milp.total_cost,
        total_cost_delta=total_cost_delta,
        trucks=milp.trucks,
        trucks_delta=trucks_delta,
        coverage_count=milp.coverage_count,
        coverage_text=coverage_text,
        terminal_capacity_facts=terminal_facts,
        assignment_facts=assignment_facts,
        allowed_explanations=(
            "Mention cost, fleet, coverage, active terminals, assignments, and capacity facts present here.",
        ),
        forbidden_explanations=(
            "Do not invent operational causes that are absent from computed facts.",
            "Do not recompute or change any numeric value.",
        ),
    )


def _active_constraints(params) -> list[str]:
    constraints: list[str] = []
    if any(not active for active in params.terminals_active.values()):
        closed = [terminal for terminal, active in params.terminals_active.items() if not active]
        constraints.append("closed_terminals=" + ",".join(closed))
    if params.min_coverage_count is not None:
        constraints.append(f"min_coverage_count={params.min_coverage_count}")
    if params.budget is not None:
        constraints.append(f"budget={params.budget}")
    if params.volume_redistribution:
        constraints.append("volume_redistribution=true")
    if params.terminal_demand_multipliers:
        constraints.append("terminal_demand_multipliers")
    if params.terminal_volume_caps:
        constraints.append("terminal_volume_caps")
    return constraints
