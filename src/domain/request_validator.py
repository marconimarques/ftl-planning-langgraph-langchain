"""Deterministic validation and solver-input construction."""

from __future__ import annotations

from .data_types import ScenarioParams
from .loader import NetworkData
from .workflow_types import NormalizedPlanningRequest, SolverInput, ValidationResult


ALLOWED_OBJECTIVES = {"minimize_cost", "maximize_coverage", "minimize_fleet"}


def validate_request(
    request: NormalizedPlanningRequest,
    network: NetworkData,
) -> tuple[ValidationResult, SolverInput | None]:
    errors: list[str] = []
    warnings: list[str] = []

    if request.requires_clarification:
        return (
            ValidationResult(
                valid=False,
                needs_clarification=True,
                clarification_question=request.clarification_question,
            ),
            None,
        )

    if request.objective not in ALLOWED_OBJECTIVES:
        errors.append(f"Unsupported objective: {request.objective}")

    known_terminals = set(network.terminal_ids)
    unknown_closed = sorted(set(request.closed_terminals) - known_terminals)
    if unknown_closed:
        errors.append(f"Unknown terminal(s): {', '.join(unknown_closed)}")

    if set(request.closed_terminals) >= known_terminals:
        errors.append("At least one terminal must remain active.")

    for label, value in (
        ("coverage_from_pct", request.coverage_from_pct),
        ("coverage_to_pct", request.coverage_to_pct),
    ):
        if value is not None and not 0 < value <= 100:
            errors.append(f"{label} must be greater than 0 and at most 100.")

    if (request.coverage_from_pct is None) ^ (request.coverage_to_pct is None):
        errors.append("Coverage comparisons require both source and target levels.")

    if request.budget is not None and request.budget < 0:
        errors.append("Budget must be non-negative.")

    for terminal_id, multiplier in request.terminal_demand_multipliers.items():
        if terminal_id not in known_terminals:
            errors.append(f"Unknown demand terminal: {terminal_id}")
        if multiplier < 0:
            errors.append(f"Demand multiplier for {terminal_id} must be non-negative.")

    for terminal_id, cap in request.terminal_volume_caps.items():
        if terminal_id not in known_terminals:
            errors.append(f"Unknown volume-cap terminal: {terminal_id}")
        if not 0 <= cap <= 1:
            errors.append(f"Volume cap for {terminal_id} must be between 0 and 1.")

    known_costs = set(network.variable_cost_components) | set(network.fixed_cost_components)
    unknown_costs = sorted(set(request.cost_component_overrides) - known_costs)
    if unknown_costs:
        errors.append(f"Unknown cost component(s): {', '.join(unknown_costs)}")

    if errors:
        return ValidationResult(valid=False, errors=errors, warnings=warnings), None

    params = ScenarioParams(
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
        terminals_active={t: t not in set(request.closed_terminals) for t in network.terminal_ids},
        min_coverage_count=request.min_coverage_count,
        budget=request.budget,
        objective=request.objective,
        volume_redistribution=request.volume_redistribution,
        served_cps=list(network.cp_ids),
        coverage_count_a=request.coverage_count_a,
        coverage_count_b=request.coverage_count_b,
        variable_cost_components=dict(network.variable_cost_components),
        fixed_cost_components=dict(network.fixed_cost_components),
        terminal_demand_multipliers=dict(request.terminal_demand_multipliers),
        terminal_volume_caps=dict(request.terminal_volume_caps),
    )

    for key, value in request.cost_component_overrides.items():
        if key in params.variable_cost_components:
            params.variable_cost_components[key] = value
            params.variable_cost_per_km = round(sum(params.variable_cost_components.values()), 4)
        elif key in params.fixed_cost_components:
            params.fixed_cost_components[key] = value
            params.fixed_cost_per_truck_month = round(sum(params.fixed_cost_components.values()), 2)

    request_type = (
        "coverage_comparison"
        if request.coverage_from_pct is not None and request.coverage_to_pct is not None
        else "single_scenario"
    )
    return (
        ValidationResult(valid=True, warnings=warnings),
        SolverInput(
            scenario_params=params,
            request_type=request_type,
            coverage_pct_from=request.coverage_from_pct,
            coverage_pct_to=request.coverage_to_pct,
        ),
    )
