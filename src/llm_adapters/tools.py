"""Compatibility solver helpers used by legacy LLM adapter paths."""

from __future__ import annotations

import json
from typing import Optional

from ..domain.data_types import ScenarioParams
from ..domain.loader import load_network_data


def _resolve_scenario_params(
    network,
    *,
    payload=None,
    speed_loaded=None,
    speed_empty=None,
    availability=None,
    overtime_hours=None,
    overtime_cost=None,
    variable_cost_per_km=None,
    fixed_cost_per_truck_month=None,
    working_days=None,
    net_driving_hours=None,
    closed_terminals="",
    var_cost_multipliers="",
    fix_cost_multipliers="",
    terminal_demand_multipliers="",
    terminal_volume_caps="",
):
    """Build ScenarioParams, applying cost multipliers to baseline components.

    Multipliers operate on baseline values from network data — not on
    explicitly passed variable_cost_per_km / fixed_cost_per_truck_month.
    Returns (params, var_components_if_changed, fix_components_if_changed).
    """
    var_components = dict(network.variable_cost_components)
    fix_components = dict(network.fixed_cost_components)

    any_var_mul = False
    if var_cost_multipliers:
        for key, mult in json.loads(var_cost_multipliers).items():
            if key in var_components:
                var_components[key] = round(var_components[key] * mult, 4)
                any_var_mul = True

    any_fix_mul = False
    if fix_cost_multipliers:
        for key, mult in json.loads(fix_cost_multipliers).items():
            if key in fix_components:
                fix_components[key] = round(fix_components[key] * mult, 2)
                any_fix_mul = True

    resolved_var = (
        round(sum(var_components.values()), 4)
        if any_var_mul
        else (variable_cost_per_km if variable_cost_per_km is not None else network.variable_cost_per_km)
    )
    resolved_fix = (
        round(sum(fix_components.values()), 2)
        if any_fix_mul
        else (fixed_cost_per_truck_month if fixed_cost_per_truck_month is not None else network.fixed_cost_per_truck_month)
    )

    terminals_active = {t: True for t in network.terminal_ids}
    if closed_terminals:
        for tid in (t.strip() for t in closed_terminals.split(",")):
            if tid in terminals_active:
                terminals_active[tid] = False

    params = ScenarioParams(
        payload=payload if payload is not None else network.payload,
        speed_loaded=speed_loaded if speed_loaded is not None else network.speed_loaded,
        speed_empty=speed_empty if speed_empty is not None else network.speed_empty,
        availability=availability if availability is not None else network.availability,
        overtime_hours=overtime_hours if overtime_hours is not None else network.overtime_hours,
        overtime_cost=overtime_cost if overtime_cost is not None else network.overtime_cost,
        variable_cost_per_km=resolved_var,
        fixed_cost_per_truck_month=resolved_fix,
        working_days=working_days if working_days is not None else network.working_days,
        net_driving_hours=net_driving_hours if net_driving_hours is not None else network.net_driving_hours,
        terminals_active=terminals_active,
        variable_cost_components=var_components if any_var_mul else {},
        fixed_cost_components=fix_components if any_fix_mul else {},
        terminal_demand_multipliers=json.loads(terminal_demand_multipliers) if terminal_demand_multipliers else {},
        terminal_volume_caps=json.loads(terminal_volume_caps) if terminal_volume_caps else {},
    )

    return params, (var_components if any_var_mul else {}), (fix_components if any_fix_mul else {})


def _result_to_dict(result) -> dict:
    return {
        "feasible": result.feasible,
        "trucks": result.trucks,
        "total_cost": result.total_cost,
        "total_km": result.total_km,
        "fixed_cost": result.fixed_cost,
        "variable_cost": result.variable_cost,
        "overtime_cost": result.overtime_cost_total,
        "coverage_count": result.coverage_count,
        "served_cps": result.served_cps,
        "assignments": result.assignments,
        "infeasibility_reason": result.infeasibility_reason,
    }


def run_milp_solver(
    payload: Optional[float] = None,
    speed_loaded: Optional[float] = None,
    speed_empty: Optional[float] = None,
    availability: Optional[float] = None,
    overtime_hours: Optional[float] = None,
    overtime_cost: Optional[float] = None,
    variable_cost_per_km: Optional[float] = None,
    fixed_cost_per_truck_month: Optional[float] = None,
    working_days: Optional[int] = None,
    net_driving_hours: Optional[float] = None,
    closed_terminals: str = "",
    min_coverage_count: Optional[int] = None,
    budget: Optional[float] = None,
    objective: str = "minimize_cost",
    volume_redistribution: bool = False,
    var_cost_multipliers: str = "",
    fix_cost_multipliers: str = "",
    terminal_demand_multipliers: str = "",
    terminal_volume_caps: str = "",
) -> str:
    """Run the MILP fleet optimization solver for a single scenario.

    Use this for any single-scenario optimization: baseline, what-if scenarios
    (payload, speed, availability, overtime, costs, terminal demand changes),
    coverage constraints, budget constraints, fleet minimization, terminal closure,
    or redistribution combined with parameter changes.
    For cost difference between two coverage levels → use compare_coverage_costs.
    For baseline redistribution only → use run_redistribution.

    Parameters
    ----------
    payload : float, optional
        Truck payload capacity in tons. Defaults to network baseline.
    speed_loaded : float, optional
        Average speed when carrying cargo in km/h. Defaults to network baseline.
    speed_empty : float, optional
        Average speed when returning empty in km/h. Defaults to network baseline.
    availability : float, optional
        Fraction of time trucks are operational, 0–1. Defaults to network baseline.
    overtime_hours : float, optional
        Extra driver hours per day beyond net driving hours. Defaults to network baseline.
    overtime_cost : float, optional
        Cost per overtime hour in USD. Defaults to network baseline.
    variable_cost_per_km : float, optional
        Total variable cost per km in USD. Defaults to network baseline.
        Ignored when var_cost_multipliers is set — multipliers always operate on baseline.
    fixed_cost_per_truck_month : float, optional
        Fixed cost per truck per month in USD. Defaults to network baseline.
        Ignored when fix_cost_multipliers is set — multipliers always operate on baseline.
    working_days : int, optional
        Working days per month. Defaults to network baseline.
    net_driving_hours : float, optional
        Net effective driving hours per day. Defaults to network baseline.
    closed_terminals : str
        Comma-separated terminal IDs to deactivate (e.g. "TB" or "TB,TC").
        Empty string means all terminals are active.
    min_coverage_count : int, optional
        Minimum number of collection points to serve. None = serve all CPs.
    budget : float, optional
        Maximum total monthly operational cost in USD.
    objective : str
        Solver objective: 'minimize_cost', 'maximize_coverage', or 'minimize_fleet'.
    volume_redistribution : bool
        Whether to allow volume redistribution across terminals.
    var_cost_multipliers : str
        JSON object mapping variable cost component keys to multipliers.
        Example: '{"tractor_maintenance": 1.10, "trailer_maintenance": 1.10}'.
        Python applies multipliers to baseline values. The result includes
        computed_variable_cost_components — copy those values to var_cost_* in your output.
    fix_cost_multipliers : str
        JSON object mapping fixed cost component keys to multipliers.
        Example: '{"tractor_depreciation": 1.05}'.
        Python applies multipliers to baseline values. The result includes
        computed_fixed_cost_components — copy those values to fix_cost_* in your output.
    terminal_demand_multipliers : str
        JSON object mapping terminal IDs to demand multipliers.
        Applied to ALL CP→terminal demand flows for that terminal.
        Example: '{"TC": 0.85}' for a 15% reduction at TC.
        '{"TA": 1.20}' for a 20% increase at TA.
    terminal_volume_caps : str
        JSON object mapping terminal IDs to volume cap fractions (0–1).
        Only used with volume_redistribution=True. Caps how much of a terminal's
        historical volume it may receive; the solver redirects the excess.
        Example: '{"TA": 0.85}' means TA receives at most 85% of its historical volume.
        Omit or pass '{}' for all queries except explicit redirect-between-terminals requests.

    Returns
    -------
    str
        JSON string with the MILP result. When multipliers are applied, also includes
        computed_variable_cost_components / computed_variable_cost_per_km and/or
        computed_fixed_cost_components / computed_fixed_cost_per_truck_month.
    """
    from ..models.solver import run_milp_solver as _run_solver

    network = load_network_data()
    params, var_comps, fix_comps = _resolve_scenario_params(
        network,
        payload=payload,
        speed_loaded=speed_loaded,
        speed_empty=speed_empty,
        availability=availability,
        overtime_hours=overtime_hours,
        overtime_cost=overtime_cost,
        variable_cost_per_km=variable_cost_per_km,
        fixed_cost_per_truck_month=fixed_cost_per_truck_month,
        working_days=working_days,
        net_driving_hours=net_driving_hours,
        closed_terminals=closed_terminals,
        var_cost_multipliers=var_cost_multipliers,
        fix_cost_multipliers=fix_cost_multipliers,
        terminal_demand_multipliers=terminal_demand_multipliers,
        terminal_volume_caps=terminal_volume_caps,
    )
    params.min_coverage_count = min_coverage_count
    params.budget = budget
    params.objective = objective
    params.volume_redistribution = volume_redistribution

    result = _run_solver(network, params)
    out = _result_to_dict(result)

    if var_comps:
        out["computed_variable_cost_components"] = var_comps
        out["computed_variable_cost_per_km"] = params.variable_cost_per_km
    if fix_comps:
        out["computed_fixed_cost_components"] = fix_comps
        out["computed_fixed_cost_per_truck_month"] = params.fixed_cost_per_truck_month

    return json.dumps(out)


def compare_coverage_costs(
    pct_from: float,
    pct_to: float,
    payload: Optional[float] = None,
    speed_loaded: Optional[float] = None,
    speed_empty: Optional[float] = None,
    availability: Optional[float] = None,
    overtime_hours: Optional[float] = None,
    overtime_cost: Optional[float] = None,
    variable_cost_per_km: Optional[float] = None,
    fixed_cost_per_truck_month: Optional[float] = None,
    working_days: Optional[int] = None,
    net_driving_hours: Optional[float] = None,
    closed_terminals: str = "",
    var_cost_multipliers: str = "",
    fix_cost_multipliers: str = "",
    terminal_demand_multipliers: str = "",
) -> str:
    """Compare the minimum cost at two coverage levels and return the cost difference.

    Use this when the user asks 'What does it cost to go from X% to Y% coverage?'
    or any question about the cost difference between two coverage levels.
    Always preserve the user's direction: X is pct_from and Y is pct_to.
    The returned delta is pct_to minus pct_from, so reductions can be negative.
    Python converts percentages to CP counts using ceil(pct/100 × n_cps) and runs
    the solver twice (minimize_cost at each level). No manual arithmetic needed.

    Parameters
    ----------
    pct_from : float
        Starting coverage percentage from the user's question (e.g. 100 for "from 100%").
    pct_to : float
        Target coverage percentage from the user's question (e.g. 65 for "to 65%").
    payload : float, optional
        Truck payload capacity in tons. Defaults to network baseline.
    speed_loaded : float, optional
        Average speed loaded in km/h. Defaults to network baseline.
    speed_empty : float, optional
        Average speed empty in km/h. Defaults to network baseline.
    availability : float, optional
        Truck availability fraction 0–1. Defaults to network baseline.
    overtime_hours : float, optional
        Extra driver hours per day. Defaults to network baseline.
    overtime_cost : float, optional
        Cost per overtime hour in USD. Defaults to network baseline.
    variable_cost_per_km : float, optional
        Total variable cost per km in USD. Defaults to network baseline.
    fixed_cost_per_truck_month : float, optional
        Fixed cost per truck per month in USD. Defaults to network baseline.
    working_days : int, optional
        Working days per month. Defaults to network baseline.
    net_driving_hours : float, optional
        Net driving hours per day. Defaults to network baseline.
    closed_terminals : str
        Comma-separated terminal IDs to deactivate. Empty = all active.
    var_cost_multipliers : str
        JSON object mapping variable cost component keys to multipliers.
    fix_cost_multipliers : str
        JSON object mapping fixed cost component keys to multipliers.
    terminal_demand_multipliers : str
        JSON object mapping terminal IDs to demand multipliers.

    Returns
    -------
    str
        JSON with coverage_count_a/b, cost_a, cost_b, cost_difference, trucks_a/b,
        and all target-level result fields (trucks, total_cost, total_km, fixed_cost,
        variable_cost, overtime_cost, coverage_count, served_cps, assignments).
        Copy these directly to milp_result and scenario_params in your output.
    """
    from ..workflow.solver_service import run_coverage_comparison

    network = load_network_data()
    params, _, _ = _resolve_scenario_params(
        network,
        payload=payload,
        speed_loaded=speed_loaded,
        speed_empty=speed_empty,
        availability=availability,
        overtime_hours=overtime_hours,
        overtime_cost=overtime_cost,
        variable_cost_per_km=variable_cost_per_km,
        fixed_cost_per_truck_month=fixed_cost_per_truck_month,
        working_days=working_days,
        net_driving_hours=net_driving_hours,
        closed_terminals=closed_terminals,
        var_cost_multipliers=var_cost_multipliers,
        fix_cost_multipliers=fix_cost_multipliers,
        terminal_demand_multipliers=terminal_demand_multipliers,
    )

    cov = run_coverage_comparison(network, params, pct_from, pct_to)

    if not cov.feasible:
        reasons: list[str] = []
        if not cov.result_a.feasible:
            reasons.append(cov.result_a.infeasibility_reason)
        if not cov.result_b.feasible:
            reasons.append(cov.result_b.infeasibility_reason)
        return json.dumps({"feasible": False, "infeasibility_reason": "; ".join(filter(None, reasons))})

    rb = cov.result_b
    return json.dumps({
        "feasible": True,
        "coverage_count_a": cov.coverage_count_a,
        "coverage_count_b": cov.coverage_count_b,
        "cost_a": cov.cost_a,
        "cost_b": cov.cost_b,
        "cost_difference": cov.cost_difference,
        "trucks_a": cov.trucks_a,
        "trucks_b": cov.trucks_b,
        # Target-level (B) result — copy to milp_result fields
        "trucks": rb.trucks,
        "total_cost": rb.total_cost,
        "total_km": rb.total_km,
        "fixed_cost": rb.fixed_cost,
        "variable_cost": rb.variable_cost,
        "overtime_cost": rb.overtime_cost_total,
        "coverage_count": rb.coverage_count,
        "served_cps": rb.served_cps,
        "assignments": rb.assignments,
        "infeasibility_reason": "",
    })


def run_redistribution() -> str:
    """Run the solver with volume redistribution using baseline network parameters.

    Use this when the user asks 'Is there a gain from redistributing volumes?',
    'What if volume is optimally redistributed across terminals?', or any query
    about redistribution gain without other parameter changes.
    Always uses baseline operational parameters — Python enforces this regardless
    of prior queries. For redistribution combined with parameter changes, use
    run_milp_solver with volume_redistribution=True instead.

    Returns
    -------
    str
        JSON string with the MILP result for the redistribution scenario.
    """
    from ..models.solver import run_milp_solver as _run_solver

    network = load_network_data()
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
        terminals_active={t: True for t in network.terminal_ids},
        volume_redistribution=True,
        objective="minimize_cost",
    )
    result = _run_solver(network, params)
    return json.dumps(_result_to_dict(result))


def load_network_data_tool() -> str:
    """Load and return the network data (collection points, terminals, distances, demand).

    Returns a JSON summary of the network for the agent's context.

    Returns
    -------
    str
        JSON string with network summary.
    """
    network = load_network_data()

    demand_summary = {
        cp: {t: network.demand[cp][t] for t in network.terminal_ids}
        for cp in network.cp_ids
    }

    return json.dumps(
        {
            "collection_points": {
                cp: {
                    "name": network.cp_names[cp],
                    "capacity_tons_month": network.cp_capacities[cp],
                    "load_time_hrs": network.cp_load_times[cp],
                }
                for cp in network.cp_ids
            },
            "terminals": {
                t: {
                    "name": network.terminal_names[t],
                    "capacity_tons_month": network.terminal_capacities[t],
                    "unload_time_hrs": network.terminal_unload_times[t],
                }
                for t in network.terminal_ids
            },
            "distances_km": {
                cp: network.distances[cp] for cp in network.cp_ids
            },
            "monthly_demand_tons": demand_summary,
            "defaults": {
                "speed_loaded_kmh": network.speed_loaded,
                "speed_empty_kmh": network.speed_empty,
                "payload_tons": network.payload,
                "availability": network.availability,
                "variable_cost_per_km": network.variable_cost_per_km,
                "fuel_cost_per_km": network.fuel_cost_per_km,
                "fixed_cost_per_truck_month": network.fixed_cost_per_truck_month,
                "net_driving_hours": network.net_driving_hours,
                "overtime_hours": network.overtime_hours,
                "overtime_cost_per_hour": network.overtime_cost,
                "working_days_per_month": network.working_days,
            },
            "cost_components": {
                "variable_per_km": dict(network.variable_cost_components),
                "fixed_per_truck_month": dict(network.fixed_cost_components),
            },
            "lever_limits": network.lever_limits,
        }
    )
