"""MILP fleet optimization solver using Pyomo + HiGHS.

Lower bound — minimizes trucks, cost, or maximizes coverage subject to
constraints. All decision variables are resolved here.
"""

from __future__ import annotations

import math
import concurrent.futures

from pyomo.environ import (
    Binary,
    ConcreteModel,
    Constraint,
    NonNegativeIntegers,
    NonNegativeReals,
    Objective,
    Set,
    SolverFactory,
    Var,
    maximize,
    minimize,
    value,
)

from ..domain.data_types import MILPResult, ScenarioParams
from ..domain.loader import NetworkData

# One dedicated thread — isolates HiGHS from asyncio event-loop conflicts
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)


def _eff_demand(
    demand: dict[str, dict[str, float]],
    multipliers: dict[str, float],
    cp: str,
    t: str,
) -> float:
    """Return effective demand for a CP→terminal pair after applying any multiplier."""
    return demand[cp].get(t, 0.0) * multipliers.get(t, 1.0)


def run_milp_solver(network: NetworkData, params: ScenarioParams) -> MILPResult:
    """Run the MILP solver, dispatching to a background thread for safety."""
    future = _executor.submit(_run_single, network, params)
    return future.result()


def _run_single(network: NetworkData, params: ScenarioParams) -> MILPResult:
    """Build and solve a single MILP model."""
    cp_ids = network.cp_ids
    terminal_ids = network.terminal_ids
    active_terminals = {t for t, on in params.terminals_active.items() if on}

    if not active_terminals:
        return MILPResult(
            feasible=False,
            infeasibility_reason="No active terminals available.",
        )

    monthly_cap_avail = params.monthly_capacity * params.availability

    # Apply availability → maintenance cost sensitivity.
    # For each sensitive component, cost increases proportionally with availability above baseline.
    eff_variable_cost_per_km = params.variable_cost_per_km
    if network.availability_sensitivity:
        delta_pp = (params.availability - network.availability) * 100
        if abs(delta_pp) > 1e-6:
            for cost_key, sensitivity_per_pp in network.availability_sensitivity.items():
                baseline_val = network.variable_cost_components.get(cost_key, 0.0)
                current_val = params.variable_cost_components.get(cost_key, baseline_val)
                eff_variable_cost_per_km += current_val * delta_pp * sensitivity_per_pp

    # When working_days changes, terminals and CPs operate more (or fewer) days
    # too — scale their monthly capacity limits proportionally.
    days_scale = params.working_days / network.working_days

    ct: dict[tuple[str, str], float] = {}
    for cp in cp_ids:
        for t in terminal_ids:
            if t not in active_terminals:
                continue
            ct[(cp, t)] = params.cycle_time(
                network.distances[cp][t],
                network.cp_load_times[cp],
                network.terminal_unload_times[t],
            )

    m = ConcreteModel()
    m.CPs = Set(initialize=cp_ids)
    m.Terminals = Set(initialize=terminal_ids)
    m.serve = Var(m.CPs, domain=Binary)
    m.trucks = Var(domain=NonNegativeIntegers)

    if params.volume_redistribution:
        # ── Redistribution: optimal single-terminal assignment ────────────────
        # Each served CP consolidates 100% of its demand to one terminal.
        # The solver chooses which terminal to minimise cost while respecting
        # both CP and terminal capacity limits.
        total_cp_demand: dict[str, float] = {
            cp: sum(
                _eff_demand(network.demand, params.terminal_demand_multipliers, cp, t)
                for t in terminal_ids
            )
            for cp in cp_ids
        }
        active_pairs = [
            (cp, t) for cp in cp_ids for t in terminal_ids if t in active_terminals
        ]

        m.assign = Var(m.CPs, m.Terminals, domain=Binary)
        m.volume = Var(m.CPs, m.Terminals, domain=NonNegativeReals)

        for cp in cp_ids:
            for t in terminal_ids:
                if t not in active_terminals:
                    m.assign[cp, t].fix(0)
                    m.volume[cp, t].fix(0)

        # Exactly one terminal per served CP
        def c_assign_rule(m, cp):
            return sum(m.assign[cp, t] for t in terminal_ids) == m.serve[cp]
        m.c_assign = Constraint(m.CPs, rule=c_assign_rule)

        total_km = sum(
            m.volume[cp, t] / params.payload * 2 * network.distances[cp][t]
            for cp, t in active_pairs
        )
        fleet_rhs = sum(m.volume[cp, t] * ct[(cp, t)] for cp, t in active_pairs)

        if not params.skip_capacity_constraints:
            def c_term_cap_rule(m, t):
                if t not in active_terminals:
                    return Constraint.Skip
                return (
                    sum(m.volume[cp, t] for cp in cp_ids)
                    <= network.terminal_capacities[t] * days_scale
                )
            m.c_term_cap = Constraint(m.Terminals, rule=c_term_cap_rule)

        # Volume caps limit incoming volume per terminal; excess is redistributed (not dropped).
        if params.terminal_volume_caps:
            hist_vol_by_terminal = {
                t: sum(network.demand[cp].get(t, 0.0) for cp in cp_ids)
                for t in params.terminal_volume_caps
            }
            def c_term_vol_cap_rule(m, t):
                if t not in active_terminals or t not in params.terminal_volume_caps:
                    return Constraint.Skip
                return sum(m.volume[cp, t] for cp in cp_ids) <= hist_vol_by_terminal[t] * params.terminal_volume_caps[t]
            m.c_term_vol_cap = Constraint(m.Terminals, rule=c_term_vol_cap_rule)

        def c_vol_link_rule(m, cp, t):
            if (cp, t) not in active_pairs:
                return Constraint.Skip
            return m.volume[cp, t] <= total_cp_demand[cp] * m.assign[cp, t]
        m.c_vol_link = Constraint(m.CPs, m.Terminals, rule=c_vol_link_rule)

        # 100% of each CP's demand must flow to its assigned terminal.
        # Without this, the solver could under-route individual CPs while
        # keeping the global total intact, making terminal utilisation appear
        # infeasible in the display even when internal volumes are within bounds.
        def c_vol_cp_demand_rule(m, cp):
            vols = [m.volume[cp, t] for t in terminal_ids if (cp, t) in active_pairs]
            if not vols:
                return Constraint.Skip
            return sum(vols) == total_cp_demand[cp] * m.serve[cp]
        m.c_vol_cp_demand = Constraint(m.CPs, rule=c_vol_cp_demand_rule)

        if not params.skip_capacity_constraints:
            def c_vol_cp_cap_rule(m, cp):
                vols = [m.volume[cp, t] for t in terminal_ids if (cp, t) in active_pairs]
                if not vols:
                    return Constraint.Skip
                return sum(vols) <= network.cp_capacities[cp] * days_scale
            m.c_vol_cp_cap = Constraint(m.CPs, rule=c_vol_cp_cap_rule)


    else:
        # ── As-is baseline: multi-terminal historical routing ─────────────────
        # Each served CP routes demand on ALL its historical lanes, matching
        # the same as-is reality that Lane-by-Lane and WCT model. This gives
        # a meaningful baseline so redistribution can show genuine improvement.
        lane_pairs = [
            (cp, t) for cp in cp_ids for t in terminal_ids
            if t in active_terminals and network.demand[cp].get(t, 0.0) > 0
        ]

        total_km = sum(
            _eff_demand(network.demand, params.terminal_demand_multipliers, cp, t)
            * m.serve[cp] / params.payload * 2 * network.distances[cp][t]
            for cp, t in lane_pairs
        )
        fleet_rhs = sum(
            _eff_demand(network.demand, params.terminal_demand_multipliers, cp, t)
            * m.serve[cp] * ct[(cp, t)]
            for cp, t in lane_pairs
        )

        if not params.skip_capacity_constraints:
            def c_term_cap_rule(m, t):
                if t not in active_terminals:
                    return Constraint.Skip
                flow = sum(
                    _eff_demand(network.demand, params.terminal_demand_multipliers, cp, t)
                    * m.serve[cp]
                    for cp in cp_ids
                    if network.demand[cp].get(t, 0.0) > 0
                )
                return flow <= network.terminal_capacities[t] * days_scale
            m.c_term_cap = Constraint(m.Terminals, rule=c_term_cap_rule)

    # ── Shared: cost expression ───────────────────────────────────────────────
    total_cost = (
        m.trucks * params.fixed_cost_per_truck_month
        + total_km * eff_variable_cost_per_km
        + m.trucks * params.working_days * params.overtime_hours * params.overtime_cost
    )

    # ── Shared: fleet sizing ──────────────────────────────────────────────────
    m.c_fleet = Constraint(expr=m.trucks * monthly_cap_avail >= fleet_rhs)

    # ── Shared: objective ─────────────────────────────────────────────────────
    if params.objective == "maximize_coverage":
        m.obj = Objective(
            expr=sum(m.serve[cp] for cp in cp_ids) - 1e-4 * m.trucks,
            sense=maximize,
        )
    elif params.objective == "minimize_fleet":
        m.obj = Objective(expr=m.trucks, sense=minimize)
    else:
        m.obj = Objective(expr=total_cost, sense=minimize)

    # ── Shared: coverage ──────────────────────────────────────────────────────
    # For maximize_coverage the objective already drives CP count up — imposing
    # len(cp_ids) as the floor defeats the purpose and makes budget-constrained
    # queries always infeasible. Use 1 as the floor so the solver is free to
    # find the true maximum. For cost/fleet minimization, default to all CPs.
    if params.min_coverage_count is not None:
        min_cov = params.min_coverage_count
    elif params.objective == "maximize_coverage":
        min_cov = 1
    else:
        min_cov = len(cp_ids)
    m.c_coverage = Constraint(expr=sum(m.serve[cp] for cp in cp_ids) >= min_cov)

    # ── Shared: budget ────────────────────────────────────────────────────────
    if params.budget is not None:
        m.c_budget = Constraint(expr=total_cost <= params.budget)

    # ── Solve ─────────────────────────────────────────────────────────────────
    solver = SolverFactory("highs")
    solver.config.tee = False
    solver.options["output_flag"] = False
    solver.options["time_limit"] = 10
    solver.options["mip_rel_gap"] = 0.005
    solver.options["threads"] = 1
    try:
        result = solver.solve(m)
    except Exception as exc:
        exc_str = str(exc)
        exc_type = type(exc).__name__
        if "feasible" in exc_str.lower() or "NoFeasibleSolution" in exc_type:
            return MILPResult(
                feasible=False,
                infeasibility_reason="No feasible solution found for the given constraints.",
            )
        raise

    term_cond = str(getattr(getattr(result, "solver", None), "termination_condition", "optimal"))
    if "optimal" not in term_cond.lower() and "feasible" not in term_cond.lower():
        try:
            _ = value(m.trucks)
        except Exception:
            return MILPResult(
                feasible=False,
                infeasibility_reason=f"Solver status: {term_cond}",
            )

    # ── Extract results ───────────────────────────────────────────────────────
    trucks_val = max(0, int(round(value(m.trucks))))
    assignments: dict[str, str] = {}
    served_cps: list[str] = []

    volumes: dict[str, dict[str, float]] = {}
    if params.volume_redistribution:
        for cp in cp_ids:
            if value(m.serve[cp]) > 0.5:
                served_cps.append(cp)
                for t in terminal_ids:
                    if t in active_terminals and value(m.assign[cp, t]) > 0.5:
                        assignments[cp] = t
        # Populate volumes so the display matrix shows the redistribution result
        for cp in served_cps:
            assigned_t = assignments.get(cp)
            if assigned_t:
                vol = value(m.volume[cp, assigned_t])
                volumes[cp] = {assigned_t: round(vol, 2)}
        km_val = sum(
            value(m.volume[cp, t]) / params.payload * 2 * network.distances[cp][t]
            for cp, t in active_pairs
        )
    else:
        for cp in cp_ids:
            if value(m.serve[cp]) > 0.5:
                served_cps.append(cp)
                cp_vol = {
                    t: _eff_demand(network.demand, params.terminal_demand_multipliers, cp, t)
                    for (c, t) in lane_pairs if c == cp
                }
                if cp_vol:
                    volumes[cp] = cp_vol
        km_val = sum(
            _eff_demand(network.demand, params.terminal_demand_multipliers, cp, t)
            / params.payload * 2 * network.distances[cp][t]
            for cp, t in lane_pairs
            if value(m.serve[cp]) > 0.5
        )

    fixed_c = trucks_val * params.fixed_cost_per_truck_month
    var_c = km_val * eff_variable_cost_per_km
    ot_c = trucks_val * params.working_days * params.overtime_hours * params.overtime_cost

    terminal_overflows: dict[str, float] = {}
    cp_overflows: dict[str, float] = {}
    if params.skip_capacity_constraints:
        for t in terminal_ids:
            if t not in active_terminals:
                continue
            if params.volume_redistribution:
                eff = sum(
                    value(m.volume[cp, t])
                    for cp in cp_ids
                    if (cp, t) in active_pairs
                )
            else:
                eff = sum(
                    _eff_demand(network.demand, params.terminal_demand_multipliers, cp, t)
                    for cp in cp_ids
                    if cp in served_cps and network.demand[cp].get(t, 0.0) > 0
                )
            if eff > 0:
                terminal_overflows[t] = eff

        for cp in served_cps:
            if params.volume_redistribution:
                eff = sum(
                    value(m.volume[cp, t])
                    for t in terminal_ids
                    if (cp, t) in active_pairs
                )
            else:
                eff = sum(
                    _eff_demand(network.demand, params.terminal_demand_multipliers, cp, t)
                    for t in terminal_ids
                    if t in active_terminals and network.demand[cp].get(t, 0.0) > 0
                )
            if eff > 0:
                cp_overflows[cp] = eff

    return MILPResult(
        feasible=True,
        trucks=trucks_val,
        total_cost=round(fixed_c + var_c + ot_c, 2),
        total_km=round(km_val, 2),
        fixed_cost=round(fixed_c, 2),
        variable_cost=round(var_c, 2),
        overtime_cost_total=round(ot_c, 2),
        coverage_count=len(served_cps),
        served_cps=sorted(served_cps),
        assignments=assignments,
        volumes=volumes,
        terminal_overflows=terminal_overflows,
        cp_overflows=cp_overflows,
    )
