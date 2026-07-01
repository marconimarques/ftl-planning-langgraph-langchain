"""Weighted Cycle Time fleet sizing model.

Middle estimate — sizes a single shared fleet for all active lanes using
a demand-weighted average cycle time.
"""

from __future__ import annotations

import math

from ..domain.data_types import ModelResult, ScenarioParams
from ..domain.loader import NetworkData


def run_weighted_cycle_time(network: NetworkData, params: ScenarioParams) -> ModelResult:
    """Compute weighted-cycle-time fleet sizing for the given scenario parameters."""
    monthly_cap = params.monthly_capacity

    active_cps = set(params.served_cps) if params.served_cps else set(network.cp_ids)
    active_terminals = {t for t, on in params.terminals_active.items() if on}

    total_demand = 0.0
    weighted_num = 0.0
    total_km = 0.0

    use_redistribution = (
        params.volume_redistribution and bool(params.milp_assignments)
    )

    for cp_id in network.cp_ids:
        if cp_id not in active_cps:
            continue

        if use_redistribution:
            # One lane per CP: all demand flows to the MILP-assigned terminal.
            assigned_t = params.milp_assignments.get(cp_id)
            if not assigned_t or assigned_t not in active_terminals:
                continue
            lanes = [(assigned_t, sum(
                v * params.terminal_demand_multipliers.get(t_id, 1.0)
                for t_id, v in network.demand[cp_id].items()
            ))]
        else:
            lanes = [
                (t_id, network.demand[cp_id].get(t_id, 0.0) * params.terminal_demand_multipliers.get(t_id, 1.0))
                for t_id in network.terminal_ids
                if t_id in active_terminals
                and network.demand[cp_id].get(t_id, 0.0) > 0
            ]

        for t_id, demand in lanes:
            dist = network.distances[cp_id][t_id]
            load_t = network.cp_load_times[cp_id]
            unload_t = network.terminal_unload_times[t_id]
            ct = params.cycle_time(dist, load_t, unload_t)
            trips = math.ceil(demand / params.payload)

            total_demand += demand
            weighted_num += demand * ct
            total_km += trips * 2 * dist

    if total_demand <= 0:
        return ModelResult(
            model_name="Weighted Cycle Time",
            trucks=0,
            total_km=0.0,
            fixed_cost=0.0,
            variable_cost=0.0,
            overtime_cost_total=0.0,
            total_cost=0.0,
        )

    weighted_cycle = weighted_num / total_demand
    operational = math.ceil(total_demand * weighted_cycle / monthly_cap)
    total_trucks = math.ceil(operational / params.availability)

    fixed_cost = total_trucks * params.fixed_cost_per_truck_month
    variable_cost = total_km * params.variable_cost_per_km
    ot_cost = (
        total_trucks * params.working_days * params.overtime_hours * params.overtime_cost
    )

    return ModelResult(
        model_name="Weighted Cycle Time",
        trucks=total_trucks,
        total_km=total_km,
        fixed_cost=fixed_cost,
        variable_cost=variable_cost,
        overtime_cost_total=ot_cost,
        total_cost=fixed_cost + variable_cost + ot_cost,
        weighted_cycle_time=weighted_cycle,
    )
