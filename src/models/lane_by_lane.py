"""Lane-by-Lane fleet sizing model.

Upper bound — most conservative. Each CP×terminal lane gets its own
dedicated fleet. Total fleet = sum across all active lanes.
"""

from __future__ import annotations

import math

from ..domain.data_types import LaneResult, ModelResult, ScenarioParams
from ..domain.loader import NetworkData


def run_lane_by_lane(network: NetworkData, params: ScenarioParams) -> ModelResult:
    """Compute lane-by-lane fleet sizing for the given scenario parameters."""
    monthly_cap = params.monthly_capacity

    active_cps = set(params.served_cps) if params.served_cps else set(network.cp_ids)
    active_terminals = {t for t, on in params.terminals_active.items() if on}

    lane_results: list[LaneResult] = []

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
            demand = sum(
                v * params.terminal_demand_multipliers.get(t_id, 1.0)
                for t_id, v in network.demand[cp_id].items()
            )
            lanes_to_process = [(assigned_t, demand)]
        else:
            lanes_to_process = [
                (t_id, network.demand[cp_id].get(t_id, 0.0) * params.terminal_demand_multipliers.get(t_id, 1.0))
                for t_id in network.terminal_ids
                if t_id in active_terminals
                and network.demand[cp_id].get(t_id, 0.0) > 0
            ]

        for t_id, demand in lanes_to_process:
            dist = network.distances[cp_id][t_id]
            load_t = network.cp_load_times[cp_id]
            unload_t = network.terminal_unload_times[t_id]
            ct = params.cycle_time(dist, load_t, unload_t)

            operational = math.ceil(demand * ct / monthly_cap)
            trucks = math.ceil(operational / params.availability)
            trips = math.ceil(demand / params.payload)
            km = trips * 2 * dist

            lane_results.append(
                LaneResult(
                    cp_id=cp_id,
                    terminal_id=t_id,
                    distance_km=dist,
                    cycle_time_hours=ct,
                    monthly_demand_tons=demand,
                    trucks_needed=trucks,
                    trips_per_month=trips,
                    total_km_month=km,
                )
            )

    total_trucks = sum(r.trucks_needed for r in lane_results)
    total_km = sum(r.total_km_month for r in lane_results)
    fixed_cost = total_trucks * params.fixed_cost_per_truck_month
    variable_cost = total_km * params.variable_cost_per_km
    ot_cost = (
        total_trucks * params.working_days * params.overtime_hours * params.overtime_cost
    )

    return ModelResult(
        model_name="Lane-by-Lane",
        trucks=total_trucks,
        total_km=total_km,
        fixed_cost=fixed_cost,
        variable_cost=variable_cost,
        overtime_cost_total=ot_cost,
        total_cost=fixed_cost + variable_cost + ot_cost,
        lane_results=lane_results,
    )
