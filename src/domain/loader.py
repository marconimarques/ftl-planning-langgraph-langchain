"""Network data loader — reads all 5 Excel files into a structured layer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent.parent.parent / "data"

_cache: "NetworkData | None" = None


@dataclass
class NetworkData:
    """Raw structured data loaded from the Excel files."""

    cp_ids: list[str]
    cp_names: dict[str, str]
    cp_capacities: dict[str, float]
    cp_load_times: dict[str, float]

    terminal_ids: list[str]
    terminal_names: dict[str, str]
    terminal_capacities: dict[str, float]
    terminal_unload_times: dict[str, float]

    distances: dict[str, dict[str, float]]
    demand: dict[str, dict[str, float]]

    speed_loaded: float
    speed_empty: float
    payload: float
    availability: float

    variable_cost_per_km: float
    fuel_cost_per_km: float
    fixed_cost_per_truck_month: float

    # Individual cost components preserved from Excel (ordered dicts)
    variable_cost_components: dict[str, float]   # cost_type → $/km
    fixed_cost_components: dict[str, float]       # cost_type → $/truck/month

    net_driving_hours: float
    overtime_hours: float
    overtime_cost: float
    working_days: int

    # Lever limits loaded from data/lever_limits.xlsx (empty dict if file absent)
    lever_limits: dict

    # Availability → maintenance cost sensitivity: cost_type → fractional increase per 1 pp of availability.
    # E.g. 0.005 means +0.5% cost per +1 pp availability above baseline.
    availability_sensitivity: dict[str, float]

    def total_cp_demand(self, cp_id: str) -> float:
        return sum(self.demand[cp_id].values())

    def total_network_demand(self) -> float:
        return sum(
            v for cp in self.demand.values() for v in cp.values()
        )


def load_network_data(force_reload: bool = False) -> NetworkData:
    global _cache
    if _cache is not None and not force_reload:
        return _cache

    cps_df = pd.read_excel(DATA_DIR / "nodes.xlsx", sheet_name="collection_points")
    terms_df = pd.read_excel(DATA_DIR / "nodes.xlsx", sheet_name="terminals")

    cp_ids: list[str] = list(cps_df["id"])
    cp_names = dict(zip(cps_df["id"], cps_df["name"]))
    cp_capacities = dict(zip(cps_df["id"], cps_df["capacity_tons_month"].astype(float)))
    cp_load_times = dict(zip(cps_df["id"], cps_df["load_time_hrs"].astype(float)))

    terminal_ids: list[str] = list(terms_df["id"])
    terminal_names = dict(zip(terms_df["id"], terms_df["name"]))
    terminal_capacities = dict(
        zip(terms_df["id"], terms_df["capacity_tons_month"].astype(float))
    )
    terminal_unload_times = dict(
        zip(terms_df["id"], terms_df["unload_time_hrs"].astype(float))
    )

    dist_df = pd.read_excel(DATA_DIR / "distances.xlsx", sheet_name="cp_to_terminal")
    distances: dict[str, dict[str, float]] = {cp: {} for cp in cp_ids}
    for _, row in dist_df.iterrows():
        distances[row["from_cp_id"]][row["to_terminal_id"]] = float(row["distance_km"])

    demand_df = pd.read_excel(DATA_DIR / "cargo_demand.xlsx", sheet_name="monthly_demand")
    demand: dict[str, dict[str, float]] = {cp: {} for cp in cp_ids}
    for _, row in demand_df.iterrows():
        demand[row["collection_point_id"]][row["terminal_id"]] = float(row["tons_per_month"])

    specs_df = pd.read_excel(DATA_DIR / "truck_specs.xlsx", sheet_name="parameters")
    specs = dict(zip(specs_df["parameter"], specs_df["value"].astype(float)))

    var_df = pd.read_excel(DATA_DIR / "truck_costs.xlsx", sheet_name="variable_costs")
    fix_df = pd.read_excel(DATA_DIR / "truck_costs.xlsx", sheet_name="fixed_costs")
    variable_cost_components: dict[str, float] = dict(
        zip(var_df["cost_type"], var_df["cost_per_km"].astype(float))
    )
    fixed_cost_components: dict[str, float] = dict(
        zip(fix_df["cost_type"], fix_df["cost_per_month"].astype(float))
    )
    variable_cost_per_km = sum(variable_cost_components.values())
    fuel_cost_per_km = variable_cost_components.get("fuel", 0.0)
    fixed_cost_per_truck_month = sum(fixed_cost_components.values())

    policy_df = pd.read_excel(DATA_DIR / "driver_policy.xlsx", sheet_name="policy")
    policy = dict(zip(policy_df["parameter"], policy_df["value"].astype(float)))

    try:
        avail_sens_df = pd.read_excel(DATA_DIR / "truck_costs.xlsx", sheet_name="availability_sensitivity")
        availability_sensitivity: dict[str, float] = dict(
            zip(avail_sens_df["cost_type"], avail_sens_df["sensitivity_per_pp"].astype(float))
        )
    except (ValueError, KeyError):
        availability_sensitivity = {}

    try:
        ops_df = pd.read_excel(DATA_DIR / "lever_limits.xlsx", sheet_name="operational_limits")
        sav_df = pd.read_excel(DATA_DIR / "lever_limits.xlsx", sheet_name="cost_savings")
        _ops_has_threshold = "moderate_threshold" in ops_df.columns
        _sav_has_threshold = "moderate_threshold" in sav_df.columns
        lever_limits: dict = {
            "operational": {
                row["param"]: {
                    "min": float(row["min_value"]),
                    "max": float(row["max_value"]),
                    "unit": str(row["unit"]),
                    "moderate_threshold": float(row["moderate_threshold"]) if _ops_has_threshold else 0.67,
                }
                for _, row in ops_df.iterrows()
            },
            "cost_savings": {
                row["cost_type"]: {
                    "category": str(row["category"]),
                    "max_saving_pct": float(row["max_saving_pct"]),
                    "moderate_threshold": float(row["moderate_threshold"]) if _sav_has_threshold else 0.67,
                }
                for _, row in sav_df.iterrows()
            },
        }
    except (FileNotFoundError, ValueError, KeyError):
        lever_limits = {}

    _cache = NetworkData(
        cp_ids=cp_ids,
        cp_names=cp_names,
        cp_capacities=cp_capacities,
        cp_load_times=cp_load_times,
        terminal_ids=terminal_ids,
        terminal_names=terminal_names,
        terminal_capacities=terminal_capacities,
        terminal_unload_times=terminal_unload_times,
        distances=distances,
        demand=demand,
        speed_loaded=float(specs["speed_loaded"]),
        speed_empty=float(specs["speed_empty"]),
        payload=float(specs["payload"]),
        availability=float(specs["availability"]),
        variable_cost_per_km=variable_cost_per_km,
        fuel_cost_per_km=fuel_cost_per_km,
        fixed_cost_per_truck_month=fixed_cost_per_truck_month,
        variable_cost_components=variable_cost_components,
        fixed_cost_components=fixed_cost_components,
        net_driving_hours=float(policy["net_driving_hours"]),
        overtime_hours=float(policy["overtime_allowed"]),
        overtime_cost=float(policy["overtime_cost"]),
        working_days=int(policy["working_days_per_month"]),
        lever_limits=lever_limits,
        availability_sensitivity=availability_sensitivity,
    )
    return _cache
