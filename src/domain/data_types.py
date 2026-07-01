"""Shared data types for the truck fleet planning system."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ScenarioParams:
    """Parameters for a scenario run — resolved by the OR Agent."""

    payload: float = 30.0
    speed_loaded: float = 45.0
    speed_empty: float = 60.0
    availability: float = 0.85
    overtime_hours: float = 0.0
    overtime_cost: float = 1370.0
    variable_cost_per_km: float = 3.97
    fixed_cost_per_truck_month: float = 23075.0
    working_days: int = 24
    net_driving_hours: float = 9.0
    terminals_active: dict[str, bool] = field(default_factory=dict)
    min_coverage_count: Optional[int] = None
    budget: Optional[float] = None
    objective: str = "minimize_cost"
    volume_redistribution: bool = False
    is_baseline: bool = False
    served_cps: list[str] = field(default_factory=list)
    coverage_count_a: Optional[int] = None
    coverage_count_b: Optional[int] = None
    milp_assignments: dict[str, str] = field(default_factory=dict)
    # Individual cost components — populated when the agent changes specific items.
    # When non-empty, variable_cost_per_km / fixed_cost_per_truck_month equal their sums.
    variable_cost_components: dict[str, float] = field(default_factory=dict)
    fixed_cost_components: dict[str, float] = field(default_factory=dict)
    # Per-terminal demand multipliers: terminal_id → multiplier (1.0 = no change).
    # Applied to all CP→terminal demand flows for that terminal.
    terminal_demand_multipliers: dict[str, float] = field(default_factory=dict)
    # Per-terminal volume caps (redistribution mode only): terminal_id → fraction of
    # historical volume the terminal may receive (e.g. 0.85 = at most 85%).
    # The solver redistributes excess demand to other terminals instead of discarding it.
    terminal_volume_caps: dict[str, float] = field(default_factory=dict)
    # When True, solver runs without c_term_cap / c_vol_cp_cap constraints to
    # produce an action map rather than a feasibility-constrained plan.
    skip_capacity_constraints: bool = False

    @property
    def effective_hours(self) -> float:
        return self.net_driving_hours + self.overtime_hours

    @property
    def monthly_capacity(self) -> float:
        return self.effective_hours * self.working_days * self.payload

    def cycle_time(self, distance_km: float, load_time: float, unload_time: float) -> float:
        return (
            load_time
            + distance_km / self.speed_loaded
            + unload_time
            + distance_km / self.speed_empty
        )


@dataclass
class LaneResult:
    """Per-lane result for Lane-by-Lane model."""

    cp_id: str
    terminal_id: str
    distance_km: float
    cycle_time_hours: float
    monthly_demand_tons: float
    trucks_needed: int
    trips_per_month: int
    total_km_month: float


@dataclass
class ModelResult:
    """Aggregate result from Lane-by-Lane or Weighted Cycle Time."""

    model_name: str
    trucks: int
    total_km: float
    fixed_cost: float
    variable_cost: float
    overtime_cost_total: float
    total_cost: float
    lane_results: list[LaneResult] = field(default_factory=list)
    weighted_cycle_time: Optional[float] = None


@dataclass
class MILPResult:
    """Result from the MILP solver."""

    feasible: bool
    trucks: int = 0
    total_cost: float = 0.0
    total_km: float = 0.0
    fixed_cost: float = 0.0
    variable_cost: float = 0.0
    overtime_cost_total: float = 0.0
    coverage_count: int = 0
    served_cps: list[str] = field(default_factory=list)
    assignments: dict[str, str] = field(default_factory=dict)
    volumes: dict[str, dict[str, float]] = field(default_factory=dict)
    infeasibility_reason: str = ""
    trucks_a: Optional[int] = None
    trucks_b: Optional[int] = None
    cost_a: Optional[float] = None
    cost_b: Optional[float] = None
    cost_difference: Optional[float] = None
    coverage_count_a: Optional[int] = None
    coverage_count_b: Optional[int] = None
    # Populated only when skip_capacity_constraints=True: terminal_id → effective demand routed
    terminal_overflows: dict[str, float] = field(default_factory=dict)
    # Populated only when skip_capacity_constraints=True: cp_id → effective demand routed
    cp_overflows: dict[str, float] = field(default_factory=dict)


@dataclass
class CoverageComparisonResult:
    """Result wrapper for a two-level coverage comparison (two solver runs).

    Created by run_coverage_comparison. Callers can use isinstance(result,
    CoverageComparisonResult) instead of checking cost_a is not None.
    """
    result_a: MILPResult
    result_b: MILPResult
    coverage_count_a: int
    coverage_count_b: int

    @property
    def feasible(self) -> bool:
        return self.result_a.feasible and self.result_b.feasible

    @property
    def cost_difference(self) -> float:
        return self.result_b.total_cost - self.result_a.total_cost

    @property
    def trucks_a(self) -> int:
        return self.result_a.trucks

    @property
    def trucks_b(self) -> int:
        return self.result_b.trucks

    @property
    def cost_a(self) -> float:
        return self.result_a.total_cost

    @property
    def cost_b(self) -> float:
        return self.result_b.total_cost


@dataclass
class PipelineResult:
    """Full pipeline result — all three models + LLM insight."""

    scenario_params: ScenarioParams
    lbl_result: Optional[ModelResult]
    wct_result: Optional[ModelResult]
    milp_result: MILPResult
    insight: str = ""
    query_number: int = 1
    query_text: str = ""
    coverage_comparison: Optional[CoverageComparisonResult] = None
