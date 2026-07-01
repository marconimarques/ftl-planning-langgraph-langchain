"""Capacity gate — pre-check for terminal and CP demand overflow before running the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field

from .data_types import ScenarioParams
from .loader import NetworkData


@dataclass
class TerminalOverflow:
    terminal_id: str
    effective_demand: float
    capacity: float
    overflow_pct: float  # (effective / capacity - 1) * 100


@dataclass
class CPOverflow:
    cp_id: str
    effective_demand: float
    capacity: float
    overflow_pct: float  # (effective / capacity - 1) * 100


@dataclass
class CapacityCheckResult:
    has_overflow: bool  # True when any terminal or CP exceeds capacity
    overflowing_terminals: list[TerminalOverflow] = field(default_factory=list)
    ok_terminals: list[str] = field(default_factory=list)
    overflowing_cps: list[CPOverflow] = field(default_factory=list)


def check_capacity(
    network: NetworkData,
    params: ScenarioParams,
) -> CapacityCheckResult:
    """Check whether effective demand exceeds capacity for any active terminal or CP.

    Only triggered when terminal_demand_multipliers contains at least one value > 1.0.
    Demand reductions (multiplier < 1) cannot produce overflow by definition.
    """
    if not params.terminal_demand_multipliers:
        return CapacityCheckResult(has_overflow=False)
    if not any(v > 1.0 for v in params.terminal_demand_multipliers.values()):
        return CapacityCheckResult(has_overflow=False)

    # ── Terminal check ────────────────────────────────────────────────────────
    overflowing_terminals: list[TerminalOverflow] = []
    ok_terminals: list[str] = []

    for t_id in network.terminal_ids:
        if not params.terminals_active.get(t_id, True):
            continue

        multiplier = params.terminal_demand_multipliers.get(t_id, 1.0)
        effective_demand = sum(
            network.demand[cp].get(t_id, 0.0) * multiplier
            for cp in network.cp_ids
            if network.demand[cp].get(t_id, 0.0) > 0
        )
        capacity = network.terminal_capacities[t_id]

        if effective_demand > capacity:
            overflow_pct = (effective_demand / capacity - 1.0) * 100
            overflowing_terminals.append(TerminalOverflow(
                terminal_id=t_id,
                effective_demand=effective_demand,
                capacity=capacity,
                overflow_pct=overflow_pct,
            ))
        else:
            ok_terminals.append(t_id)

    # ── CP check ──────────────────────────────────────────────────────────────
    overflowing_cps: list[CPOverflow] = []

    for cp in network.cp_ids:
        effective_demand = sum(
            network.demand[cp].get(t_id, 0.0) * params.terminal_demand_multipliers.get(t_id, 1.0)
            for t_id in network.terminal_ids
            if params.terminals_active.get(t_id, True)
            and network.demand[cp].get(t_id, 0.0) > 0
        )
        capacity = network.cp_capacities[cp]

        if capacity > 0 and effective_demand > capacity:
            overflow_pct = (effective_demand / capacity - 1.0) * 100
            overflowing_cps.append(CPOverflow(
                cp_id=cp,
                effective_demand=effective_demand,
                capacity=capacity,
                overflow_pct=overflow_pct,
            ))

    has_overflow = bool(overflowing_terminals or overflowing_cps)

    return CapacityCheckResult(
        has_overflow=has_overflow,
        overflowing_terminals=overflowing_terminals,
        ok_terminals=ok_terminals,
        overflowing_cps=overflowing_cps,
    )
