"""Deterministic normalization for planning requests."""

from __future__ import annotations

import math
import unicodedata

from .loader import NetworkData
from .workflow_types import DraftPlanningRequest, NormalizedPlanningRequest


def _key(value: str) -> str:
    folded = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    return folded.lower().replace("-", " ").replace("_", " ").strip()


def _terminal_aliases(network: NetworkData) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for terminal_id in network.terminal_ids:
        aliases[_key(terminal_id)] = terminal_id
        aliases[_key(network.terminal_names.get(terminal_id, terminal_id))] = terminal_id
    return aliases


def _cost_aliases(network: NetworkData) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for name in (*network.variable_cost_components.keys(), *network.fixed_cost_components.keys()):
        aliases[_key(name)] = name
        aliases[_key(name.replace("_", " "))] = name
    aliases.update(
        {
            "fuel": "fuel",
            "combustivel": "fuel",
            "diesel": "fuel",
            "driver wage": "driver_wage",
            "motorista": "driver_wage",
        }
    )
    return aliases


def _pct_to_count(value: float | None, total_cps: int) -> int | None:
    if value is None:
        return None
    return math.ceil(value / 100 * total_cps)


def normalize_request(
    draft: DraftPlanningRequest,
    network: NetworkData,
) -> NormalizedPlanningRequest:
    """Normalize aliases, units, and explicit defaults."""
    terminal_aliases = _terminal_aliases(network)
    cost_aliases = _cost_aliases(network)

    closed_terminals = [
        terminal_aliases.get(_key(item), item.strip())
        for item in draft.closed_terminals
        if str(item).strip()
    ]

    demand_multipliers = {
        terminal_aliases.get(_key(term), term.strip()): float(mult)
        for term, mult in draft.terminal_demand_multipliers.items()
    }
    volume_caps = {
        terminal_aliases.get(_key(term), term.strip()): float(cap)
        for term, cap in draft.terminal_volume_caps.items()
    }
    cost_overrides = {
        cost_aliases.get(_key(name), name.strip()): float(value)
        for name, value in draft.cost_component_overrides.items()
    }

    objective = draft.objective or (
        "maximize_coverage" if draft.budget is not None else "minimize_cost"
    )
    coverage_count_a = _pct_to_count(draft.coverage_from_pct, len(network.cp_ids))
    coverage_count_b = _pct_to_count(draft.coverage_to_pct, len(network.cp_ids))

    return NormalizedPlanningRequest(
        scenario_type=draft.scenario_type or "what_if",
        objective=objective,
        coverage_from_pct=draft.coverage_from_pct,
        coverage_to_pct=draft.coverage_to_pct,
        coverage_count_a=coverage_count_a,
        coverage_count_b=coverage_count_b,
        min_coverage_count=coverage_count_b,
        budget=draft.budget,
        closed_terminals=closed_terminals,
        terminal_demand_multipliers=demand_multipliers,
        terminal_volume_caps=volume_caps,
        cost_component_overrides=cost_overrides,
        volume_redistribution=bool(draft.volume_redistribution),
        requires_clarification=draft.requires_clarification,
        clarification_question=draft.clarification_question,
    )
