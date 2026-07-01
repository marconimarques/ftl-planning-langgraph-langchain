"""Deterministic shock-response solver services."""

from __future__ import annotations

import copy
import math
import re
from typing import Any

from ..llm_adapters.shock_response_agent import ShockResponseOutput, StrategyResult
from ..domain.data_types import ScenarioParams
from ..domain.loader import NetworkData
from ..models.solver import run_milp_solver
from ..app.i18n import t


_NON_RESPONSE_FIELDS = {
    "min_coverage_count",
    "coverage_count",
    "coverage_count_a",
    "coverage_count_b",
    "budget",
    "net_driving_hours",
}
_TARGET_STRATEGY_COUNT = 5
_MAX_STRATEGY_COUNT = 10
_NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "um": 1,
    "uma": 1,
    "dois": 2,
    "duas": 2,
    "tres": 3,
    "três": 3,
    "quatro": 4,
    "cinco": 5,
    "seis": 6,
    "sete": 7,
    "oito": 8,
    "nove": 9,
    "dez": 10,
}
_COUNT_TOKEN = r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten|um|uma|dois|duas|tres|três|quatro|cinco|seis|sete|oito|nove|dez)"
_ALT_TOKEN = r"(?:alternatives?|options?|estrat[eé]gias?|alternativas?|op[cç][oõ]es?)"
_COMPONENT_LABELS = {
    "pt": {
        "fuel": "combustivel",
        "tires": "pneus",
        "tractor_maintenance": "manutencao trator",
        "trailer_maintenance": "manutencao carreta",
        "maintenance": "manutencao",
        "others": "outros",
        "driver_wage": "salario motorista",
        "insurance": "seguro",
        "monitoring": "monitoramento",
        "depreciation": "depreciacao",
        "ipva_tax": "IPVA",
    },
    "en": {
        "fuel": "fuel",
        "tires": "tires",
        "tractor_maintenance": "tractor maintenance",
        "trailer_maintenance": "trailer maintenance",
        "maintenance": "maintenance",
        "others": "others",
        "driver_wage": "driver wage",
        "insurance": "insurance",
        "monitoring": "monitoring",
        "depreciation": "depreciation",
        "ipva_tax": "IPVA",
    },
}


def _strategy_count_from_token(token: str) -> int | None:
    value = _NUMBER_WORDS.get(token.lower())
    if value is not None:
        return value
    try:
        value = int(token)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _bounded_strategy_count(value: int | None) -> int:
    if value is None:
        return _TARGET_STRATEGY_COUNT
    return max(1, min(value, _MAX_STRATEGY_COUNT))


def parse_requested_strategy_count(query: str) -> int:
    """Return the requested number of shock alternatives, defaulting to five.

    "At least X" keeps the default when X is lower than five, while exact forms
    such as "3 alternatives" or "best 5 alternatives" use the requested count.
    """
    text = (query or "").lower()

    at_least_match = re.search(
        rf"(?:at\s+least|pelo\s+menos|no\s+m[ií]nimo)\s+{_COUNT_TOKEN}\s+{_ALT_TOKEN}",
        text,
        flags=re.IGNORECASE,
    )
    if at_least_match:
        requested = _strategy_count_from_token(at_least_match.group(1))
        return max(_TARGET_STRATEGY_COUNT, _bounded_strategy_count(requested))

    ranked_match = re.search(
        rf"(?:best|top|melhores?|principais)\s+{_COUNT_TOKEN}\s+{_ALT_TOKEN}",
        text,
        flags=re.IGNORECASE,
    )
    if ranked_match:
        return _bounded_strategy_count(_strategy_count_from_token(ranked_match.group(1)))

    exact_match = re.search(
        rf"\b{_COUNT_TOKEN}\s+{_ALT_TOKEN}",
        text,
        flags=re.IGNORECASE,
    )
    if exact_match:
        return _bounded_strategy_count(_strategy_count_from_token(exact_match.group(1)))

    return _TARGET_STRATEGY_COUNT


def build_params_from_changes(
    network: NetworkData,
    changes: dict[str, Any],
) -> ScenarioParams:
    """Build ScenarioParams from explicit shock/strategy changes."""
    changes = changes if isinstance(changes, dict) else {}
    variable_components = dict(network.variable_cost_components)
    fixed_components = dict(network.fixed_cost_components)

    for key, multiplier in _normalize_cost_multipliers(
        network,
        _mapping_or_empty(changes.get("var_cost_multipliers")),
        "variable",
    ).items():
        if key in variable_components:
            variable_components[key] = round(variable_components[key] * float(multiplier), 4)
    for key, multiplier in _normalize_cost_multipliers(
        network,
        _mapping_or_empty(changes.get("fix_cost_multipliers")),
        "fixed",
    ).items():
        if key in fixed_components:
            fixed_components[key] = round(fixed_components[key] * float(multiplier), 2)

    terminals_active = {terminal: True for terminal in network.terminal_ids}
    for terminal in changes.get("closed_terminals") or []:
        if terminal in terminals_active:
            terminals_active[terminal] = False

    variable_cost_per_km = (
        round(sum(variable_components.values()), 4)
        if variable_components
        else float(changes.get("variable_cost_per_km", network.variable_cost_per_km))
    )
    fixed_cost_per_truck_month = (
        round(sum(fixed_components.values()), 2)
        if fixed_components
        else float(changes.get("fixed_cost_per_truck_month", network.fixed_cost_per_truck_month))
    )

    params = ScenarioParams(
        payload=float(changes.get("payload", network.payload)),
        speed_loaded=float(changes.get("speed_loaded", network.speed_loaded)),
        speed_empty=float(changes.get("speed_empty", network.speed_empty)),
        availability=float(changes.get("availability", network.availability)),
        overtime_hours=float(changes.get("overtime_hours", network.overtime_hours)),
        overtime_cost=float(changes.get("overtime_cost", network.overtime_cost)),
        variable_cost_per_km=variable_cost_per_km,
        fixed_cost_per_truck_month=fixed_cost_per_truck_month,
        working_days=int(changes.get("working_days", network.working_days)),
        net_driving_hours=float(changes.get("net_driving_hours", network.net_driving_hours)),
        terminals_active=terminals_active,
        min_coverage_count=changes.get("min_coverage_count"),
        budget=changes.get("budget"),
        objective=changes.get("objective", "minimize_cost"),
        volume_redistribution=bool(changes.get("volume_redistribution", False)),
        variable_cost_components=variable_components,
        fixed_cost_components=fixed_components,
        terminal_demand_multipliers=dict(_mapping_or_empty(changes.get("terminal_demand_multipliers"))),
        terminal_volume_caps=dict(_mapping_or_empty(changes.get("terminal_volume_caps"))),
    )
    return params


def _mapping_or_empty(value: Any) -> dict:
    return dict(value) if isinstance(value, dict) else {}


def _norm_key(value: Any) -> str:
    return str(value).lower().replace("-", "_").replace(" ", "_").strip()


def _normalize_cost_multipliers(
    network: NetworkData,
    multipliers: dict,
    cost_type: str,
) -> dict[str, float]:
    """Normalize generic LLM cost names into concrete component keys."""
    components = (
        network.variable_cost_components
        if cost_type == "variable"
        else network.fixed_cost_components
    )
    normalized: dict[str, float] = {}
    maintenance_keys = [
        key
        for key in ("tractor_maintenance", "trailer_maintenance")
        if key in network.variable_cost_components
    ]
    generic_maintenance = {
        "maintenance",
        "maintenance_cost",
        "maintenance_costs",
        "manutencao",
        "custos_de_manutencao",
        "custo_de_manutencao",
    }

    for raw_key, raw_multiplier in multipliers.items():
        value = _numeric(raw_multiplier)
        if value is None:
            continue
        key = _norm_key(raw_key)
        if cost_type == "variable" and key in generic_maintenance:
            for maintenance_key in maintenance_keys:
                normalized[maintenance_key] = value
            continue
        if raw_key in components:
            normalized[str(raw_key)] = value
        elif key in components:
            normalized[key] = value
    return normalized


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _changes_are_valid_response_levers(network: NetworkData, changes: dict[str, Any]) -> bool:
    """Return whether candidate changes are allowed as shock-response levers."""
    if not isinstance(changes, dict):
        return False
    if any(field in changes for field in _NON_RESPONSE_FIELDS):
        return False

    operational_limits = (network.lever_limits or {}).get("operational", {})
    for field, limit in operational_limits.items():
        if field not in changes:
            continue
        value = _numeric(changes[field])
        if value is None:
            continue
        min_value = limit.get("min")
        max_value = limit.get("max")
        if min_value is not None and value < float(min_value):
            return False
        if max_value is not None and value > float(max_value):
            return False

    for field in ("payload", "speed_loaded", "speed_empty", "availability", "working_days", "overtime_hours"):
        if field not in changes:
            continue
        value = _numeric(changes[field])
        baseline = _numeric(getattr(network, field, None))
        if value is None or baseline is None or value <= baseline:
            return False
    if "overtime_cost" in changes:
        value = _numeric(changes["overtime_cost"])
        baseline = _numeric(network.overtime_cost)
        if value is None or baseline is None or value >= baseline:
            return False

    terminal_volume_caps = changes.get("terminal_volume_caps")
    if terminal_volume_caps is not None and not isinstance(terminal_volume_caps, dict):
        return False
    for cap in _mapping_or_empty(terminal_volume_caps).values():
        value = _numeric(cap)
        if value is None or not 0 <= value <= 1:
            return False

    terminal_demand_multipliers = changes.get("terminal_demand_multipliers")
    if terminal_demand_multipliers is not None and not isinstance(terminal_demand_multipliers, dict):
        return False
    for multiplier in _mapping_or_empty(terminal_demand_multipliers).values():
        value = _numeric(multiplier)
        if value is None or value < 0:
            return False

    return True


def _pct_delta(new_value: float, baseline: float) -> float:
    if baseline == 0:
        return 0.0
    return (new_value / baseline - 1.0) * 100.0


def _fmt_pct(value: float) -> str:
    rounded = round(value, 1)
    if rounded.is_integer():
        return f"{int(rounded)}%"
    return f"{rounded:.1f}%"


def _fmt_decimal(value: float, decimals: int = 2) -> str:
    rounded = round(value, decimals)
    if rounded.is_integer():
        return str(int(rounded))
    text = f"{rounded:.{decimals}f}".rstrip("0").rstrip(".")
    return text


_ABSOLUTE_VALUE_FORMATTERS = {
    "payload": lambda v, language: f"{_fmt_decimal(v)} {'toneladas' if language == 'pt' else 'tons'}",
    "speed_loaded": lambda v, language: f"{_fmt_decimal(v)} km/h",
    "speed_empty": lambda v, language: f"{_fmt_decimal(v)} km/h",
    "availability": lambda v, language: _fmt_pct(v * 100),
}


def _component_label(component: str, language: str) -> str:
    labels = _COMPONENT_LABELS.get(language, _COMPONENT_LABELS["en"])
    return labels.get(component, component.replace("_", " "))


def _strategy_name_from_changes(changes: dict[str, Any], network: NetworkData, language: str) -> str:
    """Build a deterministic label from the actual solved response levers."""
    if not isinstance(changes, dict) or not changes:
        return t("shock_strategy_default", language)

    labels: list[str] = []
    operational_labels = {
        "payload": ("shock_strategy_payload", network.payload),
        "speed_loaded": ("shock_strategy_speed_loaded", network.speed_loaded),
        "speed_empty": ("shock_strategy_speed_empty", network.speed_empty),
        "availability": ("shock_strategy_availability", network.availability),
        "overtime_hours": ("shock_strategy_overtime_hours", network.overtime_hours),
    }
    for field, (label_key, baseline) in operational_labels.items():
        if field not in changes:
            continue
        value = _numeric(changes[field])
        base = _numeric(baseline)
        if value is None or base is None:
            continue
        label = t(label_key, language)
        absolute_formatter = _ABSOLUTE_VALUE_FORMATTERS.get(field)
        if absolute_formatter is not None:
            connector = t("shock_strategy_connector_to", language)
            labels.append(f"{label} {connector} {absolute_formatter(value, language)}")
        else:
            pct = _pct_delta(value, base)
            connector = t("shock_strategy_connector", language)
            labels.append(f"{label} {connector} {_fmt_pct(abs(pct))}")

    if "working_days" in changes:
        wd_value = _numeric(changes["working_days"])
        if wd_value is not None:
            days = int(wd_value)
            if days >= 30:
                drivers = 3
            elif days >= 27:
                drivers = 2
            else:
                drivers = 1
            if language == "pt":
                labels.append(f"Jornada {days} dias ({drivers} motoristas/caminhão)")
            else:
                labels.append(f"{days}-day schedule ({drivers} drivers/truck)")

    for field, label_key in {
        "var_cost_multipliers": "shock_strategy_var_cost",
        "fix_cost_multipliers": "shock_strategy_fix_cost",
    }.items():
        multipliers = _mapping_or_empty(changes.get(field))
        if not multipliers:
            continue
        normalized = _normalize_cost_multipliers(
            network,
            multipliers,
            "variable" if field.startswith("var") else "fixed",
        )
        reductions = [abs((1.0 - value) * 100.0) for value in normalized.values() if value < 1.0]
        if reductions:
            label = t(label_key, language)
            connector = t("shock_strategy_connector", language)
            components = ", ".join(
                _component_label(component, language)
                for component, value in normalized.items()
                if value < 1.0
            )
            component_text = f" ({components})" if components else ""
            labels.append(f"{label}{component_text} {connector} {_fmt_pct(max(reductions))}")

    if "volume_redistribution" in changes:
        labels.append(t("shock_strategy_redistribution", language))

    if labels:
        return " + ".join(labels)
    return t("shock_strategy_default", language)


def _is_improving_strategy(strategy: StrategyResult) -> bool:
    """A ranked reaction must be feasible and reduce cost versus the pure shock."""
    return (
        math.isfinite(strategy.cost)
        and math.isfinite(strategy.cost_recovered)
        and strategy.cost_recovered > 0
        and strategy.trucks > 0
    )


def _lever_count(changes: dict) -> int:
    """Count the number of independent operational levers in a strategy's param changes.

    Dict values (e.g. var_cost_multipliers) contribute one count per component so that
    a 2-component cost change is treated as 2 levers, not 1.
    """
    count = 0
    for value in (changes or {}).values():
        if isinstance(value, dict):
            count += max(len(value), 1)
        else:
            count += 1
    return max(count, 1)


def _lever_intensity(
    field: str,
    value: Any,
    baseline: Any,
    network: NetworkData,
) -> float:
    """Fraction of (max_value − baseline) being used for this lever (0.0–1.0)."""
    limits = ((network.lever_limits or {}).get("operational") or {}).get(field, {})
    max_val = _numeric(limits.get("max"))
    base = _numeric(baseline)
    val = _numeric(value)
    if max_val is None or base is None or val is None or max_val <= base:
        return 0.0
    return max(0.0, min(1.0, (val - base) / (max_val - base)))


def _cost_lever_intensity(component: str, multiplier: Any, network: NetworkData) -> float:
    """Fraction of max_saving_pct being requested for this cost component (0.0–1.0)."""
    savings = ((network.lever_limits or {}).get("cost_savings") or {}).get(component, {})
    max_saving = _numeric(savings.get("max_saving_pct"))
    val = _numeric(multiplier)
    if max_saving is None or max_saving <= 0 or val is None:
        return 0.0
    return max(0.0, min(1.0, (1.0 - val) / max_saving))


def _effort_score(changes: dict, network: NetworkData) -> float:
    """Sum of squared intensities across all levers (Herfindahl-style concentration).

    Penalizes concentrated asks: one lever at 100% scores 1.0 effort while three
    levers at 33% each score only 0.33 — distributed moderate asks cost less.
    """
    _OP_BASELINES = [
        ("availability",   network.availability),
        ("payload",        network.payload),
        ("speed_loaded",   network.speed_loaded),
        ("speed_empty",    network.speed_empty),
        ("overtime_hours", network.overtime_hours),
        ("working_days",   network.working_days),
    ]
    intensities: list[float] = []
    for field, baseline in _OP_BASELINES:
        if field in changes:
            intensities.append(_lever_intensity(field, changes[field], baseline, network))
    for cost_field, category in (
        ("var_cost_multipliers", "variable"),
        ("fix_cost_multipliers", "fixed"),
    ):
        normalized = _normalize_cost_multipliers(
            network,
            _mapping_or_empty(changes.get(cost_field)),
            category,
        )
        for component, multiplier in normalized.items():
            intensities.append(_cost_lever_intensity(component, multiplier, network))
    return sum(i ** 2 for i in intensities) if intensities else 1.0


def _score_strategy(
    strategy: StrategyResult,
    shock_delta: float,
    network: NetworkData | None = None,
) -> float:
    """Score a strategy by sufficiency vs. execution effort.

    With network: uses effort = sum of squared intensities per lever so that
    distributed moderate asks (3 × 33%) are cheaper than one extreme ask (1 × 100%).
    Without network: falls back to lever_count penalty (backward compat).
    """
    if shock_delta <= 0:
        return strategy.cost_recovered
    ratio = strategy.cost_recovered / shock_delta
    sufficiency = min(ratio, 1.5)
    if network is not None:
        complexity = 0.15 * _effort_score(strategy.params_changed, network)
    else:
        complexity = 0.1 * (_lever_count(strategy.params_changed) - 1)
    return sufficiency - complexity


def _rank_improving_strategies(
    strategies: list[StrategyResult],
    shock_delta: float = 0.0,
    network: NetworkData | None = None,
) -> list[StrategyResult]:
    improving = [s for s in strategies if _is_improving_strategy(s)]
    if shock_delta > 0:
        return sorted(
            improving,
            key=lambda s: _score_strategy(s, shock_delta, network),
            reverse=True,
        )
    return sorted(improving, key=lambda s: s.cost_recovered, reverse=True)


_OPERATIONAL_LEVER_FIELDS = frozenset(
    {"availability", "payload", "speed_loaded", "speed_empty", "overtime_hours", "working_days"}
)


def _stakeholder_profile(changes: dict) -> str:
    """Classify which departments a strategy touches.

    Returns 'operations', 'procurement', or 'cross'.
    """
    has_ops = any(k in changes for k in _OPERATIONAL_LEVER_FIELDS)
    has_cost = (
        bool(_mapping_or_empty(changes.get("var_cost_multipliers")))
        or bool(_mapping_or_empty(changes.get("fix_cost_multipliers")))
    )
    if has_ops and not has_cost:
        return "operations"
    if has_cost and not has_ops:
        return "procurement"
    return "cross"


def _is_moderate(changes: dict, network: NetworkData) -> bool:
    """True if every lever in the strategy is used below its moderate_threshold.

    Thresholds are read from lever_limits.xlsx; default 0.67 if absent.
    """
    _OP_BASELINES = [
        ("availability",   network.availability),
        ("payload",        network.payload),
        ("speed_loaded",   network.speed_loaded),
        ("speed_empty",    network.speed_empty),
        ("overtime_hours", network.overtime_hours),
        ("working_days",   network.working_days),
    ]
    for field, baseline in _OP_BASELINES:
        if field not in changes:
            continue
        limits = ((network.lever_limits or {}).get("operational") or {}).get(field, {})
        threshold = float(limits.get("moderate_threshold", 0.67))
        if _lever_intensity(field, changes[field], baseline, network) > threshold:
            return False
    for cost_field, category in (
        ("var_cost_multipliers", "variable"),
        ("fix_cost_multipliers", "fixed"),
    ):
        normalized = _normalize_cost_multipliers(
            network,
            _mapping_or_empty(changes.get(cost_field)),
            category,
        )
        for component, multiplier in normalized.items():
            savings = ((network.lever_limits or {}).get("cost_savings") or {}).get(component, {})
            threshold = float(savings.get("moderate_threshold", 0.67))
            if _cost_lever_intensity(component, multiplier, network) > threshold:
                return False
    return True


def _lever_signature(changes: dict[str, Any]) -> frozenset:
    """Identify which specific lever(s) a strategy pulls, ignoring intensity.

    Two candidates that both touch only "payload" (at whatever magnitude) share
    this signature; a candidate combining payload with a fuel-cost cut does not.
    Used to avoid filling the visible list with several variants of the same
    single lever while a genuinely different alternative gets crowded out.
    """
    keys: set[str] = set()
    for field in _OPERATIONAL_LEVER_FIELDS:
        if field in changes:
            keys.add(field)
    for cost_field in ("var_cost_multipliers", "fix_cost_multipliers"):
        for component in _mapping_or_empty(changes.get(cost_field)):
            keys.add(f"{cost_field}:{component}")
    if "volume_redistribution" in changes:
        keys.add("volume_redistribution")
    return frozenset(keys)


def _select_with_coverage(
    ranked_all: list[StrategyResult],
    target_count: int,
    network: NetworkData,
) -> list[StrategyResult]:
    """Guarantee stakeholder profile coverage before filling by score.

    Slot 0 — best-scoring pure Operations option (single-department ownership)
    Slot 1 — best-scoring pure Procurement/Finance option (single-department ownership)
    Slot 2 — best-scoring cross-functional option where ALL levers are moderate
              (every lever below its moderate_threshold — distributed asks)
    Slots 3+ — top-remaining by score, preferring a lever combination not
              already shown; only repeats a lever signature once no distinct
              alternative remains.

    If a guaranteed slot has no matching candidate, it is skipped and the slot
    is filled from the score-ordered remainder instead.
    """
    selected: list[StrategyResult] = []
    used: set[int] = set()
    used_signatures: set[frozenset] = set()

    def _first(predicate) -> StrategyResult | None:
        for s in ranked_all:
            if id(s) not in used and predicate(s):
                return s
        return None

    def _take(s: StrategyResult | None) -> None:
        if s is not None:
            selected.append(s)
            used.add(id(s))
            used_signatures.add(_lever_signature(s.params_changed))

    _take(_first(lambda s: _stakeholder_profile(s.params_changed) == "operations"))
    _take(_first(lambda s: _stakeholder_profile(s.params_changed) == "procurement"))
    _take(_first(
        lambda s: _stakeholder_profile(s.params_changed) == "cross"
        and _is_moderate(s.params_changed, network)
    ))

    for s in ranked_all:
        if len(selected) >= target_count:
            break
        if id(s) in used:
            continue
        if _lever_signature(s.params_changed) in used_signatures:
            continue
        _take(s)

    for s in ranked_all:
        if len(selected) >= target_count:
            break
        if id(s) not in used:
            _take(s)

    return selected


def _change_signature(changes: dict[str, Any]) -> tuple:
    """Stable enough signature to avoid solving duplicate candidate plans."""
    return tuple(
        sorted(
            (str(key), repr(value))
            for key, value in (changes if isinstance(changes, dict) else {}).items()
        )
    )


_SIMILARITY_BUCKET_WIDTH = 0.1
_OPERATIONAL_BASELINE_FIELDS = (
    "availability", "payload", "speed_loaded", "speed_empty", "overtime_hours", "working_days",
)


def _similarity_cluster_key(changes: dict[str, Any], network: NetworkData) -> tuple:
    """Group strategies that use the same lever(s) at a near-identical intensity.

    Two candidates touching the same field(s) within _SIMILARITY_BUCKET_WIDTH of
    each other's feasible-range intensity read as the same strategy to a user
    (e.g. payload -> 30.6t vs 30.66t), even though their raw values differ.
    """
    parts: list[tuple] = []
    for field in _OPERATIONAL_BASELINE_FIELDS:
        if field not in changes:
            continue
        baseline = getattr(network, field, None)
        intensity = _lever_intensity(field, changes[field], baseline, network)
        bucket = round(intensity / _SIMILARITY_BUCKET_WIDTH)
        parts.append((field, bucket))
    for cost_field, category in (
        ("var_cost_multipliers", "variable"),
        ("fix_cost_multipliers", "fixed"),
    ):
        normalized = _normalize_cost_multipliers(
            network, _mapping_or_empty(changes.get(cost_field)), category,
        )
        for component, multiplier in sorted(normalized.items()):
            intensity = _cost_lever_intensity(component, multiplier, network)
            bucket = round(intensity / _SIMILARITY_BUCKET_WIDTH)
            parts.append((cost_field, component, bucket))
    if "volume_redistribution" in changes:
        parts.append(("volume_redistribution", bool(changes["volume_redistribution"])))
    return tuple(sorted(parts))


def _dedupe_near_identical_strategies(
    strategies: list[StrategyResult],
    network: NetworkData,
    shock_delta: float,
) -> list[StrategyResult]:
    """Keep only the best-scoring candidate within each near-identical cluster.

    Prevents the top-N list from showing what looks like the same strategy
    twice (e.g. "payload -> 30.6t" and "payload -> 30.66t") at the expense of
    a genuinely distinct alternative.
    """
    best_by_cluster: dict[tuple, StrategyResult] = {}
    for strategy in strategies:
        key = _similarity_cluster_key(strategy.params_changed, network)
        current = best_by_cluster.get(key)
        if current is None:
            best_by_cluster[key] = strategy
            continue
        challenger_score = _score_strategy(strategy, shock_delta, network)
        current_score = _score_strategy(current, shock_delta, network)
        if challenger_score > current_score:
            best_by_cluster[key] = strategy
    return list(best_by_cluster.values())


def _shocked_cost_keys(shock_params: dict[str, Any], field: str) -> set[str]:
    cost_type = "variable" if field.startswith("var") else "fixed"
    raw = _mapping_or_empty(shock_params.get(field))
    return {_norm_key(key) for key in raw} | set(raw.keys())


def _available_saving_candidates(
    network: NetworkData,
    shock_params: dict[str, Any],
    category: str,
) -> list[dict[str, Any]]:
    savings = (network.lever_limits or {}).get("cost_savings", {})
    field = "var_cost_multipliers" if category == "variable" else "fix_cost_multipliers"
    components = (
        network.variable_cost_components
        if category == "variable"
        else network.fixed_cost_components
    )
    shocked_keys = _shocked_cost_keys(shock_params, field)
    candidates: list[dict[str, Any]] = []

    for component, limit in savings.items():
        if str(limit.get("category")) != category:
            continue
        if component not in components or component in shocked_keys or _norm_key(component) in shocked_keys:
            continue
        max_saving = _numeric(limit.get("max_saving_pct"))
        if max_saving is None or max_saving <= 0:
            continue
        candidates.append({field: {component: round(1.0 - max_saving, 4)}})
    return candidates


def _operational_candidate(
    network: NetworkData,
    shock_params: dict[str, Any],
    field: str,
) -> dict[str, Any] | None:
    # legacy; use _operational_candidates_sweep for multi-level exploration
    if field in shock_params:
        return None
    limit = ((network.lever_limits or {}).get("operational") or {}).get(field, {})
    max_value = _numeric(limit.get("max"))
    baseline = _numeric(getattr(network, field, None))
    if max_value is None or baseline is None or max_value <= baseline:
        return None
    if field == "working_days":
        value = int(max_value)
        if value >= 30:
            return {"working_days": 30, "fix_cost_multipliers": {"driver_wage": 3.0}}
        if value >= 27:
            return {"working_days": 27, "fix_cost_multipliers": {"driver_wage": 2.0}}
        return None
    return {field: max_value}


_SWEEP_INTENSITIES = (0.33, 0.67, 1.0)


def _operational_candidates_sweep(
    network: NetworkData,
    shock_params: dict[str, Any],
    field: str,
) -> list[dict[str, Any]]:
    """Return candidates for `field` at multiple intensity levels within the feasible range.

    Generates candidates at 33%, 67%, and 100% of (baseline → max) so the solver
    can find the minimum intervention sufficient to compensate the shock rather than
    always proposing the ceiling value.
    """
    if field in shock_params:
        return []
    limit = ((network.lever_limits or {}).get("operational") or {}).get(field, {})
    max_value = _numeric(limit.get("max"))
    baseline = _numeric(getattr(network, field, None))
    if max_value is None or baseline is None or max_value <= baseline:
        return []

    if field == "working_days":
        candidates: list[dict[str, Any]] = []
        if int(max_value) >= 27:
            candidates.append({"working_days": 27, "fix_cost_multipliers": {"driver_wage": 2.0}})
        if int(max_value) >= 30:
            candidates.append({"working_days": 30, "fix_cost_multipliers": {"driver_wage": 3.0}})
        return candidates

    feasible_range = max_value - baseline
    result: list[dict[str, Any]] = []
    for intensity in _SWEEP_INTENSITIES:
        value = round(baseline + intensity * feasible_range, 4)
        if value > baseline:
            result.append({field: value})
    return result


def _enforce_working_days_driver_wage(
    candidate_changes: dict[str, Any],
    network: NetworkData,
) -> dict[str, Any]:
    """Auto-inject the correct driver_wage multiplier when working_days exceeds baseline.

    Increasing working_days requires additional drivers; without the wage multiplier the
    solver underestimates fixed cost and the strategy result is invalid.
    """
    working_days = _numeric(candidate_changes.get("working_days"))
    if working_days is None or working_days <= float(network.working_days):
        return candidate_changes

    fix_mults = dict(_mapping_or_empty(candidate_changes.get("fix_cost_multipliers")))
    if "driver_wage" in fix_mults:
        return candidate_changes

    days = int(working_days)
    if days >= 30:
        fix_mults["driver_wage"] = 3.0
    elif days >= 27:
        fix_mults["driver_wage"] = 2.0
    else:
        return candidate_changes

    return {**candidate_changes, "fix_cost_multipliers": fix_mults}


def _fallback_strategy_changes(network: NetworkData, shock_params: dict[str, Any]) -> list[dict[str, Any]]:
    """Generate bounded deterministic alternatives when the LLM supplies too few usable candidates.

    Uses a multi-level intensity sweep (33%, 67%, 100% of feasible range) per operational lever
    so the solver can find the minimum sufficient intervention rather than always the ceiling.
    """
    candidates: list[dict[str, Any]] = []
    for field in ("availability", "overtime_hours", "payload", "speed_loaded", "speed_empty", "working_days"):
        candidates.extend(_operational_candidates_sweep(network, shock_params, field))

    variable_savings = _available_saving_candidates(network, shock_params, "variable")
    fixed_savings = _available_saving_candidates(network, shock_params, "fixed")
    candidates.extend(variable_savings)
    candidates.extend(fixed_savings)

    operational = [
        change
        for change in candidates
        if any(key in change for key in ("availability", "overtime_hours", "payload", "speed_loaded", "speed_empty"))
    ]
    for op in operational[:3]:
        if variable_savings:
            candidates.append({**op, **variable_savings[0]})
        if fixed_savings:
            candidates.append({**op, **fixed_savings[0]})
    if len(variable_savings) >= 2:
        merged = {"var_cost_multipliers": {}}
        for change in variable_savings[:2]:
            merged["var_cost_multipliers"].update(change["var_cost_multipliers"])
        candidates.append(merged)
    if len(fixed_savings) >= 2:
        merged = {"fix_cost_multipliers": {}}
        for change in fixed_savings[:2]:
            merged["fix_cost_multipliers"].update(change["fix_cost_multipliers"])
        candidates.append(merged)

    unique: list[dict[str, Any]] = []
    seen: set[tuple] = set()
    for change in candidates:
        signature = _change_signature(change)
        if signature not in seen and _changes_are_valid_response_levers(network, change):
            seen.add(signature)
            unique.append(change)
    return unique


def _solve_candidate_changes(
    network: NetworkData,
    shock_params_dict: dict[str, Any],
    candidate_changes: dict[str, Any],
    shock_result,
    language: str,
) -> StrategyResult | None:
    candidate_changes = _enforce_working_days_driver_wage(candidate_changes, network)
    if not _changes_are_valid_response_levers(network, candidate_changes):
        return None
    combined_changes = {**shock_params_dict, **candidate_changes}
    result = run_milp_solver(network, build_params_from_changes(network, combined_changes))
    return StrategyResult(
        strategy_name=_strategy_name_from_changes(candidate_changes, network, language),
        params_changed=candidate_changes,
        trucks=result.trucks if result.feasible else 0,
        cost=result.total_cost if result.feasible else float("inf"),
        cost_recovered=(
            shock_result.total_cost - result.total_cost
            if shock_result.feasible and result.feasible
            else float("-inf")
        ),
    )


def solve_strategy_candidates(output, network: NetworkData):
    """Rerun shock strategy candidates in Python and recompute rankings.

    Compatibility helper for pre-migration ShockResponseOutput objects. New graph
    execution uses solve_shock_candidate_plan.
    """
    updated = copy.deepcopy(output)
    shock_cost = float(getattr(updated, "shock_cost", 0.0) or 0.0)

    solved: list[StrategyResult] = []
    for strategy in list(getattr(updated, "strategies", []) or []):
        changes = dict(getattr(strategy, "params_changed", {}) or {})
        changes = _enforce_working_days_driver_wage(changes, network)
        if not _changes_are_valid_response_levers(network, changes):
            continue
        params = build_params_from_changes(network, changes)
        result = run_milp_solver(network, params)
        solved.append(
            StrategyResult(
                strategy_name=_strategy_name_from_changes(changes, network, "pt"),
                params_changed=changes,
                trucks=result.trucks if result.feasible else 0,
                cost=result.total_cost if result.feasible else float("inf"),
                cost_recovered=shock_cost - result.total_cost if result.feasible else float("-inf"),
            )
        )

    try:
        updated.strategies = _rank_improving_strategies(solved)
    except Exception:
        setattr(updated, "strategies", solved)

    redistribution = getattr(updated, "redistribution_strategy", None)
    if redistribution is not None:
        changes = dict(getattr(redistribution, "params_changed", {}) or {})
        params = build_params_from_changes(network, changes)
        result = run_milp_solver(network, params)
        updated.redistribution_strategy = StrategyResult(
            strategy_name=_strategy_name_from_changes(changes, network, "pt"),
            params_changed=changes,
            trucks=result.trucks if result.feasible else 0,
            cost=result.total_cost if result.feasible else float("inf"),
            cost_recovered=shock_cost - result.total_cost if result.feasible else float("-inf"),
        )

    return updated


def solve_shock_candidate_plan(
    plan,
    network: NetworkData,
    language: str = "pt",
    target_strategy_count: int = _TARGET_STRATEGY_COUNT,
) -> ShockResponseOutput:
    """Solve a parser-only shock candidate plan deterministically in Python."""
    target_strategy_count = _bounded_strategy_count(target_strategy_count)
    baseline_params = build_params_from_changes(network, {})
    baseline_result = run_milp_solver(network, baseline_params)

    shock_params_dict = dict(getattr(plan, "shock_params", {}) or {})
    shock_params = build_params_from_changes(network, shock_params_dict)
    shock_result = run_milp_solver(network, shock_params)

    redistribution_changes = {**shock_params_dict, "volume_redistribution": True}
    redistribution_result = run_milp_solver(network, build_params_from_changes(network, redistribution_changes))
    redistribution_name = t("shock_strategy_redistribution", language)
    redistribution_strategy = StrategyResult(
        strategy_name=redistribution_name,
        params_changed={"volume_redistribution": True},
        trucks=redistribution_result.trucks if redistribution_result.feasible else 0,
        cost=redistribution_result.total_cost if redistribution_result.feasible else float("inf"),
        cost_recovered=(
            shock_result.total_cost - redistribution_result.total_cost
            if shock_result.feasible and redistribution_result.feasible
            else float("-inf")
        ),
    )

    solved_strategies: list[StrategyResult] = []
    seen_changes: set[tuple] = set()
    for candidate in list(getattr(plan, "strategies", []) or []):
        candidate_changes = dict(getattr(candidate, "params_changed", {}) or {})
        signature = _change_signature(candidate_changes)
        if signature in seen_changes:
            continue
        seen_changes.add(signature)
        solved = _solve_candidate_changes(
            network,
            shock_params_dict,
            candidate_changes,
            shock_result,
            language,
        )
        if solved is not None:
            solved_strategies.append(solved)

    for candidate_changes in _fallback_strategy_changes(network, shock_params_dict):
        signature = _change_signature(candidate_changes)
        if signature in seen_changes:
            continue
        seen_changes.add(signature)
        solved = _solve_candidate_changes(
            network,
            shock_params_dict,
            candidate_changes,
            shock_result,
            language,
        )
        if solved is not None:
            solved_strategies.append(solved)

    shock_delta = (
        shock_result.total_cost - baseline_result.total_cost
        if shock_result.feasible and baseline_result.feasible
        else 0.0
    )
    deduped_strategies = _dedupe_near_identical_strategies(solved_strategies, network, shock_delta)
    ranked_all = _rank_improving_strategies(deduped_strategies, shock_delta, network)
    narrative = _build_deterministic_shock_narrative(ranked_all, shock_delta, language)
    return ShockResponseOutput(
        shock_description=getattr(plan, "shock_description", ""),
        baseline_cost=baseline_result.total_cost if baseline_result.feasible else 0.0,
        baseline_trucks=baseline_result.trucks if baseline_result.feasible else 0,
        shock_cost=shock_result.total_cost if shock_result.feasible else 0.0,
        shock_trucks=shock_result.trucks if shock_result.feasible else 0,
        redistribution_strategy=redistribution_strategy,
        strategies=_select_with_coverage(ranked_all, target_strategy_count, network),
        narrative=narrative,
        candidates_evaluated=len(solved_strategies),
    )


def _build_deterministic_shock_narrative(
    strategies: list[StrategyResult],
    shock_cost_increase: float,
    language: str,
) -> str:
    ranked = sorted(strategies, key=lambda item: item.cost_recovered, reverse=True)
    if not ranked:
        return (
            "Nenhuma estratégia candidata viável foi resolvida."
            if language == "pt"
            else "No feasible candidate strategy was solved."
        )
    best = ranked[0]
    pct = (best.cost_recovered / shock_cost_increase * 100) if shock_cost_increase > 0 else 0.0
    if language == "pt":
        return (
            f"{best.strategy_name} lidera ao recuperar ${best.cost_recovered:,.0f}/mês "
            f"({pct:.1f}% do incremento do shock). A ordenação foi calculada pelos custos resolvidos em Python."
        )
    return (
        f"{best.strategy_name} ranks first by recovering ${best.cost_recovered:,.0f}/month "
        f"({pct:.1f}% of the shock cost increase). The ranking was computed from Python-solved costs."
    )
