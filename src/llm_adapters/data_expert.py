"""Data Expert Agent — cross-scenario comparative analysis for the full session."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from pydantic import BaseModel, Field

from ..domain.data_types import PipelineResult, ScenarioParams
from ..domain.loader import NetworkData
from ..llm.model_factory import invoke_structured
from .model_factory import MODEL_REGISTRY, get_api_key, make_model


# ── Session profile ───────────────────────────────────────────────────────────

@dataclass
class SessionProfile:
    has_demand_variation: bool
    has_parametric_only: bool
    has_coverage_scenarios: bool
    has_two_level: bool
    has_terminal_closure: bool
    has_redistribution: bool


def classify_session(history: list[PipelineResult]) -> SessionProfile:
    whatifs = [r for r in history if not r.scenario_params.is_baseline]
    return SessionProfile(
        has_demand_variation=any(
            any(v != 1.0 for v in r.scenario_params.terminal_demand_multipliers.values())
            for r in whatifs
        ),
        has_parametric_only=bool(whatifs) and all(
            not any(v != 1.0 for v in r.scenario_params.terminal_demand_multipliers.values())
            and r.scenario_params.terminals_active == {tid: True for tid in r.scenario_params.terminals_active}
            and not r.scenario_params.volume_redistribution
            for r in whatifs
        ),
        has_coverage_scenarios=any(
            r.scenario_params.min_coverage_count is not None
            or r.scenario_params.budget is not None
            for r in whatifs
        ),
        has_two_level=any(
            r.coverage_comparison is not None
            for r in whatifs
        ),
        has_terminal_closure=any(
            not all(r.scenario_params.terminals_active.values())
            for r in whatifs
        ),
        has_redistribution=any(r.scenario_params.volume_redistribution for r in whatifs),
    )


# ── Terminal utilization ──────────────────────────────────────────────────────

def compute_terminal_utilizations(
    result: PipelineResult,
    network: NetworkData,
) -> dict[str, Optional[float]]:
    """Returns utilization ratio per terminal. None = terminal inactive."""
    p = result.scenario_params
    r = result.milp_result
    caps = {tid: network.terminal_capacities[tid] for tid in network.terminal_ids}
    inactive = {tid for tid, active in p.terminals_active.items() if not active}

    if p.skip_capacity_constraints and r.terminal_overflows:
        return {
            tid: None if tid in inactive else r.terminal_overflows.get(tid, 0.0) / cap
            for tid, cap in caps.items()
            if cap > 0
        }

    if r.volumes:
        totals = {
            tid: sum(r.volumes.get(cp, {}).get(tid, 0.0) for cp in network.cp_ids)
            for tid in network.terminal_ids
        }
        return {
            tid: None if tid in inactive else (totals[tid] / caps[tid] if caps[tid] > 0 else 0.0)
            for tid in network.terminal_ids
            if caps.get(tid, 0.0) > 0
        }

    return {
        tid: None if tid in inactive else (
            sum(network.demand.get(cp, {}).get(tid, 0.0) for cp in network.cp_ids)
            * p.terminal_demand_multipliers.get(tid, 1.0)
            / caps[tid]
        )
        for tid in network.terminal_ids
        if caps.get(tid, 0.0) > 0
    }


# ── Serialization ─────────────────────────────────────────────────────────────

def _describe_params(
    p: ScenarioParams,
    base: Optional[ScenarioParams],
    total_cps: int,
    is_pt: bool,
) -> str:
    if base is None:
        return "padrão" if is_pt else "default"

    parts: list[str] = []

    if p.payload != base.payload:
        parts.append(f"payload={p.payload}t")
    if abs(p.speed_loaded - base.speed_loaded) > 0.01:
        pct = (p.speed_loaded / base.speed_loaded - 1) * 100
        parts.append(f"speed_loaded={'+' if pct >= 0 else ''}{pct:.0f}%")
    if abs(p.speed_empty - base.speed_empty) > 0.01:
        pct = (p.speed_empty / base.speed_empty - 1) * 100
        parts.append(f"speed_empty={'+' if pct >= 0 else ''}{pct:.0f}%")
    if abs(p.availability - base.availability) > 0.001:
        parts.append(f"availability={p.availability * 100:.0f}%")
    if abs(p.overtime_hours - base.overtime_hours) > 0.001:
        parts.append(f"overtime_hours={p.overtime_hours}")
    if p.working_days != base.working_days:
        parts.append(f"working_days={p.working_days}")

    if p.variable_cost_components and base.variable_cost_components:
        for key, val in p.variable_cost_components.items():
            bv = base.variable_cost_components.get(key, val)
            if abs(val - bv) > 0.0001 and bv > 0:
                pct = (val / bv - 1) * 100
                parts.append(f"var_{key}={'+' if pct >= 0 else ''}{pct:.0f}%")
    elif abs(p.variable_cost_per_km - base.variable_cost_per_km) > 0.001:
        pct = (p.variable_cost_per_km / base.variable_cost_per_km - 1) * 100
        parts.append(f"var_cost={'+' if pct >= 0 else ''}{pct:.0f}%")

    for tid, active in p.terminals_active.items():
        if not active and base.terminals_active.get(tid, True):
            parts.append(f"{tid} [{'INATIVO' if is_pt else 'INACTIVE'}]")

    demand_lbl = "demanda" if is_pt else "demand"
    for tid, mul in p.terminal_demand_multipliers.items():
        bm = base.terminal_demand_multipliers.get(tid, 1.0)
        if abs(mul - bm) > 0.0001:
            parts.append(f"{demand_lbl} {tid}={mul * 100:.0f}%")

    if p.min_coverage_count is not None and p.min_coverage_count != base.min_coverage_count:
        pct = int(p.min_coverage_count / total_cps * 100) if total_cps else 0
        parts.append(f"min_coverage={pct}% ({p.min_coverage_count} {'PCs' if is_pt else 'CPs'})")

    if p.budget is not None and p.budget != base.budget:
        parts.append(f"budget=${p.budget:,.0f}")

    if p.volume_redistribution and not base.volume_redistribution:
        parts.append("redistribuição=ativa" if is_pt else "redistribution=active")

    return ", ".join(parts) if parts else ("padrão" if is_pt else "default")


def serialize_scenarios(
    history: list[PipelineResult],
    network: NetworkData,
    profile: SessionProfile,
    lang: str = "pt",
) -> str:
    """Build an XML block describing all scenarios for the LLM prompt."""
    is_pt = lang == "pt"
    total_cps = len(network.cp_ids)

    baseline = next((r for r in history if r.scenario_params.is_baseline), None)
    whatifs = [r for r in history if not r.scenario_params.is_baseline]

    show_terminal_utils = not profile.has_parametric_only
    show_coverage = any(
        r.milp_result.coverage_count < total_cps
        for r in history
        if r.milp_result.feasible
    )

    base_trucks = baseline.milp_result.trucks if baseline else 0
    base_cost = baseline.milp_result.total_cost if baseline else 0.0

    scenario_tags: list[str] = []

    # Baseline
    if baseline:
        param_note = "padrão (todos a 100%)" if is_pt else "default (all at 100%)"
        lines = [
            '<scenario id="Q0" is_baseline="true">',
            f"  <params>{param_note}</params>",
            f"  <fleet>{baseline.milp_result.trucks}</fleet>",
            f"  <cost>{baseline.milp_result.total_cost:.0f}</cost>",
        ]
        if show_terminal_utils:
            utils = compute_terminal_utilizations(baseline, network)
            util_parts = []
            for tid in network.terminal_ids:
                u = utils.get(tid)
                if u is None:
                    util_parts.append(f'{tid}={"INATIVO" if is_pt else "INACTIVE"}')
                elif u >= 0.95:
                    lbl = "ALERTA" if is_pt else "NEAR_CAP"
                    util_parts.append(f"{tid}={u * 100:.0f}% {lbl}")
                else:
                    util_parts.append(f"{tid}={u * 100:.0f}%")
            lines.append(f"  <terminal_utils>{' | '.join(util_parts)}</terminal_utils>")
        if show_coverage:
            lines.append(f"  <cps_served>{baseline.milp_result.coverage_count}/{total_cps}</cps_served>")
        if baseline.query_text:
            lines.append(f"  <query>{baseline.query_text}</query>")
        lines.append("</scenario>")
        scenario_tags.append("\n".join(lines))

    # What-if scenarios
    two_level_tags: list[str] = []
    for i, result in enumerate(whatifs, 1):
        p = result.scenario_params
        r = result.milp_result
        label = f"Q{i}"

        # Two-level coverage: separate block
        if result.coverage_comparison is not None:
            cov = result.coverage_comparison
            count_a = cov.coverage_count_a
            count_b = cov.coverage_count_b
            cost_a = cov.cost_a
            cost_b = cov.cost_b
            delta_pcs = count_b - count_a
            delta_cost = cost_b - cost_a
            pct_a = count_a / total_cps * 100 if total_cps else 0
            pct_b = count_b / total_cps * 100 if total_cps else 0
            per_pc = abs(delta_cost / delta_pcs) if delta_pcs != 0 else 0.0
            tl_lines = [
                f'<coverage_comparison id="{label}">',
                f"  <level_a pct=\"{pct_a:.0f}\" cps=\"{count_a}\" cost=\"{cost_a:.0f}\"/>",
                f"  <level_b pct=\"{pct_b:.0f}\" cps=\"{count_b}\" cost=\"{cost_b:.0f}\"/>",
                f"  <delta_a_to_b delta_cps=\"{'+' if delta_pcs >= 0 else ''}{delta_pcs}\" "
                f"delta_cost=\"{'+' if delta_cost >= 0 else ''}{delta_cost:.0f}\" "
                f"cost_per_cp=\"{per_pc:.0f}\"/>",
            ]
            if result.query_text:
                tl_lines.append(f"  <query>{result.query_text}</query>")
            tl_lines.append("</coverage_comparison>")
            two_level_tags.append("\n".join(tl_lines))
            scenario_tags.append(
                f'<scenario id="{label}" type="two_level_coverage">'
                f"<query>{result.query_text}</query></scenario>"
                if result.query_text else
                f'<scenario id="{label}" type="two_level_coverage"/>'
            )
            continue

        # Normal what-if
        changed = _describe_params(p, baseline.scenario_params if baseline else None, total_cps, is_pt)
        dt = r.trucks - base_trucks
        dc = r.total_cost - base_cost
        lines = [
            f'<scenario id="{label}">',
            f"  <params_changed>{changed}</params_changed>",
            f"  <fleet>{r.trucks}</fleet>",
            f"  <delta_fleet>{'+' if dt >= 0 else ''}{dt}</delta_fleet>",
            f"  <cost>{r.total_cost:.0f}</cost>",
            f"  <delta_cost>{'+' if dc >= 0 else ''}{dc:.0f}</delta_cost>",
        ]
        if result.query_text:
            lines.append(f"  <query>{result.query_text}</query>")
        if show_terminal_utils:
            utils = compute_terminal_utilizations(result, network)
            util_parts = []
            for tid in network.terminal_ids:
                u = utils.get(tid)
                if u is None:
                    util_parts.append(f'{tid}={"INATIVO" if is_pt else "INACTIVE"}')
                elif u > 1.0:
                    util_parts.append(f"{tid}={u * 100:.0f}% OVERFLOW")
                elif u >= 0.95:
                    lbl = "ALERTA" if is_pt else "NEAR_CAP"
                    util_parts.append(f"{tid}={u * 100:.0f}% {lbl}")
                else:
                    util_parts.append(f"{tid}={u * 100:.0f}%")
            lines.append(f"  <terminal_utils>{' | '.join(util_parts)}</terminal_utils>")
        if show_coverage:
            lines.append(f"  <cps_served>{r.coverage_count}/{total_cps}</cps_served>")
        if p.skip_capacity_constraints:
            note = "sem restrições de capacidade" if is_pt else "no capacity constraints"
            lines.append(f"  <note>{note}</note>")
        if p.volume_redistribution:
            note = "redistribuição ativa" if is_pt else "redistribution active"
            lines.append(f"  <note>{note}</note>")
        lines.append("</scenario>")
        scenario_tags.append("\n".join(lines))

    parts = ["<scenarios>\n" + "\n\n".join(scenario_tags) + "\n</scenarios>"]
    if two_level_tags:
        parts.append("<coverage_comparisons>\n" + "\n\n".join(two_level_tags) + "\n</coverage_comparisons>")
    return "\n\n".join(parts)


# ── Pydantic output models ────────────────────────────────────────────────────

class TableRow(BaseModel):
    label: str
    trucks: int
    cost: float
    delta_trucks: Optional[int] = None
    delta_cost: Optional[float] = None
    terminal_utils: dict[str, Optional[float]] = Field(default_factory=dict)
    coverage_count: Optional[int] = None
    inactive_terminals: list[str] = Field(default_factory=list)
    query_text: str = ""


class DataExpertOutput(BaseModel):
    table_rows: list[TableRow]
    narrative: str
    two_level_text: list[str] = Field(default_factory=list)


class _NarrativeOnly(BaseModel):
    narrative: str


# ── Pre-computation helpers ───────────────────────────────────────────────────

def _repair_parameter_direction_narrative(
    narrative: str,
    history: list[PipelineResult],
    lang: str,
) -> str:
    baseline = next((r for r in history if r.scenario_params.is_baseline), None)
    if not baseline or not narrative:
        return narrative

    base_availability = baseline.scenario_params.availability
    has_availability_increase = any(
        not r.scenario_params.is_baseline
        and r.scenario_params.availability > base_availability + 0.001
        for r in history
    )
    has_availability_decrease = any(
        not r.scenario_params.is_baseline
        and r.scenario_params.availability < base_availability - 0.001
        for r in history
    )
    if not has_availability_increase or has_availability_decrease:
        return narrative

    if lang == "pt":
        replacements = {
            "redução da disponibilidade": "aumento da disponibilidade",
            "redução de disponibilidade": "aumento de disponibilidade",
            "queda da disponibilidade": "aumento da disponibilidade",
            "queda de disponibilidade": "aumento de disponibilidade",
            "reduzir a disponibilidade": "aumentar a disponibilidade",
            "reduziu a disponibilidade": "aumentou a disponibilidade",
        }
    else:
        replacements = {
            "availability reduction": "availability increase",
            "reduction in availability": "increase in availability",
            "availability drop": "availability increase",
            "drop in availability": "increase in availability",
            "reduced availability": "increased availability",
            "decreased availability": "increased availability",
        }

    repaired = narrative
    for bad, good in replacements.items():
        repaired = repaired.replace(bad, good)
        repaired = repaired.replace(bad.capitalize(), good.capitalize())
    return repaired


def _build_table_rows(
    history: list[PipelineResult],
    network: NetworkData,
    profile: SessionProfile,
) -> list[TableRow]:
    baseline = next((r for r in history if r.scenario_params.is_baseline), None)
    whatifs = [r for r in history if not r.scenario_params.is_baseline]
    rows: list[TableRow] = []

    if baseline:
        utils = compute_terminal_utilizations(baseline, network)
        rows.append(TableRow(
            label="Baseline",
            trucks=baseline.milp_result.trucks,
            cost=baseline.milp_result.total_cost,
            terminal_utils=utils,
            coverage_count=baseline.milp_result.coverage_count if profile.has_coverage_scenarios else None,
            query_text=baseline.query_text,
        ))

    base_trucks = baseline.milp_result.trucks if baseline else 0
    base_cost = baseline.milp_result.total_cost if baseline else 0.0

    for i, result in enumerate(whatifs, 1):
        p = result.scenario_params
        r = result.milp_result
        if result.coverage_comparison is not None:
            continue
        utils = compute_terminal_utilizations(result, network)
        inactive = [tid for tid, active in p.terminals_active.items() if not active]
        rows.append(TableRow(
            label=f"Q{i}",
            trucks=r.trucks,
            cost=r.total_cost,
            delta_trucks=r.trucks - base_trucks if baseline else None,
            delta_cost=r.total_cost - base_cost if baseline else None,
            terminal_utils=utils,
            coverage_count=r.coverage_count if profile.has_coverage_scenarios else None,
            inactive_terminals=inactive,
            query_text=result.query_text,
        ))

    return rows


def _build_two_level_text(
    history: list[PipelineResult],
    network: NetworkData,
    lang: str,
) -> list[str]:
    is_pt = lang == "pt"
    total_cps = len(network.cp_ids)
    whatifs = [r for r in history if not r.scenario_params.is_baseline]
    blocks: list[str] = []

    for i, result in enumerate(whatifs, 1):
        if result.coverage_comparison is None:
            continue
        cov = result.coverage_comparison
        p = result.scenario_params
        label = f"Q{i}"
        count_a = cov.coverage_count_a
        count_b = cov.coverage_count_b
        cost_a = cov.cost_a
        cost_b = cov.cost_b
        delta_pcs = count_b - count_a
        delta_cost = cost_b - cost_a
        pct_a = count_a / total_cps * 100 if total_cps else 0
        pct_b = count_b / total_cps * 100 if total_cps else 0
        per_pc_str = ""
        if delta_pcs != 0:
            per_pc = abs(delta_cost / delta_pcs)
            per_pc_str = (
                f"   (${per_pc:,.0f}/mês por PC adicional)"
                if is_pt else
                f"   (${per_pc:,.0f}/month per additional CP)"
            )
        sign_pc = "+" if delta_pcs >= 0 else ""
        sign_cost = "+" if delta_cost >= 0 else "-"

        if is_pt:
            lines = [
                f"── Comparação de Cobertura ({label})",
                f"  Nível A ({pct_a:.0f}%): {count_a} PCs   Custo: ${cost_a:,.0f}/mês",
                f"  Nível B ({pct_b:.0f}%): {count_b} PCs  Custo: ${cost_b:,.0f}/mês",
                f"  Delta A→B: {sign_pc}{delta_pcs} PCs   {sign_cost}${abs(delta_cost):,.0f}/mês{per_pc_str}",
            ]
        else:
            lines = [
                f"── Coverage Comparison ({label})",
                f"  Level A ({pct_a:.0f}%): {count_a} CPs   Cost: ${cost_a:,.0f}/month",
                f"  Level B ({pct_b:.0f}%): {count_b} CPs  Cost: ${cost_b:,.0f}/month",
                f"  Delta A→B: {sign_pc}{delta_pcs} CPs   {sign_cost}${abs(delta_cost):,.0f}/month{per_pc_str}",
            ]
        blocks.append("\n".join(lines))

    return blocks


# ── Profile hint ──────────────────────────────────────────────────────────────

def _build_profile_hint(profile: SessionProfile, lang: str) -> str:
    is_pt = lang == "pt"
    parts: list[str] = []

    if profile.has_parametric_only:
        parts.append(
            "Sessão paramétrica: nenhum cenário altera demanda ou routing. Trate como sensibilidades independentes quando as perguntas testarem premissas distintas."
            if is_pt else
            "Parametric session: no scenario changes demand or routing. Treat as independent sensitivities when the questions test distinct assumptions."
        )
    elif profile.has_demand_variation and profile.has_coverage_scenarios:
        parts.append(
            "Sessão mista: múltiplos tipos de variação. Priorize padrões que cruzam ≥2 cenários."
            if is_pt else
            "Mixed session: multiple variation types. Prioritise patterns spanning ≥2 scenarios."
        )
    elif profile.has_demand_variation:
        parts.append(
            "Sessão de demanda: cenários variam multipliers por terminal. Foco em threshold de overflow e comparação de estratégias."
            if is_pt else
            "Demand session: scenarios vary per-terminal multipliers. Focus on overflow thresholds and strategy comparison."
        )
    elif profile.has_coverage_scenarios:
        parts.append(
            "Sessão de cobertura: cenários variam cobertura mínima ou orçamento. Foco em fronteira de eficiência custo/PC."
            if is_pt else
            "Coverage session: scenarios vary minimum coverage or budget. Focus on cost/CP efficiency frontier."
        )

    if profile.has_terminal_closure:
        parts.append(
            "Há cenário(s) com terminal fechado."
            if is_pt else
            "There are terminal-closure scenario(s)."
        )
    if profile.has_redistribution:
        parts.append(
            "Há cenário(s) com redistribuição de volume."
            if is_pt else
            "There are volume-redistribution scenario(s)."
        )
    if profile.has_two_level:
        parts.append(
            "Há comparação de dois níveis de cobertura."
            if is_pt else
            "There is a two-level coverage comparison."
        )

    return " ".join(parts)


# ── Pre-computed pattern verification ────────────────────────────────────────

def _is_parametric_probe_session(whatifs: list[PipelineResult]) -> bool:
    if len(whatifs) < 2:
        return False
    return all(
        not any(v != 1.0 for v in r.scenario_params.terminal_demand_multipliers.values())
        and all(r.scenario_params.terminals_active.values())
        and not r.scenario_params.volume_redistribution
        and r.scenario_params.min_coverage_count is None
        and r.scenario_params.budget is None
        and r.coverage_comparison is None
        for r in whatifs
    )


def _format_parametric_change_group(
    label: str,
    p: ScenarioParams,
    base: ScenarioParams,
    is_pt: bool,
) -> tuple[list[str], list[str]]:
    deteriorations: list[str] = []
    improvements: list[str] = []

    if p.payload < base.payload - 0.001:
        deteriorations.append(
            f"{label} queda no payload de {base.payload:.0f}t para {p.payload:.0f}t"
            if is_pt else
            f"{label} payload decreased from {base.payload:.0f}t to {p.payload:.0f}t"
        )
    elif p.payload > base.payload + 0.001:
        improvements.append(
            f"{label} aumento no payload de {base.payload:.0f}t para {p.payload:.0f}t"
            if is_pt else
            f"{label} payload increased from {base.payload:.0f}t to {p.payload:.0f}t"
        )

    if p.availability < base.availability - 0.001:
        deteriorations.append(
            f"{label} queda na disponibilidade de {base.availability * 100:.0f}% para {p.availability * 100:.0f}%"
            if is_pt else
            f"{label} availability decreased from {base.availability * 100:.0f}% to {p.availability * 100:.0f}%"
        )
    elif p.availability > base.availability + 0.001:
        improvements.append(
            f"{label} aumento de disponibilidade de {base.availability * 100:.0f}% para {p.availability * 100:.0f}%"
            if is_pt else
            f"{label} availability increased from {base.availability * 100:.0f}% to {p.availability * 100:.0f}%"
        )

    component_groups = (
        ("custo variável", "variable cost", p.variable_cost_components, base.variable_cost_components),
        ("custo fixo", "fixed cost", p.fixed_cost_components, base.fixed_cost_components),
    )
    for pt_name, en_name, current, baseline_components in component_groups:
        if not current or not baseline_components:
            continue
        for key, val in current.items():
            base_val = baseline_components.get(key)
            if base_val is None or base_val <= 0 or abs(val - base_val) <= 0.0001:
                continue
            pct = (val / base_val - 1) * 100
            target = deteriorations if pct > 0 else improvements
            target.append(
                f"{label} aumento de {pct:.0f}% em {pt_name} {key}"
                if is_pt and pct > 0 else
                f"{label} redução de {abs(pct):.0f}% em {pt_name} {key}"
                if is_pt else
                f"{label} {en_name} {key} increased {pct:.0f}%"
                if pct > 0 else
                f"{label} {en_name} {key} decreased {abs(pct):.0f}%"
            )

    return deteriorations, improvements

def _precompute_pattern_facts(
    history: list[PipelineResult],
    network: NetworkData,
    lang: str,
) -> str:
    """Build a verified-facts block for the LLM prompt. All arithmetic is done here."""
    is_pt = lang == "pt"
    baseline = next((r for r in history if r.scenario_params.is_baseline), None)
    whatifs = [r for r in history if not r.scenario_params.is_baseline]
    lines: list[str] = []

    if baseline and whatifs:
        change_lines: list[str] = []
        base_params = baseline.scenario_params
        for i, result in enumerate(whatifs, 1):
            p = result.scenario_params
            changes: list[str] = []
            if abs(p.payload - base_params.payload) > 0.001:
                direction = (
                    "queda" if is_pt and p.payload < base_params.payload
                    else "aumento" if is_pt
                    else "decreased" if p.payload < base_params.payload
                    else "increased"
                )
                changes.append(
                    f"payload {direction} de {base_params.payload:.0f}t para {p.payload:.0f}t"
                    if is_pt else
                    f"payload {direction} from {base_params.payload:.0f}t to {p.payload:.0f}t"
                )
            if abs(p.availability - base_params.availability) > 0.001:
                direction = (
                    "aumentou" if is_pt and p.availability > base_params.availability
                    else "caiu" if is_pt
                    else "increased" if p.availability > base_params.availability
                    else "decreased"
                )
                changes.append(
                    f"disponibilidade {direction} de {base_params.availability * 100:.0f}% para {p.availability * 100:.0f}%"
                    if is_pt else
                    f"availability {direction} from {base_params.availability * 100:.0f}% to {p.availability * 100:.0f}%"
                )
            if p.variable_cost_components and base_params.variable_cost_components:
                for key, val in p.variable_cost_components.items():
                    base_val = base_params.variable_cost_components.get(key)
                    if base_val is None or base_val <= 0 or abs(val - base_val) <= 0.0001:
                        continue
                    pct = (val / base_val - 1) * 100
                    direction = (
                        "aumentou" if is_pt and pct > 0
                        else "reduziu" if is_pt
                        else "increased" if pct > 0
                        else "decreased"
                    )
                    changes.append(
                        f"custo variável {key} {direction} {'+' if pct >= 0 else ''}{pct:.0f}%"
                        if is_pt else
                        f"variable cost {key} {direction} {'+' if pct >= 0 else ''}{pct:.0f}%"
                    )
            if changes:
                change_lines.append(f"Q{i}: " + "; ".join(changes))
        if change_lines:
            lines.append(
                "MUDANÇAS DE PREMISSA: " + " | ".join(change_lines)
                if is_pt else
                "PARAMETER CHANGES: " + " | ".join(change_lines)
            )

        if _is_parametric_probe_session(whatifs):
            deteriorations: list[str] = []
            improvements: list[str] = []
            favorable_results: list[str] = []
            for i, result in enumerate(whatifs, 1):
                label = f"Q{i}"
                det, imp = _format_parametric_change_group(
                    label,
                    result.scenario_params,
                    base_params,
                    is_pt,
                )
                deteriorations.extend(det)
                improvements.extend(imp)
                dt = result.milp_result.trucks - baseline.milp_result.trucks
                dc = result.milp_result.total_cost - baseline.milp_result.total_cost
                if dt < 0 and dc < 0 and imp:
                    improvement_text = "; ".join(imp)
                    favorable_results.append(
                        f"{improvement_text} reduziu frota e custo vs baseline"
                        if is_pt else
                        f"{improvement_text} reduced fleet and cost vs baseline"
                    )
            lines.append(
                "INTERPRETAÇÃO DA SESSÃO: sensibilidades independentes; não trate como sequência causal única nem infira correlação entre variáveis sem fato verificado."
                if is_pt else
                "SESSION INTERPRETATION: independent sensitivity probes; do not treat as a single causal sequence or infer correlation between variables without a verified fact."
            )
            if deteriorations:
                lines.append(
                    "DETERIORAÇÕES TESTADAS: " + "; ".join(deteriorations)
                    if is_pt else
                    "TESTED DETERIORATIONS: " + "; ".join(deteriorations)
                )
            if improvements:
                lines.append(
                    "MELHORIAS/ALAVANCAS TESTADAS: " + "; ".join(improvements)
                    if is_pt else
                    "TESTED IMPROVEMENTS/LEVERS: " + "; ".join(improvements)
                )
            if favorable_results:
                lines.append(
                    "ALAVANCAS FAVORÁVEIS: " + "; ".join(favorable_results)
                    if is_pt else
                    "FAVORABLE LEVERS: " + "; ".join(favorable_results)
                )
            lines.append(
                "USO ANALÍTICO: quantificar impactos individuais de premissas distintas; não inferir que os cenários tenham correlação entre si."
                if is_pt else
                "ANALYTICAL USE: quantify individual impacts of distinct assumptions; do not infer that the scenarios are correlated."
            )

    # ── Overflow check ────────────────────────────────────────────────────────
    overflow_found: list[str] = []
    all_labeled = (
        [("Q0/Baseline", baseline)] if baseline else []
    ) + [(f"Q{i}", r) for i, r in enumerate(whatifs, 1)]
    for label, result in all_labeled:
        if result is None:
            continue
        for tid, u in compute_terminal_utilizations(result, network).items():
            if u is not None and u > 1.0:
                overflow_found.append(f"{label}/{tid}={u * 100:.0f}%")
    if overflow_found:
        tag = "sim" if is_pt else "yes"
        lines.append(f"OVERFLOW: {tag} — {', '.join(overflow_found)}")
    else:
        lines.append(
            "OVERFLOW: nenhum — padrões 'concentração de risco' e 'threshold não-linear' NÃO existem nesta sessão. Não os reporte."
            if is_pt else
            "OVERFLOW: none — patterns 'risk concentration' and 'non-linear threshold' DO NOT exist in this session. Do not report them."
        )

    # ── Near-capacity check (95–100%) ─────────────────────────────────────────
    near_cap_found: list[str] = []
    for label, result in all_labeled:
        if result is None:
            continue
        for tid, u in compute_terminal_utilizations(result, network).items():
            if u is not None and 0.95 <= u < 1.0:
                near_cap_found.append(f"{label}/{tid}={u * 100:.0f}%")
    if near_cap_found:
        tag = "sim" if is_pt else "yes"
        lines.append(f"NEAR_CAP: {tag} — {', '.join(near_cap_found)}")
    else:
        lines.append(
            "NEAR_CAP: nenhum — nenhum terminal entre 95% e 100% de capacidade. Não reporte alerta de proximidade de limite."
            if is_pt else
            "NEAR_CAP: none — no terminal between 95% and 100% capacity. Do not report near-capacity alerts."
        )

    # ── Divergência frota×custo ───────────────────────────────────────────────
    if baseline:
        base_trucks = baseline.milp_result.trucks
        base_cost = baseline.milp_result.total_cost
        divs: list[str] = []
        for i, r in enumerate(whatifs, 1):
            if r.coverage_comparison is not None:
                continue
            dt = r.milp_result.trucks - base_trucks
            dc = r.milp_result.total_cost - base_cost
            if (dt > 0 and dc < 0) or (dt < 0 and dc > 0):
                divs.append(
                    f"Q{i}: frota{'+' if dt >= 0 else ''}{dt}, custo{'+' if dc >= 0 else ''}${dc:,.0f}/mês"
                    if is_pt else
                    f"Q{i}: fleet{'+' if dt >= 0 else ''}{dt}, cost{'+' if dc >= 0 else ''}${dc:,.0f}/month"
                )
        lbl_div = "DIVERGÊNCIA frota×custo" if is_pt else "FLEET×COST DIVERGENCE"
        if divs:
            lines.append(f"{lbl_div}: {'; '.join(divs)}")
        else:
            lines.append(
                f"{lbl_div}: nenhuma — frota e custo movem na mesma direção em todos os cenários. Não reporte divergência."
                if is_pt else
                f"{lbl_div}: none — fleet and cost move in the same direction in all scenarios. Do not report divergence."
            )

    # ── Folga estrutural ──────────────────────────────────────────────────────
    # Terminal < 70% in ALL scenarios where it is active
    terminal_max_utils: dict[str, float] = {}
    for result in history:
        for tid, u in compute_terminal_utilizations(result, network).items():
            if u is not None:
                terminal_max_utils[tid] = max(terminal_max_utils.get(tid, 0.0), u)
    folga = [
        f"{tid} (max={v * 100:.0f}%)"
        for tid, v in terminal_max_utils.items()
        if v < 0.70
    ]
    lbl_folga = "FOLGA ESTRUTURAL" if is_pt else "STRUCTURAL SLACK"
    if folga:
        lines.append(
            f"{lbl_folga}: {', '.join(folga)} — abaixo de 70% em todos os cenários em que estão ativos"
            if is_pt else
            f"{lbl_folga}: {', '.join(folga)} — below 70% in all scenarios where active"
        )
    else:
        lines.append(
            f"{lbl_folga}: nenhum terminal abaixo de 70% em todos os cenários. Não reporte folga estrutural."
            if is_pt else
            f"{lbl_folga}: no terminal below 70% in all scenarios. Do not report structural slack."
        )

    # ── Impact ranking ────────────────────────────────────────────────────────
    if baseline and whatifs:
        base_trucks = baseline.milp_result.trucks
        base_cost = baseline.milp_result.total_cost
        impacts = [
            (f"Q{i}", r.milp_result.trucks - base_trucks, r.milp_result.total_cost - base_cost)
            for i, r in enumerate(whatifs, 1)
            if r.coverage_comparison is None
        ]
        if impacts:
            by_fleet = sorted(impacts, key=lambda x: abs(x[1]), reverse=True)
            by_cost = sorted(impacts, key=lambda x: abs(x[2]), reverse=True)
            fleet_rank = "; ".join(
                f"{lbl}: {'+' if dt >= 0 else ''}{dt}" for lbl, dt, _ in by_fleet
            )
            suffix = "/mês" if is_pt else "/month"
            cost_rank = "; ".join(
                f"{lbl}: {'+' if dc >= 0 else ''}${dc:,.0f}{suffix}" for lbl, _, dc in by_cost
            )
            lbl_fr = "RANKING IMPACTO NA FROTA (maior→menor)" if is_pt else "FLEET IMPACT RANKING (largest→smallest)"
            lbl_cr = "RANKING IMPACTO NO CUSTO (maior→menor)" if is_pt else "COST IMPACT RANKING (largest→smallest)"
            lines.append(f"{lbl_fr}: {fleet_rank}")
            lines.append(f"{lbl_cr}: {cost_rank}")

    header = (
        "FATOS VERIFICADOS (calculados pelo sistema — use SOMENTE estes valores, não faça nenhuma conta):"
        if is_pt else
        "VERIFIED FACTS (computed by the system — use ONLY these values, do not compute anything):"
    )
    return header + "\n" + "\n".join(lines)


# ── System prompts ────────────────────────────────────────────────────────────

_SYSTEM_PROMPT_PT = """\
## Papel
Você é um analista de decisão em logística de frota. Recebe dados de cenários de planejamento, \
as perguntas originais do usuário para cada cenário, e um bloco de FATOS VERIFICADOS pré-calculados \
pelo sistema. Sua tarefa é sintetizar uma narrativa de 3 a 4 frases que oriente a tomada de decisão.

## Contrato do Agente
- Papel: sintetizador comparativo de cenários já executados.
- Entrada: histórico serializado de cenários, classificação da sessão e FATOS VERIFICADOS calculados pelo Python.
- Limite de decisão: comparar somente cenários existentes; não criar cenários novos, não recomputar métricas e não usar memória implícita fora do histórico recebido.
- Padrão de saída: JSON com uma narrativa curta, fundamentada nos fatos verificados e útil para decisão.

## Decomposição obrigatória da tarefa
1. Leia a classificação da sessão para entender o tipo de análise comparativa.
2. Use as perguntas originais para nomear a intenção de cada cenário.
3. Verifique quais padrões foram confirmados em FATOS VERIFICADOS.
4. Escolha somente os padrões confirmados que ajudam a decisão.
5. Escreva a narrativa seguindo a estrutura obrigatória, sem fazer novas contas.
6. Faça uma checagem final: nenhum padrão não confirmado, nenhum número calculado, nenhum cenário inventado.

## REGRA ABSOLUTA 1 — sem aritmética
NUNCA compute razões, divisões, multiplicações, percentuais ou médias a partir dos dados brutos.
Use APENAS os números que já aparecem no bloco <scenarios> ou no bloco FATOS VERIFICADOS.

## REGRA ABSOLUTA 2 — respeite as precondições
Cada padrão só existe se o bloco FATOS VERIFICADOS o confirmar explicitamente.
Se o bloco diz "não existe nesta sessão", "nenhum" ou "não reporte" → NÃO reporte esse padrão.

## REGRA ABSOLUTA 3 — use o contexto da pergunta
Cada cenário tem uma tag <query> com a pergunta original do usuário. Use esse contexto para nomear \
a intenção do usuário na narrativa (ex: "a hipótese de 2 motoristas/caminhão" em vez de "Q1").

## REGRA ABSOLUTA 4 — período monetário obrigatório
Todos os valores monetários desta sessão são custos mensais. Inclua "/mês" imediatamente após \
cada cifra em dólares na narrativa, sem exceção (ex: "$7,27 milhões/mês", "$126,7 mil/mês").

## REGRA ABSOLUTA 5 — direção das premissas
Quando FATOS VERIFICADOS trouxer "MUDANÇAS DE PREMISSA", preserve exatamente a direção indicada. \
Se a disponibilidade aparece como "aumentou de 85% para 88%", é proibido chamar isso de redução, queda \
ou deterioração.

## REGRA ABSOLUTA 6 — sessões sem estrutura causal
Quando FATOS VERIFICADOS trouxer "INTERPRETAÇÃO DA SESSÃO: sensibilidades independentes", trate os \
cenários como testes individuais de premissas. Não force uma correlação, uma sequência causal única, \
nem uma direção estratégica única. Use ranking apenas como tamanho de impacto, não como prova de relação \
entre as premissas.

## Classificação da sessão
{profile_hint}

## Padrões disponíveis (use apenas os confirmados pelo bloco FATOS VERIFICADOS)
1. Concentração de risco — OVERFLOW confirmado
2. Divergência frota×custo — DIVERGÊNCIA confirmada (frota↓+custo↑ ou frota↑+custo↓)
3. Folga estrutural — FOLGA ESTRUTURAL confirmada (terminal <70% em todos os cenários ativos)
4. Threshold não-linear — OVERFLOW em dois cenários consecutivos cruzando 100%
5. Ranking de sensibilidade — use RANKING IMPACTO do bloco de fatos
6. Fronteira de cobertura — cenários com min_coverage ou orçamento
7. Fechamento vs. absorção — cenários com terminal INATIVO
8. Ganho de redistribuição — cenários com redistribuição=ativa
9. Alerta de proximidade de limite — NEAR_CAP confirmado (terminal entre 95% e 100% de capacidade)

## Estrutura obrigatória da narrativa (3 a 4 frases)
Se FATOS VERIFICADOS indicar sensibilidades independentes:
1. Diga que a sessão reúne sensibilidades independentes, não uma sequência causal única.
2. Separe deteriorações testadas de melhorias/alavancas favoráveis usando os blocos DETERIORAÇÕES TESTADAS e ALAVANCAS FAVORÁVEIS.
3. Conclua que o uso principal é quantificar impactos individuais de premissas distintas, não inferir correlação.
Caso contrário:
1. **Veredito competitivo**: alguma variação melhorou frota E custo simultaneamente em relação ao baseline? Se não, diga isso explicitamente.
2. **Alavanca dominante**: qual variável (da <query> do cenário com maior impacto no RANKING) gerou o maior efeito, e em qual dimensão (frota ou custo)?
3. **Direção operacional**: somente sugira uma direção se houver fato verificado que conecte os cenários; se não houver, diga que os cenários devem ser lidos individualmente.
4. (Opcional) Apenas se houver um trade-off relevante não coberto nas frases anteriores.

## Anti-padrões proibidos
- Computar qualquer valor não fornecido nos blocos de dados
- Reportar padrão que FATOS VERIFICADOS diz não existir
- Listar números sem conclusão operacional
- Reafirmar o baseline como insight
- Nomear cenário apenas pelo label (Q1, Q2) sem mencionar sua intenção original

## Formato de saída
Retorne JSON com:
- narrative: string com a narrativa completa de 3 a 4 frases. Máximo 120 palavras.\
"""

_SYSTEM_PROMPT_EN = """\
## Role
You are a fleet logistics decision analyst. You receive planning scenario data, the user's original \
queries for each scenario, and a VERIFIED FACTS block pre-computed by the system. Your task is to \
synthesise a 3–4 sentence narrative that guides the decision-making process.

## Agent Contract
- Role: comparative synthesizer for scenarios that have already been run.
- Input: serialized scenario history, session classification, and VERIFIED FACTS computed by Python.
- Decision boundary: compare only existing scenarios; do not create new scenarios, recompute metrics, or use implicit memory outside the supplied history.
- Output standard: JSON with a short narrative grounded in verified facts and useful for decision-making.

## Required Task Decomposition
1. Read the session classification to understand the type of comparative analysis.
2. Use the original user questions to name each scenario's intent.
3. Check which patterns were confirmed in VERIFIED FACTS.
4. Select only confirmed patterns that help the decision.
5. Write the narrative following the required structure, without doing new arithmetic.
6. Run a final check: no unconfirmed pattern, no computed number, no invented scenario.

## ABSOLUTE RULE 1 — no arithmetic
NEVER compute ratios, divisions, multiplications, percentages, or averages from raw data.
Use ONLY numbers that already appear in the <scenarios> block or the VERIFIED FACTS block.

## ABSOLUTE RULE 2 — respect preconditions
Each pattern only exists if the VERIFIED FACTS block confirms it explicitly.
If the block says "does not exist in this session", "none", or "do not report" → do NOT report that pattern.

## ABSOLUTE RULE 3 — use query context
Each scenario has a <query> tag with the user's original question. Use that context to name the \
user's intent in the narrative (e.g. "the 2-drivers/truck hypothesis" instead of "Q1").

## ABSOLUTE RULE 4 — mandatory time basis on monetary figures
All monetary values in this session are monthly costs. Append "/month" immediately after every \
dollar figure in the narrative, without exception (e.g. "$7.27 million/month", "$126.7k/month").

## ABSOLUTE RULE 5 — parameter-change direction
When VERIFIED FACTS includes "PARAMETER CHANGES", preserve the stated direction exactly. If availability \
"increased from 85% to 88%", never describe it as a reduction, drop, or deterioration.

## ABSOLUTE RULE 6 — sessions without causal structure
When VERIFIED FACTS includes "SESSION INTERPRETATION: independent sensitivity probes", treat the \
scenarios as individual assumption tests. Do not force a correlation, a single causal sequence, or a \
single strategic direction. Use rankings only as impact size, not as evidence that assumptions are related.

## Session classification
{profile_hint}

## Available patterns (use only those confirmed by the VERIFIED FACTS block)
1. Risk concentration — OVERFLOW confirmed
2. Fleet×cost divergence — DIVERGENCE confirmed (fleet↓+cost↑ or fleet↑+cost↓)
3. Structural slack — STRUCTURAL SLACK confirmed (terminal <70% in all active scenarios)
4. Non-linear threshold — OVERFLOW in two consecutive scenarios crossing 100%
5. Sensitivity ranking — use IMPACT RANKING from verified facts
6. Coverage frontier — scenarios with min_coverage or budget
7. Closure vs. absorption — scenarios with INACTIVE terminal
8. Redistribution gain — scenarios with redistribution=active
9. Near-capacity alert — NEAR_CAP confirmed (terminal between 95% and 100% capacity)

## Required narrative structure (3–4 sentences)
If VERIFIED FACTS indicates independent sensitivity probes:
1. State that the session contains independent sensitivities, not a single causal sequence.
2. Separate tested deteriorations from favorable improvements/levers using TESTED DETERIORATIONS and FAVORABLE LEVERS.
3. Conclude that the main use is quantifying individual impacts of distinct assumptions, not inferring correlation.
Otherwise:
1. **Competitive verdict**: did any variation improve both fleet AND cost simultaneously vs baseline? If not, state this explicitly.
2. **Dominant lever**: which variable (from the <query> of the highest-impact scenario in RANKING) produced the largest effect, and on which dimension (fleet or cost)?
3. **Operational direction**: suggest a direction only when a verified fact connects the scenarios; otherwise say the scenarios should be read individually.
4. (Optional) Only if there is a relevant trade-off not covered in the sentences above.

## Forbidden anti-patterns
- Computing any value not provided in the data blocks
- Reporting a pattern VERIFIED FACTS says does not exist
- Listing numbers without an operational conclusion
- Restating the baseline as an insight
- Naming a scenario only by its label (Q1, Q2) without mentioning its original intent

## Output format
Return JSON with:
- narrative: string with the full 3–4 sentence narrative. Maximum 120 words.\
"""


# ── Agent entry point ─────────────────────────────────────────────────────────

def run_data_expert_agent(
    serialized_scenarios: str,
    profile: SessionProfile,
    terminal_ids: list[str],
    lang: str,
    model: str,
    history: list[PipelineResult],
    network: NetworkData,
) -> DataExpertOutput:
    """Run the Data Expert: pre-compute table rows in Python, ask LLM only for insights."""
    table_rows = _build_table_rows(history, network, profile)
    two_level_text = _build_two_level_text(history, network, lang)

    provider, model_id = MODEL_REGISTRY.get(model, ("anthropic", "claude-haiku-4-5-20251001"))
    api_key = get_api_key(provider)

    profile_hint = _build_profile_hint(profile, lang)
    system_prompt = (
        _SYSTEM_PROMPT_PT if lang == "pt" else _SYSTEM_PROMPT_EN
    ).format(profile_hint=profile_hint)

    llm_config = make_model(provider, model_id, api_key, 2048, system_prompt)

    verified_facts = _precompute_pattern_facts(history, network, lang)

    if lang == "pt":
        user_prompt = f"CENÁRIOS:\n\n{serialized_scenarios}\n\n{verified_facts}"
    else:
        user_prompt = f"SCENARIOS:\n\n{serialized_scenarios}\n\n{verified_facts}"

    try:
        from ..app.feedback import load_examples, format_few_shot_block
        examples = load_examples("data_expert", lang, "session_analysis")
        few_shot_block = format_few_shot_block(examples, lang)
        if few_shot_block:
            user_prompt += few_shot_block
    except Exception:
        pass

    structured = invoke_structured(llm_config, user_prompt, _NarrativeOnly)
    narrative = structured.narrative if structured else ""
    narrative = _repair_parameter_direction_narrative(narrative, history, lang)

    return DataExpertOutput(
        table_rows=table_rows,
        narrative=narrative,
        two_level_text=two_level_text,
    )
