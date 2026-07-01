"""Rich CLI display — tables, panels, and formatted output."""

from __future__ import annotations

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box
import sys

from ..domain.data_types import (
    MILPResult,
    ModelResult,
    PipelineResult,
    ScenarioParams,
)
from ..llm_adapters.shock_response_agent import ShockResponseOutput
from .i18n import t

# Palette — optimised for black/dark terminal backgrounds
BLUE         = "bold dark_orange"
HEADER_STYLE = "bold white on dark_orange"
BORDER_STYLE = "dark_orange"
VALUE_STYLE  = "white"
LABEL_STYLE  = "color(250)"        # light-gray label text (readable on black)
MUTED_STYLE  = "color(244)"        # secondary / hint text (darker gray, still legible)
DELTA_POS    = "bright_green"      # improvement (fewer trucks / lower cost)
DELTA_NEG    = "bright_red"        # regression
DELTA_ZERO   = "color(250)"        # no change
MSG_OK       = "bright_green"      # success feedback
MSG_ERR      = "bright_red"        # error / invalid input
MSG_WARN     = "yellow"            # warning / disabled
MSG_INFO     = "bright_cyan"       # informational (baseline, model switch …)
MSG_QUIET    = "color(250)"        # low-priority feedback (quit, hints)


console = Console(highlight=False)


def _fmt_cost(val: float) -> str:
    return f"${val:,.0f}"


def _fmt_delta_cost(delta: float) -> tuple[str, str]:
    if delta > 0:
        return f"+{_fmt_cost(delta)}", DELTA_NEG
    if delta < 0:
        return f"-${abs(delta):,.0f}", DELTA_POS
    return "-", DELTA_ZERO


def _fmt_delta_trucks(delta: int) -> tuple[str, str]:
    if delta > 0:
        return f"+{delta}", DELTA_NEG
    if delta < 0:
        return str(delta), DELTA_POS
    return "-", DELTA_ZERO


def util_color_style(ratio: float) -> str:
    """Rich style string for a utilisation ratio (0–1): green/yellow/red/muted."""
    if ratio >= 0.95:
        return "bright_red"
    if ratio >= 0.70:
        return "yellow"
    if ratio >= 0.40:
        return "bright_green"
    return MUTED_STYLE


def _compute_weighted_volume_stats(
    network: "object",
    volumes: dict[str, dict[str, float]],
) -> tuple[float, float]:
    """Return weighted average cycle time and one-way distance for displayed volumes."""
    from ..domain.loader import NetworkData

    net: NetworkData = network
    total_volume = weighted_ct = weighted_dist = 0.0
    for cp, cp_vols in volumes.items():
        for t_id, volume in cp_vols.items():
            if volume <= 0:
                continue
            dist = net.distances[cp][t_id]
            ct = (
                net.cp_load_times[cp]
                + dist / net.speed_loaded
                + net.terminal_unload_times[t_id]
                + dist / net.speed_empty
            )
            total_volume += volume
            weighted_ct += volume * ct
            weighted_dist += volume * dist
    if total_volume <= 0:
        return 0.0, 0.0
    return weighted_ct / total_volume, weighted_dist / total_volume


def _print_weighted_volume_stats(
    network: "object",
    volumes: dict[str, dict[str, float]],
    language: str,
) -> None:
    ct_h, dist_km = _compute_weighted_volume_stats(network, volumes)
    if ct_h <= 0:
        return
    is_pt = language == "pt"
    ct_lbl = "Tempo médio ponderado de ciclo" if is_pt else "Weighted avg cycle time"
    dist_lbl = "Distância média ponderada" if is_pt else "Weighted avg distance"
    console.print(
        f"  [{MUTED_STYLE}]{ct_lbl}:[/{MUTED_STYLE}] [bold white]{ct_h:.2f} h[/bold white]"
        f"   [{MUTED_STYLE}]{dist_lbl}:[/{MUTED_STYLE}] [bold white]{dist_km:.0f} km[/bold white]"
    )


def print_summary_table(
    result: PipelineResult,
    baseline: "PipelineResult | None",
    language: str,
) -> None:
    """Print the 3-row summary table (± delta columns if not baseline)."""
    is_base = baseline is None or result is baseline
    query_label = f"Q{result.query_number}"
    dash = "-"

    title = (
        t("baseline_table_title", language)
        if is_base
        else t("table_title", language, query=query_label)
    )

    tbl = Table(
        title=title,
        box=box.SIMPLE_HEAD,
        border_style=BORDER_STYLE,
        header_style=HEADER_STYLE,
        title_style=BLUE,
        show_lines=False,
    )

    tbl.add_column(t("col_model", language), style=VALUE_STYLE, min_width=22)
    tbl.add_column(t("col_trucks", language), style=VALUE_STYLE, justify="right")
    if not is_base:
        tbl.add_column(t("col_delta_trucks", language), justify="right")
    tbl.add_column(t("col_cost", language), style=VALUE_STYLE, justify="right")
    if not is_base:
        tbl.add_column(t("col_delta_cost", language), justify="right")

    rows = [
        (t("row_lbl", language), result.lbl_result, baseline.lbl_result if baseline else None),
        (t("row_wct", language), result.wct_result, baseline.wct_result if baseline else None),
        (t("row_milp", language), None, None),
    ]
    is_two_level = result.coverage_comparison is not None

    for label, model_res, base_res in rows:
        if label == t("row_milp", language):
            _add_milp_row(tbl, result.milp_result,
                          baseline.milp_result if baseline else None,
                          label, is_base, language,
                          coverage_comparison=result.coverage_comparison)
        elif model_res is not None:
            if is_two_level:
                continue
            trucks_str = str(model_res.trucks)
            cost_str = _fmt_cost(model_res.total_cost)
            if is_base:
                tbl.add_row(label, trucks_str, cost_str)
            else:
                base_trucks = base_res.trucks if base_res else 0
                base_cost = base_res.total_cost if base_res else 0.0
                dt = model_res.trucks - base_trucks
                dc = model_res.total_cost - base_cost
                dt_str, dt_style = _fmt_delta_trucks(dt)
                dc_str, dc_style = _fmt_delta_cost(dc)
                tbl.add_row(
                    label,
                    trucks_str,
                    Text(dt_str, style=dt_style),
                    cost_str,
                    Text(dc_str, style=dc_style),
                )

    console.print(tbl)


def _add_milp_row(
    tbl: Table,
    milp: MILPResult,
    base_milp: "MILPResult | None",
    label: str,
    is_base: bool,
    language: str,
    *,
    coverage_comparison=None,
) -> None:
    infeasible_str = t("no_solution", language)
    dash = t("milp_infeasible_short", language)

    if not milp.feasible:
        if is_base:
            tbl.add_row(label, dash, f"[dim]{infeasible_str}[/dim]")
        else:
            tbl.add_row(label, dash, dash, f"[dim]{infeasible_str}[/dim]", dash)
        return

    trucks_str = str(milp.trucks)
    cost_str = _fmt_cost(milp.total_cost)

    if is_base or base_milp is None:
        tbl.add_row(label, trucks_str, cost_str)
    else:
        if coverage_comparison is not None:
            cov = coverage_comparison
            dt = cov.trucks_b - cov.trucks_a
            dc = cov.cost_difference
            dt_str, dt_style = _fmt_delta_trucks(dt)
            dc_str, dc_style = _fmt_delta_cost(dc)
            tbl.add_row(
                label,
                str(cov.trucks_b),
                Text(dt_str, style=dt_style),
                _fmt_cost(cov.cost_b),
                Text(dc_str, style=dc_style),
            )
            return
        if not base_milp.feasible:
            tbl.add_row(label, trucks_str, dash, cost_str, dash)
            return
        dt = milp.trucks - base_milp.trucks
        dc = milp.total_cost - base_milp.total_cost
        dt_str, dt_style = _fmt_delta_trucks(dt)
        dc_str, dc_style = _fmt_delta_cost(dc)
        tbl.add_row(
            label,
            trucks_str,
            Text(dt_str, style=dt_style),
            cost_str,
            Text(dc_str, style=dc_style),
        )


def print_param_recap(
    params: ScenarioParams,
    baseline_params: ScenarioParams,
    language: str,
) -> None:
    """Print the parameter-changed recap for what-if scenarios."""
    changes: list[tuple[str, str, str]] = []

    bp = baseline_params
    p = params

    if p.payload != bp.payload:
        changes.append((
            t("param_payload", language),
            f"{bp.payload} t",
            f"{p.payload} t",
        ))
    if p.speed_loaded != bp.speed_loaded:
        changes.append((
            t("param_speed_loaded", language),
            f"{bp.speed_loaded} km/h",
            f"{p.speed_loaded} km/h",
        ))
    if p.speed_empty != bp.speed_empty:
        changes.append((
            t("param_speed_empty", language),
            f"{bp.speed_empty} km/h",
            f"{p.speed_empty} km/h",
        ))
    if p.availability != bp.availability:
        changes.append((
            t("param_availability", language),
            f"{bp.availability*100:.0f}%",
            f"{p.availability*100:.0f}%",
        ))
    if p.overtime_hours != bp.overtime_hours:
        changes.append((
            t("param_overtime", language),
            f"{bp.overtime_hours} h",
            f"{p.overtime_hours} h",
        ))
    if p.working_days != bp.working_days:
        changes.append((
            t("param_working_days", language),
            str(bp.working_days),
            str(p.working_days),
        ))
    # Variable cost components — diff individually when both params carry them.
    if p.variable_cost_components and bp.variable_cost_components:
        for comp_key, new_val in p.variable_cost_components.items():
            old_val = bp.variable_cost_components.get(comp_key, new_val)
            if abs(new_val - old_val) > 0.0001:
                label = t(f"param_var_{comp_key}", language)
                changes.append((label, f"${old_val:.4f}/km", f"${new_val:.4f}/km"))
    elif abs(p.variable_cost_per_km - bp.variable_cost_per_km) > 0.001:
        changes.append((
            t("param_variable_cost", language),
            f"${bp.variable_cost_per_km:.2f}",
            f"${p.variable_cost_per_km:.2f}",
        ))

    # Fixed cost components — diff individually when both params carry them.
    if p.fixed_cost_components and bp.fixed_cost_components:
        for comp_key, new_val in p.fixed_cost_components.items():
            old_val = bp.fixed_cost_components.get(comp_key, new_val)
            if abs(new_val - old_val) > 0.01:
                label = t(f"param_fix_{comp_key}", language)
                changes.append((label, f"${old_val:,.2f}/mês", f"${new_val:,.2f}/mês"))
    elif abs(p.fixed_cost_per_truck_month - bp.fixed_cost_per_truck_month) > 1:
        changes.append((
            t("param_fixed_cost", language),
            f"${bp.fixed_cost_per_truck_month:,.0f}",
            f"${p.fixed_cost_per_truck_month:,.0f}",
        ))
    for tid in p.terminals_active:
        p_active = p.terminals_active.get(tid, True)
        bp_active = bp.terminals_active.get(tid, True)
        if p_active != bp_active:
            old_val = t("active", language) if bp_active else t("inactive", language)
            new_val = t("active", language) if p_active else t("inactive", language)
            changes.append((f"terminal {tid}", old_val, new_val))
    for tid, mul in p.terminal_demand_multipliers.items():
        bp_mul = bp.terminal_demand_multipliers.get(tid, 1.0)
        if abs(mul - bp_mul) > 0.0001:
            changes.append((
                f"{t('param_terminal_demand', language)} {tid}",
                f"{bp_mul * 100:.0f}%",
                f"{mul * 100:.0f}%",
            ))
    for tid, cap in p.terminal_volume_caps.items():
        bp_cap = bp.terminal_volume_caps.get(tid, 1.0)
        if abs(cap - bp_cap) > 0.0001:
            changes.append((
                f"{t('param_terminal_volume_cap', language)} {tid}",
                f"{bp_cap * 100:.0f}%",
                f"{cap * 100:.0f}%",
            ))

    if not changes:
        return

    tbl = Table(
        title=t("param_recap_title", language),
        box=box.MINIMAL,
        border_style=BORDER_STYLE,
        header_style=HEADER_STYLE,
        title_style=BLUE,
        show_header=False,
    )
    tbl.add_column("param", style=LABEL_STYLE)
    tbl.add_column("from", justify="right", style=MUTED_STYLE)
    tbl.add_column("arrow", justify="center")
    tbl.add_column("to", justify="left", style="bold white")

    for name, old, new in changes:
        tbl.add_row(name, old, "->", new)

    console.print(tbl)


def print_served_cps(
    milp: "MILPResult",
    all_cp_ids: list[str],
    language: str,
) -> None:
    """Print served / unserved CP lists when coverage is partial."""
    if not milp.feasible or not milp.served_cps:
        return
    total = len(all_cp_ids)
    served = sorted(milp.served_cps)
    unserved = sorted(set(all_cp_ids) - set(served))
    if not unserved:
        return  # full coverage — nothing to highlight

    tbl = Table(
        title=t("served_cps_title", language, served=len(served), total=total),
        box=box.SIMPLE_HEAD,
        border_style=BORDER_STYLE,
        header_style=HEADER_STYLE,
        title_style=BLUE,
        show_header=False,
        padding=(0, 1),
    )
    tbl.add_column("label", style=LABEL_STYLE, min_width=18)
    tbl.add_column("cps",   style="white")

    tbl.add_row(
        t("served_cps_served", language),
        "  ".join(served),
    )
    tbl.add_row(
        Text(t("served_cps_unserved", language), style=MSG_ERR),
        Text("  ".join(unserved), style=MSG_ERR),
    )
    console.print(tbl)


def print_insight(insight: str, language: str) -> None:
    """Print the LLM insight as a conversational reply."""
    label = "Assistente" if language == "pt" else "Assistant"
    console.print(f"[white]{label}> {escape(insight)}[/white]")


def print_infeasible_explanation(
    milp: MILPResult, params: ScenarioParams, language: str
) -> None:
    """Print the no-feasible-solution explanation block."""
    lines = [
        f"[bold]{t('infeasible_header', language)}[/bold]",
        milp.infeasibility_reason or "",
        "",
        t("infeasible_suggestion", language),
    ]
    console.print(
        Panel(
            "\n".join(lines),
            border_style="red",
            padding=(0, 1),
        )
    )


def print_detail(
    model_name: str,
    model_result: "ModelResult | MILPResult | None",
    language: str,
    milp: bool = False,
    total_cps: int = 0,
) -> None:
    """Print detail metrics for one model."""
    if model_result is None:
        console.print(f"[dim]{t('detail_not_available', language)}[/dim]")
        return

    tbl = Table(
        title=t("detail_title", language, model=model_name),
        box=box.SIMPLE,
        border_style=BORDER_STYLE,
        header_style=HEADER_STYLE,
        title_style=BLUE,
        show_header=False,
    )
    tbl.add_column("metric", style=LABEL_STYLE, min_width=28)
    tbl.add_column("value", style="bold white", justify="right")

    if milp and isinstance(model_result, MILPResult):
        mr: MILPResult = model_result
        if not mr.feasible:
            console.print(f"[dim]{t('no_solution', language)}[/dim]")
            return
        tbl.add_row(t("detail_trucks", language), str(mr.trucks))
        tbl.add_row(t("detail_km", language), f"{mr.total_km:,.0f} km")
        tbl.add_row(t("detail_fixed", language), _fmt_cost(mr.fixed_cost))
        tbl.add_row(t("detail_variable", language), _fmt_cost(mr.variable_cost))
        tbl.add_row(t("detail_overtime", language), _fmt_cost(mr.overtime_cost_total))
        tbl.add_row(t("detail_total", language), _fmt_cost(mr.total_cost))
        coverage_str = f"{mr.coverage_count}/{total_cps}" if total_cps > 0 else str(mr.coverage_count)
        tbl.add_row(t("detail_coverage", language), coverage_str)
        if mr.cost_difference is not None:
            tbl.add_row(
                t("cost_diff_label_short", language),
                f"${mr.cost_difference:+,.0f}",
            )
    elif isinstance(model_result, ModelResult):
        mr2: ModelResult = model_result
        tbl.add_row(t("detail_trucks", language), str(mr2.trucks))
        tbl.add_row(t("detail_km", language), f"{mr2.total_km:,.0f} km")
        tbl.add_row(t("detail_fixed", language), _fmt_cost(mr2.fixed_cost))
        tbl.add_row(t("detail_variable", language), _fmt_cost(mr2.variable_cost))
        tbl.add_row(t("detail_overtime", language), _fmt_cost(mr2.overtime_cost_total))
        tbl.add_row(t("detail_total", language), _fmt_cost(mr2.total_cost))
        if mr2.weighted_cycle_time is not None:
            tbl.add_row(
                t("detail_weighted_ct", language),
                f"{mr2.weighted_cycle_time:.2f} h",
            )

    console.print(tbl)


def print_network_relocate(
    result: "PipelineResult",
    network: "object",
    language: str,
) -> None:
    """Print the optimised volume redistribution matrix with capacity and utilisation."""
    from ..domain.loader import NetworkData

    net: NetworkData = network
    assignments = result.milp_result.assignments
    is_pt = language == "pt"

    if not assignments:
        print_message(t("no_relocate_result", language), style=MSG_QUIET)
        return

    tbl = Table(
        title=t("relocate_grid_title", language),
        box=box.SIMPLE_HEAD, border_style=BORDER_STYLE,
        header_style=HEADER_STYLE, title_style=BLUE,
        show_lines=False,
    )

    pc_col  = "PC" if is_pt else "CP"
    cap_sfx = "t/mês" if is_pt else "t/mo"

    tbl.add_column(pc_col, style=LABEL_STYLE, width=6, no_wrap=True)
    tbl.add_column(
        f"Cap. {pc_col}\n({cap_sfx})",
        justify="right", style="white", min_width=8,
    )
    for t_id in net.terminal_ids:
        t_cap = net.terminal_capacities[t_id]
        tbl.add_column(
            f"[bold]{t_id}[/bold]\n{t_cap:,.0f} t",
            justify="right", style="white", min_width=9,
        )
    tbl.add_column(
        f"[bold]Volume[/bold]\n({cap_sfx})",
        justify="right", style="bold white", min_width=9,
    )
    tbl.add_column("Util %", justify="right", min_width=7)

    term_totals   = {t_id: 0.0 for t_id in net.terminal_ids}
    grand_vol     = 0.0
    grand_cp_cap  = 0.0

    for cp in net.cp_ids:
        cp_cap     = net.cp_capacities[cp]
        grand_cp_cap += cp_cap
        assigned_t = assignments.get(cp)
        cp_demand  = sum(net.demand[cp].values())

        row = [cp, f"{cp_cap:,.0f}"]
        if assigned_t:
            term_totals[assigned_t] += cp_demand
            grand_vol += cp_demand
            util = cp_demand / cp_cap if cp_cap > 0 else 0.0
            u_s  = util_color_style(util)
            for t_id in net.terminal_ids:
                if t_id == assigned_t:
                    row.append(f"[bold bright_white]{cp_demand:,.0f}[/bold bright_white]")
                else:
                    row.append(f"[color(240)]—[/color(240)]")
            row.append(f"{cp_demand:,.0f}")
            row.append(f"[{u_s}]{util*100:.0f}%[/{u_s}]")
        else:
            for _ in net.terminal_ids:
                row.append(f"[{MUTED_STYLE}]n/a[/{MUTED_STYLE}]")
            row.append(f"[{MUTED_STYLE}]—[/{MUTED_STYLE}]")
            row.append(f"[{MUTED_STYLE}]—[/{MUTED_STYLE}]")
        tbl.add_row(*row)

    tbl.add_section()

    # Totals
    ov_util = grand_vol / grand_cp_cap if grand_cp_cap > 0 else 0.0
    ov_s    = util_color_style(ov_util)
    totals  = ["[bold]Total[/bold]", f"[bold]{grand_cp_cap:,.0f}[/bold]"]
    for t_id in net.terminal_ids:
        totals.append(f"[bold]{term_totals[t_id]:,.0f}[/bold]")
    totals += [f"[bold]{grand_vol:,.0f}[/bold]", f"[bold {ov_s}]{ov_util*100:.0f}%[/bold {ov_s}]"]
    tbl.add_row(*totals)

    # Terminal capacity
    t_cap_lbl = (
        f"[{MUTED_STYLE}]Cap. terminal[/{MUTED_STYLE}]" if is_pt
        else f"[{MUTED_STYLE}]Terminal cap.[/{MUTED_STYLE}]"
    )
    t_cap_row = [t_cap_lbl, ""]
    total_t_cap = 0.0
    for t_id in net.terminal_ids:
        tc = net.terminal_capacities[t_id]
        total_t_cap += tc
        t_cap_row.append(f"{tc:,.0f}")
    t_cap_row += [f"{total_t_cap:,.0f}", ""]
    tbl.add_row(*t_cap_row)

    # Terminal utilisation
    t_util_lbl = (
        f"[{MUTED_STYLE}]Util. terminal[/{MUTED_STYLE}]" if is_pt
        else f"[{MUTED_STYLE}]Terminal util.[/{MUTED_STYLE}]"
    )
    t_util_row = [t_util_lbl, ""]
    for t_id in net.terminal_ids:
        tc  = net.terminal_capacities[t_id]
        tu  = term_totals[t_id] / tc if tc > 0 else 0.0
        ts  = util_color_style(tu)
        t_util_row.append(f"[{ts}]{tu*100:.0f}%[/{ts}]")
    t_util_row += ["", ""]
    tbl.add_row(*t_util_row)

    console.print(tbl)

    # Weighted avg cycle time and distance
    wct_result = result.wct_result
    ct_h = wct_result.weighted_cycle_time if (wct_result and wct_result.weighted_cycle_time is not None) else None

    total_demand_d = weighted_dist_d = 0.0
    for cp in net.cp_ids:
        assigned_t = assignments.get(cp)
        if not assigned_t:
            continue
        cp_demand = sum(net.demand[cp].values())
        total_demand_d  += cp_demand
        weighted_dist_d += cp_demand * net.distances[cp][assigned_t]
    wdist_km = weighted_dist_d / total_demand_d if total_demand_d > 0 else 0.0

    ct_lbl   = "Tempo médio ponderado de ciclo" if is_pt else "Weighted avg cycle time"
    dist_lbl = "Distância média ponderada"       if is_pt else "Weighted avg distance"
    parts = []
    if ct_h is not None:
        parts.append(
            f"  [{MUTED_STYLE}]{ct_lbl}:[/{MUTED_STYLE}] [bold white]{ct_h:.2f} h[/bold white]"
        )
    if wdist_km > 0:
        parts.append(
            f"[{MUTED_STYLE}]{dist_lbl}:[/{MUTED_STYLE}] [bold white]{wdist_km:.0f} km[/bold white]"
        )
    if parts:
        console.print("   ".join(parts))


def print_volume_matrix(
    milp: MILPResult,
    network: "object",
    terminals_active: dict[str, bool],
    language: str,
    title: "str | None" = None,
) -> None:
    """Print CP × terminal volume table for any scenario that changes CP routing or coverage."""
    from ..domain.loader import NetworkData

    if not milp.feasible or not milp.volumes:
        return

    net: NetworkData = network
    is_pt = language == "pt"
    closed_lbl = "fechado" if is_pt else "closed"
    if title is None:
        title = (
            "Distribuição de Volume — Terminais Ativos"
            if is_pt else
            "Volume Distribution — Active Terminals"
        )

    tbl = Table(
        title=title,
        box=box.SIMPLE_HEAD, border_style=BORDER_STYLE,
        header_style=HEADER_STYLE, title_style=BLUE,
        show_lines=False,
    )

    pc_col  = "PC" if is_pt else "CP"
    cap_sfx = "t/mês" if is_pt else "t/mo"

    tbl.add_column(pc_col, style=LABEL_STYLE, width=6, no_wrap=True)
    tbl.add_column(
        f"Cap. {pc_col}\n({cap_sfx})",
        justify="right", style="white", min_width=8,
    )
    for t_id in net.terminal_ids:
        t_cap    = net.terminal_capacities[t_id]
        is_open  = terminals_active.get(t_id, True)
        col_hdr  = (
            f"[bold]{t_id}[/bold]\n{t_cap:,.0f} t"
            if is_open else
            f"[{MUTED_STYLE}]{t_id}  ✗[/{MUTED_STYLE}]\n[{MUTED_STYLE}]{closed_lbl}[/{MUTED_STYLE}]"
        )
        tbl.add_column(col_hdr, justify="right", style="white", min_width=9)
    tbl.add_column(
        f"[bold]Volume[/bold]\n({cap_sfx})",
        justify="right", style="bold white", min_width=9,
    )
    tbl.add_column("Util %", justify="right", min_width=7)

    term_totals  = {t_id: 0.0 for t_id in net.terminal_ids}
    grand_vol    = grand_cp_cap = 0.0

    for cp in net.cp_ids:
        cp_cap   = net.cp_capacities[cp]
        grand_cp_cap += cp_cap
        cp_vols  = milp.volumes.get(cp, {})
        cp_total = sum(cp_vols.values())
        grand_vol += cp_total
        util     = cp_total / cp_cap if cp_cap > 0 else 0.0
        u_s      = util_color_style(util)

        row = [cp, f"{cp_cap:,.0f}"]
        for t_id in net.terminal_ids:
            is_open = terminals_active.get(t_id, True)
            if not is_open:
                row.append(f"[{MUTED_STYLE}]—[/{MUTED_STYLE}]")
            else:
                vol = cp_vols.get(t_id, 0.0)
                term_totals[t_id] += vol
                row.append(f"{vol:,.0f}" if vol > 0 else f"[color(240)]—[/color(240)]")
        if cp_total > 0:
            row += [f"{cp_total:,.0f}", f"[{u_s}]{util*100:.0f}%[/{u_s}]"]
        else:
            row += [f"[{MUTED_STYLE}]—[/{MUTED_STYLE}]", f"[{MUTED_STYLE}]—[/{MUTED_STYLE}]"]
        tbl.add_row(*row)

    tbl.add_section()

    # Totals
    ov_util = grand_vol / grand_cp_cap if grand_cp_cap > 0 else 0.0
    ov_s    = util_color_style(ov_util)
    totals  = ["[bold]Total[/bold]", f"[bold]{grand_cp_cap:,.0f}[/bold]"]
    for t_id in net.terminal_ids:
        is_open = terminals_active.get(t_id, True)
        totals.append(
            f"[bold]{term_totals[t_id]:,.0f}[/bold]" if is_open
            else f"[{MUTED_STYLE}]—[/{MUTED_STYLE}]"
        )
    totals += [f"[bold]{grand_vol:,.0f}[/bold]", f"[bold {ov_s}]{ov_util*100:.0f}%[/bold {ov_s}]"]
    tbl.add_row(*totals)

    # Terminal capacity
    t_cap_lbl = (
        f"[{MUTED_STYLE}]Cap. terminal[/{MUTED_STYLE}]" if is_pt
        else f"[{MUTED_STYLE}]Terminal cap.[/{MUTED_STYLE}]"
    )
    t_cap_row = [t_cap_lbl, ""]
    total_t_cap = 0.0
    for t_id in net.terminal_ids:
        is_open = terminals_active.get(t_id, True)
        if not is_open:
            t_cap_row.append(f"[{MUTED_STYLE}]—[/{MUTED_STYLE}]")
        else:
            tc = net.terminal_capacities[t_id]
            total_t_cap += tc
            t_cap_row.append(f"{tc:,.0f}")
    t_cap_row += [f"{total_t_cap:,.0f}", ""]
    tbl.add_row(*t_cap_row)

    # Terminal utilisation
    t_util_lbl = (
        f"[{MUTED_STYLE}]Util. terminal[/{MUTED_STYLE}]" if is_pt
        else f"[{MUTED_STYLE}]Terminal util.[/{MUTED_STYLE}]"
    )
    t_util_row = [t_util_lbl, ""]
    for t_id in net.terminal_ids:
        is_open = terminals_active.get(t_id, True)
        if not is_open:
            t_util_row.append(f"[{MUTED_STYLE}]—[/{MUTED_STYLE}]")
        else:
            tc  = net.terminal_capacities[t_id]
            tu  = term_totals[t_id] / tc if tc > 0 else 0.0
            ts  = util_color_style(tu)
            t_util_row.append(f"[{ts}]{tu*100:.0f}%[/{ts}]")
    t_util_row += ["", ""]
    tbl.add_row(*t_util_row)

    console.print(tbl)
    _print_weighted_volume_stats(net, milp.volumes, language)
    console.print()


def print_capacity_warning(
    check: "CapacityCheckResult",
    network: "object",
    language: str,
) -> None:
    """Print a combined warning panel listing terminals and CPs that exceed capacity."""
    from ..domain.loader import NetworkData

    net: NetworkData = network
    is_pt = language == "pt"
    title = t("capacity_exceeded_title", language)

    n_t = len(check.overflowing_terminals)
    n_cp = len(check.overflowing_cps)

    if is_pt:
        parts = []
        if n_t:
            parts.append(f"{n_t} terminal(is)")
        if n_cp:
            parts.append(f"{n_cp} PC(s)")
        detail = f"A demanda efetiva excede a capacidade em {' e '.join(parts)}."
    else:
        parts = []
        if n_t:
            parts.append(f"{n_t} terminal(s)")
        if n_cp:
            parts.append(f"{n_cp} CP(s)")
        detail = f"Effective demand exceeds capacity at {' and '.join(parts)}."

    lines: list[str] = [f"[bold]{detail}[/bold]"]

    # ── Terminals ─────────────────────────────────────────────────────────────
    if n_t or check.ok_terminals:
        lines.append("")
        lines.append(f"  [bold {LABEL_STYLE}]{'Terminais:' if is_pt else 'Terminals:'}[/bold {LABEL_STYLE}]")
        overflow_map = {o.terminal_id: o for o in check.overflowing_terminals}
        for t_id in net.terminal_ids:
            if t_id in overflow_map:
                o = overflow_map[t_id]
                lines.append(
                    f"    [bold bright_red]Terminal {t_id}[/bold bright_red]"
                    f"  {o.effective_demand:,.0f} t efet."
                    f"  cap {o.capacity:,.0f} t"
                    f"  [bright_red]+{o.overflow_pct:.1f}%  ◀[/bright_red]"
                )
            elif t_id in check.ok_terminals:
                lines.append(
                    f"    [{MUTED_STYLE}]Terminal {t_id}[/{MUTED_STYLE}]"
                    f"  [bright_green]OK ✓[/bright_green]"
                )

    # ── CPs (overflowing only — list can be long) ─────────────────────────────
    if n_cp:
        lbl = "Pontos de coleta (excedidos):" if is_pt else "Collection points (exceeded):"
        lines.append("")
        lines.append(f"  [bold {LABEL_STYLE}]{lbl}[/bold {LABEL_STYLE}]")
        for o in sorted(check.overflowing_cps, key=lambda x: x.overflow_pct, reverse=True):
            pc_lbl = "PC" if is_pt else "CP"
            lines.append(
                f"    [bold bright_red]{pc_lbl} {o.cp_id}[/bold bright_red]"
                f"  {o.effective_demand:,.0f} t efet."
                f"  cap {o.capacity:,.0f} t"
                f"  [bright_red]+{o.overflow_pct:.1f}%  ◀[/bright_red]"
            )

    console.print(
        Panel(
            "\n".join(lines),
            title=f"[bold yellow]⚠ {title}[/bold yellow]",
            border_style="yellow",
            padding=(0, 1),
        )
    )


def print_over_capacity_highlights(
    milp_result: MILPResult,
    network: "object",
    language: str,
) -> None:
    """Print the action-map table for over-capacity runs."""
    from ..domain.loader import NetworkData

    net: NetworkData = network
    overflows = milp_result.terminal_overflows
    if not overflows:
        return

    is_pt = language == "pt"
    title = t("capacity_invest_title", language)
    col_routed = t("capacity_invest_col_routed", language)
    col_cap = "Capacidade" if is_pt else "Capacity"
    col_gap = t("capacity_invest_col_gap", language)
    col_gap_pct = t("capacity_invest_col_gap_pct", language)
    col_action = "Ação sugerida" if is_pt else "Suggested action"

    tbl = Table(
        title=title,
        box=box.SIMPLE_HEAD,
        border_style=BORDER_STYLE,
        header_style=HEADER_STYLE,
        title_style=BLUE,
        show_lines=False,
    )
    tbl.add_column("Terminal", style=LABEL_STYLE, min_width=12)
    tbl.add_column(col_routed, justify="right", style=VALUE_STYLE, min_width=12)
    tbl.add_column(col_cap, justify="right", style="white", min_width=12)
    tbl.add_column(col_gap, justify="right", min_width=10)
    tbl.add_column(col_gap_pct, justify="right", min_width=7)
    tbl.add_column(col_action, style="white", min_width=22)

    for t_id in net.terminal_ids:
        eff = overflows.get(t_id)
        if eff is None:
            continue
        cap = net.terminal_capacities[t_id]
        if eff > cap:
            gap = eff - cap
            pct = (eff / cap - 1.0) * 100
            action = (
                f"+{gap:,.0f} t/mês a absorver"
                if is_pt else
                f"+{gap:,.0f} t/month to absorb"
            )
            tbl.add_row(
                f"Terminal {t_id}",
                f"{eff:,.0f} t",
                f"{cap:,.0f} t",
                Text(f"+{gap:,.0f}", style=DELTA_NEG),
                Text(f"+{pct:.1f}%", style=DELTA_NEG),
                action,
            )

    console.print(tbl)


def print_cp_over_capacity_highlights(
    milp_result: MILPResult,
    network: "object",
    language: str,
) -> None:
    """Print the CP action-map table for over-capacity runs."""
    from ..domain.loader import NetworkData

    net: NetworkData = network
    overflows = milp_result.cp_overflows
    if not overflows:
        return

    is_pt = language == "pt"
    title = t("capacity_cp_invest_title", language)
    col_routed = t("capacity_invest_col_routed", language)
    col_cap = "Capacidade" if is_pt else "Capacity"
    col_gap = t("capacity_invest_col_gap", language)
    col_gap_pct = t("capacity_invest_col_gap_pct", language)
    col_action = "Ação sugerida" if is_pt else "Suggested action"
    pc_lbl = "PC" if is_pt else "CP"

    tbl = Table(
        title=title,
        box=box.SIMPLE_HEAD,
        border_style=BORDER_STYLE,
        header_style=HEADER_STYLE,
        title_style=BLUE,
        show_lines=False,
    )
    tbl.add_column(pc_lbl, style=LABEL_STYLE, min_width=8)
    tbl.add_column(col_routed, justify="right", style=VALUE_STYLE, min_width=12)
    tbl.add_column(col_cap, justify="right", style="white", min_width=12)
    tbl.add_column(col_gap, justify="right", min_width=10)
    tbl.add_column(col_gap_pct, justify="right", min_width=7)
    tbl.add_column(col_action, style="white", min_width=22)

    any_row = False
    for cp in net.cp_ids:
        eff = overflows.get(cp)
        if eff is None:
            continue
        cap = net.cp_capacities[cp]
        if cap > 0 and eff > cap:
            gap = eff - cap
            pct = (eff / cap - 1.0) * 100
            action = (
                f"+{gap:,.0f} t/mês a absorver"
                if is_pt else
                f"+{gap:,.0f} t/month to absorb"
            )
            tbl.add_row(
                cp,
                f"{eff:,.0f} t",
                f"{cap:,.0f} t",
                Text(f"+{gap:,.0f}", style=DELTA_NEG),
                Text(f"+{pct:.1f}%", style=DELTA_NEG),
                action,
            )
            any_row = True

    if any_row:
        console.print(tbl)


def print_commands_list(language: str, rows: list[tuple[str, str]]) -> None:
    """Print the slash-commands reference table."""
    tbl = Table(
        title=t("cmd_list_title", language),
        box=box.SIMPLE_HEAD,
        border_style=BORDER_STYLE,
        header_style=HEADER_STYLE,
        title_style=BLUE,
        show_lines=False,
        padding=(0, 1),
    )
    tbl.add_column(t("cmd_col_command", language), style="bold cyan", min_width=22, no_wrap=True)
    tbl.add_column(t("cmd_col_desc",    language), style=VALUE_STYLE)

    for cmd, desc in rows:
        tbl.add_row(cmd, desc)

    console.print(tbl)


def print_message(msg: str, style: str = "white") -> None:
    console.print(f"[{style}]{msg}[/]")


def render_data_expert(
    output: "DataExpertOutput",
    profile: "SessionProfile",
    terminal_ids: list[str],
    lang: str,
    total_cps: int = 0,
) -> None:
    """Render the Data Expert comparative table, two-level blocks, and insights panel."""
    from ..llm_adapters.data_expert import DataExpertOutput, SessionProfile  # noqa: F401

    is_pt = lang == "pt"
    rows = output.table_rows

    # Terminals always inactive across every scenario are omitted from the table
    always_inactive: set[str] = {
        tid for tid in terminal_ids
        if all(row.terminal_utils.get(tid) is None for row in rows)
    }
    visible_terminals = [tid for tid in terminal_ids if tid not in always_inactive]

    show_deltas = profile.has_parametric_only and not profile.has_coverage_scenarios
    show_terminal_utils = (
        not profile.has_parametric_only
        and bool(visible_terminals)
        and (profile.has_demand_variation or profile.has_terminal_closure or profile.has_redistribution)
    )
    show_pcs = profile.has_coverage_scenarios

    # ── Build table ───────────────────────────────────────────────────────────
    title = t("data_expert_table_title", lang)
    tbl = Table(
        title=title,
        box=box.SIMPLE_HEAD,
        border_style=BORDER_STYLE,
        header_style=HEADER_STYLE,
        title_style=BLUE,
        show_lines=False,
    )

    cen_lbl = "Cen." if is_pt else "Scen."
    fleet_lbl = "Frota" if is_pt else "Fleet"
    cost_lbl = t("col_cost", lang)

    tbl.add_column(cen_lbl, style=LABEL_STYLE, min_width=9, no_wrap=True)
    tbl.add_column(fleet_lbl, justify="right", style=VALUE_STYLE, min_width=6)

    if show_deltas:
        tbl.add_column("Δ Frota" if is_pt else "Δ Fleet", justify="right", min_width=8)

    tbl.add_column(cost_lbl, justify="right", style=VALUE_STYLE, min_width=12)

    if show_deltas:
        tbl.add_column("Δ Custo" if is_pt else "Δ Cost", justify="right", min_width=11)

    if show_terminal_utils:
        util_lbl_sfx = " util"
        for tid in visible_terminals:
            tbl.add_column(f"{tid}{util_lbl_sfx}", justify="right", min_width=9)

    if show_pcs:
        tbl.add_column("PCs" if is_pt else "CPs", justify="right", min_width=7)

    # ── Add rows ──────────────────────────────────────────────────────────────
    total_for_pcs = total_cps or len(terminal_ids)

    for row in rows:
        cells: list = [row.label, str(row.trucks)]

        if show_deltas:
            if row.delta_trucks is None:
                cells.append(Text("—", style=DELTA_ZERO))
            else:
                dt_str, dt_style = _fmt_delta_trucks(row.delta_trucks)
                cells.append(Text(dt_str, style=dt_style))

        cells.append(_fmt_cost(row.cost))

        if show_deltas:
            if row.delta_cost is None:
                cells.append(Text("—", style=DELTA_ZERO))
            else:
                dc_str, dc_style = _fmt_delta_cost(row.delta_cost)
                cells.append(Text(dc_str, style=dc_style))

        if show_terminal_utils:
            for tid in visible_terminals:
                util = row.terminal_utils.get(tid)
                if util is None:
                    inactive_lbl = "INATIVO" if is_pt else "INACTIVE"
                    cells.append(Text(inactive_lbl, style=MUTED_STYLE))
                else:
                    pct = util * 100
                    pct_str = f"{pct:.0f}%"
                    if pct > 100:
                        cells.append(Text(f"{pct_str} ◀", style=DELTA_NEG))
                    else:
                        style = util_color_style(util)
                        cells.append(f"[{style}]{pct_str}[/{style}]")

        if show_pcs:
            if row.coverage_count is not None and total_for_pcs > 0:
                cells.append(f"{row.coverage_count}/{total_for_pcs}")
            else:
                cells.append("—")

        tbl.add_row(*cells)

    console.print()
    console.print(tbl)

    # ── Two-level blocks ──────────────────────────────────────────────────────
    if output.two_level_text:
        two_title = t("data_expert_two_level_title", lang)
        for block_text in output.two_level_text:
            console.print()
            console.print(Panel(
                block_text,
                title=f"[bold]{two_title}[/bold]",
                border_style=BORDER_STYLE,
                padding=(0, 2),
            ))

    # ── Query legend ──────────────────────────────────────────────────────────
    legend_rows = [r for r in rows if r.query_text]
    if legend_rows:
        is_pt_disp = lang == "pt"
        legend_lbl = "Legenda de cenários" if is_pt_disp else "Scenario legend"
        legend_lines: list[str] = []
        for r in legend_rows:
            legend_lines.append(f"  [{LABEL_STYLE}]{r.label}[/{LABEL_STYLE}]  {escape(r.query_text)}")
        console.print(
            f"[{MUTED_STYLE}]{legend_lbl}:[/{MUTED_STYLE}]\n" + "\n".join(legend_lines)
        )
        console.print()

    # ── Narrative panel ───────────────────────────────────────────────────────
    narrative = output.narrative.strip()
    insight_title = t("data_expert_insight_title", lang)
    if not narrative:
        console.print()
        msg = t("data_expert_no_patterns", lang)
        console.print(Panel(
            f"  [{MUTED_STYLE}]{escape(msg)}[/{MUTED_STYLE}]",
            title=f"[bold]{insight_title}[/bold]",
            border_style=MUTED_STYLE,
            padding=(1, 2),
        ))
        console.print()
        return

    console.print()
    console.print(Panel(
        f"  {escape(narrative)}",
        title=f"[bold]{insight_title}[/bold]",
        border_style=BLUE,
        padding=(1, 2),
    ))
    console.print()


def render_shock_response(output: ShockResponseOutput, lang: str) -> None:
    """Render the shock response ranking panel + narrative."""
    is_pt = lang == "pt"
    console.print()

    ref_label = "Sem compensação" if is_pt else "No mitigation"
    baseline_label = "Baseline"
    shock_delta = output.shock_cost - output.baseline_cost

    title_str = (
        f"Resposta ao shock: {output.shock_description}"
        if is_pt else
        f"Shock response: {output.shock_description}"
    )
    table = Table(
        title=title_str,
        box=box.SIMPLE_HEAD,
        show_header=True,
        header_style=LABEL_STYLE,
        border_style=MUTED_STYLE,
    )

    rank_col       = "#"
    strat_col      = "Estratégia"  if is_pt else "Strategy"
    trucks_col     = "Caminhões"   if is_pt else "Trucks"
    cost_col       = "Custo"       if is_pt else "Cost"
    recov_col      = "Recuperado"  if is_pt else "Recovered"
    rec_pct_col    = "Rec. %"
    delta_base_col = "Δ Baseline"

    table.add_column(rank_col,       justify="right", style=MUTED_STYLE, width=3)
    table.add_column(strat_col,      justify="left",  style=VALUE_STYLE)
    table.add_column(trucks_col,     justify="right", style=VALUE_STYLE, width=10)
    table.add_column(cost_col,       justify="right", style=VALUE_STYLE, width=12)
    table.add_column(recov_col,      justify="right", style=VALUE_STYLE, width=12)
    table.add_column(rec_pct_col,    justify="right", style=VALUE_STYLE, width=7)
    table.add_column(delta_base_col, justify="right", style=VALUE_STYLE, width=12)

    table.add_row(
        "—", baseline_label,
        str(output.baseline_trucks),
        f"${output.baseline_cost:,.0f}",
        "—", "—", "—",
        style=MUTED_STYLE,
    )
    net_shock_str = f"+${shock_delta:,.0f}" if shock_delta >= 0 else f"-${abs(shock_delta):,.0f}"
    table.add_row(
        "—", ref_label,
        str(output.shock_trucks),
        f"${output.shock_cost:,.0f}",
        "—",
        Text("0%", style=DELTA_NEG),
        Text(net_shock_str, style=DELTA_NEG),
        style="bold red",
    )
    table.add_section()

    for i, s in enumerate(sorted(output.strategies, key=lambda s: s.cost), start=1):
        cr = s.cost_recovered
        recov_str   = f"+${cr:,.0f}" if cr >= 0 else f"-${abs(cr):,.0f}"
        recov_style = DELTA_POS if cr > 0 else (DELTA_NEG if cr < 0 else DELTA_ZERO)

        if shock_delta > 0:
            ratio      = cr / shock_delta
            ratio_str  = f"{ratio * 100:.0f}%"
            ratio_style = DELTA_POS if ratio >= 1.0 else ("yellow" if ratio >= 0.5 else MUTED_STYLE)
        else:
            ratio_str, ratio_style = "—", MUTED_STYLE

        net_base = s.cost - output.baseline_cost
        if net_base <= 0:
            net_base_str   = f"-${abs(net_base):,.0f}" if net_base < 0 else "$0"
            net_base_style = DELTA_POS
        else:
            net_base_str = f"+${net_base:,.0f}"
            remaining    = net_base / shock_delta if shock_delta > 0 else 1.0
            net_base_style = "yellow" if remaining <= 0.25 else DELTA_NEG

        table.add_row(
            str(i),
            escape(s.strategy_name),
            str(s.trucks),
            f"${s.cost:,.0f}",
            Text(recov_str,   style=recov_style),
            Text(ratio_str,   style=ratio_style),
            Text(net_base_str, style=net_base_style),
        )

    console.print(table)

    if output.candidates_evaluated > len(output.strategies):
        shown = len(output.strategies)
        total = output.candidates_evaluated
        note = (
            f"  {shown} de {total} opções analisadas — melhores resultados"
            if is_pt else
            f"  {shown} of {total} options analyzed — best results"
        )
        console.print(Text(note, style=MUTED_STYLE))

    if output.redistribution_strategy:
        rs = output.redistribution_strategy
        cr = rs.cost_recovered
        recov_str   = f"+${cr:,.0f}" if cr >= 0 else f"-${abs(cr):,.0f}"
        recov_color = DELTA_POS if cr > 0 else DELTA_NEG

        if shock_delta > 0:
            ratio       = cr / shock_delta
            ratio_str   = f"{ratio * 100:.0f}%"
            ratio_color = DELTA_POS if ratio >= 1.0 else ("yellow" if ratio >= 0.5 else MUTED_STYLE)
        else:
            ratio_str, ratio_color = "—", MUTED_STYLE

        net_base = rs.cost - output.baseline_cost
        if net_base <= 0:
            net_base_str   = f"-${abs(net_base):,.0f}" if net_base < 0 else "$0"
            net_base_color = DELTA_POS
        else:
            net_base_str   = f"+${net_base:,.0f}"
            remaining      = net_base / shock_delta if shock_delta > 0 else 1.0
            net_base_color = "yellow" if remaining <= 0.25 else DELTA_NEG

        if is_pt:
            title_rs   = "★  Redistribuição otimizada"
            trucks_lbl = "Caminhões"
            cost_lbl   = "Custo"
            recov_lbl  = "Recuperado"
        else:
            title_rs   = "★  Optimized redistribution"
            trucks_lbl = "Trucks"
            cost_lbl   = "Cost"
            recov_lbl  = "Recovered"

        metrics = Text()
        metrics.append(f"{trucks_lbl}: ", style=LABEL_STYLE)
        metrics.append(f"{rs.trucks}   ", style=VALUE_STYLE)
        metrics.append(f"{cost_lbl}: ", style=LABEL_STYLE)
        metrics.append(f"${rs.cost:,.0f}   ", style=VALUE_STYLE)
        metrics.append(f"{recov_lbl}: ", style=LABEL_STYLE)
        metrics.append(f"{recov_str}   ", style=recov_color)
        metrics.append("Rec. %: ", style=LABEL_STYLE)
        metrics.append(f"{ratio_str}   ", style=ratio_color)
        metrics.append("Δ Baseline: ", style=LABEL_STYLE)
        metrics.append(net_base_str, style=net_base_color)

        console.print()
        console.print(Panel(
            metrics,
            title=f"[bold]{title_rs}[/bold]",
            border_style=DELTA_POS,
            padding=(0, 2),
        ))

    console.print()

    analysis_title = "Análise" if is_pt else "Analysis"
    console.print(Panel(
        escape(output.narrative),
        title=f"[bold]{analysis_title}[/bold]",
        border_style=BORDER_STYLE,
        padding=(0, 2),
    ))
    console.print()
