"""Static slash command handlers — /network, /requirements, /questions, /onboarding."""

from __future__ import annotations

import math

from rich.panel import Panel
from rich.table import Table
from rich import box

from ..domain.data_types import MILPResult, ScenarioParams
from ..domain.loader import NetworkData
from .display import console, BLUE, BORDER_STYLE, HEADER_STYLE, LABEL_STYLE, MUTED_STYLE, util_color_style
from .i18n import t


def handle_network(network: NetworkData, language: str) -> None:
    """Display collection points, terminals, and demand matrix."""
    _render_network(network, language)


def handle_demand_scenario(network: NetworkData, params: ScenarioParams, language: str) -> None:
    """Display the demand & capacity table with terminal_demand_multipliers applied."""
    if not params.terminal_demand_multipliers:
        return
    is_pt = language == "pt"
    title = (
        "Demanda e Capacidade — Cenário Ajustado"
        if is_pt else
        "Demand & Capacity — Adjusted Scenario"
    )
    _render_network(network, language, multipliers=params.terminal_demand_multipliers, title_override=title)


def handle_requirements(network: NetworkData, language: str) -> None:
    """Display truck specs, driver policy, and cost data."""
    _render_requirements(network, language)


def handle_limits(network: NetworkData, language: str) -> None:
    """Display lever limits — operational bounds and maximum cost savings."""
    _render_limits(network, language)


def handle_lane_costs(network: NetworkData, milp_result: MILPResult, language: str) -> None:
    """Display operating cost ranking by CP→terminal lane, lowest to highest cost/ton."""
    _render_lane_costs(network, milp_result, language)


def handle_questions(network: NetworkData, language: str) -> None:
    """Display static example questions to guide users on what they can ask."""
    _render_questions(network, language)


def handle_onboarding(network: NetworkData, language: str) -> None:
    """Display the user-friendly introduction to the prototype."""
    if language == "pt":
        _onboarding_pt(network)
    else:
        _onboarding_en(network)


# ── Network ──────────────────────────────────────────────────────────────────

def _compute_weighted_stats(
    network: NetworkData,
    multipliers: "dict[str, float] | None" = None,
) -> tuple[float, float]:
    """Return (weighted_avg_cycle_time_h, weighted_avg_one_way_dist_km) across all lanes."""
    total_demand = weighted_ct = weighted_dist = 0.0
    for cp in network.cp_ids:
        for t_id, raw_demand in network.demand[cp].items():
            demand = raw_demand * (multipliers.get(t_id, 1.0) if multipliers else 1.0)
            if demand <= 0:
                continue
            dist = network.distances[cp][t_id]
            ct = (
                network.cp_load_times[cp]
                + dist / network.speed_loaded
                + network.terminal_unload_times[t_id]
                + dist / network.speed_empty
            )
            total_demand  += demand
            weighted_ct   += demand * ct
            weighted_dist += demand * dist
    if total_demand <= 0:
        return 0.0, 0.0
    return weighted_ct / total_demand, weighted_dist / total_demand


def _render_network(
    network: NetworkData,
    language: str,
    multipliers: "dict[str, float] | None" = None,
    title_override: "str | None" = None,
) -> None:
    is_pt = language == "pt"
    console.print()

    default_title = "Demanda e Capacidade por Rota" if is_pt else "Demand & Capacity by Route"
    # ── Demand flow + capacity + utilisation matrix ───────────────────────────
    t2 = Table(
        title=title_override or default_title,
        box=box.SIMPLE_HEAD, border_style=BLUE,
        header_style=HEADER_STYLE, title_style=BLUE,
        show_lines=False,
    )

    pc_col     = "PC" if is_pt else "CP"
    cap_sfx    = "t/mês" if is_pt else "t/mo"
    demand_lbl = "Demanda" if is_pt else "Demand"

    t2.add_column(pc_col, style=LABEL_STYLE, width=6, no_wrap=True)
    t2.add_column(
        f"Cap. {pc_col}\n({cap_sfx})",
        justify="right", style="white", min_width=8,
    )
    for t_id in network.terminal_ids:
        t_cap = network.terminal_capacities[t_id]
        t2.add_column(
            f"[bold]{t_id}[/bold]\n{t_cap:,.0f} t",
            justify="right", style="white", min_width=9,
        )
    t2.add_column(
        f"[bold]{demand_lbl}[/bold]\n({cap_sfx})",
        justify="right", style="bold white", min_width=9,
    )
    t2.add_column("Util %", justify="right", min_width=7)

    term_totals  = {t_id: 0.0 for t_id in network.terminal_ids}
    grand_demand = grand_cp_cap = 0.0

    for cp in network.cp_ids:
        cp_cap = network.cp_capacities[cp]
        grand_cp_cap += cp_cap
        cp_total = sum(
            network.demand[cp].get(t_id, 0.0) * (multipliers.get(t_id, 1.0) if multipliers else 1.0)
            for t_id in network.terminal_ids
        )
        grand_demand += cp_total
        util = cp_total / cp_cap if cp_cap > 0 else 0.0
        u_s  = util_color_style(util)

        row = [cp, f"{cp_cap:,.0f}"]
        for t_id in network.terminal_ids:
            d = network.demand[cp].get(t_id, 0.0) * (multipliers.get(t_id, 1.0) if multipliers else 1.0)
            term_totals[t_id] += d
            row.append(f"{d:,.0f}" if d > 0 else f"[color(240)]—[/color(240)]")
        row.append(f"{cp_total:,.0f}")
        row.append(f"[{u_s}]{util*100:.0f}%[/{u_s}]")
        t2.add_row(*row)

    t2.add_section()

    # Totals row
    ov_util = grand_demand / grand_cp_cap if grand_cp_cap > 0 else 0.0
    ov_s    = util_color_style(ov_util)
    totals  = ["[bold]Total[/bold]", f"[bold]{grand_cp_cap:,.0f}[/bold]"]
    for t_id in network.terminal_ids:
        totals.append(f"[bold]{term_totals[t_id]:,.0f}[/bold]")
    totals += [f"[bold]{grand_demand:,.0f}[/bold]", f"[bold {ov_s}]{ov_util*100:.0f}%[/bold {ov_s}]"]
    t2.add_row(*totals)

    # Terminal capacity row
    t_cap_lbl = (
        f"[{MUTED_STYLE}]Cap. terminal[/{MUTED_STYLE}]" if is_pt
        else f"[{MUTED_STYLE}]Terminal cap.[/{MUTED_STYLE}]"
    )
    t_cap_row = [t_cap_lbl, ""]
    total_t_cap = 0.0
    for t_id in network.terminal_ids:
        tc = network.terminal_capacities[t_id]
        total_t_cap += tc
        t_cap_row.append(f"[{MUTED_STYLE}]{tc:,.0f}[/{MUTED_STYLE}]")
    t_cap_row += [f"[{MUTED_STYLE}]{total_t_cap:,.0f}[/{MUTED_STYLE}]", ""]
    t2.add_row(*t_cap_row)

    # Terminal utilisation row
    t_util_lbl = (
        f"[{MUTED_STYLE}]Util. terminal[/{MUTED_STYLE}]" if is_pt
        else f"[{MUTED_STYLE}]Terminal util.[/{MUTED_STYLE}]"
    )
    t_util_row = [t_util_lbl, ""]
    for t_id in network.terminal_ids:
        tc  = network.terminal_capacities[t_id]
        tu  = term_totals[t_id] / tc if tc > 0 else 0.0
        ts  = util_color_style(tu)
        t_util_row.append(f"[{ts}]{tu*100:.0f}%[/{ts}]")
    t_util_row += ["", ""]
    t2.add_row(*t_util_row)

    console.print(t2)

    wct_h, wdist_km = _compute_weighted_stats(network, multipliers)
    if wct_h > 0:
        ct_lbl   = "Tempo médio ponderado de ciclo" if is_pt else "Weighted avg cycle time"
        dist_lbl = "Distância média ponderada"       if is_pt else "Weighted avg distance"
        console.print(
            f"  [{MUTED_STYLE}]{ct_lbl}:[/{MUTED_STYLE}] [bold white]{wct_h:.2f} h[/bold white]"
            f"   [{MUTED_STYLE}]{dist_lbl}:[/{MUTED_STYLE}] [bold white]{wdist_km:.0f} km[/bold white]"
        )
    console.print()


# ── Requirements ─────────────────────────────────────────────────────────────

def _render_requirements(network: NetworkData, language: str) -> None:
    is_pt = language == "pt"
    per_month = "/mês" if is_pt else "/month"
    per_day = "/dia" if is_pt else "/day"

    console.print()
    tbl = Table(
        title="Especificações — Caminhão" if is_pt else "Specifications — Truck",
        box=box.SIMPLE_HEAD, border_style=BLUE, header_style=HEADER_STYLE,
        title_style=BLUE,
    )
    col_param = "Parâmetro" if is_pt else "Parameter"
    col_value = "Valor" if is_pt else "Value"
    tbl.add_column(col_param, style=LABEL_STYLE)
    tbl.add_column(col_value, style="bold white", justify="right")
    tbl.add_row("Velocidade (carregado)" if is_pt else "Speed (loaded)", f"{network.speed_loaded} km/h")
    tbl.add_row("Velocidade (vazio)" if is_pt else "Speed (empty)", f"{network.speed_empty} km/h")
    tbl.add_row("Carga útil" if is_pt else "Payload", f"{network.payload} t")
    tbl.add_row("Disponibilidade" if is_pt else "Availability", f"{network.availability*100:.0f}%")
    console.print(tbl)

    tbl2 = Table(
        title="Política do Motorista" if is_pt else "Driver Policy",
        box=box.SIMPLE_HEAD, border_style=BLUE, header_style=HEADER_STYLE,
        title_style=BLUE,
    )
    tbl2.add_column(col_param, style=LABEL_STYLE)
    tbl2.add_column(col_value, style="bold white", justify="right")
    tbl2.add_row("Horas líquidas/dia" if is_pt else "Net driving hours/day", f"{network.net_driving_hours} h")
    tbl2.add_row("Hora extra (baseline)" if is_pt else "Overtime (baseline)", f"{network.overtime_hours} h{per_day}")
    tbl2.add_row("Custo hora extra" if is_pt else "Overtime cost", f"${network.overtime_cost:,.0f}/h")
    tbl2.add_row("Dias úteis/mês" if is_pt else "Working days/month", str(network.working_days))
    console.print(tbl2)

    tbl3 = Table(
        title="Custos Operacionais" if is_pt else "Operational Costs",
        box=box.SIMPLE_HEAD, border_style=BLUE, header_style=HEADER_STYLE,
        title_style=BLUE,
    )
    col_comp = "Componente" if is_pt else "Component"
    tbl3.add_column(col_comp, style=LABEL_STYLE)
    tbl3.add_column(col_value, style="bold white", justify="right")
    var_title = "Variável" if is_pt else "Variable"
    fix_title = "Fixo" if is_pt else "Fixed"
    tbl3.add_row(f"[bold]{var_title}[/bold]", f"[bold]${network.variable_cost_per_km:.4f}/km[/bold]")
    for key, val in network.variable_cost_components.items():
        tbl3.add_row(f"  — {t(f'param_var_{key}', language)}", f"${val:.4f}/km")
    tbl3.add_row("", "")
    tbl3.add_row(f"[bold]{fix_title}[/bold]", f"[bold]${network.fixed_cost_per_truck_month:,.2f}{per_month}[/bold]")
    for key, val in network.fixed_cost_components.items():
        tbl3.add_row(f"  — {t(f'param_fix_{key}', language)}", f"${val:,.2f}{per_month}")
    console.print(tbl3)

    if network.availability_sensitivity:
        tbl4 = Table(
            title="Sensibilidade de Custo — Disponibilidade" if is_pt else "Cost Sensitivity — Availability",
            box=box.SIMPLE_HEAD, border_style=BLUE, header_style=HEADER_STYLE,
            title_style=BLUE,
        )
        tbl4.add_column(col_comp, style=LABEL_STYLE)
        tbl4.add_column(
            "+1pp disponib. =>" if is_pt else "+1pp avail. =>",
            style="bold white", justify="right",
        )
        for key, sens in network.availability_sensitivity.items():
            label = t(f"param_var_{key}", language)
            tbl4.add_row(f"  — {label}", f"+{sens * 100:.1f}%")
        console.print(tbl4)

    console.print()


# ── Lever Limits ─────────────────────────────────────────────────────────────

_OP_LABEL_PT: dict[str, tuple[str, str]] = {
    "payload":           ("Carga útil",        "t"),
    "availability":      ("Disponibilidade",   "%"),
    "overtime_hours":    ("Hora extra",         "h/dia"),
    "net_driving_hours": ("Horas líquidas/dia", "h"),
    "working_days":      ("Dias úteis/mês",     "dias"),
}
_OP_LABEL_EN: dict[str, tuple[str, str]] = {
    "payload":           ("Payload",             "t"),
    "availability":      ("Availability",        "%"),
    "overtime_hours":    ("Overtime",            "h/day"),
    "net_driving_hours": ("Net driving hours",   "h"),
    "working_days":      ("Working days/month",  "days"),
}


def _render_limits(network: NetworkData, language: str) -> None:
    is_pt = language == "pt"
    ll = network.lever_limits
    if not ll:
        msg = "Arquivo lever_limits.xlsx não encontrado." if is_pt else "lever_limits.xlsx file not found."
        console.print(f"\n[{MUTED_STYLE}]{msg}[/{MUTED_STYLE}]\n")
        return

    label_map = _OP_LABEL_PT if is_pt else _OP_LABEL_EN
    console.print()

    # ── Operational limits ────────────────────────────────────────────────────
    tbl_op = Table(
        title="Limites Operacionais" if is_pt else "Operational Limits",
        box=box.SIMPLE_HEAD, border_style=BLUE, header_style=HEADER_STYLE,
        title_style=BLUE,
    )
    tbl_op.add_column("Parâmetro" if is_pt else "Parameter", style=LABEL_STYLE)
    tbl_op.add_column("Mín" if is_pt else "Min", justify="right", style="bold white", min_width=6)
    tbl_op.add_column("Máx" if is_pt else "Max", justify="right", style="bold white", min_width=10)
    tbl_op.add_column("Unidade" if is_pt else "Unit", style="white")

    has_ndh_note = False
    for param, limits in ll.get("operational", {}).items():
        label, unit = label_map.get(param, (param, ""))
        min_v: float = limits["min"]
        max_v: float = limits["max"]

        if param == "availability":
            min_str = f"{min_v * 100:.0f}%"
            max_str = f"{max_v * 100:.0f}%"
            unit_str = "—"
        else:
            min_str = f"{min_v:g}"
            max_str = f"{max_v:g}"
            unit_str = unit

        if param == "net_driving_hours":
            label = label + " *"
            max_str = f"{max_v:g} (baseline)"
            has_ndh_note = True

        tbl_op.add_row(label, min_str, max_str, unit_str)

    console.print(tbl_op)

    if has_ndh_note:
        ndh_note = (
            "  * Horas líquidas não são uma alavanca de compensação — use hora extra para estender a jornada."
            if is_pt else
            "  * Net driving hours is not a compensation lever — use overtime hours to extend the working day."
        )
        console.print(f"[{MUTED_STYLE}]{ndh_note}[/{MUTED_STYLE}]")

    console.print()

    # ── Cost savings ─────────────────────────────────────────────────────────
    tbl_sav = Table(
        title=(
            "Savings Máximos — Procurement / Renegociação"
            if is_pt else
            "Maximum Savings — Procurement / Renegotiation"
        ),
        box=box.SIMPLE_HEAD, border_style=BLUE, header_style=HEADER_STYLE,
        title_style=BLUE,
    )
    tbl_sav.add_column("Componente" if is_pt else "Component", style=LABEL_STYLE)
    tbl_sav.add_column("Tipo" if is_pt else "Type", style="white", min_width=8)
    tbl_sav.add_column("Saving máx" if is_pt else "Max saving", justify="right", style="bold white", min_width=10)

    cost_savings = ll.get("cost_savings", {})
    var_label = "Variável" if is_pt else "Variable"
    fix_label = "Fixo" if is_pt else "Fixed"

    var_entries = [(k, v) for k, v in cost_savings.items() if v["category"] == "variable"]
    fix_entries = [(k, v) for k, v in cost_savings.items() if v["category"] == "fixed"]

    for cost_type, info in var_entries:
        comp_label = t(f"param_var_{cost_type}", language)
        pct = info["max_saving_pct"]
        pct_str = f"{pct * 100:.0f}%" if pct > 0 else f"[{MUTED_STYLE}]—[/{MUTED_STYLE}]"
        tbl_sav.add_row(f"  {comp_label}", var_label, pct_str)

    tbl_sav.add_row("", "", "")

    for cost_type, info in fix_entries:
        comp_label = t(f"param_fix_{cost_type}", language)
        pct = info["max_saving_pct"]
        pct_str = f"{pct * 100:.0f}%" if pct > 0 else f"[{MUTED_STYLE}]—[/{MUTED_STYLE}]"
        tbl_sav.add_row(f"  {comp_label}", fix_label, pct_str)

    console.print(tbl_sav)
    console.print()


# ── Lane costs ───────────────────────────────────────────────────────────────

def _render_lane_costs(network: NetworkData, milp_result: MILPResult, language: str) -> None:
    """Rank MILP-assigned CP→terminal lanes by operating cost per ton.

    Variable cost is exact per lane (demand × round-trip km × $/km).
    Fixed and overtime costs are allocated proportionally by each lane's
    demand×cycle_time contribution — the same weighting the MILP uses to size
    the fleet — so the total across all lanes reconciles with milp_result totals.
    """
    is_pt = language == "pt"
    console.print()

    # Build lanes from MILP volumes (the actual routed demand per CP→terminal).
    lanes: list[dict] = []
    total_weight = 0.0  # sum of demand×cycle_time across all lanes

    for cp_id, vol_dict in milp_result.volumes.items():
        for t_id, demand in vol_dict.items():
            if demand <= 0:
                continue
            dist = network.distances[cp_id][t_id]
            ct = (
                dist / network.speed_loaded
                + network.cp_load_times[cp_id]
                + dist / network.speed_empty
                + network.terminal_unload_times[t_id]
            )
            trips = math.ceil(demand / network.payload)
            km = trips * 2 * dist
            var_cost = km * network.variable_cost_per_km
            weight = demand * ct  # proportionality basis for fixed cost allocation
            total_weight += weight
            lanes.append({
                "cp_id": cp_id,
                "t_id": t_id,
                "dist": dist,
                "demand": demand,
                "km": km,
                "var_cost": var_cost,
                "weight": weight,
            })

    if not lanes:
        msg = "Nenhuma rota encontrada no resultado MILP." if is_pt else "No lanes found in the MILP result."
        console.print(f"[{MUTED_STYLE}]{msg}[/{MUTED_STYLE}]")
        console.print()
        return

    # Allocate fixed + overtime costs proportionally by demand×cycle_time weight.
    fixed_pool = milp_result.fixed_cost
    ot_pool = milp_result.overtime_cost_total
    for lane in lanes:
        share = lane["weight"] / total_weight if total_weight > 0 else 0.0
        lane["fix_cost"] = share * fixed_pool
        lane["ot_cost"] = share * ot_pool
        lane["total_cost"] = lane["var_cost"] + lane["fix_cost"] + lane["ot_cost"]
        lane["cost_per_ton"] = lane["total_cost"] / lane["demand"]

    lanes.sort(key=lambda x: x["cost_per_ton"])

    title = (
        "Custo Operacional por Rota — MILP Baseline (menor → maior $/ton)"
        if is_pt else
        "Operating Cost by Lane — MILP Baseline (lowest → highest $/ton)"
    )
    tbl = Table(
        title=title,
        box=box.SIMPLE_HEAD, border_style=BLUE,
        header_style=HEADER_STYLE, title_style=BLUE,
        show_lines=False,
    )

    cp_col      = "PC"         if is_pt else "CP"
    term_col    = "Terminal"
    dist_col    = "Dist. km"
    dem_col     = "Dem. t/mês" if is_pt else "Dem. t/mo"
    var_col     = "Var/mês"    if is_pt else "Var/mo"
    fix_col     = "Fixo/mês"   if is_pt else "Fixed/mo"
    total_col   = "Total/mês"  if is_pt else "Total/mo"
    cperton_col = "$/ton"

    tbl.add_column("#",         justify="right",  style="white",      width=3)
    tbl.add_column(cp_col,      style=LABEL_STYLE, no_wrap=True,      width=6)
    tbl.add_column(term_col,    style=LABEL_STYLE,                    width=9)
    tbl.add_column(dist_col,    justify="right",  style="white",      width=8)
    tbl.add_column(dem_col,     justify="right",  style="white",      width=10)
    tbl.add_column(var_col,     justify="right",  style="white",      width=11)
    tbl.add_column(fix_col,     justify="right",  style="white",      width=11)
    tbl.add_column(total_col,   justify="right",  style="white",      width=11)
    tbl.add_column(cperton_col, justify="right",  style="bold white", width=8)

    for i, lane in enumerate(lanes, start=1):
        tbl.add_row(
            str(i),
            lane["cp_id"],
            lane["t_id"],
            f"{lane['dist']:,.0f}",
            f"{lane['demand']:,.0f}",
            f"${lane['var_cost']:,.0f}",
            f"${lane['fix_cost']:,.0f}",
            f"${lane['total_cost']:,.0f}",
            f"${lane['cost_per_ton']:,.1f}",
        )

    console.print(tbl)
    note = (
        "Custo fixo alocado proporcionalmente por demanda×ciclo — base MILP."
        if is_pt else
        "Fixed cost allocated proportionally by demand×cycle time — MILP basis."
    )
    console.print(f"  [{MUTED_STYLE}]{note}[/{MUTED_STYLE}]")
    console.print()


# ── Questions ────────────────────────────────────────────────────────────────

def _render_questions(network: NetworkData, language: str) -> None:
    """Render example questions derived from live network data."""
    is_pt = language == "pt"
    t_id_1 = network.terminal_ids[0]
    t_id_2 = network.terminal_ids[1] if len(network.terminal_ids) > 1 else network.terminal_ids[0]

    if is_pt:
        note = "Exemplos ilustrativos — você pode fazer qualquer pergunta com os valores da sua operação."
        title = "Exemplos de Perguntas"
        questions = [
            # ── Parâmetros operacionais ──────────────────────────────────────
            ("1", "E se a velocidade dos caminhões cair 10 km/h e a carga útil for 28 toneladas?"),
            ("2", "E se a disponibilidade subir para 92% e o custo do combustível aumentar 8%?"),
            ("3", "E se adotarmos 2 motoristas por caminhão, aumentando os dias úteis para 27?"),
            # ── Resposta a shock ─────────────────────────────────────────────
            ("4", "Quais as alternativas para compensar uma queda de carga útil para 28t por caminhão?"),
            # ── Cobertura e frota ────────────────────────────────────────────
            ("5", "Quanto custa aumentar a cobertura de 60% para 90% dos pontos de coleta?"),
            ("6", "Qual a melhor cobertura de pontos de coleta com orçamento mensal de $4,85 milhões?"),
            ("7", "Qual a frota mínima para atender pelo menos 80% dos pontos de coleta?"),
            # ── Fechamento de terminal ───────────────────────────────────────
            ("8", f"Qual o impacto na frota e no custo se o terminal {t_id_2} for fechado?"),
            # ── Variação de demanda ──────────────────────────────────────────
            ("9",  f"Se a demanda em {t_id_1} crescer 20% e em {t_id_2} cair 15%, com redistribuição otimizada, qual a frota necessária?"),
            ("10", f"Qual o impacto de limitar o volume recebido pelo terminal {t_id_1} a 80% da sua capacidade histórica?"),
        ]
    else:
        note = "Illustrative examples — you can ask any question using figures from your own operation."
        title = "Example Questions"
        questions = [
            # ── Operational parameters ───────────────────────────────────────
            ("1", "What if truck speed drops 10 km/h and payload is 28 tonnes?"),
            ("2", "What if availability rises to 92% and fuel cost increases 8%?"),
            ("3", "What if we adopt 2 drivers per truck, increasing working days to 27?"),
            # ── Shock response ───────────────────────────────────────────────
            ("4", "What are the alternatives to compensate for a payload drop to 28 t per truck?"),
            # ── Coverage and fleet ───────────────────────────────────────────
            ("5", "What does it cost to increase service coverage from 60% to 90% of collection points?"),
            ("6", "What is the best collection point coverage within a $4.85M monthly budget?"),
            ("7", "What is the minimum fleet to serve at least 80% of collection points?"),
            # ── Terminal closure ─────────────────────────────────────────────
            ("8", f"What is the fleet and cost impact if terminal {t_id_2} is closed?"),
            # ── Demand variation ─────────────────────────────────────────────
            ("9",  f"If demand at {t_id_1} grows 20% and at {t_id_2} drops 15%, with optimal redistribution, how many trucks do we need?"),
            ("10", f"What is the impact of capping terminal {t_id_1}'s incoming volume to 80% of its historical capacity?"),
        ]

    tbl = Table(
        title=title,
        box=box.SIMPLE_HEAD, border_style=BLUE,
        header_style=HEADER_STYLE, title_style=BLUE,
    )
    tbl.add_column("#", style=LABEL_STYLE, justify="right", width=3)
    tbl.add_column("" if is_pt else "", style="white", max_width=80)

    for num, q in questions:
        tbl.add_row(num, q)

    console.print()
    console.print(f"[{MUTED_STYLE}]{note}[/{MUTED_STYLE}]")
    console.print(tbl)
    console.print()


# ── Onboarding ───────────────────────────────────────────────────────────────

def _onboarding_pt(network: NetworkData) -> None:
    terminal = network.terminal_ids[1] if len(network.terminal_ids) > 1 else network.terminal_ids[0]
    text = f"""[bold blue]FLEET Planning - Seu Assistente de Tomada de Decisão[/bold blue]

Este protótipo transforma perguntas de negócio em cenários calculados para carga lotação. Você testa hipóteses de frota, custo, cobertura, capacidade, demanda e fechamento de terminais sem montar um modelo do zero.

[bold]O que você ganha[/bold]
  [bold]•[/bold] Mais velocidade para comparar alternativas.
  [bold]•[/bold] Mais clareza para enxergar trade-offs entre frota, custo e nível de serviço.
  [bold]•[/bold] Mais confiança para discutir decisões com números consistentes.

[bold]Três modelos, três lentes[/bold]
  [bold]Corredor por corredor[/bold] mostra a visão conservadora: cada rota carrega sua própria necessidade de caminhões.
  [bold]Ciclo ponderado[/bold] aproxima o ganho de compartilhamento da frota usando o ciclo médio ponderado pela demanda.
  [bold]Otimizador MILP[/bold] procura a melhor alocação viável, respeitando cobertura, capacidades e restrições do cenário.

Use os três juntos. Se eles convergem, a decisão tende a ser robusta. Se divergem, existe uma pergunta operacional importante para investigar.

[bold yellow]Seu papel[/bold yellow]
Traga o pensamento crítico: quais hipóteses fazem sentido, que restrições são reais, qual risco é aceitável. O protótipo fica encarregado da matemática.

[bold]Comece assim[/bold]
  1. Rode [bold]/cenario-base[/bold] para criar a referência.
  2. Faça perguntas de cenário em linguagem natural.
  3. Compare frota, custo e cobertura antes de decidir.

[bold]Perguntas úteis[/bold]
  Minimize o custo atendendo pelo menos 70% dos pontos de coleta.
  E se a disponibilidade for 88%?
  O que acontece se o terminal {terminal} for fechado?

[bold]Comandos principais[/bold]
  /cenario-base  — calcular a referência
  /perguntas     — ver exemplos de análises
  /rede          — revisar demanda, PCs e terminais
  /requisitos    — revisar custos e premissas operacionais
  /detalhe       — abrir métricas de um modelo
  /exportar      — gerar .xlsx do resultado atual
"""
    console.print(Panel(text, border_style=BLUE, padding=(1, 2)))


def _onboarding_en(network: NetworkData) -> None:
    terminal = network.terminal_ids[1] if len(network.terminal_ids) > 1 else network.terminal_ids[0]
    text = f"""[bold blue]FLEET Planning - Your Decision-Making Assistant[/bold blue]

This prototype turns business questions into calculated Full Truck Load scenarios. You can test fleet, cost, coverage, capacity, demand, and terminal-closure assumptions without building a model from scratch.

[bold]What it gives you[/bold]
  [bold]•[/bold] Faster comparison of operational alternatives.
  [bold]•[/bold] Clearer trade-offs between fleet size, cost, and service level.
  [bold]•[/bold] More consistent numbers for decision discussions.

[bold]Three models, three lenses[/bold]
  [bold]Lane-by-Lane[/bold] shows the conservative view: each lane carries its own truck requirement.
  [bold]Weighted Cycle Time[/bold] estimates fleet-sharing gains through a demand-weighted average cycle.
  [bold]MILP optimizer[/bold] searches for the best feasible allocation while respecting coverage, capacity, and scenario constraints.

Use the three together. If they converge, the decision is usually robust. If they diverge, there is an operational question worth investigating.

[bold yellow]Your role[/bold yellow]
Bring the critical thinking: which assumptions are credible, which constraints are real, and what risk is acceptable. The prototype handles the math.

[bold]Start here[/bold]
  1. Run [bold]/baseline[/bold] to create the reference.
  2. Ask scenario questions in natural language.
  3. Compare fleet, cost, and coverage before deciding.

[bold]Useful questions[/bold]
  Minimize cost while serving at least 70% of the collection points.
  What if availability is 88%?
  What happens if terminal {terminal} is closed?

[bold]Main commands[/bold]
  /baseline      — calculate the reference
  /questions     — see example analyses
  /network       — review demand, CPs, and terminals
  /requirements  — review costs and operating assumptions
  /detail        — open metrics for one model
  /export        — generate an .xlsx for the current result
"""
    console.print(Panel(text, border_style=BLUE, padding=(1, 2)))
