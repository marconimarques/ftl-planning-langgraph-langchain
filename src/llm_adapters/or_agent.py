"""OR Agent — interprets natural language queries into scenario parameters."""

from __future__ import annotations

import functools
import json
import math
import re
from typing import Any, Optional

from pydantic import BaseModel, Field, create_model

from ..llm.model_factory import LLMConfig, invoke_structured
from ..domain.data_types import MILPResult, ScenarioParams
from ..domain.loader import NetworkData, load_network_data
from .tools import compare_coverage_costs
from .model_factory import make_model


def _build_system_prompt(network: NetworkData, language: str) -> str:
    """Generate the OR Agent system prompt from live network data."""
    is_pt = language == "pt"
    n_cps = len(network.cp_ids)
    cp_ids_str = ", ".join(network.cp_ids)
    terminal_ids_str = ", ".join(network.terminal_ids)
    cap_unit = "t/mês" if is_pt else "t/month"
    truck_unit = "caminhão/mês" if is_pt else "truck/month"
    cp_unit = "PCs" if is_pt else "CPs"

    # Terminal descriptions
    terminal_desc = "\n".join(
        f"- {t_id} (cap {network.terminal_capacities[t_id]:,.0f} {cap_unit})"
        for t_id in network.terminal_ids
    )

    # Coverage conversion examples
    coverage_lines = "\n".join(
        f"- {pct}% → ceil({pct / 100:.2f} × {n_cps}) = {math.ceil(pct / 100 * n_cps)} {cp_unit}"
        for pct in [60, 70, 90, 100]
    )

    # Variable cost components table
    var_comp_rows = "\n".join(
        f"| {key} | var_cost_{key} | ${val:.4f}/km |"
        for key, val in network.variable_cost_components.items()
    )

    # Fixed cost components table
    fix_comp_rows = "\n".join(
        f"| {key} | fix_cost_{key} | ${val:,.2f}/{truck_unit} |"
        for key, val in network.fixed_cost_components.items()
    )

    # Cost multiplier examples
    var_items = list(network.variable_cost_components.items())
    ex_key, ex_base = var_items[0]
    ex_new = round(ex_base * 1.10, 4)

    maint_keys = [k for k in ("tractor_maintenance", "trailer_maintenance") if k in network.variable_cost_components]
    if len(maint_keys) >= 2:
        mk1, mk2 = maint_keys[0], maint_keys[1]
        mv1 = network.variable_cost_components[mk1]
        mv2 = network.variable_cost_components[mk2]
        new_mv1 = round(mv1 * 1.05, 4)
        new_mv2 = round(mv2 * 1.05, 4)
        if is_pt:
            cost_ex2 = (
                f'\n- "custos de manutenção sobem 5%" → '
                f'var_cost_multipliers = \'{{{{"\\"{mk1}\\"": 1.05, "\\"{mk2}\\"": 1.05}}}}\'; '
                f'resultado: var_cost_{mk1} = {new_mv1}, var_cost_{mk2} = {new_mv2}'
            )
        else:
            cost_ex2 = (
                f'\n- "maintenance costs increase 5%" → '
                f'var_cost_multipliers = \'{{{{"\\"{mk1}\\"": 1.05, "\\"{mk2}\\"": 1.05}}}}\'; '
                f'result: var_cost_{mk1} = {new_mv1}, var_cost_{mk2} = {new_mv2}'
            )
    else:
        cost_ex2 = ""

    # Structural multiplier example: N drivers per truck → driver_wage × N
    driver_wage_base = network.fixed_cost_components.get("driver_wage", 0.0)
    if driver_wage_base > 0:
        driver_wage_2x = round(driver_wage_base * 2, 2)
        if is_pt:
            cost_ex3 = (
                f'\n- "2 motoristas por caminhão" (mudança estrutural, não percentual) → '
                f'fix_cost_multipliers = \'{{{{"driver_wage": 2.0}}}}\', working_days = 27; '
                f'resultado: fix_cost_driver_wage = {driver_wage_2x:,.2f}'
            )
            fix_driver_note = (
                f'Nota: driver_wage é o custo mensal de UM motorista por caminhão '
                f'(${driver_wage_base:,.2f}/{truck_unit}). '
                f'Para N motoristas por caminhão: fix_cost_multipliers = \'{{{{"driver_wage": N}}}}\'.'
            )
        else:
            cost_ex3 = (
                f'\n- "2 drivers per truck" (structural change, not a percentage) → '
                f'fix_cost_multipliers = \'{{{{"driver_wage": 2.0}}}}\', working_days = 27; '
                f'result: fix_cost_driver_wage = {driver_wage_2x:,.2f}'
            )
            fix_driver_note = (
                f'Note: driver_wage is the monthly cost of ONE driver per truck '
                f'(${driver_wage_base:,.2f}/{truck_unit}). '
                f'For N drivers per truck: fix_cost_multipliers = \'{{{{"driver_wage": N}}}}\'.'
            )
    else:
        cost_ex3 = ""
        fix_driver_note = ""

    # Terminal closure example (use second terminal when available)
    closed_ex = network.terminal_ids[1] if len(network.terminal_ids) > 1 else network.terminal_ids[0]

    # Terminal demand examples
    ex_term_a = network.terminal_ids[0]
    ex_term_b = network.terminal_ids[1] if len(network.terminal_ids) > 1 else network.terminal_ids[0]
    if is_pt:
        term_demand_section = (
            f"\n## Ajuste de Demanda por Terminal\n"
            f"Para modelar variações de volume em um terminal (redução ou aumento),\n"
            f"passe terminal_demand_multipliers como dict JSON na chamada da ferramenta E no JSON de saída.\n"
            f"Aplicado a TODOS os fluxos PC→terminal daquele terminal.\n\n"
            f"Exemplos:\n"
            f'- "demanda no {ex_term_a} reduz 15%" → terminal_demand_multipliers = {{"{ex_term_a}": 0.85}}\n'
            f'- "demanda no {ex_term_b} aumenta 20%" → terminal_demand_multipliers = {{"{ex_term_b}": 1.20}}\n'
            f'- "{ex_term_a} cai 10% e {ex_term_b} sobe 5%" → terminal_demand_multipliers = {{"{ex_term_a}": 0.90, "{ex_term_b}": 1.05}}\n\n'
            f"Importante — dois cenários com ajuste de demanda:\n"
            f"1. Roteamento histórico mantido (padrão): use run_milp_solver(terminal_demand_multipliers=...)\n"
            f"   → mostra impacto da mudança de demanda com as rotas atuais. USE SEMPRE que o usuário perguntar\n"
            f"   sobre impacto, efeito ou consequência de uma mudança de demanda.\n"
            f"2. Redistribuição ótima: use run_milp_solver(terminal_demand_multipliers=..., volume_redistribution=True)\n"
            f"   → solver realoca PCs entre terminais para minimizar custo com a nova demanda.\n"
            f"   Use SOMENTE quando o usuário pedir explicitamente redistribuição, realocação ou otimização\n"
            f"   de rotas em conjunto com a mudança de demanda. Nunca infira redistribuição por uma pergunta\n"
            f"   de impacto.\n\n"
            f"No JSON de saída inclua em scenario_params:\n"
            f'  "terminal_demand_multipliers": {{<terminal_id>: <multiplicador>}}\n'
            f"Se não houver ajuste: \"terminal_demand_multipliers\": {{}}\n\n"
            f"## Limitação de Volume por Terminal (terminal_volume_caps)\n"
            f"Use terminal_volume_caps quando o usuário quer REDIRECIONAR parte do volume de um terminal para\n"
            f"outros — a demanda total do sistema é PRESERVADA, apenas o destino muda.\n\n"
            f"Diferença crucial:\n"
            f"- terminal_demand_multipliers: REDUZ a demanda real (o volume 'desaparece' do sistema)\n"
            f"- terminal_volume_caps: LIMITA o que o terminal recebe; o excesso é redistribuído pelo solver\n\n"
            f"Use terminal_volume_caps + volume_redistribution=True quando o usuário disser:\n"
            f'- "{ex_term_a} terá sua demanda reduzida 15% e este volume deve ser redistribuído para os outros terminais"\n'
            f'- "redirecionar 15% da demanda de {ex_term_a} para {ex_term_b} e outros"\n'
            f'- "limitar recebimento de {ex_term_a} a 80% e otimizar redistribuição"\n\n'
            f"Exemplo: '{ex_term_a} reduz 15% que deve ir para os outros terminais'\n"
            f"→ terminal_volume_caps = {{'{ex_term_a}': 0.85}}, volume_redistribution = true\n"
            f"→ NÃO use terminal_demand_multipliers neste caso\n\n"
            f"No JSON de saída inclua em scenario_params:\n"
            f'  "terminal_volume_caps": {{<terminal_id>: <fração>}}\n'
            f"Se não houver limitação: \"terminal_volume_caps\": {{}}"
        )
    else:
        term_demand_section = (
            f"\n## Terminal Demand Adjustment\n"
            f"To model volume changes at a specific terminal (reduction or increase),\n"
            f"pass terminal_demand_multipliers as a JSON dict in the tool call AND in the output JSON.\n"
            f"Applied to ALL CP→terminal demand flows for that terminal.\n\n"
            f"Examples:\n"
            f'- "demand at {ex_term_a} decreases 15%" → terminal_demand_multipliers = {{"{ex_term_a}": 0.85}}\n'
            f'- "demand at {ex_term_b} increases 20%" → terminal_demand_multipliers = {{"{ex_term_b}": 1.20}}\n'
            f'- "{ex_term_a} drops 10% and {ex_term_b} increases 5%" → terminal_demand_multipliers = {{"{ex_term_a}": 0.90, "{ex_term_b}": 1.05}}\n\n'
            f"Important — two scenarios with demand adjustment:\n"
            f"1. Historical routing kept (default): use run_milp_solver(terminal_demand_multipliers=...)\n"
            f"   → shows the impact of demand change with current routes. USE ALWAYS when the user asks about\n"
            f"   the impact, effect, or consequence of a demand change.\n"
            f"2. Optimal redistribution: use run_milp_solver(terminal_demand_multipliers=..., volume_redistribution=True)\n"
            f"   → solver reassigns CPs across terminals to minimize cost under the new demand.\n"
            f"   Use ONLY when the user explicitly requests redistribution, reassignment, or route optimization\n"
            f"   alongside the demand change. Never infer redistribution from a plain impact question.\n\n"
            f"In the output JSON, include in scenario_params:\n"
            f'  "terminal_demand_multipliers": {{<terminal_id>: <multiplier>}}\n'
            f"If no adjustment: \"terminal_demand_multipliers\": {{}}\n\n"
            f"## Terminal Volume Caps (terminal_volume_caps)\n"
            f"Use terminal_volume_caps when the user wants to REDIRECT part of a terminal's volume to\n"
            f"other terminals — total system demand is PRESERVED, only the destination changes.\n\n"
            f"Crucial difference:\n"
            f"- terminal_demand_multipliers: REDUCES actual demand (volume disappears from the system)\n"
            f"- terminal_volume_caps: LIMITS what a terminal receives; the solver redistributes the excess\n\n"
            f"Use terminal_volume_caps + volume_redistribution=True when the user says:\n"
            f'- "{ex_term_a} demand is reduced 15% and this volume must be redirected to other terminals"\n'
            f'- "redirect 15% of {ex_term_a}\'s demand to {ex_term_b} and others"\n'
            f'- "limit {ex_term_a} intake to 80% and optimize redistribution"\n\n'
            f"Example: '{ex_term_a} loses 15% which must go to other terminals'\n"
            f"→ terminal_volume_caps = {{'{ex_term_a}': 0.85}}, volume_redistribution = true\n"
            f"→ Do NOT use terminal_demand_multipliers in this case\n\n"
            f"In the output JSON, include in scenario_params:\n"
            f'  "terminal_volume_caps": {{<terminal_id>: <fraction>}}\n'
            f"If no cap: \"terminal_volume_caps\": {{}}"
        )

    # Coverage-level illustrative examples
    ex_lower_pct, ex_upper_pct = 60, 90
    ex_lower_count = math.ceil(ex_lower_pct / 100 * n_cps)
    ex_upper_count = math.ceil(ex_upper_pct / 100 * n_cps)

    # Output format snippets for dynamic cost override fields
    var_override_snippet = "\n".join(
        f'    "var_cost_{key}": <float or null>,'
        for key in network.variable_cost_components
    )
    fix_override_snippet = "\n".join(
        f'    "fix_cost_{key}": <float or null>,'
        for key in network.fixed_cost_components
    )

    if is_pt:
        cost_ex1 = (
            f'- "{ex_key} sobe 10%" → var_cost_multipliers = \'{{{{"\\"{ex_key}\\"": 1.10}}}}\'; '
            f'resultado: var_cost_{ex_key} = {ex_new}'
        )
        return f"""Você é um especialista em Pesquisa Operacional especializado em otimização de frotas de caminhões.

## Contrato do Agente
- Papel: parser de cenários de planejamento de frota.
- Entrada: uma consulta do usuário e os dados atuais da rede.
- Limite de decisão: converter somente a intenção atual em `scenario_params`; não reutilizar parâmetros de consultas anteriores, não inventar resultados de solver e não estimar custos ou frota.
- Padrão de saída: JSON estruturado com parâmetros válidos para o Python validar e resolver deterministicamente.

## Rede
- {n_cps} pontos de coleta: {cp_ids_str}
- {len(network.terminal_ids)} terminais de descarga:
{terminal_desc}
- Operação: FTL — um caminhão atende um ponto de coleta por viagem, dirige até um terminal

## Parâmetros Padrão (baseline)
| Parâmetro | Valor |
|---|---|
| Carga útil | {network.payload} toneladas |
| Velocidade carregado | {network.speed_loaded} km/h |
| Velocidade vazio | {network.speed_empty} km/h |
| Disponibilidade do caminhão | {int(network.availability * 100)}% ({network.availability}) |
| Horas líquidas de direção | {network.net_driving_hours} h/dia |
| Dias úteis | {network.working_days}/mês |
| Custo variável | ${network.variable_cost_per_km:.4f}/km |
| Custo fixo | ${network.fixed_cost_per_truck_month:,.2f}/caminhão/mês |
| Hora extra | {network.overtime_hours} h/dia, ${network.overtime_cost:,.0f}/h |

## Mapeamento de Objetivos
| Tipo de cenário | objective | parâmetros-chave |
|---|---|---|
| Minimizar custo (padrão) | minimize_cost | todos os PCs ou min_coverage_count |
| Maximizar cobertura dentro do orçamento | maximize_coverage | budget |
| Minimizar tamanho da frota | minimize_fleet | min_coverage_count |

Regra is_baseline: defina is_baseline = true SOMENTE quando o usuário pedir explicitamente o cenário base (ex: "calcule o baseline", "qual o cenário padrão") sem nenhuma alteração de parâmetros, sem restrição de cobertura e sem orçamento. Para qualquer outro cenário: is_baseline = false.

Restrições combinadas: se o usuário informar cobertura mínima E orçamento simultaneamente, use objective = maximize_coverage com o orçamento como restrição e min_coverage_count como piso mínimo.

## Rotas Determinísticas Disponíveis
| Rota | Quando parametrizar |
|---|---|
| run_milp_solver | Qualquer cenário único (baseline, what-if, cobertura, orçamento, fechamento de terminal) |
| compare_coverage_costs | Diferença de custo entre dois níveis de cobertura |
| run_redistribution | Ganho de realocação de volume com parâmetros baseline (sem outras mudanças) |
| load_network_data_tool | Dados da rede para contexto adicional |

## Conversão de Cobertura (regra ceil)
Fórmula geral (aplique a qualquer porcentagem informada pelo usuário):
  min_coverage_count = ceil(pct / 100 × {n_cps})

Exemplos ilustrativos:
{coverage_lines}

## Componentes de Custo Variável (por km)
| Componente | Chave | Baseline |
|---|---|---|
{var_comp_rows}
| **Total** | variable_cost_per_km | **${network.variable_cost_per_km:.4f}/km** |

## Componentes de Custo Fixo (por caminhão/mês)
| Componente | Chave | Baseline |
|---|---|---|
{fix_comp_rows}
| **Total** | fixed_cost_per_truck_month | **${network.fixed_cost_per_truck_month:,.2f}/{truck_unit}** |

{fix_driver_note}

## Alterações de Custo (use multiplicadores — não calcule manualmente)
Para mudanças percentuais ("X aumenta Y%") ou estruturais ("N motoristas por caminhão"),
passe var_cost_multipliers ou fix_cost_multipliers como string JSON.
O Python aplica os multiplicadores aos valores baseline. Não calcule os novos valores.

> NOTA OBRIGATÓRIA: Quando a ferramenta retornar computed_variable_cost_components /
> computed_fixed_cost_components, copie esses valores para os campos var_cost_* / fix_cost_*
> no JSON de saída. Esta cópia é obrigatória — sem ela, os custos do cenário estarão errados.

{cost_ex1}{cost_ex2}{cost_ex3}

## Status dos Terminais
- Por padrão, todos os terminais estão ativos: closed_terminals = []
- Para fechar um terminal, inclua seu ID na lista closed_terminals
- Terminais disponíveis: {terminal_ids_str}
- Exemplo: {closed_ex} está fechado → closed_terminals = ["{closed_ex}"]
{term_demand_section}

## Realocação de Volume
- Consulta de realocação pura (sem outras mudanças de parâmetros) → chame run_redistribution(). Sem argumentos — usa baseline. No JSON de saída: volume_redistribution = true, todos os outros params = baseline.
- Realocação combinada com outras mudanças de parâmetros → chame run_milp_solver(..., volume_redistribution=true).
NUNCA chame run_redistribution() se qualquer parâmetro difere do baseline (custo, velocidade, disponibilidade, dias úteis, carga útil, terminal fechado ou demanda). Para redistribuição combinada com ajuste de demanda, consulte a seção "Ajuste de Demanda por Terminal".

## Dois Níveis de Cobertura (ex: "Quanto custa ir de X% para Y% de cobertura?")
Para perguntas sobre diferença de custo entre dois níveis de cobertura: chame compare_coverage_costs(pct_from=X, pct_to=Y), preservando a direção pedida pelo usuário.
NUNCA inverta: se o usuário disse "de X% para Y%", chame compare_coverage_costs(pct_from=X, pct_to=Y) — nunca pct_from=Y, pct_to=X.
O Python converte as porcentagens em contagens e executa o solver duas vezes.
Exemplo — "de {ex_lower_pct}% para {ex_upper_pct}%": compare_coverage_costs(pct_from={ex_lower_pct}, pct_to={ex_upper_pct}) → coverage_count_a={ex_lower_count}, coverage_count_b={ex_upper_count}
Exemplo — "reduzir de 100% para 75%": compare_coverage_costs(pct_from=100, pct_to=75) → cost_difference negativo se 75% for mais barato.

No JSON de saída, copie os campos do resultado da ferramenta:
- scenario_params.coverage_count_a = tool["coverage_count_a"]
- scenario_params.coverage_count_b = tool["coverage_count_b"]

IMPORTANTE: coverage_count_a, coverage_count_b devem ser null em TODAS as outras queries. Nunca preencha esses campos se não chamou compare_coverage_costs.

## Fluxo de Decisão
Cada consulta é independente (baseline por padrão). Identifique o que mudou → preencha os parâmetros da rota correta → retorne o JSON exato abaixo.

IMPORTANTE — terminal_volume_caps: use SOMENTE quando o usuário pedir explicitamente redirecionar/redistribuir volume de um terminal para outros. Em TODAS as demais queries (baseline, what-if de parâmetros, fechamento de terminal, cobertura, orçamento): terminal_volume_caps = {{}}.

## Formato de Saída
```json
{{
  "scenario_params": {{
    "payload": <float>,
    "speed_loaded": <float>,
    "speed_empty": <float>,
    "availability": <float>,
    "overtime_hours": <float>,
    "overtime_cost": <float>,
    "variable_cost_per_km": <float>,
    "fixed_cost_per_truck_month": <float>,
    "working_days": <int>,
    "net_driving_hours": <float>,
    "closed_terminals": [<terminal_id>, ...],
    "min_coverage_count": <int or null>,
    "budget": <float or null>,
    "objective": <"minimize_cost"|"maximize_coverage"|"minimize_fleet">,
    "volume_redistribution": <bool>,
    "is_baseline": <bool>,
    "coverage_count_a": <int or null — somente se compare_coverage_costs foi chamado, senão null>,
    "coverage_count_b": <int or null — somente se compare_coverage_costs foi chamado, senão null>,
    "terminal_demand_multipliers": {{<terminal_id>: <multiplier>}},
    "terminal_volume_caps": {{<terminal_id>: <fração>}},
{var_override_snippet}
{fix_override_snippet}
  }}
}}
```

Os resultados do solver são produzidos depois pelo Python a partir destes parâmetros.
"""

    # English prompt
    cost_ex1 = (
        f'- "{ex_key} increases 10%" → var_cost_multipliers = \'{{{{"\\"{ex_key}\\"": 1.10}}}}\'; '
        f'result: var_cost_{ex_key} = {ex_new}'
    )
    return f"""You are an Operations Research expert specializing in truck fleet planning optimization.

## Agent Contract
- Role: fleet-planning scenario parser.
- Input: one user query and the current network data.
- Decision boundary: convert only the current query into `scenario_params`; do not reuse parameters from prior queries, invent solver results, or estimate cost or fleet.
- Output standard: structured JSON with valid parameters for Python to validate and solve deterministically.

## Network
- {n_cps} collection points: {cp_ids_str}
- {len(network.terminal_ids)} unload terminals:
{terminal_desc}
- Operation: FTL — one truck serves one collection point per trip, drives to one terminal

## Default Parameters (baseline)
| Parameter | Value |
|---|---|
| Payload | {network.payload} tons |
| Speed loaded | {network.speed_loaded} km/h |
| Speed empty | {network.speed_empty} km/h |
| Truck availability | {int(network.availability * 100)}% ({network.availability}) |
| Net driving hours | {network.net_driving_hours} h/day |
| Working days | {network.working_days}/month |
| Variable cost | ${network.variable_cost_per_km:.4f}/km |
| Fixed cost | ${network.fixed_cost_per_truck_month:,.2f}/truck/month |
| Overtime | {network.overtime_hours} h/day, ${network.overtime_cost:,.0f}/h |

## Objective Mapping
| Scenario type | objective | key parameters |
|---|---|---|
| Minimize cost (default) | minimize_cost | all CPs or min_coverage_count |
| Maximize coverage at budget | maximize_coverage | budget |
| Minimize fleet size | minimize_fleet | min_coverage_count |

is_baseline rule: set is_baseline = true ONLY when the user explicitly asks for the baseline scenario (e.g. "calculate baseline", "what is the standard scenario") with no parameter changes, no coverage constraint, and no budget. For all other scenarios: is_baseline = false.

Combined constraints: if the user specifies both a minimum coverage level AND a budget simultaneously, use objective = maximize_coverage with the budget as the constraint and min_coverage_count as the floor.

## Available Deterministic Routes
| Route | When to parameterize |
|---|---|
| run_milp_solver | Any single scenario (baseline, what-if, coverage, budget, terminal closure) |
| compare_coverage_costs | Cost difference between two coverage levels |
| run_redistribution | Volume redistribution gain at baseline parameters (no other changes) |
| load_network_data_tool | Fetch network data for additional context |

## Coverage Conversion (ceil rule)
General formula — apply to any percentage the user specifies:
  min_coverage_count = ceil(pct / 100 × {n_cps})

Illustrative examples:
{coverage_lines}

## Variable Cost Components (per km)
| Component | Key | Baseline |
|---|---|---|
{var_comp_rows}
| **Total** | variable_cost_per_km | **${network.variable_cost_per_km:.4f}/km** |

## Fixed Cost Components (per truck/month)
| Component | Key | Baseline |
|---|---|---|
{fix_comp_rows}
| **Total** | fixed_cost_per_truck_month | **${network.fixed_cost_per_truck_month:,.2f}/{truck_unit}** |

{fix_driver_note}

## Cost Changes (use multipliers — do not compute manually)
For percentage changes ("X increases Y%") or structural changes ("N drivers per truck"),
pass var_cost_multipliers or fix_cost_multipliers as a JSON string.
Python applies multipliers to baseline values. Do not compute new values yourself.

> MANDATORY NOTE: When the tool returns computed_variable_cost_components /
> computed_fixed_cost_components, copy those values to the var_cost_* / fix_cost_* fields
> in your output. This copy is required — without it, scenario costs will be wrong.

{cost_ex1}{cost_ex2}{cost_ex3}

## Terminal Status
- By default all terminals are active: closed_terminals = []
- To close a terminal, include its ID in the closed_terminals list
- Available terminals: {terminal_ids_str}
- Example: "{closed_ex} is closed" → closed_terminals = ["{closed_ex}"]
{term_demand_section}

## Volume Redistribution
- Pure redistribution query (no other parameter changes) → call run_redistribution(). No arguments — uses baseline. In output: volume_redistribution = true, all other params = baseline.
- Redistribution combined with parameter changes → call run_milp_solver(..., volume_redistribution=True).
NEVER call run_redistribution() if any parameter deviates from baseline (cost, speed, availability, working days, payload, closed terminal, or demand). For redistribution combined with demand adjustment, see the "Terminal Demand Adjustment" section.

## Two Coverage Levels (e.g. "What does it cost to go from X% to Y% coverage?")
For questions about the cost difference between two coverage levels: call compare_coverage_costs(pct_from=X, pct_to=Y), preserving the direction requested by the user.
NEVER reverse: if the user said "from X% to Y%", call compare_coverage_costs(pct_from=X, pct_to=Y) — never pct_from=Y, pct_to=X.
Python converts percentages to CP counts and runs the solver twice.
Example — "from {ex_lower_pct}% to {ex_upper_pct}%": compare_coverage_costs(pct_from={ex_lower_pct}, pct_to={ex_upper_pct}) → coverage_count_a={ex_lower_count}, coverage_count_b={ex_upper_count}
Example — "reduce from 100% to 75%": compare_coverage_costs(pct_from=100, pct_to=75) → negative cost_difference if 75% is cheaper.

In the output JSON, copy fields from the tool result:
- scenario_params.coverage_count_a = tool["coverage_count_a"]
- scenario_params.coverage_count_b = tool["coverage_count_b"]

IMPORTANT: coverage_count_a, coverage_count_b must be null for ALL other queries. Never populate these fields unless you called compare_coverage_costs.

## Decision Flow
Every query is independent (baseline by default). Identify what changed → fill the parameters for the correct route → return the exact JSON below.

IMPORTANT — terminal_volume_caps: use ONLY when the user explicitly asks to redirect/redistribute volume from one terminal to others. For ALL other queries (baseline, parameter what-ifs, terminal closure, coverage, budget): terminal_volume_caps = {{}}.

## Output Format
```json
{{
  "scenario_params": {{
    "payload": <float>,
    "speed_loaded": <float>,
    "speed_empty": <float>,
    "availability": <float>,
    "overtime_hours": <float>,
    "overtime_cost": <float>,
    "variable_cost_per_km": <float>,
    "fixed_cost_per_truck_month": <float>,
    "working_days": <int>,
    "net_driving_hours": <float>,
    "closed_terminals": [<terminal_id>, ...],
    "min_coverage_count": <int or null>,
    "budget": <float or null>,
    "objective": <"minimize_cost"|"maximize_coverage"|"minimize_fleet">,
    "volume_redistribution": <bool>,
    "is_baseline": <bool>,
    "coverage_count_a": <int or null — only if compare_coverage_costs was called, else null>,
    "coverage_count_b": <int or null — only if compare_coverage_costs was called, else null>,
    "terminal_demand_multipliers": {{<terminal_id>: <multiplier>}},
    "terminal_volume_caps": {{<terminal_id>: <fraction>}},
{var_override_snippet}
{fix_override_snippet}
  }}
}}
```

Solver results are produced later by Python from these parameters.
"""


def _make_or_output_schema(network: NetworkData) -> type[BaseModel]:
    """Build structured output schema with network-derived defaults and dynamic cost fields."""
    # Dynamic per-component cost override fields (Optional[float] = None)
    var_override_fields: dict[str, Any] = {
        f"var_cost_{key}": (Optional[float], None)
        for key in network.variable_cost_components
    }
    fix_override_fields: dict[str, Any] = {
        f"fix_cost_{key}": (Optional[float], None)
        for key in network.fixed_cost_components
    }

    _ScenarioParamsSchema = create_model(
        "_ScenarioParamsSchema",
        payload=(float, network.payload),
        speed_loaded=(float, network.speed_loaded),
        speed_empty=(float, network.speed_empty),
        availability=(float, network.availability),
        overtime_hours=(float, network.overtime_hours),
        overtime_cost=(float, network.overtime_cost),
        variable_cost_per_km=(float, network.variable_cost_per_km),
        fixed_cost_per_truck_month=(float, network.fixed_cost_per_truck_month),
        working_days=(int, network.working_days),
        net_driving_hours=(float, network.net_driving_hours),
        closed_terminals=(list[str], Field(default_factory=list)),
        min_coverage_count=(Optional[int], None),
        budget=(Optional[float], None),
        objective=(str, "minimize_cost"),
        volume_redistribution=(bool, False),
        is_baseline=(bool, False),
        coverage_count_a=(Optional[int], None),
        coverage_count_b=(Optional[int], None),
        terminal_demand_multipliers=(dict[str, float], Field(default_factory=dict)),
        terminal_volume_caps=(dict[str, float], Field(default_factory=dict)),
        var_cost_multipliers=(dict[str, float], Field(default_factory=dict)),
        fix_cost_multipliers=(dict[str, float], Field(default_factory=dict)),
        **var_override_fields,
        **fix_override_fields,
    )

    class _ORAgentOutput(BaseModel):
        scenario_params: _ScenarioParamsSchema  # type: ignore[valid-type]

    return _ORAgentOutput


@functools.lru_cache(maxsize=4)
def _cached_system_prompt(language: str) -> str:
    network = load_network_data()
    return _build_system_prompt(network, language) + _parser_mode_override(language)


def create_or_agent(provider: str, model_id: str, api_key: str, language: str = "pt") -> LLMConfig:
    """Create the OR parser model config."""
    return make_model(provider, model_id, api_key, 4096, _cached_system_prompt(language))


def _parser_mode_override(language: str) -> str:
    if language == "pt":
        return (
            "\n\n## Modo Parser LangGraph\n"
            "Não chame ferramentas e não produza resultados de solver. Retorne somente o objeto "
            "estruturado scenario_params. O Python validará os parâmetros, escolherá a rota "
            "determinística de solver e reconciliará os resultados.\n"
            "- Para mudanças percentuais de custo, prefira var_cost_multipliers ou "
            "fix_cost_multipliers com chaves reais dos componentes. O Python aplicará os "
            "multiplicadores aos valores baseline.\n"
            "- Use campos var_cost_* ou fix_cost_* apenas quando o usuário informar um valor "
            "absoluto de componente.\n"
            "- Para cobertura X -> Y, preencha coverage_count_a e coverage_count_b na direção "
            "do usuário; o Python executará os dois solves.\n"
            "- Para redistribuição pura ou combinada, marque volume_redistribution=true; o "
            "Python escolherá a rota correta.\n"
            "- Nunca invente trucks, custos, cobertura atendida ou assignments."
        )
    return (
        "\n\n## LangGraph Parser Mode\n"
        "Do not call tools and do not produce solver results. Return only the structured "
        "scenario_params object. Python will validate parameters, choose the deterministic "
        "solver route, and reconcile outputs.\n"
        "- For percentage cost changes, prefer var_cost_multipliers or fix_cost_multipliers "
        "with real component keys. Python will apply multipliers to baseline values.\n"
        "- Use var_cost_* or fix_cost_* fields only when the user gives an absolute component value.\n"
        "- For coverage X -> Y, fill coverage_count_a and coverage_count_b in the user's direction; "
        "Python will run both solves.\n"
        "- For pure or combined redistribution, set volume_redistribution=true; Python will choose "
        "the correct route.\n"
        "- Never invent trucks, costs, served coverage, or assignments."
    )


@functools.lru_cache(maxsize=2)
def _get_or_output_schema(language: str) -> type:
    network = load_network_data()
    return _make_or_output_schema(network)


def _parse_directional_coverage_query(query: str) -> tuple[float, float] | None:
    """Extract X -> Y when the user explicitly asks a coverage transition."""
    text = query.lower()
    if "cobertura" not in text and "coverage" not in text:
        return None

    patterns = (
        r"\bde\s+(\d+(?:[,.]\d+)?)\s*%?\s+(?:para|a)\s+(\d+(?:[,.]\d+)?)\s*%?",
        r"\bfrom\s+(\d+(?:[,.]\d+)?)\s*%?\s+to\s+(\d+(?:[,.]\d+)?)\s*%?",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            pct_from = float(match.group(1).replace(",", "."))
            pct_to = float(match.group(2).replace(",", "."))
            return pct_from, pct_to
    return None


def _reconcile_directional_coverage_comparison(
    query: str,
    scenario_params: ScenarioParams,
    milp_result: MILPResult,
) -> None:
    """Make explicit coverage X -> Y questions deterministic after LLM parsing."""
    transition = _parse_directional_coverage_query(query)
    if transition is None:
        return

    pct_from, pct_to = transition
    closed_terminals = ",".join(
        tid for tid, active in scenario_params.terminals_active.items() if not active
    )
    tool_result = json.loads(
        compare_coverage_costs(
            pct_from=pct_from,
            pct_to=pct_to,
            payload=scenario_params.payload,
            speed_loaded=scenario_params.speed_loaded,
            speed_empty=scenario_params.speed_empty,
            availability=scenario_params.availability,
            overtime_hours=scenario_params.overtime_hours,
            overtime_cost=scenario_params.overtime_cost,
            variable_cost_per_km=scenario_params.variable_cost_per_km,
            fixed_cost_per_truck_month=scenario_params.fixed_cost_per_truck_month,
            working_days=scenario_params.working_days,
            net_driving_hours=scenario_params.net_driving_hours,
            closed_terminals=closed_terminals,
            terminal_demand_multipliers=json.dumps(scenario_params.terminal_demand_multipliers),
        )
    )

    milp_result.feasible = bool(tool_result.get("feasible"))
    milp_result.infeasibility_reason = tool_result.get("infeasibility_reason", "")
    if not milp_result.feasible:
        return

    scenario_params.coverage_count_a = tool_result["coverage_count_a"]
    scenario_params.coverage_count_b = tool_result["coverage_count_b"]
    scenario_params.min_coverage_count = tool_result["coverage_count"]
    scenario_params.served_cps = list(tool_result.get("served_cps", []))
    milp_result.trucks_a = tool_result["trucks_a"]
    milp_result.trucks_b = tool_result["trucks_b"]
    milp_result.cost_a = tool_result["cost_a"]
    milp_result.cost_b = tool_result["cost_b"]
    milp_result.cost_difference = tool_result["cost_difference"]
    milp_result.trucks = tool_result["trucks"]
    milp_result.total_cost = tool_result["total_cost"]
    milp_result.total_km = tool_result["total_km"]
    milp_result.fixed_cost = tool_result["fixed_cost"]
    milp_result.variable_cost = tool_result["variable_cost"]
    milp_result.overtime_cost_total = tool_result["overtime_cost"]
    milp_result.coverage_count = tool_result["coverage_count"]
    milp_result.served_cps = list(tool_result.get("served_cps", []))
    milp_result.assignments = dict(tool_result.get("assignments", {}))


def _scenario_params_from_structured(sp: Any, network: NetworkData) -> ScenarioParams:
    """Convert the parser structured output into ScenarioParams."""
    # Build per-component cost dicts, merging baseline defaults with agent-set overrides.
    var_components = dict(network.variable_cost_components)
    fix_components = dict(network.fixed_cost_components)

    any_var_set = False
    any_fix_set = False
    for key, multiplier in dict(getattr(sp, "var_cost_multipliers", {}) or {}).items():
        if key in var_components:
            try:
                var_components[key] = round(var_components[key] * float(multiplier), 4)
                any_var_set = True
            except (TypeError, ValueError):
                pass
    for key, multiplier in dict(getattr(sp, "fix_cost_multipliers", {}) or {}).items():
        if key in fix_components:
            try:
                fix_components[key] = round(fix_components[key] * float(multiplier), 2)
                any_fix_set = True
            except (TypeError, ValueError):
                pass
    for key in network.variable_cost_components:
        val = getattr(sp, f"var_cost_{key}", None)
        if val is not None:
            var_components[key] = float(val)
            any_var_set = True
    for key in network.fixed_cost_components:
        val = getattr(sp, f"fix_cost_{key}", None)
        if val is not None:
            fix_components[key] = float(val)
            any_fix_set = True

    variable_cost_per_km = (
        round(sum(var_components.values()), 4) if any_var_set else sp.variable_cost_per_km
    )
    fixed_cost_per_truck_month = (
        round(sum(fix_components.values()), 2) if any_fix_set else sp.fixed_cost_per_truck_month
    )

    # Build terminals_active from network IDs; close any listed in closed_terminals.
    terminals_active = {t: True for t in network.terminal_ids}
    for tid in sp.closed_terminals:
        if tid in terminals_active:
            terminals_active[tid] = False

    scenario_params = ScenarioParams(
        payload=sp.payload,
        speed_loaded=sp.speed_loaded,
        speed_empty=sp.speed_empty,
        availability=sp.availability,
        overtime_hours=sp.overtime_hours,
        overtime_cost=sp.overtime_cost,
        variable_cost_per_km=variable_cost_per_km,
        fixed_cost_per_truck_month=fixed_cost_per_truck_month,
        working_days=sp.working_days,
        net_driving_hours=sp.net_driving_hours,
        terminals_active=terminals_active,
        min_coverage_count=sp.min_coverage_count,
        budget=sp.budget,
        objective=sp.objective,
        volume_redistribution=sp.volume_redistribution,
        is_baseline=sp.is_baseline,
        served_cps=list(network.cp_ids),
        coverage_count_a=sp.coverage_count_a,
        coverage_count_b=sp.coverage_count_b,
        variable_cost_components=var_components,
        fixed_cost_components=fix_components,
        terminal_demand_multipliers=dict(sp.terminal_demand_multipliers) if sp.terminal_demand_multipliers else {},
        terminal_volume_caps=dict(sp.terminal_volume_caps) if sp.terminal_volume_caps else {},
    )

    return scenario_params


def _run_solver_for_params(
    query: str,
    network: NetworkData,
    scenario_params: ScenarioParams,
) -> MILPResult:
    """Run the deterministic Python solver path selected from parsed params."""
    from ..models.solver import run_milp_solver as _run_solver
    from ..workflow.solver_service import run_coverage_comparison

    transition = _parse_directional_coverage_query(query)
    if transition is None:
        result = _run_solver(network, scenario_params)
        if result.served_cps:
            scenario_params.served_cps = list(result.served_cps)
        if result.feasible and result.assignments:
            scenario_params.milp_assignments = dict(result.assignments)
        return result

    pct_from, pct_to = transition
    cov = run_coverage_comparison(network, scenario_params, pct_from, pct_to)
    if cov.feasible:
        scenario_params.coverage_count_a = cov.coverage_count_a
        scenario_params.coverage_count_b = cov.coverage_count_b
        scenario_params.served_cps = list(cov.result_b.served_cps)
        if cov.result_b.assignments:
            scenario_params.milp_assignments = dict(cov.result_b.assignments)
    return cov.result_b


def run_or_agent(
    agent: LLMConfig, query: str
) -> tuple[MILPResult, ScenarioParams]:
    """Legacy compatibility wrapper: parse, then run deterministic solver routing.

    New workflow code should call parse_or_agent and route solver execution through
    workflow.solver_service instead.
    """
    network = load_network_data()
    scenario_params = parse_or_agent(agent, query, network)
    milp_result = _run_solver_for_params(query, network, scenario_params)
    return milp_result, scenario_params


def parse_or_agent(
    agent: LLMConfig,
    query: str,
    network: NetworkData | None = None,
) -> ScenarioParams:
    """Run the stateless OR parser and return ScenarioParams only."""
    network = network or load_network_data()
    output = invoke_structured(agent, query, _get_or_output_schema("default"))
    return _scenario_params_from_structured(output.scenario_params, network)
