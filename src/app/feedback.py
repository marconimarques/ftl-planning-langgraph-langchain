"""Feedback loop — capture, persist, and inject few-shot examples for agent improvement."""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..domain.data_types import MILPResult, ScenarioParams

_FEEDBACK_DIR = Path("data/feedback")
_MAX_INJECT = 3


def detect_scenario_type(
    params: ScenarioParams,
    milp_result: MILPResult,
    coverage_comparison=None,
) -> str:
    """Derive the scenario type label used as a filter key for few-shot injection."""
    if params.is_baseline:
        return "baseline"
    if coverage_comparison is not None:
        return "two_level_cost_diff"
    if params.skip_capacity_constraints:
        return "capacity_overflow"
    if params.terminal_volume_caps and params.volume_redistribution:
        return "volume_cap_redistribution"
    if params.terminal_demand_multipliers and params.volume_redistribution:
        return "demand_change_redistribution"
    if params.volume_redistribution:
        return "volume_redistribution"
    if params.terminal_demand_multipliers:
        return "demand_change"
    if any(not v for v in params.terminals_active.values()):
        return "terminal_closure"
    if params.objective == "maximize_coverage":
        return "maximize_coverage_budget"
    if params.objective == "minimize_fleet":
        return "minimize_fleet"
    if params.min_coverage_count is not None:
        return "minimize_cost_coverage"
    return "parametric_whatif"


def save_feedback(
    *,
    agent: str,
    lang: str,
    rating: int,
    scenario_type: str,
    query: str,
    key_facts: dict,
    output: str,
    correction: Optional[str],
) -> None:
    """Append one feedback record to the agent's JSONL file."""
    _FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "id": str(uuid.uuid4()),
        "ts": datetime.now().isoformat(timespec="seconds"),
        "agent": agent,
        "lang": lang,
        "rating": rating,
        "scenario_type": scenario_type,
        "query": query,
        "key_facts": key_facts,
        "output": output,
        "correction": correction or None,
    }
    path = _FEEDBACK_DIR / f"{agent}_{lang}.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_examples(
    agent: str,
    lang: str,
    scenario_type: str,
    max_examples: int = _MAX_INJECT,
) -> list[dict]:
    """Load up to max_examples few-shot records for the given agent/lang/scenario_type.

    Returns approved agent outputs (rating 3) and any written user contribution.
    Agent outputs rated 1 or 2 remain in the log, but are not used as examples
    unless the user provided a correction/improvement text.
    Falls back to PT examples when no examples exist for the requested language.
    """
    path = _FEEDBACK_DIR / f"{agent}_{lang}.jsonl"
    if not path.exists() and lang != "pt":
        path = _FEEDBACK_DIR / f"{agent}_pt.jsonl"
    if not path.exists():
        return []
    records: list[dict] = []
    try:
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                has_user_contribution = bool((rec.get("correction") or "").strip())
                is_approved_agent_output = rec.get("rating", 0) >= 3
                if rec.get("scenario_type") == scenario_type and (has_user_contribution or is_approved_agent_output):
                    records.append(rec)
    except OSError:
        return []
    records.sort(key=lambda r: r.get("ts", ""), reverse=True)
    return records[:max_examples]


def format_few_shot_block(examples: list[dict], lang: str) -> str:
    """Format examples as a prompt block for injection before the final instruction."""
    if not examples:
        return ""
    is_pt = lang == "pt"
    tag = "exemplos_referencia" if is_pt else "reference_examples"
    intro = (
        "Exemplos de respostas bem avaliadas para este tipo de cenário — use como referência de estilo, tom e profundidade:"
        if is_pt
        else "High-quality example responses for this scenario type — use as style, tone, and depth reference:"
    )
    lines: list[str] = [f"\n\n<{tag}>", intro]
    for i, rec in enumerate(examples, 1):
        canonical = rec.get("correction") or rec.get("output", "")
        query = rec.get("query", "")
        source = (
            "contribuição do usuário" if is_pt and rec.get("correction")
            else "resposta aprovada" if is_pt
            else "user contribution" if rec.get("correction")
            else "approved response"
        )
        lines.append(f"\n[{i}] {source}  query: {query!r}")
        lines.append(f"response: {canonical!r}")
    lines.append(f"</{tag}>")
    return "\n".join(lines)
