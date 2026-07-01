"""Audit record helpers."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
from typing import Any

from .workflow_types import AuditRecord


def audit_to_dict(record: AuditRecord) -> dict[str, Any]:
    """Convert an audit record to plain Python containers."""
    return asdict(record) if is_dataclass(record) else dict(record)


def append_audit_jsonl(record: AuditRecord, path: Path) -> None:
    """Append an audit record as one JSONL entry."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = audit_to_dict(record)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def load_audit_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load audit JSONL entries as dictionaries, skipping malformed lines."""
    if not path.exists():
        return []

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                records.append(payload)
    return records
