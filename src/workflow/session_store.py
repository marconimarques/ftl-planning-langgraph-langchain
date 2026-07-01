"""Session history and audit persistence boundary."""

from __future__ import annotations

from ..domain.data_types import PipelineResult
from ..domain.workflow_types import AuditRecord


class InMemorySessionStore:
    def __init__(self) -> None:
        self.scenario_history: list[PipelineResult] = []
        self.audit_records: list[AuditRecord] = []
        self.shock_strategy_history: list[list[dict]] = []

    def append_scenario(self, result: PipelineResult) -> None:
        self.scenario_history.append(result)

    def append_audit(self, record: AuditRecord) -> None:
        self.audit_records.append(record)

    def append_shock_strategies(self, strategies: list[dict]) -> None:
        """Persist the params_changed dicts from one shock analysis round."""
        self.shock_strategy_history.append(list(strategies))

    def last_audit(self) -> AuditRecord | None:
        return self.audit_records[-1] if self.audit_records else None
