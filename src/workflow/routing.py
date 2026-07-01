"""Conditional routing helpers for graph nodes."""

from __future__ import annotations

from ..domain.workflow_types import FleetPlanningState


def route_after_intent(state: FleetPlanningState) -> str:
    return state["intent"].label


def route_after_validation(state: FleetPlanningState) -> str:
    validation = state["validation"]
    if validation.valid:
        return "valid"
    if validation.needs_clarification:
        return "needs_clarification"
    return "invalid"


def route_after_solver(state: FleetPlanningState) -> str:
    result = state["primary_result"]
    if result.feasible:
        return "feasible"
    return "infeasible"


def route_after_consistency(state: FleetPlanningState) -> str:
    return "pass" if state["consistency"].passed else "fail"
