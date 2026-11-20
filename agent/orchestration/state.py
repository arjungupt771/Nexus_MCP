"""Structured LangGraph state for multi-server orchestration."""

from __future__ import annotations

from typing import Any, TypedDict

from agent.llm.models import Message, TokenUsage
from agent.orchestration.aggregator import AggregatedResult
from agent.orchestration.models import (
    ExecutionPlan,
    ExecutionReport,
    Intent,
    PlanStep,
    StepRecord,
    TaskPlan,
)
from agent.orchestration.tools import ToolInfo


class OrchestrationState(TypedDict, total=False):
    request_id: str
    user_input: str
    history: list[Message]
    intent: Intent | None
    tasks: TaskPlan | None
    candidates: list[ToolInfo]
    plan: ExecutionPlan | None
    plan_ids: list[str]
    plan_issues: list[str]
    plan_attempts: int
    layers: list[list[str]]
    report: ExecutionReport | None
    # Accumulated across recovery rounds.
    steps: dict[str, PlanStep]
    records: dict[str, StepRecord]
    superseded: list[str]
    replans: int
    aggregated: AggregatedResult | None
    final_response: str
    usage: TokenUsage
    stop_reason: str
    error: dict[str, Any] | None
