"""Aggregate step results from different servers into one view (7.3)."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agent.orchestration.models import (
    PlanStep,
    StepRecord,
    StepState,
    WorkflowStatus,
)

LLM_SOURCE = "llm"


class StepSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_id: str
    description: str = ""
    tool: str | None = None
    state: StepState
    data_preview: str | None = None
    error: str | None = None
    deduplicated: bool = False


class AggregatedResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: WorkflowStatus
    # server_id -> summaries (synthesize steps are grouped under "llm").
    by_server: dict[str, list[StepSummary]] = Field(default_factory=dict)
    succeeded: list[str] = Field(default_factory=list)
    failed: dict[str, str] = Field(default_factory=dict)
    skipped: dict[str, str] = Field(default_factory=dict)
    # Recovery (replan) rounds that were needed to get here.
    recoveries: int = 0

    @property
    def servers_used(self) -> list[str]:
        return sorted(s for s in self.by_server if s != LLM_SOURCE)

    def digest(self, max_chars: int = 6000) -> str:
        """Plain-text rendering handed to the model for the final answer."""
        lines = [f"Workflow status: {self.status.value}."]

        if self.recoveries:
            lines.append(f"(Recovered after {self.recoveries} replanning round(s).)")

        for server, summaries in self.by_server.items():
            lines.append(f"\n## {server}")

            for s in summaries:
                head = f"- [{s.state.value}] {s.step_id}: {s.description or s.tool}"
                lines.append(head)

                if s.data_preview:
                    lines.append(f"  result: {s.data_preview}")

                if s.error:
                    lines.append(f"  problem: {s.error}")

        text = "\n".join(lines)
        return text if len(text) <= max_chars else text[: max_chars - 20] + "\n...[truncated]"


def _unwrap(data: Any) -> Any:
    """Strip the Nexus MCP ``{"success": true, "data": ...}`` envelope."""
    if isinstance(data, dict) and data.get("success") is True and "data" in data:
        return data["data"]

    return data


class ResultAggregator:
    def __init__(self, *, max_preview_chars: int = 1500) -> None:
        if max_preview_chars < 50:
            raise ValueError("max_preview_chars must be at least 50")

        self._max_preview = max_preview_chars

    def _preview(self, data: Any) -> str | None:
        if data is None:
            return None

        text = json.dumps(_unwrap(data), default=str, ensure_ascii=False, separators=(",", ":"))
        return (
            text if len(text) <= self._max_preview else text[: self._max_preview] + "...[truncated]"
        )

    def aggregate(
        self,
        steps: Mapping[str, PlanStep],
        records: Mapping[str, StepRecord],
        status: WorkflowStatus,
    ) -> AggregatedResult:
        result = AggregatedResult(status=status)

        for step_id, record in records.items():
            step = steps.get(step_id)
            source = (
                LLM_SOURCE if record.kind.value == "synthesize" else (record.server_id or "unknown")
            )
            error = None

            if record.state is StepState.SKIPPED:
                error = record.skip_reason
                result.skipped[step_id] = error or "skipped"
            elif not record.succeeded:
                error = f"{record.error_code}: {record.error_message}"
                result.failed[step_id] = error
            else:
                result.succeeded.append(step_id)

            result.by_server.setdefault(source, []).append(
                StepSummary(
                    step_id=step_id,
                    description=step.description if step else "",
                    tool=record.tool,
                    state=record.state,
                    data_preview=self._preview(record.data) if record.succeeded else None,
                    error=error,
                    deduplicated=record.deduplicated,
                )
            )

        return result
