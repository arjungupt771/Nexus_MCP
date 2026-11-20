"""Plan, execution-state and result models for multi-server orchestration.

Planning produces an ``ExecutionPlan`` (pure data). Execution produces an
``ExecutionReport``. Neither knows about the other's implementation, so each
can be tested alone.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from agent.orchestration.references import find_refs

STEP_ID_PATTERN = r"^[A-Za-z][A-Za-z0-9_]{0,31}$"


class StepKind(StrEnum):
    TOOL = "tool"
    # Produces text with the LLM from (resolved) inputs, e.g. to summarize
    # results from several servers before writing them somewhere.
    SYNTHESIZE = "synthesize"


class PlanStep(BaseModel):
    """One unit of work.

    ``arguments`` may contain references to earlier steps:
    ``{"$ref": "s1", "path": "data.items.0.name"}`` or
    ``{"$template": "Found {s1.data.total_count} repos"}``.
    Referenced steps become dependencies automatically.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=STEP_ID_PATTERN)
    description: str = ""
    kind: StepKind = StepKind.TOOL
    tool: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)
    instruction: str | None = None
    timeout_seconds: float | None = Field(default=None, gt=0, le=600)
    retries: int = Field(default=0, ge=0, le=3)
    # Identical tool+arguments calls in one workflow are normally executed
    # once; set True for deliberate repeats of side-effecting tools.
    allow_duplicate: bool = False

    @field_validator("depends_on")
    @classmethod
    def _unique_deps(cls, value: list[str]) -> list[str]:
        return list(dict.fromkeys(value))

    @model_validator(mode="after")
    def _check_kind(self) -> PlanStep:
        if self.kind is StepKind.TOOL and not self.tool:
            raise ValueError(f"step '{self.id}': tool steps require 'tool'.")

        if self.kind is StepKind.SYNTHESIZE and not self.instruction:
            raise ValueError(f"step '{self.id}': synthesize steps require 'instruction'.")

        if self.id in self.depends_on:
            raise ValueError(f"step '{self.id}' cannot depend on itself.")

        return self


def topological_layers(dependencies: dict[str, list[str]]) -> list[list[str]]:
    """Group ids into levels that can run in parallel; raises on cycles/unknowns."""
    unknown = sorted({d for deps in dependencies.values() for d in deps} - dependencies.keys())

    if unknown:
        raise ValueError(f"unknown dependencies: {', '.join(unknown)}")

    remaining = {k: set(v) for k, v in dependencies.items()}
    layers: list[list[str]] = []

    while remaining:
        ready = sorted(k for k, deps in remaining.items() if not deps)

        if not ready:
            raise ValueError(f"dependency cycle among: {', '.join(sorted(remaining))}")

        layers.append(ready)

        for key in ready:
            del remaining[key]

        for deps in remaining.values():
            deps.difference_update(ready)

    return layers


class ExecutionPlan(BaseModel):
    """A validated dependency graph of steps."""

    model_config = ConfigDict(extra="forbid")

    plan_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    goal: str = ""
    steps: list[PlanStep] = Field(default_factory=list)
    # Ids of already-completed steps (from an earlier plan) this plan may reference.
    inputs: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_graph(self) -> ExecutionPlan:
        ids = [s.id for s in self.steps]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})

        if duplicates:
            raise ValueError(f"duplicate step ids: {', '.join(duplicates)}")

        overlap = sorted(set(ids) & set(self.inputs))

        if overlap:
            raise ValueError(f"step ids reuse completed ids: {', '.join(overlap)}")

        known = set(ids) | set(self.inputs)

        for step in self.steps:
            refs = find_refs(step.arguments)
            missing = sorted(refs - known)

            if missing:
                raise ValueError(
                    f"step '{step.id}' references unknown step(s): {', '.join(missing)}"
                )

            # Referenced steps are implicit dependencies.
            for ref in sorted(refs):
                if ref not in step.depends_on:
                    step.depends_on.append(ref)

        self.layers()
        return self

    def layers(self) -> list[list[str]]:
        """Parallel execution levels (dependencies on ``inputs`` are pre-satisfied)."""
        own = {s.id for s in self.steps}
        unknown = sorted({d for s in self.steps for d in s.depends_on} - own - set(self.inputs))

        if unknown:
            raise ValueError(f"unknown dependencies: {', '.join(unknown)}")

        return topological_layers({s.id: [d for d in s.depends_on if d in own] for s in self.steps})

    def step(self, step_id: str) -> PlanStep:
        return next(s for s in self.steps if s.id == step_id)


class Intent(BaseModel):
    """What the user wants, as understood before any planning."""

    model_config = ConfigDict(extra="forbid")

    goal: str = Field(description="One-sentence restatement of the user's goal.")
    needs_tools: bool = Field(description="True only if external tools are required to satisfy it.")
    server_hints: list[str] = Field(
        default_factory=list, description="Server ids likely to be needed (may be empty)."
    )


class Task(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=STEP_ID_PATTERN)
    description: str = Field(min_length=1)
    depends_on: list[str] = Field(default_factory=list)


class TaskPlan(BaseModel):
    """Tool-agnostic breakdown of a request."""

    model_config = ConfigDict(extra="forbid")

    tasks: list[Task] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate(self) -> TaskPlan:
        ids = [t.id for t in self.tasks]

        if len(ids) != len(set(ids)):
            raise ValueError("duplicate task ids")

        topological_layers({t.id: t.depends_on for t in self.tasks})
        return self


# ---------------------------------------------------------------- execution state


class StepState(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


TERMINAL_FAILURES = {StepState.FAILED, StepState.TIMED_OUT, StepState.SKIPPED, StepState.CANCELLED}


class WorkflowStatus(StrEnum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    PARTIAL = "partial"
    FAILED = "failed"
    CANCELLED = "cancelled"


def _now() -> datetime:
    return datetime.now(UTC)


class StepRecord(BaseModel):
    """Live and final state of one step."""

    model_config = ConfigDict(extra="forbid")

    step_id: str
    kind: StepKind = StepKind.TOOL
    tool: str | None = None
    server_id: str | None = None
    tool_name: str | None = None
    state: StepState = StepState.PENDING
    attempts: int = 0
    execution_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    tool_request_id: str | None = None
    resolved_arguments: dict[str, Any] = Field(default_factory=dict)
    data: Any = None
    error_code: str | None = None
    error_message: str | None = None
    retryable: bool = False
    skip_reason: str | None = None
    deduplicated: bool = False
    dedupe_key: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    duration_ms: float = 0.0

    @property
    def succeeded(self) -> bool:
        return self.state is StepState.SUCCEEDED


class ExecutionReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    execution_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    plan_id: str
    status: WorkflowStatus = WorkflowStatus.RUNNING
    steps: dict[str, StepRecord] = Field(default_factory=dict)
    started_at: datetime = Field(default_factory=_now)
    finished_at: datetime | None = None
    duration_ms: float = 0.0

    def ids_in(self, *states: StepState) -> list[str]:
        return [i for i, r in self.steps.items() if r.state in states]

    @property
    def failed_ids(self) -> list[str]:
        return self.ids_in(StepState.FAILED, StepState.TIMED_OUT)

    @property
    def has_failures(self) -> bool:
        return any(r.state in TERMINAL_FAILURES for r in self.steps.values())

    def finalize(self, *, cancelled: bool = False) -> None:
        states = [r.state for r in self.steps.values()]
        succeeded = sum(s is StepState.SUCCEEDED for s in states)

        if cancelled:
            self.status = WorkflowStatus.CANCELLED
        elif succeeded == len(states):
            self.status = WorkflowStatus.SUCCEEDED
        elif succeeded == 0:
            self.status = WorkflowStatus.FAILED
        else:
            self.status = WorkflowStatus.PARTIAL

        self.finished_at = _now()
        self.duration_ms = (self.finished_at - self.started_at).total_seconds() * 1000


StopReason = Literal["completed", "partial", "failed", "error", "no_tools_needed"]
