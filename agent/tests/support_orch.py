"""Fakes for orchestration tests."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any

from agent.llm import (
    LLMError,
    LLMResponse,
    ModelConfig,
    ModelInfo,
    TokenUsage,
)
from agent.llm.models import ToolCall
from agent.orchestration.models import ExecutionPlan, PlanStep
from agent.tools.bridge import ToolOutcome


def ok(data: Any, tool: str = "x__y") -> ToolOutcome:
    server, _, name = tool.partition("__")
    return ToolOutcome(
        call_id="c",
        spec_name=tool,
        server_id=server,
        tool_name=name or tool,
        success=True,
        content="{}",
        data={"success": True, "data": data},
        request_id="req",
    )


def fail(
    code: str = "TOOL_EXECUTION_FAILED", *, retryable: bool = False, message: str = "boom"
) -> ToolOutcome:
    return ToolOutcome(
        call_id="c",
        spec_name="x__y",
        success=False,
        error_code=code,
        retryable=retryable,
        content=json.dumps({"success": False, "error": {"code": code, "message": message}}),
    )


def domain_fail(message: str = "repo not found", kind: str = "github_api_error") -> ToolOutcome:
    return ToolOutcome(
        call_id="c",
        spec_name="x__y",
        success=True,
        content="{}",
        data={"success": False, "error": {"type": kind, "message": message}},
    )


class FakeRunner:
    """Records calls, concurrency and timing; behaviours are scripted per tool."""

    def __init__(self) -> None:
        self.behaviors: dict[str, Callable[[dict], ToolOutcome | list[ToolOutcome]]] = {}
        self.delays: dict[str, float] = {}
        self.calls: list[tuple[str, dict]] = []
        self.finished: list[str] = []
        self.timeouts: list[float | None] = []
        self.active = 0
        self.peak = 0
        self.cancelled = 0
        self._sequences: dict[str, list[ToolOutcome]] = {}

    def sequence(self, tool: str, *outcomes: ToolOutcome) -> None:
        self._sequences[tool] = list(outcomes)

    async def execute(self, call: ToolCall, *, timeout_seconds: float | None = None) -> ToolOutcome:
        self.calls.append((call.name, dict(call.arguments)))
        self.timeouts.append(timeout_seconds)
        self.active += 1
        self.peak = max(self.peak, self.active)

        try:
            await asyncio.sleep(self.delays.get(call.name, 0))

            if call.name in self._sequences and self._sequences[call.name]:
                outcome = self._sequences[call.name].pop(0)
            elif call.name in self.behaviors:
                outcome = self.behaviors[call.name](call.arguments)
            else:
                outcome = ok({"tool": call.name, "args": call.arguments}, call.name)

            self.finished.append(call.name)
            return outcome
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        finally:
            self.active -= 1

    def count(self, tool: str) -> int:
        return sum(1 for name, _ in self.calls if name == tool)


def step(
    step_id: str, tool: str | None = None, deps: list[str] | None = None, **kwargs: Any
) -> PlanStep:
    return PlanStep(id=step_id, tool=tool or f"srv__{step_id}", depends_on=deps or [], **kwargs)


def plan(*steps: PlanStep, **kwargs: Any) -> ExecutionPlan:
    return ExecutionPlan(steps=list(steps), **kwargs)


class ScriptedProvider:
    """Provider stand-in: text script + structured-output script."""

    name = "scripted"

    def __init__(
        self, texts: list[LLMResponse | LLMError] | None = None, structured: list[Any] | None = None
    ):
        self.config = ModelConfig(provider="scripted", model="scripted-1")
        self.texts = list(texts or [])
        self.structured = list(structured or [])
        self.text_requests: list[tuple[list, Any]] = []
        self.structured_requests: list[tuple[list, type]] = []

    async def generate(self, messages, *, tools=None):
        self.text_requests.append((list(messages), tools))
        item = self.texts.pop(0)

        if isinstance(item, Exception):
            raise item

        return item

    async def generate_structured(self, messages, schema):
        self.structured_requests.append((list(messages), schema))
        item = self.structured.pop(0)

        if isinstance(item, Exception):
            raise item

        return item if isinstance(item, schema) else schema.model_validate(item)

    async def stream(self, messages, *, tools=None):
        yield  # pragma: no cover

    def model_info(self) -> ModelInfo:
        return ModelInfo(provider="scripted", model="scripted-1")

    async def health_check(self) -> bool:
        return True

    async def aclose(self) -> None:
        return None


def text(content: str, tokens: int = 10) -> LLMResponse:
    return LLMResponse(content=content, usage=TokenUsage.of(tokens, 1))
