"""Helpers for memory tests: deterministic clock, builders, fake runners."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

from agent.memory import (
    MemoryCreate,
    MemoryManager,
    MemoryRecord,
    MemorySettings,
    MemoryType,
)
from agent.memory.models import MemoryKind
from agent.memory.text import content_hash, keyword_set


class Clock:
    """Controllable clock so recency/retention tests are deterministic."""

    def __init__(self, start: datetime | None = None) -> None:
        self.now = start or datetime(2026, 11, 17, 12, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **kwargs: float) -> None:
        self.now += timedelta(**kwargs)


def record(
    content: str = "User prefers dark mode",
    *,
    user_id: str = "u1",
    memory_type: MemoryType = MemoryType.LONG_TERM,
    kind: MemoryKind = MemoryKind.PREFERENCE,
    importance: float = 0.7,
    clock: Clock | None = None,
    **kwargs: Any,
) -> MemoryRecord:
    now = (clock or Clock())()
    return MemoryRecord(
        user_id=user_id,
        type=memory_type,
        kind=kind,
        content=content,
        importance=importance,
        keywords=keyword_set(content),
        content_hash=content_hash(content),
        created_at=kwargs.pop("created_at", now),
        updated_at=kwargs.pop("updated_at", now),
        **kwargs,
    )


def create(content: str, **kwargs: Any) -> MemoryCreate:
    return MemoryCreate(content=content, user_id=kwargs.pop("user_id", "u1"), **kwargs)


def manager(
    clock: Clock | None = None, settings: MemorySettings | None = None, **kwargs: Any
) -> MemoryManager:
    """In-memory SQLite manager with an injected clock on every component."""
    built = MemoryManager.in_memory(settings, **kwargs)

    if clock is not None:
        built.memory._clock = clock
        built.memory.retriever._clock = clock
        built.conversations._clock = clock

    return built


class FakeRunner:
    """Stand-in for Agent/Orchestrator with scriptable results."""

    def __init__(
        self,
        reply: str = "ok",
        *,
        stop_reason: str = "completed",
        outcomes: list | None = None,
        raises: Exception | None = None,
    ) -> None:
        self.reply = reply
        self.stop_reason = stop_reason
        self.outcomes = outcomes or []
        self.raises = raises
        self.calls: list[dict[str, Any]] = []

    async def run(self, request: str, /, *, history=None, request_id=None):
        self.calls.append({"request": request, "history": history, "request_id": request_id})

        if self.raises:
            raise self.raises

        return SimpleNamespace(
            response=self.reply,
            stop_reason=self.stop_reason,
            tool_outcomes=self.outcomes,
            request_id=request_id,
            replans=0,
        )


def outcome(
    tool: str,
    *,
    server: str = "notes",
    ok: bool = True,
    args: dict | None = None,
    code: str | None = None,
):
    return SimpleNamespace(
        tool_name=tool,
        spec_name=f"{server}__{tool}",
        server_id=server,
        success=ok,
        error_code=code,
        arguments=args or {},
        data={"ok": ok},
    )
