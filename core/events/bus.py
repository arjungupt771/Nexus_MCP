"""Event bus abstraction and in-memory implementation."""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Protocol, runtime_checkable

from core.events.models import RegistryEvent

logger = logging.getLogger(__name__)

EventHandler = Callable[[RegistryEvent], Awaitable[None] | None]


@runtime_checkable
class EventBus(Protocol):
    """Contract for publishing and observing registry events."""

    async def publish(self, event: RegistryEvent) -> RegistryEvent:
        """Publish an event; returns it with its sequence number."""
        ...

    def subscribe(self, handler: EventHandler) -> Callable[[], None]:
        """Register a handler; returns a function that unsubscribes it."""
        ...


class NullEventBus:
    """Discards events. Default when no observer is configured."""

    async def publish(self, event: RegistryEvent) -> RegistryEvent:
        return event

    def subscribe(self, handler: EventHandler) -> Callable[[], None]:
        return lambda: None


class InMemoryEventBus:
    """Async in-process bus with handler isolation and replay history.

    * A failing handler never affects publishers or other handlers.
    * ``stream()`` yields events through a bounded queue (oldest dropped on
      overflow) suitable for WebSocket / SSE fan-out.
    * ``recent()`` supports resuming after a reconnect.
    """

    def __init__(self, *, history_size: int = 500) -> None:
        if history_size < 1:
            raise ValueError("history_size must be at least 1")

        self._handlers: list[EventHandler] = []
        self._queues: list[asyncio.Queue[RegistryEvent]] = []
        self._history: deque[RegistryEvent] = deque(maxlen=history_size)
        self._sequence = 0

    async def publish(self, event: RegistryEvent) -> RegistryEvent:
        # No await between sequence assignment and history append, so
        # ordering is consistent under concurrent publishers.
        self._sequence += 1
        event = event.model_copy(update={"sequence": self._sequence})
        self._history.append(event)

        for queue in list(self._queues):
            if queue.full():
                queue.get_nowait()

            queue.put_nowait(event)

        for handler in list(self._handlers):
            try:
                outcome = handler(event)

                if asyncio.iscoroutine(outcome):
                    await outcome

            except Exception:
                logger.exception("Event handler failed for %s", event.type.value)

        return event

    def subscribe(self, handler: EventHandler) -> Callable[[], None]:
        self._handlers.append(handler)

        def unsubscribe() -> None:
            if handler in self._handlers:
                self._handlers.remove(handler)

        return unsubscribe

    @asynccontextmanager
    async def stream(self, *, max_queue: int = 256) -> AsyncIterator[AsyncIterator[RegistryEvent]]:
        """Open a live stream: ``async with bus.stream() as events``."""
        queue: asyncio.Queue[RegistryEvent] = asyncio.Queue(maxsize=max_queue)
        self._queues.append(queue)

        async def _iterate() -> AsyncIterator[RegistryEvent]:
            while True:
                yield await queue.get()

        try:
            yield _iterate()
        finally:
            self._queues.remove(queue)

    def recent(self, *, since: int = 0) -> list[RegistryEvent]:
        """Return retained events with ``sequence > since``."""
        return [e for e in self._history if e.sequence > since]
