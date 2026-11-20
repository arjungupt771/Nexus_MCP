"""Short-term (working) memory: bounded, in-process, never persisted (8.1).

Holds the recent turns, the current execution context and *intermediate tool
results* of a conversation. Long-lived facts are promoted to persistent memory
only by the extraction pipeline.
"""

from __future__ import annotations

from collections import OrderedDict, deque
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict

from agent.memory.models import utcnow


class ToolResultEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    server_id: str | None = None
    tool_name: str
    success: bool
    preview: str = ""
    at: datetime


class ShortTermMemory:
    def __init__(
        self, conversation_id: str, *, max_messages: int = 20, max_tool_results: int = 20
    ) -> None:
        self.conversation_id = conversation_id
        self._messages: deque[tuple[str, str]] = deque(maxlen=max_messages)
        self._tool_results: deque[ToolResultEntry] = deque(maxlen=max_tool_results)
        self.execution_context: dict[str, Any] = {}

    def add_message(self, role: str, content: str) -> None:
        self._messages.append((role, content))

    def recent_messages(self, n: int | None = None) -> list[tuple[str, str]]:
        items = list(self._messages)
        return items[-n:] if n else items

    def add_tool_result(
        self,
        tool_name: str,
        *,
        server_id: str | None,
        success: bool,
        preview: str = "",
        max_preview: int = 300,
    ) -> None:
        self._tool_results.append(
            ToolResultEntry(
                server_id=server_id,
                tool_name=tool_name,
                success=success,
                preview=preview[:max_preview],
                at=utcnow(),
            )
        )

    @property
    def tool_results(self) -> list[ToolResultEntry]:
        return list(self._tool_results)

    def set_context(self, **values: Any) -> None:
        self.execution_context.update(values)

    def clear_execution(self) -> None:
        """Reset per-request state (context and tool results); keep recent messages."""
        self.execution_context.clear()
        self._tool_results.clear()

    def clear(self) -> None:
        self._messages.clear()
        self.clear_execution()


class ShortTermMemoryManager:
    """LRU registry so idle conversations cannot grow memory without bound."""

    def __init__(
        self, *, max_conversations: int = 200, max_messages: int = 20, max_tool_results: int = 20
    ) -> None:
        self._items: OrderedDict[str, ShortTermMemory] = OrderedDict()
        self._max_conversations = max_conversations
        self._max_messages = max_messages
        self._max_tool_results = max_tool_results

    def get(self, conversation_id: str) -> ShortTermMemory:
        memory = self._items.get(conversation_id)

        if memory is None:
            memory = ShortTermMemory(
                conversation_id,
                max_messages=self._max_messages,
                max_tool_results=self._max_tool_results,
            )
            self._items[conversation_id] = memory

        self._items.move_to_end(conversation_id)

        while len(self._items) > self._max_conversations:
            self._items.popitem(last=False)

        return memory

    def drop(self, conversation_id: str) -> bool:
        return self._items.pop(conversation_id, None) is not None

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, conversation_id: str) -> bool:
        return conversation_id in self._items
