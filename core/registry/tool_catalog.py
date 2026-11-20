from __future__ import annotations

import asyncio

from core.models import ToolDefinition


class InMemoryToolCatalog:
    """Async-safe in-memory implementation of ToolCatalog."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}
        self._lock = asyncio.Lock()

    async def upsert_tools(
        self,
        server_id: str,
        tools: list[ToolDefinition],
    ) -> list[ToolDefinition]:
        async with self._lock:
            incoming: dict[str, ToolDefinition] = {}

            for tool in tools:
                if tool.server_id != server_id:
                    raise ValueError(
                        f"Tool '{tool.name}' belongs to server "
                        f"'{tool.server_id}', not '{server_id}'."
                    )

                qualified_name = tool.qualified_name

                if qualified_name in incoming:
                    raise ValueError(f"Duplicate tool '{qualified_name}' in the same update.")

                incoming[qualified_name] = tool.model_copy(deep=True)

            # Replace this server's complete tool snapshot.
            # This removes tools that disappeared during rediscovery.
            updated = {
                key: value for key, value in self._tools.items() if value.server_id != server_id
            }

            updated.update(incoming)
            self._tools = updated

            return [tool.model_copy(deep=True) for tool in incoming.values()]

    async def get_tool(
        self,
        qualified_name: str,
    ) -> ToolDefinition | None:
        async with self._lock:
            tool = self._tools.get(qualified_name)

            if tool is None:
                return None

            return tool.model_copy(deep=True)

    async def list_tools(
        self,
        server_id: str | None = None,
    ) -> list[ToolDefinition]:
        async with self._lock:
            tools = [
                tool
                for tool in self._tools.values()
                if server_id is None or tool.server_id == server_id
            ]

            return [tool.model_copy(deep=True) for tool in tools]

    async def remove_server_tools(
        self,
        server_id: str,
    ) -> int:
        async with self._lock:
            existing_count = len(self._tools)

            self._tools = {
                key: tool for key, tool in self._tools.items() if tool.server_id != server_id
            }

            return existing_count - len(self._tools)

    async def count(self) -> int:
        async with self._lock:
            return len(self._tools)

    async def clear(self) -> None:
        async with self._lock:
            self._tools.clear()

    def __repr__(self) -> str:
        return "InMemoryToolCatalog()"
