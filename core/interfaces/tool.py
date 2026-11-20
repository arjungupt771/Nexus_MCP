
from __future__ import annotations

from typing import Protocol, runtime_checkable

from core.models import ToolDefinition


@runtime_checkable
class ToolCatalog(Protocol):
    """Contract for storing and retrieving discovered MCP tools."""

    async def upsert_tools(
        self,
        server_id: str,
        tools: list[ToolDefinition],
    ) -> list[ToolDefinition]:
        """Add or replace the discovered tools belonging to a server."""
        ...

    async def get_tool(
        self,
        qualified_name: str,
    ) -> ToolDefinition | None:
        """Retrieve a tool using its server-qualified name."""
        ...

    async def list_tools(
        self,
        server_id: str | None = None,
    ) -> list[ToolDefinition]:
        """List all tools or restrict the results to one server."""
        ...

    async def remove_server_tools(
        self,
        server_id: str,
    ) -> int:
        """Remove all catalog entries belonging to a server."""
        ...
