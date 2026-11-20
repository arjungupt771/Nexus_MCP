
"""Transport abstraction for MCP communication."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class MCPTransport(Protocol):
    """Contract for an active connection to an MCP server."""

    async def connect(self) -> None:
        """Establish the transport connection."""
        ...

    async def disconnect(self) -> None:
        """Close the transport connection and release resources."""
        ...

    async def list_tools(
        self,
        *,
        cursor: str | None = None,
    ) -> Any:
        """List MCP tools, optionally continuing from a cursor."""
        ...

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        *,
        read_timeout_seconds: float | None = None,
    ) -> Any:
        """Invoke an MCP tool and return its raw SDK response."""
        ...
