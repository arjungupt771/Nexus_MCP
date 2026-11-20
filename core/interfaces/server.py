
from __future__ import annotations

from typing import Protocol, runtime_checkable

from core.interfaces.transport import MCPTransport
from core.models import ServerDescriptor
from core.models.server import ServerStatus


@runtime_checkable
class MCPServerConnector(Protocol):
    """Creates transport connections for configured MCP servers."""

    async def connect(
        self,
        server: ServerDescriptor,
    ) -> MCPTransport:
        """Create and establish a connection to an MCP server."""
        ...


@runtime_checkable
class ServerRegistry(Protocol):
    """Contract for managing registered MCP server descriptors."""

    async def register(
        self,
        server: ServerDescriptor,
    ) -> ServerDescriptor:
        """Register a new server."""
        ...

    async def unregister(
        self,
        server_id: str,
    ) -> ServerDescriptor:
        """Remove a registered server."""
        ...

    async def get(
        self,
        server_id: str,
    ) -> ServerDescriptor | None:
        """Retrieve a server by its unique identifier."""
        ...

    async def list_servers(self) -> list[ServerDescriptor]:
        """Return all registered servers."""
        ...

    async def update_status(
            self,
            server_id: str,
            status: ServerStatus,
            ) -> ServerDescriptor:
         """Update and return a registered server's connection status."""
         ...


@runtime_checkable
class MutableServerRegistry(ServerRegistry, Protocol):
    """Registry that also supports atomic partial updates (Phase 5)."""

    async def update_fields(
        self,
        server_id: str,
        **changes: object,
    ) -> ServerDescriptor:
        """Atomically apply validated field changes to a descriptor."""
        ...
