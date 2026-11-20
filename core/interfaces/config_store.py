"""Persistent storage contract for MCP server configuration."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from core.models.server_config import ServerRecord


@runtime_checkable
class ServerConfigStore(Protocol):
    """Durable storage for server registrations.

    Implementations must be safe for concurrent use and must never persist
    secrets (``ServerRegistration`` already guarantees references only).
    SQLite is the first implementation; PostgreSQL can implement the same
    contract without touching callers.
    """

    async def initialize(self) -> None:
        """Create or migrate storage. Safe to call repeatedly."""
        ...

    async def add(self, record: ServerRecord) -> ServerRecord:
        """Insert a record; raises ServerAlreadyRegisteredError on duplicates."""
        ...

    async def update(self, record: ServerRecord) -> ServerRecord:
        """Replace an existing record; raises ServerNotFoundError if absent."""
        ...

    async def get(self, server_id: str) -> ServerRecord | None:
        """Return a record or None."""
        ...

    async def list_records(self) -> list[ServerRecord]:
        """Return all records ordered by creation time."""
        ...

    async def delete(self, server_id: str) -> bool:
        """Delete a record; returns whether it existed."""
        ...

    async def close(self) -> None:
        """Release resources."""
        ...
