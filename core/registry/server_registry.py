from __future__ import annotations

import asyncio

from core.exceptions import (
    ServerAlreadyRegisteredError,
    ServerNotFoundError,
)
from core.models import ServerDescriptor
from core.models.server import ServerStatus


class InMemoryServerRegistry:
    """Async-safe in-memory implementation of ServerRegistry."""

    def __init__(self) -> None:
        self._servers: dict[str, ServerDescriptor] = {}
        self._lock = asyncio.Lock()

    async def register(
        self,
        server: ServerDescriptor,
    ) -> ServerDescriptor:
        async with self._lock:
            server_id = server.server_id

            if server_id in self._servers:
                raise ServerAlreadyRegisteredError(
                    f"Server '{server_id}' is already registered.",
                    details={"server_id": server_id},
                )

            stored = server.model_copy(deep=True)
            self._servers[server_id] = stored

            return stored.model_copy(deep=True)

    async def unregister(
        self,
        server_id: str,
    ) -> ServerDescriptor:
        async with self._lock:
            server = self._servers.pop(server_id, None)

            if server is None:
                raise ServerNotFoundError(
                    f"Server '{server_id}' is not registered.",
                    details={"server_id": server_id},
                )

            return server.model_copy(deep=True)

    async def get(
        self,
        server_id: str,
    ) -> ServerDescriptor | None:
        async with self._lock:
            server = self._servers.get(server_id)

            if server is None:
                return None

            return server.model_copy(deep=True)

    async def list_servers(self) -> list[ServerDescriptor]:
        async with self._lock:
            return [server.model_copy(deep=True) for server in self._servers.values()]

    async def contains(self, server_id: str) -> bool:
        async with self._lock:
            return server_id in self._servers

    async def count(self) -> int:
        async with self._lock:
            return len(self._servers)

    async def clear(self) -> None:
        async with self._lock:
            self._servers.clear()

    async def update_status(
        self,
        server_id: str,
        status: ServerStatus,
    ) -> ServerDescriptor:
        async with self._lock:
            server = self._servers.get(server_id)
            if server is None:
                raise ServerNotFoundError(
                    f"Server '{server_id}' is not registered.",
                    details={"server_id": server_id},
                )

            updated = server.model_copy(update={"status": status})
            self._servers[server_id] = updated

            return updated.model_copy(deep=True)

    async def update_fields(
        self,
        server_id: str,
        **changes: object,
    ) -> ServerDescriptor:
        """Atomically apply validated changes to a descriptor."""
        async with self._lock:
            server = self._servers.get(server_id)

            if server is None:
                raise ServerNotFoundError(
                    f"Server '{server_id}' is not registered.",
                    details={"server_id": server_id},
                )

            if "server_id" in changes and changes["server_id"] != server_id:
                raise ValueError("server_id cannot be changed.")

            # Re-validate the merged data so invalid updates never land.
            merged = {**server.model_dump(), **changes}
            updated = ServerDescriptor.model_validate(merged)
            self._servers[server_id] = updated

            return updated.model_copy(deep=True)

    def __repr__(self) -> str:
        return "InMemoryServerRegistry()"
