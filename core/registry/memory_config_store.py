"""In-memory ServerConfigStore for tests and ephemeral deployments."""

from __future__ import annotations

import asyncio

from core.exceptions import ServerAlreadyRegisteredError, ServerNotFoundError
from core.models.server_config import ServerRecord


class InMemoryServerConfigStore:
    def __init__(self) -> None:
        self._records: dict[str, ServerRecord] = {}
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        return None

    async def add(self, record: ServerRecord) -> ServerRecord:
        async with self._lock:
            server_id = record.registration.server_id

            if server_id in self._records:
                raise ServerAlreadyRegisteredError(
                    f"Server '{server_id}' is already registered.",
                    details={"server_id": server_id},
                )

            self._records[server_id] = record.model_copy(deep=True)
            return record

    async def update(self, record: ServerRecord) -> ServerRecord:
        async with self._lock:
            server_id = record.registration.server_id

            if server_id not in self._records:
                raise ServerNotFoundError(
                    f"Server '{server_id}' is not registered.",
                    details={"server_id": server_id},
                )

            self._records[server_id] = record.model_copy(deep=True)
            return record

    async def get(self, server_id: str) -> ServerRecord | None:
        async with self._lock:
            record = self._records.get(server_id)
            return record.model_copy(deep=True) if record else None

    async def list_records(self) -> list[ServerRecord]:
        async with self._lock:
            ordered = sorted(self._records.values(), key=lambda r: r.created_at)
            return [r.model_copy(deep=True) for r in ordered]

    async def delete(self, server_id: str) -> bool:
        async with self._lock:
            return self._records.pop(server_id, None) is not None

    async def close(self) -> None:
        return None
