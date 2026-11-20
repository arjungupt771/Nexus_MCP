"""Vector index adapters: in-memory (tests/ephemeral) and Qdrant (optional)."""

from __future__ import annotations

import uuid
from typing import Any

from agent.memory.embeddings import cosine
from agent.memory.errors import MemoryStorageError
from agent.memory.models import MemoryType


class InMemoryVectorIndex:
    """Non-persistent exact index. Handy for tests and throwaway sessions."""

    def __init__(self) -> None:
        self._items: dict[str, tuple[list[float], str, MemoryType]] = {}

    async def upsert(
        self, memory_id: str, vector: list[float], *, user_id: str, memory_type: MemoryType
    ) -> None:
        self._items[memory_id] = (list(vector), user_id, memory_type)

    async def delete(self, memory_ids: list[str]) -> None:
        for memory_id in memory_ids:
            self._items.pop(memory_id, None)

    async def search(
        self,
        vector: list[float],
        *,
        user_id: str,
        types: list[MemoryType] | None = None,
        limit: int = 20,
    ) -> list[tuple[str, float]]:
        scored = [
            (memory_id, cosine(vector, stored))
            for memory_id, (stored, owner, kind) in self._items.items()
            if owner == user_id and (not types or kind in types)
        ]
        return sorted(scored, key=lambda item: (-item[1], item[0]))[:limit]

    async def close(self) -> None:
        return None

    def __len__(self) -> int:
        return len(self._items)


class QdrantVectorIndex:
    """Optional Qdrant adapter (``pip install qdrant-client``).

    Pass an ``AsyncQdrantClient``: ``AsyncQdrantClient(url=...)`` for a server,
    or ``AsyncQdrantClient(":memory:")`` for the embedded local mode. Memory ids
    are stored as UUIDs; the original id is kept in the payload.
    """

    def __init__(
        self, client: Any, *, collection: str = "nexus_memories", dimensions: int = 256
    ) -> None:
        try:
            from qdrant_client import models
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise MemoryStorageError(
                "qdrant-client is not installed; install it with 'pip install qdrant-client'."
            ) from exc

        self._models = models
        self._client = client
        self._collection = collection
        self._dimensions = dimensions
        self._ready = False

    @staticmethod
    def _point_id(memory_id: str) -> str:
        return str(uuid.UUID(memory_id))

    async def _ensure(self) -> None:
        if self._ready:
            return

        try:
            if not await self._client.collection_exists(self._collection):
                await self._client.create_collection(
                    self._collection,
                    vectors_config=self._models.VectorParams(
                        size=self._dimensions, distance=self._models.Distance.COSINE
                    ),
                )

            self._ready = True
        except Exception as exc:
            raise MemoryStorageError(
                "Qdrant is unavailable.", details={"cause_type": type(exc).__name__}
            ) from exc

    async def upsert(
        self, memory_id: str, vector: list[float], *, user_id: str, memory_type: MemoryType
    ) -> None:
        await self._ensure()

        try:
            await self._client.upsert(
                self._collection,
                points=[
                    self._models.PointStruct(
                        id=self._point_id(memory_id),
                        vector=vector,
                        payload={
                            "memory_id": memory_id,
                            "user_id": user_id,
                            "type": memory_type.value,
                        },
                    )
                ],
            )
        except Exception as exc:
            raise MemoryStorageError(
                "Qdrant upsert failed.", details={"cause_type": type(exc).__name__}
            ) from exc

    async def delete(self, memory_ids: list[str]) -> None:
        if not memory_ids:
            return

        await self._ensure()

        try:
            await self._client.delete(
                self._collection,
                points_selector=self._models.PointIdsList(
                    points=[self._point_id(i) for i in memory_ids]
                ),
            )
        except Exception as exc:
            raise MemoryStorageError(
                "Qdrant delete failed.", details={"cause_type": type(exc).__name__}
            ) from exc

    async def search(
        self,
        vector: list[float],
        *,
        user_id: str,
        types: list[MemoryType] | None = None,
        limit: int = 20,
    ) -> list[tuple[str, float]]:
        await self._ensure()
        must = [
            self._models.FieldCondition(key="user_id", match=self._models.MatchValue(value=user_id))
        ]

        if types:
            must.append(
                self._models.FieldCondition(
                    key="type", match=self._models.MatchAny(any=[t.value for t in types])
                )
            )

        try:
            response = await self._client.query_points(
                self._collection,
                query=vector,
                query_filter=self._models.Filter(must=must),
                limit=limit,
                with_payload=True,
            )
        except Exception as exc:
            raise MemoryStorageError(
                "Qdrant search failed.", details={"cause_type": type(exc).__name__}
            ) from exc

        return [(p.payload["memory_id"], float(p.score)) for p in response.points if p.payload]

    async def close(self) -> None:
        try:
            await self._client.close()
        except Exception:  # pragma: no cover - best effort
            pass
