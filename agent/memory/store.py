"""Storage contracts. Everything above this layer is backend-independent."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from agent.memory.models import (
    ConversationRecord,
    MemoryFilter,
    MemoryRecord,
    MemoryType,
    StoredMessage,
)


@runtime_checkable
class MemoryStore(Protocol):
    """Durable memory records. SQLite is the default; others can implement this."""

    async def initialize(self) -> None: ...

    async def add(self, record: MemoryRecord) -> MemoryRecord:
        """Insert. Raises MemoryConflictError if (user, type, content_hash) exists."""
        ...

    async def get(self, memory_id: str) -> MemoryRecord | None: ...

    async def get_many(self, memory_ids: list[str]) -> list[MemoryRecord]: ...

    async def update(self, record: MemoryRecord) -> MemoryRecord:
        """Replace by id. Raises MemoryNotFoundError / MemoryConflictError."""
        ...

    async def delete(self, memory_id: str) -> bool: ...

    async def delete_where(self, flt: MemoryFilter) -> list[str]:
        """Delete everything matching (expired included); returns deleted ids."""
        ...

    async def delete_expired(self, now: datetime, user_id: str | None = None) -> list[str]: ...

    async def find_by_hash(
        self, user_id: str, memory_type: MemoryType, content_hash: str
    ) -> MemoryRecord | None: ...

    async def list(
        self,
        flt: MemoryFilter,
        *,
        limit: int = 100,
        offset: int = 0,
        now: datetime | None = None,
    ) -> list[MemoryRecord]:
        """Newest first (created_at desc, id) so paging is deterministic."""
        ...

    async def search_keywords(
        self,
        terms: list[str],
        flt: MemoryFilter,
        *,
        limit: int,
        now: datetime | None = None,
    ) -> list[tuple[MemoryRecord, int]]:
        """Records matching any term with the number of distinct terms matched."""
        ...

    async def count(self, flt: MemoryFilter, *, now: datetime | None = None) -> int: ...

    async def list_user_ids(self) -> list[str]: ...

    async def touch(self, memory_ids: list[str], when: datetime) -> None: ...

    async def close(self) -> None: ...


@runtime_checkable
class ConversationStore(Protocol):
    async def initialize(self) -> None: ...

    async def create(self, conversation: ConversationRecord) -> ConversationRecord: ...

    async def get(self, conversation_id: str) -> ConversationRecord | None: ...

    async def list(
        self,
        user_id: str,
        *,
        include_archived: bool = False,
        limit: int = 50,
        offset: int = 0,
    ) -> list[ConversationRecord]: ...

    async def update(self, conversation: ConversationRecord) -> ConversationRecord: ...

    async def delete(self, conversation_id: str) -> bool: ...

    async def append_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        metadata: dict[str, str] | None = None,
        created_at: datetime | None = None,
    ) -> StoredMessage:
        """Atomically assign the next ``seq`` and bump the conversation counters."""
        ...

    async def messages(
        self,
        conversation_id: str,
        *,
        after_seq: int = 0,
        limit: int | None = None,
        newest_first: bool = False,
    ) -> list[StoredMessage]: ...

    async def delete_messages_through(self, conversation_id: str, seq: int) -> int: ...

    async def delete_messages(self, conversation_id: str) -> int: ...


@runtime_checkable
class VectorIndex(Protocol):
    """Similarity search over memory embeddings (SQLite default, Qdrant optional)."""

    async def upsert(
        self, memory_id: str, vector: list[float], *, user_id: str, memory_type: MemoryType
    ) -> None: ...

    async def delete(self, memory_ids: list[str]) -> None: ...

    async def search(
        self,
        vector: list[float],
        *,
        user_id: str,
        types: list[MemoryType] | None = None,
        limit: int = 20,
    ) -> list[tuple[str, float]]:
        """(memory_id, cosine similarity) best first."""
        ...

    async def close(self) -> None: ...
