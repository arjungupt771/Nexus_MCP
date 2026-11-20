"""SQLite persistence for memories, vectors and conversations.

Uses SQLAlchemy Core (already a project dependency) with blocking calls moved
to worker threads, the same approach as the Phase 5 registry store. Pointing
the URL at PostgreSQL later needs no change to callers.
"""

from __future__ import annotations

import array
import asyncio
import heapq
import json
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, TypeVar

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
    and_,
    create_engine,
    delete,
    func,
    insert,
    or_,
    select,
    update,
)
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.pool import StaticPool

from agent.memory.embeddings import cosine
from agent.memory.errors import (
    ConversationNotFoundError,
    MemoryConflictError,
    MemoryNotFoundError,
    MemoryStorageError,
    MemorySystemError,
)
from agent.memory.models import (
    ConversationRecord,
    MemoryFilter,
    MemoryKind,
    MemoryRecord,
    MemorySource,
    MemoryType,
    StoredMessage,
    utcnow,
)

logger = logging.getLogger(__name__)
SCHEMA_VERSION = 1
T = TypeVar("T")

metadata = MetaData()

meta = Table(
    "nexus_memory_meta",
    metadata,
    Column("key", String(50), primary_key=True),
    Column("value", String(100), nullable=False),
)
memories = Table(
    "nexus_memories",
    metadata,
    Column("id", String(40), primary_key=True),
    Column("user_id", String(100), nullable=False),
    Column("type", String(20), nullable=False),
    Column("kind", String(20), nullable=False),
    Column("content", Text, nullable=False),
    Column("source", String(20), nullable=False),
    Column("importance", Float, nullable=False),
    Column("confidence", Float, nullable=False),
    Column("conversation_id", String(40)),
    Column("metadata", Text, nullable=False, default="{}"),
    Column("keywords", Text, nullable=False, default="[]"),
    Column("content_hash", String(64), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    Column("last_accessed_at", DateTime(timezone=True)),
    Column("access_count", Integer, nullable=False, default=0),
    Column("expires_at", DateTime(timezone=True)),
    Column("pinned", Boolean, nullable=False, default=False),
    UniqueConstraint("user_id", "type", "content_hash", name="uq_memory_dedupe"),
    Index("ix_memory_user_type", "user_id", "type"),
    Index("ix_memory_conversation", "conversation_id"),
)
memory_keywords = Table(
    "nexus_memory_keywords",
    metadata,
    Column("memory_id", String(40), ForeignKey("nexus_memories.id"), nullable=False),
    Column("term", String(64), nullable=False),
    Index("ix_keyword_term", "term"),
    Index("ix_keyword_memory", "memory_id"),
)
memory_vectors = Table(
    "nexus_memory_vectors",
    metadata,
    Column("memory_id", String(40), primary_key=True),
    Column("dim", Integer, nullable=False),
    Column("vector", LargeBinary, nullable=False),
)
conversations = Table(
    "nexus_conversations",
    metadata,
    Column("id", String(40), primary_key=True),
    Column("user_id", String(100), nullable=False),
    Column("title", String(300), nullable=False, default=""),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    Column("archived_at", DateTime(timezone=True)),
    Column("summary", Text, nullable=False, default=""),
    Column("summarized_through", Integer, nullable=False, default=0),
    Column("message_count", Integer, nullable=False, default=0),
    Column("memory_enabled", Boolean, nullable=False, default=True),
    Column("metadata", Text, nullable=False, default="{}"),
    Index("ix_conversation_user", "user_id"),
)
messages = Table(
    "nexus_messages",
    metadata,
    Column("conversation_id", String(40), nullable=False),
    Column("seq", Integer, nullable=False),
    Column("role", String(20), nullable=False),
    Column("content", Text, nullable=False),
    Column("metadata", Text, nullable=False, default="{}"),
    Column("created_at", DateTime(timezone=True), nullable=False),
    UniqueConstraint("conversation_id", "seq", name="uq_message_seq"),
    Index("ix_message_conversation", "conversation_id", "seq"),
)


def _aware(value: datetime | None) -> datetime | None:
    return value if value is None or value.tzinfo else value.replace(tzinfo=UTC)


def _pack(vector: list[float]) -> bytes:
    return array.array("f", vector).tobytes()


def _unpack(blob: bytes) -> list[float]:
    values = array.array("f")
    values.frombytes(blob)
    return list(values)


class SQLiteBackend:
    """Shared engine; the three stores below all use one instance."""

    def __init__(self, url: str = "sqlite:///./nexus_memory.db") -> None:
        self._url = url
        self._engine: Engine | None = None
        self._lock = asyncio.Lock()
        # An in-memory SQLite database is one shared connection: serialize access.
        self._serial = asyncio.Lock() if url in {"sqlite://", "sqlite:///:memory:"} else None

    @classmethod
    def in_memory(cls) -> SQLiteBackend:
        return cls("sqlite://")

    def _create_engine(self) -> Engine:
        if self._url in {"sqlite://", "sqlite:///:memory:"}:
            return create_engine(
                "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
            )

        return create_engine(self._url, connect_args={"check_same_thread": False})

    async def engine(self) -> Engine:
        async with self._lock:
            if self._engine is None:
                engine = self._create_engine()

                def _setup() -> None:
                    metadata.create_all(engine)

                    with engine.begin() as conn:
                        if (
                            conn.execute(
                                select(meta.c.value).where(meta.c.key == "version")
                            ).scalar()
                            is None
                        ):
                            conn.execute(
                                insert(meta).values(key="version", value=str(SCHEMA_VERSION))
                            )

                try:
                    await asyncio.to_thread(_setup)
                except SQLAlchemyError as exc:
                    engine.dispose()
                    raise MemoryStorageError(
                        "Memory storage could not be initialized.",
                        details={"cause_type": type(exc).__name__},
                    ) from exc

                self._engine = engine

            return self._engine

    async def run(self, fn: Callable[[Engine], T]) -> T:
        engine = await self.engine()

        try:
            if self._serial is not None:
                async with self._serial:
                    return await asyncio.to_thread(fn, engine)

            return await asyncio.to_thread(fn, engine)
        except MemorySystemError:
            raise
        except SQLAlchemyError as exc:
            # Type only: SQL parameters could contain memory content.
            logger.error("Memory storage failure: %s", type(exc).__name__)
            raise MemoryStorageError(
                "Memory storage failed.", details={"cause_type": type(exc).__name__}
            ) from exc

    async def close(self) -> None:
        async with self._lock:
            engine, self._engine = self._engine, None

        if engine is not None:
            await asyncio.to_thread(engine.dispose)


# ------------------------------------------------------------------ memories


def _conditions(flt: MemoryFilter, now: datetime) -> list[Any]:
    c = memories.c
    clauses: list[Any] = [c.user_id == flt.user_id]

    if flt.types:
        clauses.append(c.type.in_([t.value for t in flt.types]))

    if flt.kinds:
        clauses.append(c.kind.in_([k.value for k in flt.kinds]))

    if flt.sources:
        clauses.append(c.source.in_([s.value for s in flt.sources]))

    if flt.conversation_id is not None:
        clauses.append(c.conversation_id == flt.conversation_id)

    if flt.min_importance is not None:
        clauses.append(c.importance >= flt.min_importance)

    if flt.max_importance is not None:
        clauses.append(c.importance < flt.max_importance)

    if flt.created_after is not None:
        clauses.append(c.created_at >= flt.created_after)

    if flt.created_before is not None:
        clauses.append(c.created_at <= flt.created_before)

    if flt.pinned is not None:
        clauses.append(c.pinned == flt.pinned)

    if not flt.include_expired:
        clauses.append(or_(c.expires_at.is_(None), c.expires_at > now))

    return clauses


def _record(row: Any) -> MemoryRecord:
    return MemoryRecord(
        id=row.id,
        user_id=row.user_id,
        type=MemoryType(row.type),
        kind=MemoryKind(row.kind),
        content=row.content,
        source=MemorySource(row.source),
        importance=row.importance,
        confidence=row.confidence,
        conversation_id=row.conversation_id,
        metadata=json.loads(row.metadata),
        keywords=json.loads(row.keywords),
        content_hash=row.content_hash,
        created_at=_aware(row.created_at),
        updated_at=_aware(row.updated_at),
        last_accessed_at=_aware(row.last_accessed_at),
        access_count=row.access_count,
        expires_at=_aware(row.expires_at),
        pinned=bool(row.pinned),
    )


def _values(record: MemoryRecord) -> dict[str, Any]:
    return {
        "id": record.id,
        "user_id": record.user_id,
        "type": record.type.value,
        "kind": record.kind.value,
        "content": record.content,
        "source": record.source.value,
        "importance": record.importance,
        "confidence": record.confidence,
        "conversation_id": record.conversation_id,
        "metadata": json.dumps(record.metadata),
        "keywords": json.dumps(record.keywords),
        "content_hash": record.content_hash,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
        "last_accessed_at": record.last_accessed_at,
        "access_count": record.access_count,
        "expires_at": record.expires_at,
        "pinned": record.pinned,
    }


def _metadata_match(record: MemoryRecord, wanted: dict[str, str]) -> bool:
    return all(record.metadata.get(k) == v for k, v in wanted.items())


class SQLiteMemoryStore:
    def __init__(self, backend: SQLiteBackend) -> None:
        self._db = backend

    async def initialize(self) -> None:
        await self._db.engine()

    @staticmethod
    def _write_keywords(conn: Any, record: MemoryRecord) -> None:
        conn.execute(delete(memory_keywords).where(memory_keywords.c.memory_id == record.id))

        if record.keywords:
            conn.execute(
                insert(memory_keywords),
                [{"memory_id": record.id, "term": term} for term in record.keywords],
            )

    async def add(self, record: MemoryRecord) -> MemoryRecord:
        def _op(engine: Engine) -> None:
            try:
                with engine.begin() as conn:
                    conn.execute(insert(memories).values(**_values(record)))
                    self._write_keywords(conn, record)
            except IntegrityError as exc:
                raise MemoryConflictError(
                    "An identical memory already exists.", details={"memory_id": record.id}
                ) from exc

        await self._db.run(_op)
        return record

    async def get(self, memory_id: str) -> MemoryRecord | None:
        def _op(engine: Engine) -> MemoryRecord | None:
            with engine.connect() as conn:
                row = conn.execute(select(memories).where(memories.c.id == memory_id)).first()

            return _record(row) if row else None

        return await self._db.run(_op)

    async def get_many(self, memory_ids: list[str]) -> list[MemoryRecord]:
        if not memory_ids:
            return []

        def _op(engine: Engine) -> list[MemoryRecord]:
            with engine.connect() as conn:
                rows = conn.execute(select(memories).where(memories.c.id.in_(memory_ids))).all()

            return [_record(r) for r in rows]

        return await self._db.run(_op)

    async def update(self, record: MemoryRecord) -> MemoryRecord:
        values = _values(record)
        values.pop("id")
        values.pop("created_at")

        def _op(engine: Engine) -> None:
            try:
                with engine.begin() as conn:
                    result = conn.execute(
                        update(memories).where(memories.c.id == record.id).values(**values)
                    )

                    if result.rowcount == 0:
                        raise MemoryNotFoundError(
                            "Memory not found.", details={"memory_id": record.id}
                        )

                    self._write_keywords(conn, record)
            except IntegrityError as exc:
                raise MemoryConflictError(
                    "Another memory already has this content.", details={"memory_id": record.id}
                ) from exc

        await self._db.run(_op)
        return record

    @staticmethod
    def _delete_ids(conn: Any, ids: list[str]) -> None:
        for start in range(0, len(ids), 500):
            chunk = ids[start : start + 500]
            conn.execute(delete(memory_keywords).where(memory_keywords.c.memory_id.in_(chunk)))
            conn.execute(delete(memory_vectors).where(memory_vectors.c.memory_id.in_(chunk)))
            conn.execute(delete(memories).where(memories.c.id.in_(chunk)))

    async def delete(self, memory_id: str) -> bool:
        def _op(engine: Engine) -> bool:
            with engine.begin() as conn:
                exists = conn.execute(
                    select(memories.c.id).where(memories.c.id == memory_id)
                ).first()

                if exists:
                    self._delete_ids(conn, [memory_id])

            return exists is not None

        return await self._db.run(_op)

    async def delete_where(self, flt: MemoryFilter) -> list[str]:
        flt = flt.model_copy(update={"include_expired": True})

        def _op(engine: Engine) -> list[str]:
            with engine.begin() as conn:
                rows = conn.execute(select(memories).where(and_(*_conditions(flt, utcnow())))).all()
                ids = [r.id for r in rows if _metadata_match(_record(r), flt.metadata)]
                self._delete_ids(conn, ids)

            return ids

        return await self._db.run(_op)

    async def delete_expired(self, now: datetime, user_id: str | None = None) -> list[str]:
        def _op(engine: Engine) -> list[str]:
            clauses = [memories.c.expires_at.is_not(None), memories.c.expires_at <= now]

            if user_id is not None:
                clauses.append(memories.c.user_id == user_id)

            with engine.begin() as conn:
                ids = [
                    r.id for r in conn.execute(select(memories.c.id).where(and_(*clauses))).all()
                ]
                self._delete_ids(conn, ids)

            return ids

        return await self._db.run(_op)

    async def find_by_hash(
        self, user_id: str, memory_type: MemoryType, content_hash: str
    ) -> MemoryRecord | None:
        def _op(engine: Engine) -> MemoryRecord | None:
            with engine.connect() as conn:
                row = conn.execute(
                    select(memories).where(
                        memories.c.user_id == user_id,
                        memories.c.type == memory_type.value,
                        memories.c.content_hash == content_hash,
                    )
                ).first()

            return _record(row) if row else None

        return await self._db.run(_op)

    async def list(
        self,
        flt: MemoryFilter,
        *,
        limit: int = 100,
        offset: int = 0,
        now: datetime | None = None,
    ) -> list[MemoryRecord]:
        moment = now or utcnow()

        def _op(engine: Engine) -> list[MemoryRecord]:
            query = (
                select(memories)
                .where(and_(*_conditions(flt, moment)))
                .order_by(memories.c.created_at.desc(), memories.c.id)
            )

            if not flt.metadata:
                query = query.limit(limit).offset(offset)

            with engine.connect() as conn:
                records = [_record(r) for r in conn.execute(query).all()]

            if flt.metadata:
                records = [r for r in records if _metadata_match(r, flt.metadata)]
                return records[offset : offset + limit]

            return records

        return await self._db.run(_op)

    async def search_keywords(
        self,
        terms: list[str],
        flt: MemoryFilter,
        *,
        limit: int,
        now: datetime | None = None,
    ) -> list[tuple[MemoryRecord, int]]:
        if not terms:
            return []

        moment = now or utcnow()

        def _op(engine: Engine) -> list[tuple[MemoryRecord, int]]:
            matched = func.count(func.distinct(memory_keywords.c.term)).label("matched")
            query = (
                select(memories, matched)
                .join(memory_keywords, memory_keywords.c.memory_id == memories.c.id)
                .where(and_(memory_keywords.c.term.in_(terms), *_conditions(flt, moment)))
                .group_by(memories.c.id)
                .order_by(matched.desc(), memories.c.importance.desc(), memories.c.id)
            )

            if not flt.metadata:
                query = query.limit(limit)

            with engine.connect() as conn:
                rows = conn.execute(query).all()

            hits = [(_record(r), int(r.matched)) for r in rows]
            hits = [h for h in hits if _metadata_match(h[0], flt.metadata)]
            return hits[:limit]

        return await self._db.run(_op)

    async def count(self, flt: MemoryFilter, *, now: datetime | None = None) -> int:
        moment = now or utcnow()

        def _op(engine: Engine) -> int:
            with engine.connect() as conn:
                if flt.metadata:
                    rows = conn.execute(
                        select(memories).where(and_(*_conditions(flt, moment)))
                    ).all()
                    return sum(_metadata_match(_record(r), flt.metadata) for r in rows)

                return int(
                    conn.execute(
                        select(func.count())
                        .select_from(memories)
                        .where(and_(*_conditions(flt, moment)))
                    ).scalar()
                    or 0
                )

        return await self._db.run(_op)

    async def list_user_ids(self) -> list[str]:
        def _op(engine: Engine) -> list[str]:
            with engine.connect() as conn:
                return sorted(
                    r[0] for r in conn.execute(select(memories.c.user_id).distinct()).all()
                )

        return await self._db.run(_op)

    async def touch(self, memory_ids: list[str], when: datetime) -> None:
        if not memory_ids:
            return

        def _op(engine: Engine) -> None:
            with engine.begin() as conn:
                conn.execute(
                    update(memories)
                    .where(memories.c.id.in_(memory_ids))
                    .values(last_accessed_at=when, access_count=memories.c.access_count + 1)
                )

        await self._db.run(_op)

    async def close(self) -> None:
        await self._db.close()


# ------------------------------------------------------------------ vectors


class SQLiteVectorIndex:
    """Persistent brute-force cosine index living in the same database.

    Exact and dependency-free; comfortable for thousands of memories per user.
    Use ``QdrantVectorIndex`` for larger corpora.
    """

    def __init__(self, backend: SQLiteBackend) -> None:
        self._db = backend

    async def upsert(
        self, memory_id: str, vector: list[float], *, user_id: str, memory_type: MemoryType
    ) -> None:
        def _op(engine: Engine) -> None:
            with engine.begin() as conn:
                conn.execute(delete(memory_vectors).where(memory_vectors.c.memory_id == memory_id))
                conn.execute(
                    insert(memory_vectors).values(
                        memory_id=memory_id, dim=len(vector), vector=_pack(vector)
                    )
                )

        await self._db.run(_op)

    async def delete(self, memory_ids: list[str]) -> None:
        if not memory_ids:
            return

        def _op(engine: Engine) -> None:
            with engine.begin() as conn:
                conn.execute(
                    delete(memory_vectors).where(memory_vectors.c.memory_id.in_(memory_ids))
                )

        await self._db.run(_op)

    async def search(
        self,
        vector: list[float],
        *,
        user_id: str,
        types: list[MemoryType] | None = None,
        limit: int = 20,
    ) -> list[tuple[str, float]]:
        def _op(engine: Engine) -> list[tuple[str, float]]:
            query = (
                select(memory_vectors.c.memory_id, memory_vectors.c.dim, memory_vectors.c.vector)
                .join(memories, memories.c.id == memory_vectors.c.memory_id)
                .where(memories.c.user_id == user_id)
            )

            if types:
                query = query.where(memories.c.type.in_([t.value for t in types]))

            with engine.connect() as conn:
                rows = conn.execute(query).all()

            scored = (
                (cosine(vector, _unpack(r.vector)), r.memory_id)
                for r in rows
                if r.dim == len(vector)
            )
            best = heapq.nsmallest(limit, ((-s, mid) for s, mid in scored))
            return [(mid, -neg) for neg, mid in best]

        return await self._db.run(_op)

    async def close(self) -> None:
        return None


# ------------------------------------------------------------------ conversations


def _conversation(row: Any) -> ConversationRecord:
    return ConversationRecord(
        id=row.id,
        user_id=row.user_id,
        title=row.title,
        created_at=_aware(row.created_at),
        updated_at=_aware(row.updated_at),
        archived_at=_aware(row.archived_at),
        summary=row.summary,
        summarized_through=row.summarized_through,
        message_count=row.message_count,
        memory_enabled=bool(row.memory_enabled),
        metadata=json.loads(row.metadata),
    )


def _message(row: Any) -> StoredMessage:
    return StoredMessage(
        conversation_id=row.conversation_id,
        seq=row.seq,
        role=row.role,
        content=row.content,
        metadata=json.loads(row.metadata),
        created_at=_aware(row.created_at),
    )


class SQLiteConversationStore:
    def __init__(self, backend: SQLiteBackend) -> None:
        self._db = backend

    async def initialize(self) -> None:
        await self._db.engine()

    @staticmethod
    def _conv_values(c: ConversationRecord) -> dict[str, Any]:
        return {
            "id": c.id,
            "user_id": c.user_id,
            "title": c.title,
            "created_at": c.created_at,
            "updated_at": c.updated_at,
            "archived_at": c.archived_at,
            "summary": c.summary,
            "summarized_through": c.summarized_through,
            "message_count": c.message_count,
            "memory_enabled": c.memory_enabled,
            "metadata": json.dumps(c.metadata),
        }

    async def create(self, conversation: ConversationRecord) -> ConversationRecord:
        def _op(engine: Engine) -> None:
            with engine.begin() as conn:
                conn.execute(insert(conversations).values(**self._conv_values(conversation)))

        await self._db.run(_op)
        return conversation

    async def get(self, conversation_id: str) -> ConversationRecord | None:
        def _op(engine: Engine) -> ConversationRecord | None:
            with engine.connect() as conn:
                row = conn.execute(
                    select(conversations).where(conversations.c.id == conversation_id)
                ).first()

            return _conversation(row) if row else None

        return await self._db.run(_op)

    async def list(
        self, user_id: str, *, include_archived: bool = False, limit: int = 50, offset: int = 0
    ) -> list[ConversationRecord]:
        def _op(engine: Engine) -> list[ConversationRecord]:
            query = select(conversations).where(conversations.c.user_id == user_id)

            if not include_archived:
                query = query.where(conversations.c.archived_at.is_(None))

            query = query.order_by(conversations.c.updated_at.desc(), conversations.c.id)

            with engine.connect() as conn:
                return [
                    _conversation(r) for r in conn.execute(query.limit(limit).offset(offset)).all()
                ]

        return await self._db.run(_op)

    async def update(self, conversation: ConversationRecord) -> ConversationRecord:
        values = self._conv_values(conversation)
        values.pop("id")
        values.pop("created_at")

        def _op(engine: Engine) -> None:
            with engine.begin() as conn:
                result = conn.execute(
                    update(conversations)
                    .where(conversations.c.id == conversation.id)
                    .values(**values)
                )

                if result.rowcount == 0:
                    raise ConversationNotFoundError(
                        "Conversation not found.", details={"conversation_id": conversation.id}
                    )

        await self._db.run(_op)
        return conversation

    async def delete(self, conversation_id: str) -> bool:
        def _op(engine: Engine) -> bool:
            with engine.begin() as conn:
                conn.execute(delete(messages).where(messages.c.conversation_id == conversation_id))
                result = conn.execute(
                    delete(conversations).where(conversations.c.id == conversation_id)
                )
                return result.rowcount > 0

        return await self._db.run(_op)

    async def append_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        metadata: dict[str, str] | None = None,
        created_at: datetime | None = None,
    ) -> StoredMessage:
        stamp = created_at or utcnow()

        def _op(engine: Engine) -> StoredMessage:
            for _ in range(8):  # concurrent writers may race on seq; retry
                try:
                    with engine.begin() as conn:
                        exists = conn.execute(
                            select(conversations.c.id).where(conversations.c.id == conversation_id)
                        ).first()

                        if not exists:
                            raise ConversationNotFoundError(
                                "Conversation not found.",
                                details={"conversation_id": conversation_id},
                            )

                        last = conn.execute(
                            select(func.max(messages.c.seq)).where(
                                messages.c.conversation_id == conversation_id
                            )
                        ).scalar()
                        stored = StoredMessage(
                            conversation_id=conversation_id,
                            seq=(last or 0) + 1,
                            role=role,
                            content=content,
                            metadata=metadata or {},
                            created_at=stamp,
                        )
                        conn.execute(
                            insert(messages).values(
                                conversation_id=conversation_id,
                                seq=stored.seq,
                                role=role,
                                content=content,
                                metadata=json.dumps(stored.metadata),
                                created_at=stored.created_at,
                            )
                        )
                        conn.execute(
                            update(conversations)
                            .where(conversations.c.id == conversation_id)
                            .values(
                                message_count=conversations.c.message_count + 1,
                                updated_at=stored.created_at,
                            )
                        )
                        return stored
                except IntegrityError:
                    continue

            raise MemoryStorageError("Could not allocate a message sequence number.")

        return await self._db.run(_op)

    async def messages(
        self,
        conversation_id: str,
        *,
        after_seq: int = 0,
        limit: int | None = None,
        newest_first: bool = False,
    ) -> list[StoredMessage]:
        def _op(engine: Engine) -> list[StoredMessage]:
            query = select(messages).where(
                messages.c.conversation_id == conversation_id, messages.c.seq > after_seq
            )
            query = query.order_by(messages.c.seq.desc() if newest_first else messages.c.seq)

            if limit is not None:
                query = query.limit(limit)

            with engine.connect() as conn:
                return [_message(r) for r in conn.execute(query).all()]

        return await self._db.run(_op)

    async def delete_messages_through(self, conversation_id: str, seq: int) -> int:
        def _op(engine: Engine) -> int:
            with engine.begin() as conn:
                result = conn.execute(
                    delete(messages).where(
                        messages.c.conversation_id == conversation_id, messages.c.seq <= seq
                    )
                )
                remaining = conn.execute(
                    select(func.count())
                    .select_from(messages)
                    .where(messages.c.conversation_id == conversation_id)
                ).scalar()
                conn.execute(
                    update(conversations)
                    .where(conversations.c.id == conversation_id)
                    .values(message_count=int(remaining or 0))
                )
                return result.rowcount

        return await self._db.run(_op)

    async def delete_messages(self, conversation_id: str) -> int:
        def _op(engine: Engine) -> int:
            with engine.begin() as conn:
                result = conn.execute(
                    delete(messages).where(messages.c.conversation_id == conversation_id)
                )
                conn.execute(
                    update(conversations)
                    .where(conversations.c.id == conversation_id)
                    .values(message_count=0)
                )
                return result.rowcount

        return await self._db.run(_op)
