"""MemoryService: the single write/read/control surface for memories (8.2, 8.7)."""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime, timedelta

from pydantic import BaseModel, ConfigDict

from agent.memory.embeddings import Embedder
from agent.memory.errors import (
    MemoryConflictError,
    MemoryNotFoundError,
    MemoryStorageError,
    MemoryValidationError,
)
from agent.memory.extraction import ImportanceScorer, MemoryCandidate, resolve_type_kind
from agent.memory.models import (
    MemoryCreate,
    MemoryFilter,
    MemoryKind,
    MemoryRecord,
    MemorySource,
    MemoryType,
    MemoryUpdate,
    ScoredMemory,
    WriteResult,
    utcnow,
)
from agent.memory.privacy import SensitiveDataFilter
from agent.memory.retrieval import MemoryContext, MemoryRetriever, build_context
from agent.memory.settings import MemorySettings, RetentionPolicy
from agent.memory.store import MemoryStore, VectorIndex
from agent.memory.text import content_hash, keyword_set

logger = logging.getLogger(__name__)
_REINFORCEMENT = 0.02


class RetentionReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expired: int = 0
    aged: int = 0
    low_importance: int = 0
    over_capacity: int = 0

    @property
    def total(self) -> int:
        return self.expired + self.aged + self.low_importance + self.over_capacity


class MemoryService:
    def __init__(
        self,
        store: MemoryStore,
        vectors: VectorIndex | None = None,
        embedder: Embedder | None = None,
        settings: MemorySettings | None = None,
        *,
        privacy: SensitiveDataFilter | None = None,
        scorer: ImportanceScorer | None = None,
        clock: Callable[[], datetime] = utcnow,
    ) -> None:
        self.settings = settings or MemorySettings()
        self._store = store
        self._vectors = vectors
        self._embedder = embedder
        self.privacy = privacy or SensitiveDataFilter(self.settings.privacy)
        self._scorer = scorer or ImportanceScorer()
        self._clock = clock
        self.retriever = MemoryRetriever(store, vectors, embedder, self.settings.retrieval, clock)

    # ------------------------------------------------------------------ vectors

    async def _index(self, record: MemoryRecord) -> bool:
        """Embed and index; failure leaves the memory stored and keyword-searchable."""
        if not (self._vectors and self._embedder):
            return True

        try:
            vector = (await self._embedder.embed([record.content]))[0]
            await self._vectors.upsert(
                record.id, vector, user_id=record.user_id, memory_type=record.type
            )
            return True
        except Exception:
            logger.warning("Vector indexing failed for memory %s", record.id, exc_info=True)
            return False

    async def _unindex(self, ids: list[str]) -> None:
        if self._vectors and ids:
            try:
                await self._vectors.delete(ids)
            except Exception:
                logger.warning("Vector deletion failed for %d memories", len(ids), exc_info=True)

    async def reindex(self, user_id: str) -> int:
        """Rebuild vectors for a user (e.g. after switching embedders or restoring Qdrant)."""
        records = await self._store.list(
            MemoryFilter(user_id=user_id, include_expired=True), limit=100_000
        )
        done = 0

        for record in records:
            if await self._index(record):
                done += 1

        return done

    # ------------------------------------------------------------------ create

    async def add_memory(self, create: MemoryCreate) -> WriteResult:
        """Add a memory, merging duplicates and refusing sensitive content."""
        content = " ".join(create.content.split())

        if not content:
            raise MemoryValidationError("Memory content cannot be empty.")

        content = content[: self.settings.max_content_chars]
        check = self.privacy.inspect(content)

        if check.blocked:
            return WriteResult(rejected_reason=check.reason)

        content = check.text
        memory_type, kind = resolve_type_kind(content, create.type, create.kind)
        now = self._clock()
        candidate = MemoryCandidate(
            content,
            memory_type,
            kind,
            create.source,
            explicit=create.source is MemorySource.EXPLICIT,
            metadata=create.metadata,
        )
        importance = (
            create.importance if create.importance is not None else self._scorer.score(candidate)
        )
        digest = content_hash(content)

        existing = await self._store.find_by_hash(create.user_id, memory_type, digest)

        if existing is None:
            existing = await self._near_duplicate(create.user_id, memory_type, kind, content)

        if existing is not None:
            return WriteResult(
                record=await self._merge(existing, importance, create, now), deduplicated=True
            )

        record = MemoryRecord(
            user_id=create.user_id,
            type=memory_type,
            kind=kind,
            content=content,
            source=create.source,
            importance=importance,
            conversation_id=create.conversation_id,
            metadata=dict(create.metadata),
            keywords=keyword_set(content),
            content_hash=digest,
            created_at=now,
            updated_at=now,
            expires_at=now + timedelta(days=create.ttl_days) if create.ttl_days else None,
            pinned=create.pinned,
        )

        try:
            await self._store.add(record)
        except MemoryConflictError:  # lost a race with an identical concurrent write
            winner = await self._store.find_by_hash(create.user_id, memory_type, digest)

            if winner is None:
                raise

            return WriteResult(
                record=await self._merge(winner, importance, create, now), deduplicated=True
            )

        if not await self._index(record):
            record = record.model_copy(
                update={"metadata": {**record.metadata, "vector_indexed": "false"}}
            )
            await self._store.update(record)

        return WriteResult(record=record, created=True)

    async def add_candidates(
        self, user_id: str, candidates: list[MemoryCandidate], *, conversation_id: str | None = None
    ) -> list[WriteResult]:
        """Persist *approved* extraction candidates (privacy is re-checked on write)."""
        results = []

        for c in candidates:
            results.append(
                await self.add_memory(
                    MemoryCreate(
                        content=c.content,
                        user_id=user_id,
                        type=c.type,
                        kind=c.kind,
                        importance=c.importance,
                        source=c.source,
                        conversation_id=c.conversation_id or conversation_id,
                        metadata=c.metadata,
                        ttl_days=c.ttl_days,
                    )
                )
            )

        return results

    async def _near_duplicate(
        self, user_id: str, memory_type: MemoryType, kind: MemoryKind, content: str
    ) -> MemoryRecord | None:
        if not (self._vectors and self._embedder):
            return None

        try:
            vector = (await self._embedder.embed([content]))[0]
            hits = await self._vectors.search(vector, user_id=user_id, types=[memory_type], limit=3)
        except Exception:
            return None

        close = [mid for mid, sim in hits if sim >= self.settings.near_duplicate_similarity]

        for record in await self._store.get_many(close):
            if record.kind is kind:
                return record

        return None

    async def _merge(
        self, existing: MemoryRecord, importance: float, create: MemoryCreate, now: datetime
    ) -> MemoryRecord:
        merged = existing.model_copy(
            update={
                "importance": round(
                    min(1.0, max(existing.importance, importance) + _REINFORCEMENT), 3
                ),
                "updated_at": now,
                "metadata": {**create.metadata, **existing.metadata},
                "pinned": existing.pinned or create.pinned,
            }
        )
        await self._store.update(merged)
        return merged

    # ------------------------------------------------------------------ read

    async def get_memory(self, user_id: str, memory_id: str) -> MemoryRecord | None:
        record = await self._store.get(memory_id)
        return record if record and record.user_id == user_id else None  # no cross-user reads

    async def list_memories(
        self, flt: MemoryFilter, *, limit: int = 100, offset: int = 0
    ) -> list[MemoryRecord]:
        return await self._store.list(flt, limit=limit, offset=offset, now=self._clock())

    async def retrieve(self, user_id: str, text: str, **kwargs: object) -> list[ScoredMemory]:
        return await self.retriever.retrieve(user_id, text, **kwargs)  # type: ignore[arg-type]

    async def context_for(
        self, user_id: str, text: str, *, budget_tokens: int | None = None, **kwargs: object
    ) -> MemoryContext:
        hits = await self.retrieve(user_id, text, **kwargs)
        return build_context(hits, budget_tokens or self.settings.context_token_budget)

    # ------------------------------------------------------------------ update / delete

    async def update_memory(
        self, user_id: str, memory_id: str, change: MemoryUpdate
    ) -> MemoryRecord:
        record = await self.get_memory(user_id, memory_id)

        if record is None:
            raise MemoryNotFoundError("Memory not found.", details={"memory_id": memory_id})

        fields: dict[str, object] = {"updated_at": self._clock()}
        content_changed = False

        if change.content is not None:
            content = " ".join(change.content.split())[: self.settings.max_content_chars]
            check = self.privacy.inspect(content)

            if check.blocked:
                raise MemoryValidationError(
                    "Updated content contains sensitive information and cannot be stored.",
                    details={"categories": list(check.categories)},
                )

            fields.update(
                content=check.text,
                content_hash=content_hash(check.text),
                keywords=keyword_set(check.text),
            )
            content_changed = True

        if change.kind is not None:
            fields["kind"] = change.kind

        if change.importance is not None:
            fields["importance"] = change.importance

        if change.metadata is not None:
            fields["metadata"] = dict(change.metadata)

        if change.pinned is not None:
            fields["pinned"] = change.pinned

        if change.clear_expiry:
            fields["expires_at"] = None
        elif change.ttl_days is not None:
            fields["expires_at"] = self._clock() + timedelta(days=change.ttl_days)

        updated = MemoryRecord.model_validate({**record.model_dump(), **fields})
        await self._store.update(updated)  # raises MemoryConflictError on duplicate content

        if content_changed and not await self._index(updated):
            updated = updated.model_copy(
                update={"metadata": {**updated.metadata, "vector_indexed": "false"}}
            )
            await self._store.update(updated)

        return updated

    async def delete_memory(self, user_id: str, memory_id: str) -> bool:
        if await self.get_memory(user_id, memory_id) is None:
            return False

        deleted = await self._store.delete(memory_id)
        await self._unindex([memory_id])
        return deleted

    async def delete_matching(self, flt: MemoryFilter) -> int:
        ids = await self._store.delete_where(flt)
        await self._unindex(ids)
        return len(ids)

    async def clear_conversation_memory(self, user_id: str, conversation_id: str) -> int:
        """Delete every memory that was created from one conversation."""
        return await self.delete_matching(
            MemoryFilter(user_id=user_id, conversation_id=conversation_id)
        )

    async def forget_user(self, user_id: str) -> int:
        """Delete all of a user's memories (including pinned and expired)."""
        return await self.delete_matching(MemoryFilter(user_id=user_id))

    async def upsert_summary(self, user_id: str, conversation_id: str, text: str) -> WriteResult:
        """Keep exactly one summary memory per conversation (replace, don't pile up)."""
        existing = await self._store.list(
            MemoryFilter(
                user_id=user_id, conversation_id=conversation_id, kinds=[MemoryKind.SUMMARY]
            ),
            limit=1,
        )

        if existing:
            check = self.privacy.inspect(text[:1500])

            if check.blocked:
                return WriteResult(rejected_reason=check.reason)

            try:
                updated = await self.update_memory(
                    user_id, existing[0].id, MemoryUpdate(content=check.text)
                )
            except MemoryConflictError:
                return WriteResult(record=existing[0], deduplicated=True)

            return WriteResult(record=updated)

        return await self.add_memory(
            MemoryCreate(
                content=text[:1500],
                user_id=user_id,
                type=MemoryType.LONG_TERM,
                kind=MemoryKind.SUMMARY,
                source=MemorySource.SUMMARY,
                conversation_id=conversation_id,
                importance=0.5,
            )
        )

    # ------------------------------------------------------------------ retention (8.3, 8.7)

    def configure_retention(self, policy: RetentionPolicy) -> None:
        """Replace the retention policy at runtime (applies on the next ``apply_retention``)."""
        self.settings = self.settings.model_copy(update={"retention": policy})

    async def apply_retention(self, user_id: str | None = None) -> RetentionReport:
        users = [user_id] if user_id else await self._store.list_user_ids()
        report = RetentionReport()

        for user in users:
            report.expired += await self._drop(
                await self._store.delete_expired(self._clock(), user)
            )
            report = await self._retain_user(user, report)

        return report

    async def _drop(self, ids: list[str]) -> int:
        await self._unindex(ids)
        return len(ids)

    async def _retain_user(self, user_id: str, report: RetentionReport) -> RetentionReport:
        policy = self.settings.retention
        now = self._clock()

        for memory_type, max_age in policy.max_age_days.items():
            if max_age is not None:
                flt = MemoryFilter(
                    user_id=user_id,
                    types=[memory_type],
                    pinned=False,
                    created_before=now - timedelta(days=max_age),
                )
                report.aged += await self.delete_matching(flt)

        if policy.low_importance_max_age_days is not None:
            flt = MemoryFilter(
                user_id=user_id,
                pinned=False,
                max_importance=policy.low_importance_floor,
                created_before=now - timedelta(days=policy.low_importance_max_age_days),
            )
            report.low_importance += await self.delete_matching(flt)

        half_lives = self.settings.retrieval.recency_half_life_days

        for memory_type, cap in policy.max_records_per_user.items():
            if cap is None:
                continue

            flt = MemoryFilter(user_id=user_id, types=[memory_type])
            total = await self._store.count(flt, now=now)

            if total <= cap:
                continue

            candidates = await self._store.list(
                MemoryFilter(user_id=user_id, types=[memory_type], pinned=False),
                limit=100_000,
                now=now,
            )

            def keep_value(
                r: MemoryRecord, hl: float = half_lives[memory_type]
            ) -> tuple[float, str]:
                age = max(0.0, (now - r.updated_at).total_seconds() / 86400)
                return (0.7 * r.importance + 0.3 * 0.5 ** (age / hl), r.id)

            victims = sorted(candidates, key=keep_value)[: total - cap]
            report.over_capacity += await self._drop_records([v.id for v in victims])

        return report

    async def _drop_records(self, ids: list[str]) -> int:
        count = 0

        for memory_id in ids:
            if await self._store.delete(memory_id):
                count += 1

        await self._unindex(ids)
        return count

    async def stats(self, user_id: str) -> dict[str, int]:
        now = self._clock()
        out = {"total": await self._store.count(MemoryFilter(user_id=user_id), now=now)}

        for memory_type in MemoryType:
            out[memory_type.value] = await self._store.count(
                MemoryFilter(user_id=user_id, types=[memory_type]), now=now
            )

        return out


__all__ = ["MemoryService", "MemoryStorageError", "RetentionReport"]
