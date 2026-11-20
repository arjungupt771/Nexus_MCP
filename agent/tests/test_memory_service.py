from __future__ import annotations

from datetime import timedelta
from types import SimpleNamespace

import pytest
from support_memory import Clock, create, manager

from agent.memory import (
    HashingEmbedder,
    MemoryConflictError,
    MemoryFilter,
    MemoryKind,
    MemoryNotFoundError,
    MemorySettings,
    MemorySource,
    MemoryStorageError,
    MemoryType,
    MemoryUpdate,
    MemoryValidationError,
    RetentionPolicy,
)
from agent.memory.extraction import classify, resolve_type_kind

# ------------------------------------------------------------------ creation & classification


@pytest.mark.parametrize(
    ("content", "memory_type", "kind"),
    [
        ("User prefers dark mode", MemoryType.LONG_TERM, MemoryKind.PREFERENCE),
        ("User's favorite editor is Neovim", MemoryType.LONG_TERM, MemoryKind.PREFERENCE),
        (
            "User wants the assistant to always reply briefly",
            MemoryType.LONG_TERM,
            MemoryKind.PREFERENCE,
        ),
        ("Decision: use PostgreSQL for billing", MemoryType.EPISODIC, MemoryKind.DECISION),
        ("Created note 'Weekly plan' successfully", MemoryType.EPISODIC, MemoryKind.OUTCOME),
        ("Sam Lee is the user's manager", MemoryType.SEMANTIC, MemoryKind.RELATIONSHIP),
        ("User works at Acme Corp", MemoryType.LONG_TERM, MemoryKind.FACT),
        ("Kubernetes schedules containers onto nodes", MemoryType.SEMANTIC, MemoryKind.KNOWLEDGE),
    ],
)
def test_classification(content, memory_type, kind):
    assert classify(content) == (memory_type, kind)


def test_type_and_kind_resolution_rules():
    assert resolve_type_kind("x", None, MemoryKind.TASK) == (MemoryType.EPISODIC, MemoryKind.TASK)
    assert resolve_type_kind("anything", MemoryType.SEMANTIC, None)[0] is MemoryType.SEMANTIC
    assert resolve_type_kind("User prefers tea", MemoryType.LONG_TERM, None) == (
        MemoryType.LONG_TERM,
        MemoryKind.PREFERENCE,
    )
    assert resolve_type_kind("blah", MemoryType.EPISODIC, None) == (
        MemoryType.EPISODIC,
        MemoryKind.EVENT,
    )
    assert resolve_type_kind("x", MemoryType.LONG_TERM, MemoryKind.SUMMARY) == (
        MemoryType.LONG_TERM,
        MemoryKind.SUMMARY,
    )


async def test_memory_is_created_with_inferred_metadata():
    m = manager(Clock())
    result = await m.memory.add_memory(
        create("User prefers dark mode and tabs", metadata={"origin": "settings"})
    )

    r = result.record
    assert result.created and not result.deduplicated and result.stored
    assert (r.type, r.kind, r.source) == (
        MemoryType.LONG_TERM,
        MemoryKind.PREFERENCE,
        MemorySource.EXPLICIT,
    )
    assert 0 < r.importance <= 1 and r.keywords == ["prefer", "dark", "mode", "tab"]
    assert (
        r.metadata == {"origin": "settings"}
        and r.created_at == r.updated_at
        and r.id
        and r.content_hash
    )
    assert (await m.memory.get_memory("u1", r.id)) == r


async def test_explicit_type_kind_importance_ttl_and_pinning_are_respected():
    clock = Clock()
    m = manager(clock)
    r = (
        await m.memory.add_memory(
            create(
                "Standup is at 9:30",
                type=MemoryType.EPISODIC,
                kind=MemoryKind.EVENT,
                importance=0.33,
                ttl_days=2,
                pinned=True,
                conversation_id="c1",
                source=MemorySource.SYSTEM,
            )
        )
    ).record

    assert (r.type, r.kind, r.importance, r.pinned, r.conversation_id) == (
        MemoryType.EPISODIC,
        MemoryKind.EVENT,
        0.33,
        True,
        "c1",
    )
    assert r.expires_at == clock.now + timedelta(days=2) and r.source is MemorySource.SYSTEM


async def test_whitespace_is_normalized_and_content_is_bounded():
    m = manager(settings=MemorySettings(max_content_chars=60))
    r = (await m.memory.add_memory(create("User   prefers\n\ttea " + "x" * 200))).record

    assert "  " not in r.content and len(r.content) <= 60

    with pytest.raises(MemoryValidationError):
        await m.memory.add_memory(create("   "))


async def test_vectors_are_indexed_for_new_memories():
    m = manager()
    r = (await m.memory.add_memory(create("User prefers dark mode"))).record
    vector = (await HashingEmbedder().embed([r.content]))[0]

    hits = await m.memory._vectors.search(vector, user_id="u1")
    assert hits and hits[0][0] == r.id and hits[0][1] > 0.99


# ------------------------------------------------------------------ deduplication


async def test_exact_duplicates_are_merged_and_reinforced():
    m = manager(Clock())
    first = (
        await m.memory.add_memory(
            create("User prefers dark mode", importance=0.6, metadata={"a": "1"})
        )
    ).record

    second = await m.memory.add_memory(
        create("user PREFERS dark mode!!", importance=0.8, metadata={"a": "2", "b": "3"})
    )

    assert second.deduplicated and not second.created and second.record.id == first.id
    assert second.record.importance == pytest.approx(0.82)  # max(0.6, 0.8) + reinforcement
    assert second.record.metadata == {"a": "1", "b": "3"}  # existing keys win, new keys added
    assert await m.memory._store.count(MemoryFilter(user_id="u1")) == 1


async def test_near_duplicates_are_merged_via_vector_similarity():
    m = manager()
    a = (await m.memory.add_memory(create("User prefers dark mode in the editor"))).record
    b = await m.memory.add_memory(create("User prefers dark mode in the editors"))

    assert b.deduplicated and b.record.id == a.id

    different = await m.memory.add_memory(create("User prefers tabs over spaces"))
    assert different.created


async def test_same_text_in_different_scopes_is_not_a_duplicate():
    m = manager()
    await m.memory.add_memory(create("Sam is great", user_id="u1", type=MemoryType.LONG_TERM))

    assert (
        await m.memory.add_memory(create("Sam is great", user_id="u2", type=MemoryType.LONG_TERM))
    ).created
    assert (
        await m.memory.add_memory(create("Sam is great", user_id="u1", type=MemoryType.SEMANTIC))
    ).created
    assert (
        await m.memory.add_memory(create("Sam is great", user_id="u1", type=MemoryType.LONG_TERM))
    ).deduplicated


async def test_concurrent_identical_adds_produce_one_record():
    import asyncio

    m = manager()
    results = await asyncio.gather(
        *(m.memory.add_memory(create("User likes green tea")) for _ in range(10))
    )

    assert sum(r.created for r in results) == 1 and sum(r.deduplicated for r in results) == 9
    assert await m.memory._store.count(MemoryFilter(user_id="u1")) == 1


# ------------------------------------------------------------------ privacy


@pytest.mark.parametrize(
    "content",
    [
        "my password is hunter2",
        "token ghp_abcdefghijklmnopqrstuvwxyz012",
        "card 4111 1111 1111 1111",
        "I take medication daily",
    ],
)
async def test_sensitive_content_is_never_stored(content):
    m = manager()
    result = await m.memory.add_memory(create(content))

    assert not result.stored and result.rejected_reason.startswith("sensitive:")
    assert await m.memory._store.count(MemoryFilter(user_id="u1")) == 0


async def test_pii_is_redacted_before_storage():
    m = manager()
    r = (await m.memory.add_memory(create("Contact for invoices is ada@example.com"))).record

    assert "example.com" not in r.content and "[email]" in r.content


async def test_updates_cannot_smuggle_sensitive_data_in():
    m = manager()
    r = (await m.memory.add_memory(create("User prefers tea"))).record

    with pytest.raises(MemoryValidationError) as exc_info:
        await m.memory.update_memory("u1", r.id, MemoryUpdate(content="my api key is abc123"))

    assert "credential" in exc_info.value.details["categories"]
    assert (await m.memory.get_memory("u1", r.id)).content == "User prefers tea"


# ------------------------------------------------------------------ updates & deletion


async def test_update_changes_fields_reindexes_and_keeps_identity():
    clock = Clock()
    m = manager(clock)
    r = (await m.memory.add_memory(create("User prefers dark mode"))).record
    clock.advance(hours=1)

    updated = await m.memory.update_memory(
        "u1",
        r.id,
        MemoryUpdate(
            content="User prefers light mode", importance=0.95, pinned=True, metadata={"v": "2"}
        ),
    )

    assert (
        updated.id == r.id
        and updated.created_at == r.created_at
        and updated.updated_at > r.updated_at
    )
    assert (updated.content, updated.importance, updated.pinned, updated.metadata) == (
        "User prefers light mode",
        0.95,
        True,
        {"v": "2"},
    )
    assert updated.keywords == ["prefer", "light", "mode"]
    assert [h.record.id for h in await m.memory.retrieve("u1", "light mode")][:1] == [r.id]
    assert (
        await m.memory.retrieve("u1", "dark mode", include_standing=False, min_importance=0.99)
        == []
    )


async def test_update_ttl_and_clearing_expiry():
    clock = Clock()
    m = manager(clock)
    r = (await m.memory.add_memory(create("User prefers tea", ttl_days=1))).record

    longer = await m.memory.update_memory("u1", r.id, MemoryUpdate(ttl_days=30))
    assert longer.expires_at == clock.now + timedelta(days=30)
    assert (
        await m.memory.update_memory("u1", r.id, MemoryUpdate(clear_expiry=True))
    ).expires_at is None


async def test_update_conflicts_and_missing_memories():
    m = manager()
    a = (await m.memory.add_memory(create("User prefers tea"))).record
    await m.memory.add_memory(create("User prefers coffee"))

    with pytest.raises(MemoryConflictError):
        await m.memory.update_memory("u1", a.id, MemoryUpdate(content="user prefers COFFEE"))

    with pytest.raises(MemoryNotFoundError):
        await m.memory.update_memory("u1", "missing", MemoryUpdate(importance=0.1))

    with pytest.raises(ValueError):
        MemoryUpdate(importance=3)


async def test_delete_removes_record_keywords_and_vectors():
    m = manager()
    r = (await m.memory.add_memory(create("User prefers dark mode"))).record

    assert await m.memory.delete_memory("u1", r.id) is True
    assert await m.memory.delete_memory("u1", r.id) is False
    assert await m.memory.get_memory("u1", r.id) is None
    assert await m.memory.retrieve("u1", "dark mode") == []
    vector = (await HashingEmbedder().embed(["User prefers dark mode"]))[0]
    assert await m.memory._vectors.search(vector, user_id="u1") == []


async def test_users_cannot_read_change_or_delete_each_others_memories():
    m = manager()
    r = (await m.memory.add_memory(create("User prefers tea", user_id="owner"))).record

    assert await m.memory.get_memory("intruder", r.id) is None
    assert await m.memory.delete_memory("intruder", r.id) is False
    with pytest.raises(MemoryNotFoundError):
        await m.memory.update_memory("intruder", r.id, MemoryUpdate(importance=0))
    assert await m.memory.retrieve("intruder", "tea") == []
    assert (await m.memory.get_memory("owner", r.id)).importance == r.importance


async def test_clear_conversation_memory_only_removes_that_conversations_memories():
    m = manager()
    await m.memory.add_memory(create("User prefers tea", conversation_id="c1"))
    await m.memory.add_memory(create("User prefers coffee", conversation_id="c1"))
    keep = (await m.memory.add_memory(create("User prefers juice", conversation_id="c2"))).record
    other = (
        await m.memory.add_memory(create("User prefers milk", conversation_id="c1", user_id="u2"))
    ).record

    assert await m.memory.clear_conversation_memory("u1", "c1") == 2
    assert (
        await m.memory.get_memory("u1", keep.id) is not None
        and await m.memory.get_memory("u2", other.id) is not None
    )
    assert await m.memory.forget_user("u1") == 1 and await m.memory.stats("u1") == {
        "total": 0,
        "long_term": 0,
        "episodic": 0,
        "semantic": 0,
    }


async def test_summary_memory_is_replaced_not_duplicated():
    m = manager()
    first = await m.memory.upsert_summary("u1", "c1", "We discussed the billing migration plan.")
    second = await m.memory.upsert_summary(
        "u1", "c1", "We discussed billing migration and chose PostgreSQL."
    )

    assert (
        first.created
        and second.record.id == first.record.id
        and "PostgreSQL" in second.record.content
    )
    summaries = await m.memory.list_memories(MemoryFilter(user_id="u1", kinds=[MemoryKind.SUMMARY]))
    assert len(summaries) == 1
    assert (
        await m.memory.upsert_summary("u1", "c1", "password is hunter2")
    ).rejected_reason.startswith("sensitive")


# ------------------------------------------------------------------ retention


async def test_expired_memories_are_purged():
    clock = Clock()
    m = manager(clock)
    await m.memory.add_memory(create("User prefers tea", ttl_days=1))
    await m.memory.add_memory(create("User prefers coffee"))
    clock.advance(days=2)

    report = await m.memory.apply_retention()

    assert report.expired == 1 and report.total == 1
    assert [r.content for r in await m.memory.list_memories(MemoryFilter(user_id="u1"))] == [
        "User prefers coffee"
    ]


async def test_age_based_retention_spares_pinned_and_other_types():
    clock = Clock()
    m = manager(clock)
    ep = dict(type=MemoryType.EPISODIC, kind=MemoryKind.TASK, importance=0.9)
    await m.memory.add_memory(create("Old task one finished", **ep))
    await m.memory.add_memory(create("Old pinned task finished", pinned=True, **ep))
    await m.memory.add_memory(create("Old preference stays", importance=0.9))
    clock.advance(days=100)
    await m.memory.add_memory(create("Fresh task finished", **ep))

    report = await m.memory.apply_retention("u1")  # episodic max age is 90 days by default

    assert report.aged == 1
    left = {r.content for r in await m.memory.list_memories(MemoryFilter(user_id="u1"))}
    assert left == {"Old pinned task finished", "Old preference stays", "Fresh task finished"}


async def test_low_importance_memories_age_out_sooner():
    clock = Clock()
    m = manager(clock)
    await m.memory.add_memory(create("Minor passing remark", importance=0.1))
    await m.memory.add_memory(create("Important preference noted", importance=0.9))
    clock.advance(days=31)

    assert (await m.memory.apply_retention()).low_importance == 1
    assert [r.content for r in await m.memory.list_memories(MemoryFilter(user_id="u1"))] == [
        "Important preference noted"
    ]


async def test_capacity_limits_evict_the_least_valuable_first():
    clock = Clock()
    m = manager(clock)
    m.memory.configure_retention(
        RetentionPolicy(
            max_records_per_user={
                MemoryType.LONG_TERM: 3,
                MemoryType.EPISODIC: None,
                MemoryType.SEMANTIC: None,
            },
            low_importance_max_age_days=None,
        )
    )

    for text, imp in [
        ("alpha note", 0.9),
        ("bravo note", 0.2),
        ("charlie note", 0.5),
        ("delta note", 0.8),
        ("echo note", 0.3),
    ]:
        await m.memory.add_memory(create(text, importance=imp, type=MemoryType.LONG_TERM))
        clock.advance(minutes=1)

    pinned = (
        await m.memory.add_memory(
            create("zulu note", importance=0.1, pinned=True, type=MemoryType.LONG_TERM)
        )
    ).record
    report = await m.memory.apply_retention("u1")

    assert report.over_capacity == 3  # 6 records, cap 3
    left = {r.content for r in await m.memory.list_memories(MemoryFilter(user_id="u1"))}
    assert {"alpha note", "delta note", "zulu note"} == left and pinned.content in left


async def test_retention_is_configurable_at_runtime_and_covers_all_users():
    clock = Clock()
    m = manager(clock)
    await m.memory.add_memory(
        create("Task done for one", type=MemoryType.EPISODIC, kind=MemoryKind.TASK, user_id="a")
    )
    await m.memory.add_memory(
        create("Task done for two", type=MemoryType.EPISODIC, kind=MemoryKind.TASK, user_id="b")
    )
    clock.advance(days=10)

    m.configure_retention(RetentionPolicy(max_age_days={MemoryType.EPISODIC: 5.0}))
    assert m.settings.retention.max_age_days[MemoryType.EPISODIC] == 5.0

    report = await m.apply_retention()  # no user -> every user
    assert report.aged == 2


# ------------------------------------------------------------------ storage & index failures


async def test_vector_index_failure_does_not_lose_the_memory():
    m = manager()

    class BrokenIndex:
        async def upsert(self, *a, **k):
            raise MemoryStorageError("qdrant down")

        async def delete(self, ids):
            raise MemoryStorageError("qdrant down")

        async def search(self, *a, **k):
            raise MemoryStorageError("qdrant down")

        async def close(self):
            return None

    m.memory._vectors = BrokenIndex()
    m.memory.retriever._vectors = m.memory._vectors

    result = await m.memory.add_memory(create("User prefers dark mode"))

    assert result.created and result.record.metadata["vector_indexed"] == "false"
    hits = await m.memory.retrieve("u1", "dark mode")  # keyword search still works
    assert [h.record.id for h in hits] == [result.record.id] and hits[0].semantic_score == 0
    assert (
        await m.memory.delete_memory("u1", result.record.id) is True
    )  # delete tolerates the index being down


async def test_embedder_failure_degrades_to_keyword_only():
    m = manager()

    class BrokenEmbedder:
        dimensions = 8

        async def embed(self, texts):
            raise RuntimeError("model unavailable")

    m.memory._embedder = BrokenEmbedder()
    m.memory.retriever._embedder = m.memory._embedder
    result = await m.memory.add_memory(create("User prefers dark mode"))

    assert result.created and result.record.metadata.get("vector_indexed") == "false"
    assert (await m.memory.retrieve("u1", "dark mode"))[0].record.id == result.record.id


async def test_store_failures_surface_as_structured_errors():
    m = manager()

    async def boom(*a, **k):
        raise MemoryStorageError("disk full", details={"cause_type": "OperationalError"})

    m.memory._store.add = boom

    with pytest.raises(MemoryStorageError) as exc_info:
        await m.memory.add_memory(create("User prefers dark mode"))

    assert exc_info.value.to_dict()["code"] == "MEMORY_STORAGE_FAILED"


async def test_reindex_rebuilds_vectors():
    m = manager()
    r = (await m.memory.add_memory(create("User prefers dark mode"))).record
    await m.memory._vectors.delete([r.id])
    vector = (await HashingEmbedder().embed([r.content]))[0]
    assert await m.memory._vectors.search(vector, user_id="u1") == []

    assert await m.memory.reindex("u1") == 1
    assert (await m.memory._vectors.search(vector, user_id="u1"))[0][0] == r.id


async def test_service_without_vectors_works_keyword_only():
    from agent.memory import MemoryManager, SQLiteBackend

    backend = SQLiteBackend.in_memory()
    m = MemoryManager.build(backend, vector_index=None)
    m.memory._vectors = None
    m.memory.retriever._vectors = None
    r = (await m.memory.add_memory(create("User prefers dark mode"))).record

    assert [h.record.id for h in await m.memory.retrieve("u1", "dark mode")] == [r.id]
    assert SimpleNamespace  # keep import used
