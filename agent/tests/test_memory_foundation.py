from __future__ import annotations

import asyncio
import sqlite3
import uuid

import pytest
from support_memory import Clock, record

from agent.memory import (
    HashingEmbedder,
    InMemoryVectorIndex,
    MemoryConflictError,
    MemoryFilter,
    MemoryKind,
    MemoryNotFoundError,
    MemoryRecord,
    MemoryStorageError,
    MemoryType,
    PrivacyPolicy,
    QdrantVectorIndex,
    SensitiveDataFilter,
    SQLiteBackend,
    SQLiteMemoryStore,
    SQLiteVectorIndex,
)
from agent.memory.embeddings import cosine
from agent.memory.privacy import PiiMode
from agent.memory.settings import MemorySettings, RetentionPolicy
from agent.memory.store import MemoryStore, VectorIndex
from agent.memory.text import content_hash, estimate_tokens, keyword_set, tokenize

# ------------------------------------------------------------------ text & embeddings


def test_tokenizer_drops_stopwords_and_stems_consistently():
    assert tokenize("The user uses Docker containers!") == ["docker", "container"]
    assert keyword_set("deploying deploying services") == ["deploy", "service"]
    assert content_hash("I like  TEA!") == content_hash("i like tea")
    assert content_hash("tea") != content_hash("coffee")
    assert estimate_tokens("x" * 40) == 10


async def test_hashing_embedder_is_deterministic_normalized_and_overlap_aware():
    embedder = HashingEmbedder(128)
    a, b, c = await embedder.embed(
        ["deploying the billing service", "billing service deployment", "I like pasta"]
    )
    again = (await embedder.embed(["deploying the billing service"]))[0]

    assert a == again and len(a) == 128
    assert abs(sum(x * x for x in a) - 1) < 1e-9
    assert cosine(a, b) > 0.5 > cosine(a, c)
    assert cosine([], []) == 0 and cosine([1.0], [0.0]) == 0
    with pytest.raises(ValueError):
        HashingEmbedder(4)


# ------------------------------------------------------------------ privacy


@pytest.mark.parametrize(
    ("text", "category"),
    [
        ("my password is hunter2", "credential"),
        ("api key = abc123xyz", "credential"),
        ("use ghp_abcdefghijklmnopqrstuvwxyz0123", "api_key"),
        ("sk-abcdefghijklmnop1234", "api_key"),
        ("AKIAABCDEFGHIJKLMNOP", "api_key"),
        ("Authorization: Bearer abcdefghijklmnopqrstuv", "token"),
        ("eyJhbGciOiJIUzI1.eyJzdWIiOiIxMjM0NTY3.abc123", "token"),
        ("-----BEGIN RSA PRIVATE KEY-----", "private_key"),
        ("ssn 123-45-6789", "government_id"),
        ("card 4111 1111 1111 1111", "payment_card"),
        ("I take medication for it", "sensitive_topic"),
        ("my salary is large", "sensitive_topic"),
    ],
)
def test_sensitive_content_is_blocked(text, category):
    result = SensitiveDataFilter().inspect(text)

    assert (
        result.blocked and category in result.categories and result.reason.startswith("sensitive:")
    )


@pytest.mark.parametrize(
    "text", ["I prefer dark mode", "order 1234 5678 9012 3456", "use Python 3.13", "meeting at 3pm"]
)
def test_ordinary_content_passes(text):
    assert not SensitiveDataFilter().inspect(text).blocked


def test_pii_modes_and_custom_rules():
    text = "mail ada@example.com or call (415) 555-2671"
    assert SensitiveDataFilter().inspect(text).text == "mail [email] or call [phone]"
    assert not SensitiveDataFilter().inspect(text).blocked
    assert SensitiveDataFilter(PrivacyPolicy(pii_mode=PiiMode.ALLOW)).inspect(text).text == text
    blocked = SensitiveDataFilter(PrivacyPolicy(pii_mode=PiiMode.BLOCK)).inspect(text)
    assert blocked.blocked and "pii" in blocked.categories

    custom = SensitiveDataFilter(
        PrivacyPolicy(restricted_terms=("project nightfall",), allow_sensitive_topics=True)
    )
    assert custom.inspect("about Project Nightfall").categories == ("restricted_term",)
    assert not custom.inspect("I take medication").blocked


def test_restriction_phrases_and_secret_redaction():
    f = SensitiveDataFilter()

    assert f.is_restricted_request("This is OFF THE RECORD, ok?") and f.is_restricted_request(
        "please don't remember this"
    )
    assert not f.is_restricted_request("I like tea")
    redacted = f.redact_secrets("key sk-abcdefghijkl12 card 4111111111111111 hello")
    assert redacted == "key [REDACTED:api_key] card [REDACTED:payment_card] hello"


# ------------------------------------------------------------------ SQLite memory store


@pytest.fixture
async def backend(tmp_path):
    instance = SQLiteBackend(f"sqlite:///{tmp_path / 'mem.db'}")
    yield instance
    await instance.close()


@pytest.fixture
async def store(backend):
    return SQLiteMemoryStore(backend)


def test_stores_satisfy_protocols():
    b = SQLiteBackend.in_memory()
    assert isinstance(SQLiteMemoryStore(b), MemoryStore) and isinstance(
        SQLiteVectorIndex(b), VectorIndex
    )
    assert isinstance(InMemoryVectorIndex(), VectorIndex)


async def test_add_get_update_delete_roundtrip(store):
    r = record("User prefers dark mode", metadata={"k": "v"}, pinned=True)
    await store.add(r)

    loaded = await store.get(r.id)
    assert loaded == r and loaded.metadata == {"k": "v"} and loaded.keywords == r.keywords

    changed = r.model_copy(
        update={
            "importance": 0.9,
            "content": "User prefers light mode",
            "content_hash": content_hash("User prefers light mode"),
            "keywords": ["light", "mode"],
        }
    )
    await store.update(changed)
    assert (await store.get(r.id)).importance == 0.9
    assert [
        m.id for m, _ in await store.search_keywords(["light"], MemoryFilter(user_id="u1"), limit=5)
    ] == [r.id]
    assert (
        await store.search_keywords(["dark"], MemoryFilter(user_id="u1"), limit=5) == []
    )  # keyword index follows updates

    assert (
        await store.delete(r.id) is True
        and await store.delete(r.id) is False
        and await store.get(r.id) is None
    )
    with pytest.raises(MemoryNotFoundError):
        await store.update(r)


async def test_duplicate_content_per_user_and_type_conflicts(store):
    await store.add(record("User likes tea"))

    with pytest.raises(MemoryConflictError):
        await store.add(record("user likes TEA!"))  # same normalized hash

    await store.add(record("User likes tea", user_id="u2"))  # other user: fine
    await store.add(
        record("User likes tea", memory_type=MemoryType.SEMANTIC, kind=MemoryKind.KNOWLEDGE)
    )  # other type: fine
    found = await store.find_by_hash("u1", MemoryType.LONG_TERM, content_hash("User likes tea"))
    assert found is not None and found.user_id == "u1"


async def test_filters_cover_every_dimension(store):
    clock = Clock()
    old = record(
        "old fact",
        kind=MemoryKind.FACT,
        clock=clock,
        created_at=clock.now - __import__("datetime").timedelta(days=10),
    )
    await store.add(old)
    items = [
        record("pref one", importance=0.9, conversation_id="c1", metadata={"tag": "a"}),
        record(
            "fact two",
            kind=MemoryKind.FACT,
            importance=0.4,
            conversation_id="c2",
            metadata={"tag": "b"},
        ),
        record(
            "episode three",
            memory_type=MemoryType.EPISODIC,
            kind=MemoryKind.TASK,
            importance=0.6,
            pinned=True,
        ),
        record("other user", user_id="u2"),
    ]

    for item in items:
        await store.add(item)

    async def ids(**kw):
        return {r.content for r in await store.list(MemoryFilter(user_id="u1", **kw))}

    assert await ids() == {"old fact", "pref one", "fact two", "episode three"}
    assert await ids(types=[MemoryType.EPISODIC]) == {"episode three"}
    assert await ids(kinds=[MemoryKind.FACT]) == {"fact two", "old fact"}
    assert await ids(conversation_id="c1") == {"pref one"}
    assert await ids(min_importance=0.6) == {"pref one", "episode three", "old fact"}
    assert await ids(max_importance=0.6) == {"fact two"}
    assert await ids(metadata={"tag": "b"}) == {"fact two"}
    assert await ids(pinned=True) == {"episode three"}
    assert await ids(created_before=clock.now - __import__("datetime").timedelta(days=5)) == {
        "old fact"
    }
    assert await ids(created_after=clock.now - __import__("datetime").timedelta(days=5)) == {
        "pref one",
        "fact two",
        "episode three",
    }
    assert await store.count(MemoryFilter(user_id="u1")) == 4 and await store.list_user_ids() == [
        "u1",
        "u2",
    ]


async def test_listing_order_and_paging_are_deterministic(store):
    clock = Clock()

    for i in range(5):
        await store.add(
            record(
                f"memory number {i}",
                clock=clock,
                created_at=clock.now + __import__("datetime").timedelta(minutes=i),
            )
        )

    page1 = await store.list(MemoryFilter(user_id="u1"), limit=2)
    page2 = await store.list(MemoryFilter(user_id="u1"), limit=2, offset=2)

    assert [r.content for r in page1] == ["memory number 4", "memory number 3"]
    assert [r.content for r in page2] == ["memory number 2", "memory number 1"]


async def test_expired_memories_are_hidden_and_purgeable(store):
    clock = Clock()
    delta = __import__("datetime").timedelta
    await store.add(record("expired item", clock=clock, expires_at=clock.now - delta(days=1)))
    await store.add(record("live item", clock=clock, expires_at=clock.now + delta(days=1)))

    flt = MemoryFilter(user_id="u1")
    assert [r.content for r in await store.list(flt, now=clock.now)] == ["live item"]
    assert (
        len(await store.list(flt.model_copy(update={"include_expired": True}), now=clock.now)) == 2
    )
    assert (
        len(await store.delete_expired(clock.now)) == 1
        and await store.count(flt.model_copy(update={"include_expired": True})) == 1
    )


async def test_keyword_search_ranks_by_matched_terms_and_cascades_on_delete(store):
    a = record("docker kubernetes deployment guide")
    b = record("docker basics")
    await store.add(a)
    await store.add(b)

    terms = keyword_set("docker kubernetes")  # the retriever passes stemmed terms
    hits = await store.search_keywords(terms, MemoryFilter(user_id="u1"), limit=5)
    assert [(h.id, n) for h, n in hits] == [(a.id, 2), (b.id, 1)]

    await store.delete_where(MemoryFilter(user_id="u1", pinned=False))
    assert await store.search_keywords(terms, MemoryFilter(user_id="u1"), limit=5) == []


async def test_touch_updates_access_statistics(store):
    clock = Clock()
    r = record()
    await store.add(r)
    await store.touch([r.id], clock.now)
    await store.touch([r.id], clock.now)

    loaded = await store.get(r.id)
    assert loaded.access_count == 2 and loaded.last_accessed_at == clock.now


async def test_data_persists_across_backends_and_ids_are_per_record(tmp_path):
    url = f"sqlite:///{tmp_path / 'p.db'}"
    first = SQLiteBackend(url)
    r = record()
    await SQLiteMemoryStore(first).add(r)
    await first.close()

    second = SQLiteBackend(url)
    assert (await SQLiteMemoryStore(second).get(r.id)).content == r.content
    await second.close()


async def test_storage_failures_become_structured_errors(tmp_path):
    path = tmp_path / "broken.db"
    backend = SQLiteBackend(f"sqlite:///{path}")
    store = SQLiteMemoryStore(backend)
    await store.initialize()
    connection = sqlite3.connect(path)
    connection.execute("DROP TABLE nexus_memories")
    connection.commit()
    connection.close()

    with pytest.raises(MemoryStorageError) as exc_info:
        await store.get("x")

    assert (
        exc_info.value.details == {"cause_type": "OperationalError"}
        and exc_info.value.code == "MEMORY_STORAGE_FAILED"
    )
    await backend.close()


async def test_concurrent_identical_adds_have_one_winner(store):
    results = await asyncio.gather(
        *(store.add(record("same content")) for _ in range(8)), return_exceptions=True
    )

    assert sum(not isinstance(r, Exception) for r in results) == 1
    assert all(isinstance(r, MemoryConflictError) for r in results if isinstance(r, Exception))


# ------------------------------------------------------------------ vector indexes


async def vector_roundtrip(index):
    ids = [uuid.uuid4().hex for _ in range(4)]
    await index.upsert(ids[0], [1.0, 0.0, 0.0, 0.0], user_id="u1", memory_type=MemoryType.LONG_TERM)
    await index.upsert(ids[1], [0.7, 0.7, 0.0, 0.0], user_id="u1", memory_type=MemoryType.EPISODIC)
    await index.upsert(ids[2], [0.0, 0.0, 1.0, 0.0], user_id="u1", memory_type=MemoryType.LONG_TERM)
    await index.upsert(
        ids[3], [1.0, 0.0, 0.0, 0.0], user_id="u2", memory_type=MemoryType.LONG_TERM
    )  # someone else's

    hits = await index.search([1.0, 0.0, 0.0, 0.0], user_id="u1", limit=3)
    assert [h[0] for h in hits[:2]] == [ids[0], ids[1]] and hits[0][1] == pytest.approx(
        1.0, abs=1e-4
    )
    assert ids[3] not in [h[0] for h in hits]  # user isolation
    assert [
        h[0]
        for h in await index.search([1.0, 0.0, 0.0, 0.0], user_id="u1", types=[MemoryType.EPISODIC])
    ] == [ids[1]]
    assert len(await index.search([1.0, 0.0, 0.0, 0.0], user_id="u1", limit=1)) == 1

    await index.upsert(
        ids[0], [0.0, 1.0, 0.0, 0.0], user_id="u1", memory_type=MemoryType.LONG_TERM
    )  # overwrite
    assert (await index.search([0.0, 1.0, 0.0, 0.0], user_id="u1"))[0][0] in (ids[0], ids[1])
    await index.delete([ids[0], ids[1]])
    assert ids[0] not in [h[0] for h in await index.search([1.0, 0.0, 0.0, 0.0], user_id="u1")]
    await index.delete([])


async def test_in_memory_vector_index():
    index = InMemoryVectorIndex()
    await vector_roundtrip(index)
    assert len(index) >= 1


async def test_sqlite_vector_index(backend, store):
    index = SQLiteVectorIndex(backend)
    # Vectors join against memories for user/type filtering, so create matching rows.
    ids = {}

    async def put(content, user, mtype):
        r = record(
            content,
            user_id=user,
            memory_type=mtype,
            kind=MemoryKind.EVENT if mtype is MemoryType.EPISODIC else MemoryKind.FACT,
        )
        await store.add(r)
        return r.id

    a = await put("alpha item", "u1", MemoryType.LONG_TERM)
    b = await put("beta item", "u1", MemoryType.EPISODIC)
    c = await put("gamma item", "u2", MemoryType.LONG_TERM)
    await index.upsert(a, [1.0, 0.0], user_id="u1", memory_type=MemoryType.LONG_TERM)
    await index.upsert(b, [0.6, 0.8], user_id="u1", memory_type=MemoryType.EPISODIC)
    await index.upsert(c, [1.0, 0.0], user_id="u2", memory_type=MemoryType.LONG_TERM)

    assert [h[0] for h in await index.search([1.0, 0.0], user_id="u1")] == [a, b]
    assert [
        h[0] for h in await index.search([1.0, 0.0], user_id="u1", types=[MemoryType.EPISODIC])
    ] == [b]
    assert await index.search([1.0, 0.0, 0.0], user_id="u1") == []  # dimension mismatch is ignored
    await store.delete(a)  # deleting a memory removes its vector too
    assert [h[0] for h in await index.search([1.0, 0.0], user_id="u1")] == [b]
    assert ids == {}


async def test_qdrant_adapter_against_the_real_client_in_local_mode():
    qdrant = pytest.importorskip("qdrant_client")
    index = QdrantVectorIndex(qdrant.AsyncQdrantClient(":memory:"), dimensions=4)

    await vector_roundtrip(index)
    await index.close()


async def test_qdrant_failures_are_structured():
    pytest.importorskip("qdrant_client")

    class Broken:
        async def collection_exists(self, name):
            raise ConnectionError("qdrant down")

    index = QdrantVectorIndex(Broken(), dimensions=4)

    with pytest.raises(MemoryStorageError) as exc_info:
        await index.search([1.0, 0, 0, 0], user_id="u")

    assert exc_info.value.details["cause_type"] == "ConnectionError"


# ------------------------------------------------------------------ settings


def test_settings_validation():
    MemorySettings()

    for bad in (
        {"keep_recent_messages": 20, "summarize_after_messages": 10},
        {"min_importance_to_store": 2},
        {"surprise": 1},
    ):
        with pytest.raises(ValueError):
            MemorySettings(**bad)

    for bad in (
        {"max_age_days": {MemoryType.EPISODIC: -1}},
        {"max_records_per_user": {MemoryType.EPISODIC: 0}},
        {"max_messages_per_conversation": 1},
    ):
        with pytest.raises(ValueError):
            RetentionPolicy(**bad)


def test_memory_record_rejects_kind_that_does_not_match_type():
    with pytest.raises(ValueError, match="does not belong"):
        MemoryRecord(
            user_id="u",
            type=MemoryType.EPISODIC,
            kind=MemoryKind.PREFERENCE,
            content="x",
            content_hash="h",
        )
