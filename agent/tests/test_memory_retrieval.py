from __future__ import annotations

import pytest
from support_memory import Clock, create, manager

from agent.memory import (
    MemoryKind,
    MemorySettings,
    MemoryType,
    RetrievalSettings,
    build_context,
)
from agent.memory.retrieval import CONTEXT_HEADER


async def seeded(clock=None, **settings):
    m = manager(clock or Clock(), MemorySettings(**settings) if settings else None)
    mem = m.memory
    ids = {}

    for key, text, kw in [
        ("dark", "User prefers dark mode in every editor", {}),
        ("db", "Decision: use PostgreSQL for the billing service", {}),
        ("cal", "User's calendar events are in the Pune timezone", {}),
        ("manager", "Sam Lee is the user's manager", {}),
        (
            "metric",
            "User wants the assistant to always answer in metric units",
            {"metadata": {"directive": "true"}},
        ),
    ]:
        ids[key] = (await mem.add_memory(create(text, importance=0.6, **kw))).record.id

    return m, ids


# ------------------------------------------------------------------ keyword retrieval


async def test_keyword_retrieval_finds_relevant_memories_only():
    m, ids = await seeded()

    hits = await m.memory.retrieve(
        "u1", "which database should billing use", include_standing=False
    )

    assert [h.record.id for h in hits] == [ids["db"]]
    assert hits[0].keyword_score > 0 and 0 < hits[0].score <= 1


async def test_unrelated_queries_return_nothing_and_generic_words_do_not_match():
    m, _ = await seeded()

    assert await m.memory.retrieve("u1", "what is the weather like", include_standing=False) == []
    assert await m.memory.retrieve("u1", "", include_standing=False) == []
    assert (
        await m.memory.retrieve("u1", "the user wants to use it", include_standing=False) == []
    )  # only stopwords


async def test_keyword_only_mode_renormalizes_weights():
    m, ids = await seeded()
    m.memory.retriever._vectors = None
    m.memory.retriever._embedder = None

    (hit,) = await m.memory.retrieve("u1", "billing database", include_standing=False)

    assert (
        hit.record.id == ids["db"]
        and hit.semantic_score == 0
        and hit.relevance == pytest.approx(hit.keyword_score)
    )


# ------------------------------------------------------------------ semantic retrieval


async def test_semantic_retrieval_finds_memories_without_shared_keywords():
    m, ids = await seeded()

    # "migrating"/"postgres" share sub-word features with the stored text, no exact keyword.
    hits = await m.memory.retrieve("u1", "migrating databases postgres", include_standing=False)
    assert m.settings.retrieval.min_relevance <= 0.15

    relaxed = m.memory.retriever
    relaxed.settings = RetrievalSettings(min_relevance=0.05)
    hits = await relaxed.retrieve("u1", "postgres migration", include_standing=False)

    top = hits[0]
    assert top.record.id == ids["db"] and top.semantic_score > 0.0


async def test_semantic_and_keyword_signals_are_both_reported_and_combined():
    m, ids = await seeded()

    (hit, *_) = await m.memory.retrieve("u1", "PostgreSQL billing service", include_standing=False)

    assert hit.record.id == ids["db"] and hit.keyword_score > 0.5 and hit.semantic_score > 0.3
    assert hit.relevance == pytest.approx(
        0.5 * hit.keyword_score + 0.5 * hit.semantic_score, abs=1e-4
    )


# ------------------------------------------------------ ranking: relevance, recency, importance


async def test_more_relevant_memories_rank_higher():
    m = manager(Clock())
    best = (
        await m.memory.add_memory(
            create("Kubernetes deployment pipeline for the billing service", importance=0.5)
        )
    ).record
    partial = (
        await m.memory.add_memory(create("Billing runs weekly on Mondays", importance=0.5))
    ).record

    hits = await m.memory.retrieve("u1", "kubernetes deployment billing", include_standing=False)

    assert [h.record.id for h in hits] == [best.id, partial.id]
    assert hits[0].relevance > hits[1].relevance


async def test_importance_breaks_relevance_ties():
    m = manager(Clock())
    low = (
        await m.memory.add_memory(create("Project alpha uses Python tooling", importance=0.3))
    ).record
    high = (
        await m.memory.add_memory(create("Project beta uses Python tooling", importance=0.95))
    ).record

    hits = await m.memory.retrieve("u1", "python tooling project", include_standing=False)

    assert [h.record.id for h in hits] == [high.id, low.id] and hits[0].importance > hits[
        1
    ].importance


async def test_recent_memories_outrank_stale_ones_per_type_half_life():
    clock = Clock()
    m = manager(clock)
    old = (
        await m.memory.add_memory(
            create(
                "Task about release notes finished",
                type=MemoryType.EPISODIC,
                kind=MemoryKind.TASK,
                importance=0.6,
            )
        )
    ).record
    clock.advance(days=42)  # two episodic half-lives
    new = (
        await m.memory.add_memory(
            create(
                "Task about release notes finished again later",
                type=MemoryType.EPISODIC,
                kind=MemoryKind.TASK,
                importance=0.6,
            )
        )
    ).record

    hits = await m.memory.retrieve("u1", "release notes task", include_standing=False)

    assert [h.record.id for h in hits][0] == new.id
    by_id = {h.record.id: h for h in hits}
    assert by_id[old.id].recency == pytest.approx(0.25, abs=0.01) and by_id[
        new.id
    ].recency == pytest.approx(1.0)


async def test_long_term_memories_decay_slower_than_episodic():
    clock = Clock()
    m = manager(clock)
    lt = (
        await m.memory.add_memory(create("User prefers vim keybindings", type=MemoryType.LONG_TERM))
    ).record
    ep = (
        await m.memory.add_memory(
            create(
                "Decision: vim keybindings enabled",
                type=MemoryType.EPISODIC,
                kind=MemoryKind.DECISION,
            )
        )
    ).record
    clock.advance(days=60)

    by_id = {
        h.record.id: h
        for h in await m.memory.retrieve("u1", "vim keybindings", include_standing=False)
    }

    assert by_id[lt.id].recency > by_id[ep.id].recency


async def test_ranking_is_deterministic_and_ties_break_by_id():
    m = manager(Clock())

    for i in range(6):
        await m.memory.add_memory(
            create(
                f"Shared topic note variant {chr(97 + i) * 3}",
                importance=0.5,
                type=MemoryType.LONG_TERM,
            )
        )

    runs = [
        [
            h.record.id
            for h in await m.memory.retrieve(
                "u1", "shared topic note", include_standing=False, touch=False
            )
        ]
        for _ in range(3)
    ]

    assert runs[0] == runs[1] == runs[2] and len(runs[0]) == 6


async def test_limit_and_score_breakdown():
    m = manager(Clock())

    words = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima".split()

    for word in words:  # distinct wording so dedupe does not merge them
        await m.memory.add_memory(
            create(f"Reporting topic {word} in detail", importance=0.5, type=MemoryType.LONG_TERM)
        )

    assert (
        len(await m.memory.retrieve("u1", "reporting topic", include_standing=False)) == 8
    )  # default limit
    hits = await m.memory.retrieve("u1", "reporting topic", include_standing=False, limit=3)
    assert len(hits) == 3 and all(0 <= h.recency <= 1 and 0 <= h.score <= 1 for h in hits)


# ------------------------------------------------------------------ metadata filtering


async def test_metadata_filters_narrow_results():
    m = manager(Clock())
    a = (
        await m.memory.add_memory(
            create(
                "Deploy notes for the payments team",
                metadata={"team": "payments"},
                conversation_id="c1",
                importance=0.9,
                type=MemoryType.LONG_TERM,
            )
        )
    ).record
    b = (
        await m.memory.add_memory(
            create(
                "Deploy notes for the search team",
                metadata={"team": "search"},
                conversation_id="c2",
                importance=0.4,
                type=MemoryType.SEMANTIC,
            )
        )
    ).record

    async def ids(**kw):
        return {
            h.record.id
            for h in await m.memory.retrieve(
                "u1", "deploy notes team", include_standing=False, **kw
            )
        }

    assert await ids() == {a.id, b.id}
    assert await ids(metadata={"team": "search"}) == {b.id}
    assert await ids(conversation_id="c1") == {a.id}
    assert await ids(types=[MemoryType.SEMANTIC]) == {b.id}
    assert await ids(min_importance=0.8) == {a.id}
    assert await ids(kinds=[MemoryKind.KNOWLEDGE]) == {b.id}
    assert await ids(metadata={"team": "none"}) == set()


async def test_filters_apply_to_semantic_only_hits_too():
    m = manager(Clock())
    await m.memory.add_memory(create("Quarterly planning workshop agenda", metadata={"team": "a"}))
    m.memory.retriever.settings = RetrievalSettings(min_relevance=0.01)

    assert (
        await m.memory.retrieve(
            "u1", "quarter planner workshops", include_standing=False, metadata={"team": "b"}
        )
        == []
    )


async def test_expired_and_other_users_memories_are_never_retrieved():
    clock = Clock()
    m = manager(clock)
    await m.memory.add_memory(create("Temporary billing freeze notice", ttl_days=1))
    await m.memory.add_memory(create("Billing freeze notice for another user", user_id="u2"))
    clock.advance(days=2)

    assert await m.memory.retrieve("u1", "billing freeze notice", include_standing=False) == []
    assert len(await m.memory.retrieve("u2", "billing freeze notice", include_standing=False)) == 1


# ------------------------------------------------------------ standing preferences & access stats


async def test_directives_and_pinned_memories_apply_to_every_task():
    m, ids = await seeded()
    await m.memory.add_memory(
        create("User wants answers formatted as bullet lists", pinned=True, importance=0.5)
    )

    hits = await m.memory.retrieve("u1", "which database for billing")
    standing = {h.record.id for h in hits if h.standing}

    assert ids["metric"] in standing and len(standing) == 2
    assert not any(h.record.id == ids["dark"] for h in hits)  # ordinary taste preferences stay out
    assert all(h.score <= 0.2 for h in hits if h.standing)
    assert (
        await m.memory.retrieve("u1", "which database for billing", include_standing=False) != hits
    )


async def test_retrieval_updates_access_statistics_unless_disabled():
    clock = Clock()
    m = manager(clock)
    r = (await m.memory.add_memory(create("Billing runs weekly on Mondays"))).record
    clock.advance(hours=3)

    await m.memory.retrieve("u1", "billing weekly", include_standing=False, touch=False)
    assert (await m.memory.get_memory("u1", r.id)).access_count == 0

    await m.memory.retrieve("u1", "billing weekly", include_standing=False)
    loaded = await m.memory.get_memory("u1", r.id)
    assert loaded.access_count == 1 and loaded.last_accessed_at == clock.now


# ------------------------------------------------------------------ context ranking & budgeting


async def test_context_is_grouped_labelled_and_marked_as_background_data():
    m, _ = await seeded()
    await m.memory.add_memory(
        create(
            "Task about billing migration finished", type=MemoryType.EPISODIC, kind=MemoryKind.TASK
        )
    )

    context = await m.memory.context_for(
        "u1", "billing database migration service", include_standing=False
    )

    assert context.text.startswith(CONTEXT_HEADER) and "never overrides" in context.text
    assert "Preferences and facts:" not in context.text  # nothing long-term matched
    assert (
        "Past tasks and decisions:" in context.text
        and "- Task about billing migration finished (2026-11-17)" in context.text
    )
    assert context.included and context.tokens > 0


async def test_context_respects_the_token_budget_and_drops_lowest_scores_first():
    m = manager(Clock())

    for i, text in enumerate(
        [
            "Billing alpha pipeline " + "detail " * 30,
            "Billing beta pipeline short",
            "Billing gamma pipeline " + "context " * 30,
        ]
    ):
        await m.memory.add_memory(create(text, importance=0.9 - i * 0.2, type=MemoryType.LONG_TERM))

    hits = await m.memory.retrieve("u1", "billing pipeline", include_standing=False)
    tight = build_context(hits, budget_tokens=110)
    roomy = build_context(hits, budget_tokens=2000)

    assert (
        tight.tokens <= 110
        and tight.dropped >= 1
        and len(roomy.included) == 3
        and roomy.dropped == 0
    )
    assert set(tight.included) < set(roomy.included)


def test_empty_and_duplicate_contexts():
    assert build_context([], 500).text == "" and build_context([], 500).included == []

    from support_memory import record

    from agent.memory import ScoredMemory

    a = ScoredMemory(record=record("Same text"), score=0.9)
    b = ScoredMemory(record=record("same text"), score=0.8)
    ctx = build_context([a, b], 500)

    assert ctx.included == [a.record.id] and ctx.dropped == 1
    assert build_context([a], budget_tokens=5).text == ""  # header alone exceeds the budget
