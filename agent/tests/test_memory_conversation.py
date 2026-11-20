from __future__ import annotations

import asyncio

import pytest
from support_memory import Clock, manager
from support_orch import ScriptedProvider, text

from agent.llm import LLMRateLimitError, Role
from agent.memory import (
    ConversationNotFoundError,
    ExtractiveSummarizer,
    LLMSummarizer,
    MemoryManager,
    MemorySettings,
    MemoryValidationError,
    RetentionPolicy,
    ShortTermMemory,
    ShortTermMemoryManager,
    StoredMessage,
)

U = "u1"


async def conversation(m, **kwargs):
    return await m.conversations.create(U, **kwargs)


async def fill(m, cid, n, *, words="alpha beta gamma delta epsilon"):
    for i in range(n):
        await m.conversations.add_message(
            U, cid, "user" if i % 2 == 0 else "assistant", f"turn {i}. {words}"
        )


# ------------------------------------------------------------------ lifecycle


async def test_conversations_have_unique_ids_and_can_be_listed_renamed_and_archived():
    clock = Clock()
    m = manager(clock)
    a = await conversation(m, title="First")
    clock.advance(minutes=1)
    b = await conversation(m, title="Second")

    assert a.id != b.id and a.memory_enabled and not a.archived
    assert [c.id for c in await m.conversations.list(U)] == [
        b.id,
        a.id,
    ]  # most recently updated first

    assert (await m.conversations.rename(U, a.id, "Renamed")).title == "Renamed"
    archived = await m.conversations.archive(U, a.id)
    assert archived.archived and [c.id for c in await m.conversations.list(U)] == [b.id]
    assert {c.id for c in await m.conversations.list(U, include_archived=True)} == {a.id, b.id}
    assert not (await m.conversations.unarchive(U, a.id)).archived


async def test_get_or_create_and_delete():
    m = manager()
    created = await m.conversations.get_or_create(U)
    again = await m.conversations.get_or_create(U, created.id)

    assert again.id == created.id
    await fill(m, created.id, 3)
    assert await m.conversations.delete(U, created.id) is True
    with pytest.raises(ConversationNotFoundError):
        await m.conversations.get(U, created.id)


async def test_conversations_are_private_to_their_owner():
    m = manager()
    c = await conversation(m)
    await fill(m, c.id, 2)

    for call in (
        m.conversations.get("intruder", c.id),
        m.conversations.history("intruder", c.id),
        m.conversations.add_message("intruder", c.id, "user", "hi"),
        m.conversations.context_window("intruder", c.id),
        m.conversations.archive("intruder", c.id),
        m.conversations.delete("intruder", c.id),
    ):
        with pytest.raises(ConversationNotFoundError):
            await call

    assert await m.conversations.list("intruder") == []
    assert len(await m.conversations.history(U, c.id)) == 2


# ------------------------------------------------------------------ message persistence


async def test_messages_are_persisted_in_order_with_sequence_numbers():
    m = manager()
    c = await conversation(m)
    await fill(m, c.id, 5)

    history = await m.conversations.history(U, c.id)
    assert [h.seq for h in history] == [1, 2, 3, 4, 5] and [h.role for h in history][:2] == [
        "user",
        "assistant",
    ]
    assert [h.seq for h in await m.conversations.history(U, c.id, after_seq=3)] == [4, 5]
    assert [h.seq for h in await m.conversations.history(U, c.id, limit=2)] == [1, 2]
    assert (await m.conversations.get(U, c.id)).message_count == 5


async def test_history_survives_reopening_the_database(tmp_path):
    url = f"sqlite:///{tmp_path / 'conv.db'}"
    first = MemoryManager.sqlite(url)
    c = await first.conversations.create(U, title="Persisted")
    await fill(first, c.id, 4)
    await first.close()

    second = MemoryManager.sqlite(url)
    assert [h.content[:6] for h in await second.conversations.history(U, c.id)] == [
        "turn 0",
        "turn 1",
        "turn 2",
        "turn 3",
    ]
    assert (await second.conversations.get(U, c.id)).title == "Persisted"
    await second.close()


async def test_message_rules_roles_redaction_limits_and_tool_messages():
    m = manager(settings=MemorySettings(max_message_chars=100))
    c = await conversation(m)

    with pytest.raises(MemoryValidationError):
        await m.conversations.add_message(U, c.id, "robot", "hi")

    assert await m.conversations.add_message(U, c.id, "user", "   ") is None
    assert (
        await m.conversations.add_message(U, c.id, "tool", "raw tool output") is None
    )  # not persisted by default

    secret = await m.conversations.add_message(
        U, c.id, "user", "my key is sk-abcdefghijklmnop1234 ok"
    )
    assert secret.content == "my key is [REDACTED:api_key] ok"
    long = await m.conversations.add_message(U, c.id, "assistant", "x" * 500)
    assert len(long.content) == 100

    keep_tools = manager(settings=MemorySettings(persist_tool_messages=True))
    c2 = await keep_tools.conversations.create(U)
    assert (await keep_tools.conversations.add_message(U, c2.id, "tool", "output")).role == "tool"


async def test_concurrent_appends_get_unique_contiguous_sequence_numbers():
    m = manager()
    c = await conversation(m)

    await asyncio.gather(
        *(m.conversations.add_message(U, c.id, "user", f"message {i}") for i in range(25))
    )

    assert [h.seq for h in await m.conversations.history(U, c.id)] == list(range(1, 26))
    assert (await m.conversations.get(U, c.id)).message_count == 25


# ------------------------------------------------------------------ context-window management


async def test_window_keeps_the_newest_messages_that_fit_the_budget():
    m = manager()
    c = await conversation(m)
    await fill(m, c.id, 30, words="word " * 20)

    small = await m.conversations.context_window(U, c.id, max_tokens=200)
    large = await m.conversations.context_window(U, c.id, max_tokens=100_000)

    assert small.truncated and small.tokens <= 200
    assert [x.seq for x in small.messages] == list(
        range(30 - len(small.messages) + 1, 31)
    )  # contiguous tail
    assert len(large.messages) == 30 and not large.truncated


async def test_latest_message_is_always_kept_even_if_it_exceeds_the_budget():
    m = manager()
    c = await conversation(m)
    await m.conversations.add_message(U, c.id, "user", "tiny")
    await m.conversations.add_message(U, c.id, "user", "huge " * 500)

    window = await m.conversations.context_window(U, c.id, max_tokens=50)

    assert [x.seq for x in window.messages] == [2] and window.truncated


async def test_window_never_loads_unbounded_history():
    m = manager(settings=MemorySettings(history_token_budget=10_000_000))
    c = await conversation(m)

    for _ in range(10):
        await asyncio.gather(
            *(m.conversations.add_message(U, c.id, "user", "x") for _ in range(50))
        )

    window = await m.conversations.context_window(U, c.id)

    assert len(window.messages) <= 400 and window.truncated  # hard fetch cap


async def test_window_converts_to_llm_messages_with_summary_first():
    m = manager()
    c = await conversation(m)
    await fill(m, c.id, 20)
    await m.conversations.update_summary(U, c.id)
    await m.conversations.add_message(U, c.id, "system", "internal note")

    window = await m.conversations.context_window(U, c.id)
    messages = window.to_llm_messages()

    assert messages[0].role is Role.USER and messages[0].content.startswith(
        "Summary of the earlier conversation"
    )
    assert messages[1].role is Role.ASSISTANT and messages[1].content == "Understood."
    assert {x.role for x in messages[2:]} <= {
        Role.USER,
        Role.ASSISTANT,
    }  # system/tool messages are not replayed


# ------------------------------------------------------------------ summaries & pruning


async def test_summary_is_due_after_enough_messages_and_keeps_recent_ones_verbatim():
    m = manager()
    c = await conversation(m)
    await fill(m, c.id, 10)
    assert not await m.conversations.needs_summary(U, c.id)
    assert await m.conversations.update_summary(U, c.id) is None  # below threshold

    await fill(m, c.id, 10)
    assert await m.conversations.needs_summary(U, c.id)
    updated = await m.conversations.update_summary(U, c.id)

    keep = m.settings.keep_recent_messages
    assert updated.summarized_through == 20 - keep and "turn 0" in updated.summary
    window = await m.conversations.context_window(U, c.id)
    assert [x.seq for x in window.messages] == list(
        range(20 - keep + 1, 21)
    )  # only unsummarized turns replayed
    assert not await m.conversations.needs_summary(U, c.id)


async def test_summaries_roll_forward_and_secrets_never_enter_them():
    m = manager()
    c = await conversation(m)
    await m.conversations.add_message(
        U, c.id, "user", "remember my token is ghp_abcdefghijklmnopqrstuvwx"
    )
    await fill(m, c.id, 19)
    first = await m.conversations.update_summary(U, c.id)
    await fill(m, c.id, 16)
    second = await m.conversations.update_summary(U, c.id)

    assert "ghp_" not in first.summary and second.summarized_through > first.summarized_through
    assert second.summary.startswith(first.summary.split("\n")[0])  # previous summary is the base


async def test_extractive_summarizer_is_bounded_and_drops_oldest_lines_first():
    s = ExtractiveSummarizer(max_chars=200, line_chars=40)
    msgs = [
        StoredMessage(
            conversation_id="c",
            seq=i,
            role="user" if i % 2 else "assistant",
            content=f"line {i}. " + "x" * 80,
        )
        for i in range(1, 30)
    ]

    out = await s.summarize("", msgs)

    assert len(out) <= 200 and "line 29" in out and "line 1." not in out
    assert out.splitlines()[0].startswith(("User:", "Assistant:"))
    skipped = await s.summarize(
        "", [StoredMessage(conversation_id="c", seq=1, role="tool", content="data")]
    )
    assert skipped == ""


async def test_llm_summarizer_and_fallback():
    msgs = [
        StoredMessage(conversation_id="c", seq=1, role="user", content="We should migrate billing.")
    ]
    ok = ScriptedProvider(texts=[text("  Goal: migrate billing.  ")])
    assert await LLMSummarizer(ok).summarize("prev", msgs) == "Goal: migrate billing."
    assert "Previous summary:\nprev" in ok.text_requests[0][0][-1].content

    failing = ScriptedProvider(texts=[LLMRateLimitError("slow", provider="p")])
    assert (await LLMSummarizer(failing).summarize("", msgs)).startswith(
        "User: We should migrate billing"
    )
    empty = ScriptedProvider(texts=[text("   ")])
    assert (await LLMSummarizer(empty).summarize("", msgs)).startswith("User:")


async def test_stored_history_is_bounded_by_pruning_summarized_messages():
    m = manager(
        settings=MemorySettings(retention=RetentionPolicy(max_messages_per_conversation=10))
    )
    c = await conversation(m)
    await fill(m, c.id, 40)
    assert (await m.conversations.get(U, c.id)).message_count == 40

    await m.conversations.update_summary(U, c.id)  # folds 32, then prunes to the cap

    refreshed = await m.conversations.get(U, c.id)
    assert refreshed.message_count == 10 and refreshed.summarized_through == 32
    assert [h.seq for h in await m.conversations.history(U, c.id)] == list(range(31, 41))


async def test_unsummarized_messages_are_never_pruned():
    m = manager(
        settings=MemorySettings(retention=RetentionPolicy(max_messages_per_conversation=10))
    )
    c = await conversation(m)
    await fill(m, c.id, 30)

    assert await m.conversations.prune(U, c.id) == 0  # nothing summarized yet, nothing deleted
    assert (await m.conversations.get(U, c.id)).message_count == 30


async def test_clear_and_archive_stale():
    clock = Clock()
    m = manager(
        clock, MemorySettings(retention=RetentionPolicy(archive_conversations_after_days=30))
    )
    old, fresh = await conversation(m), await conversation(m)
    await fill(m, old.id, 20)
    await m.conversations.update_summary(U, old.id)
    assert await m.conversations.clear(U, old.id) == 20
    cleared = await m.conversations.get(U, old.id)
    assert (cleared.summary, cleared.summarized_through, cleared.message_count) == ("", 0, 0)

    clock.advance(days=45)
    await m.conversations.add_message(U, fresh.id, "user", "recent activity")
    assert await m.conversations.archive_stale(U) == 1
    assert [c.id for c in await m.conversations.list(U)] == [fresh.id]
    assert await manager().conversations.archive_stale(U) == 0  # no archiving without a policy


async def test_memory_can_be_disabled_per_conversation():
    m = manager()
    c = await conversation(m)

    assert not (await m.conversations.set_memory_enabled(U, c.id, False)).memory_enabled
    assert (await m.conversations.get(U, c.id)).memory_enabled is False


# ------------------------------------------------------------------ short-term memory


def test_short_term_memory_is_bounded_and_holds_context_and_tool_results():
    s = ShortTermMemory("c1", max_messages=3, max_tool_results=2)

    for i in range(5):
        s.add_message("user", f"m{i}")
        s.add_tool_result(
            f"tool{i}", server_id="notes", success=i % 2 == 0, preview="p" * 1000, max_preview=50
        )

    assert [c for _, c in s.recent_messages()] == ["m2", "m3", "m4"] and s.recent_messages(1) == [
        ("user", "m4")
    ]
    assert [t.tool_name for t in s.tool_results] == ["tool3", "tool4"] and len(
        s.tool_results[0].preview
    ) == 50

    s.set_context(request_id="r1", step="plan")
    s.set_context(step="execute")
    assert s.execution_context == {"request_id": "r1", "step": "execute"}
    s.clear_execution()
    assert s.execution_context == {} and s.tool_results == [] and len(s.recent_messages()) == 3
    s.clear()
    assert s.recent_messages() == []


def test_short_term_manager_evicts_least_recently_used_conversations():
    mgr = ShortTermMemoryManager(max_conversations=2)
    a, b = mgr.get("a"), mgr.get("b")
    mgr.get("a")  # a is now most recent
    mgr.get("c")

    assert "a" in mgr and "c" in mgr and "b" not in mgr and len(mgr) == 2
    assert mgr.get("a") is a and mgr.get("b") is not b
    assert mgr.drop("a") is True and mgr.drop("a") is False
