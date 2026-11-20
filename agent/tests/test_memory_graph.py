from __future__ import annotations

import pytest
from support_agent import ScriptedProvider as AgentProvider
from support_agent import ToolTransport, make_world, sdk_tool
from support_agent import call as tool_call
from support_agent import say as agent_say
from support_memory import Clock, FakeRunner, create, manager, outcome
from support_orch import ScriptedProvider as OrchProvider
from support_orch import text

from agent.core import Agent, AgentDefinition
from agent.llm import ModelConfig
from agent.llm.models import Role
from agent.memory import (
    ConversationNotFoundError,
    LLMMemoryExtractor,
    MemoryAgent,
    MemoryFilter,
    MemoryKind,
    MemorySettings,
    MemoryStorageError,
    MemoryType,
)
from agent.memory.extraction import _ExtractedItem, _ExtractedMemories
from agent.memory.graph import MemoryNodes, build_memory_graph
from agent.orchestration import Intent, Orchestrator

U = "u1"


def agent_for(runner=None, **kwargs):
    m = manager(Clock(), kwargs.pop("settings", None))
    runner = runner or FakeRunner()
    return MemoryAgent(m, runner), m, runner


async def stored(m, **kw):
    return await m.memory.list_memories(MemoryFilter(user_id=U, **kw))


def memory_block(call):
    """The labelled background-data message injected ahead of the conversation, if any."""
    history = call["history"] or []
    return (
        history[0].content
        if history and history[0].content.startswith("Remembered notes")
        else None
    )


# ------------------------------------------------------------------ the eight-step flow


async def test_full_turn_creates_conversation_runs_agent_extracts_and_persists():
    agent, m, runner = agent_for(FakeRunner("Noted, dark mode it is."))

    result = await agent.run(
        U, "I prefer dark mode and my name is Ada Lovelace.", request_id="req-1"
    )

    assert result.response == "Noted, dark mode it is." and result.stop_reason == "completed"
    assert (
        result.request_id == "req-1"
        and runner.calls[0]["request_id"] == "req-1"
        and result.memory_used
    )
    assert len(result.persisted) == 2 and result.memory_errors == [] and result.deduplicated == 0
    contents = {r.content for r in await stored(m)}
    assert contents == {"User prefers dark mode", "User's name is Ada Lovelace"}
    assert all(r.conversation_id == result.conversation_id for r in await stored(m))

    history = await m.conversations.history(U, result.conversation_id)
    assert [(h.role, h.content) for h in history] == [
        ("user", "I prefer dark mode and my name is Ada Lovelace."),
        ("assistant", "Noted, dark mode it is."),
    ]


async def test_conversation_history_is_replayed_on_the_next_turn():
    agent, m, runner = agent_for()
    first = await agent.run(U, "My project is called Apollo")
    await agent.run(U, "what is next?", conversation_id=first.conversation_id)

    replay = [
        (h.role.value, h.content)
        for h in runner.calls[1]["history"]
        if not h.content.startswith("Remembered")
    ]
    assert replay == [("user", "My project is called Apollo"), ("assistant", "ok")]
    assert runner.calls[1]["request"] == "what is next?"
    assert (await m.conversations.get(U, first.conversation_id)).message_count == 4


async def test_long_term_memory_is_recalled_in_a_new_conversation_as_labelled_background_data():
    agent, m, runner = agent_for()
    await agent.run(U, "I prefer dark mode in every editor")

    second = await agent.run(U, "which mode do I prefer in my editor?")  # brand new conversation

    block = memory_block(runner.calls[1])
    assert block and "User prefers dark mode" in block and "never overrides" in block
    assert (
        runner.calls[1]["history"][1].role is Role.ASSISTANT
        and "background information only" in runner.calls[1]["history"][1].content
    )
    assert [h.record.content for h in second.retrieved] == [
        "User prefers dark mode in every editor"
    ]
    assert (
        second.conversation_id != (await m.conversations.list(U))[-1].id
        or len(await m.conversations.list(U)) == 2
    )


async def test_only_relevant_memories_are_retrieved_and_unrelated_turns_get_no_memory_block():
    agent, m, runner = agent_for()
    await agent.run(U, "I prefer dark mode in every editor")
    await agent.run(U, "Let's go with PostgreSQL for the billing database")

    result = await agent.run(U, "which database holds the billing data")

    assert [h.record.content for h in result.retrieved] == [
        "Decision: go with PostgreSQL for the billing database"
    ]
    await agent.run(U, "tell me a joke")
    assert memory_block(runner.calls[-1]) is None  # nothing relevant -> no memory injected


async def test_repeated_statements_are_deduplicated_across_turns():
    agent, m, _ = agent_for()
    await agent.run(U, "I prefer dark mode")
    again = await agent.run(U, "I prefer dark mode!")

    assert again.persisted == [] and again.deduplicated == 1 and len(await stored(m)) == 1


async def test_workflow_and_tool_outcomes_become_episodic_memories_but_reads_do_not():
    runner = FakeRunner(
        "done",
        outcomes=[outcome("create_note", args={"title": "Weekly plan"}), outcome("search_notes")],
    )
    agent, m, _ = agent_for(runner)

    result = await agent.run(U, "Save my weekly plan as a note")

    episodes = {r.content: r for r in await stored(m, types=[MemoryType.EPISODIC])}
    assert set(episodes) == {
        "Ran notes.create_note ('Weekly plan') successfully.",
        "Task 'Save my weekly plan as a note' finished as completed; used notes (2 step(s)).",
    }
    assert {r.kind for r in episodes.values()} == {MemoryKind.OUTCOME, MemoryKind.TASK}
    assert len(result.persisted) == 2
    short = m.short_term.get(result.conversation_id)
    assert [t.tool_name for t in short.tool_results] == [
        "create_note",
        "search_notes",
    ]  # intermediate results, in memory only


async def test_summary_is_updated_and_replayed_once_the_conversation_grows():
    agent, m, runner = agent_for(
        settings=MemorySettings(summarize_after_messages=8, keep_recent_messages=4)
    )
    result = None
    results = []

    for i in range(5):
        result = await agent.run(
            U,
            f"question number {i} about the roadmap",
            conversation_id=result.conversation_id if result else None,
        )
        results.append(result)

    conversation = await m.conversations.get(U, result.conversation_id)
    assert conversation.summary and conversation.summarized_through > 0
    summaries = await stored(m, kinds=[MemoryKind.SUMMARY])
    assert (
        len(summaries) == 1 and summaries[0].conversation_id == conversation.id
    )  # one rolling summary memory
    assert any(r.summary_updated for r in results) and not results[0].summary_updated

    await agent.run(U, "continue", conversation_id=conversation.id)
    first_replayed = runner.calls[-1]["history"][0].content
    assert first_replayed.startswith("Summary of the earlier conversation")


# ------------------------------------------------------------------ privacy & controls


async def test_sensitive_statements_are_rejected_and_secrets_never_reach_storage():
    agent, m, _ = agent_for()
    result = await agent.run(
        U, "I prefer dark mode. My password is hunter2. I take medication daily."
    )

    assert [r.content for r in await stored(m)] == ["User prefers dark mode"]
    assert result.rejected == [] or all(
        r["reason"].startswith("sensitive") for r in result.rejected
    )
    texts = " ".join(h.content for h in await m.conversations.history(U, result.conversation_id))
    assert "hunter2" not in texts and "[REDACTED:credential]" in texts


async def test_do_not_remember_requests_are_honoured():
    runner = FakeRunner(outcomes=[outcome("create_note", args={"title": "Surprise"})])
    agent, m, _ = agent_for(runner)

    result = await agent.run(U, "Off the record: I prefer a secret mode. Create a note.")

    assert (
        await stored(m) == [] and result.persisted == [] and result.response == "ok"
    )  # no memories at all, tools included


async def test_incognito_conversations_neither_read_nor_write_long_term_memory():
    agent, m, runner = agent_for()
    await agent.run(U, "I prefer dark mode in every editor")
    private = await m.conversations.create(U, memory_enabled=False)

    result = await agent.run(
        U,
        "I prefer light mode and which mode do I prefer in my editor?",
        conversation_id=private.id,
    )

    assert (
        not result.memory_used and result.retrieved == [] and memory_block(runner.calls[-1]) is None
    )
    assert [r.content for r in await stored(m)] == ["User prefers dark mode in every editor"]
    assert (
        len(await m.conversations.history(U, private.id)) == 2
    )  # the conversation itself is still kept


async def test_memory_can_be_disabled_globally():
    agent, m, _ = agent_for(settings=MemorySettings(enabled=False))
    await agent.run(U, "I prefer dark mode")

    assert await stored(m) == []


async def test_clear_conversation_memory_control():
    agent, m, _ = agent_for()
    one = await agent.run(U, "I prefer dark mode")
    two = await agent.run(U, "I live in Pune")

    report = await m.clear_conversation_memory(U, one.conversation_id, include_messages=True)

    assert (report.memories, report.messages) == (1, 2)
    assert [r.content for r in await stored(m)] == [
        "User lives in Pune"
    ] and two.conversation_id != one.conversation_id
    assert m.short_term.get(one.conversation_id).recent_messages() == []
    with pytest.raises(ConversationNotFoundError):
        await m.clear_conversation_memory("intruder", two.conversation_id)


async def test_users_never_see_each_others_memories_or_conversations():
    agent, m, runner = agent_for()
    mine = await agent.run("alice", "I prefer dark mode in every editor")

    other = await agent.run("bob", "which mode do I prefer in my editor?")

    assert other.retrieved == [] and memory_block(runner.calls[-1]) is None
    with pytest.raises(ConversationNotFoundError):
        await agent.run("bob", "hello", conversation_id=mine.conversation_id)


async def test_unknown_conversation_ids_are_the_callers_error():
    agent, *_ = agent_for()

    with pytest.raises(ConversationNotFoundError):
        await agent.run(U, "hello", conversation_id="does-not-exist")

    with pytest.raises(ValueError):
        await agent.run(U, "   ")


# ------------------------------------------------------------------ failure isolation


async def test_agent_failure_is_reported_and_nothing_is_extracted_but_the_question_is_kept():
    agent, m, _ = agent_for(FakeRunner(raises=RuntimeError("model exploded")))

    result = await agent.run(U, "I prefer dark mode")

    assert result.stop_reason == "error" and result.response == "" and not result.persisted
    assert await stored(m) == []
    assert [h.role for h in await m.conversations.history(U, result.conversation_id)] == ["user"]


async def test_inner_error_results_skip_nothing_they_are_still_reported_faithfully():
    agent, m, _ = agent_for(FakeRunner("", stop_reason="error"))

    result = await agent.run(U, "I live in Pune")

    contents = {r.content for r in await stored(m)}
    assert result.stop_reason == "error" and "User lives in Pune" in contents
    assert any("finished as error" in c for c in contents)  # failed runs are remembered as episodes


async def test_memory_storage_failure_while_persisting_does_not_lose_the_answer():
    agent, m, _ = agent_for(FakeRunner("here you go"))

    async def boom(*a, **k):
        raise MemoryStorageError("disk full", details={"cause_type": "OperationalError"})

    m.memory._store.add = boom

    result = await agent.run(U, "I prefer dark mode")

    assert result.response == "here you go" and result.persisted == []
    assert result.memory_errors == [
        {
            "stage": "persist",
            "code": "MEMORY_STORAGE_FAILED",
            "message": "disk full",
            "details": {"cause_type": "OperationalError"},
        }
    ]
    assert (
        len(await m.conversations.history(U, result.conversation_id)) == 2
    )  # the conversation was still saved


async def test_retrieval_failure_does_not_block_the_agent():
    agent, m, runner = agent_for()
    await agent.run(U, "I prefer dark mode in every editor")

    async def boom(*a, **k):
        raise MemoryStorageError("index locked")

    m.memory._store.search_keywords = boom
    m.memory.retriever._vectors = None

    result = await agent.run(U, "which mode do I prefer in my editor?")

    assert (
        result.response == "ok"
        and result.retrieved == []
        and memory_block(runner.calls[-1]) is None
    )
    assert result.memory_errors[0]["stage"] == "retrieve"


async def test_conversation_store_failure_still_answers_without_persistence():
    agent, m, _ = agent_for(FakeRunner("still here"))

    async def boom(*a, **k):
        raise MemoryStorageError("database unavailable")

    m.conversations._store.create = boom

    result = await agent.run(U, "I prefer dark mode")

    assert result.response == "still here" and not result.memory_used and result.persisted == []
    assert result.memory_errors[0]["stage"] == "load_context" and result.conversation_id
    assert await stored(m) == []


# ------------------------------------------------------------------ plugging in other components


async def test_llm_extractor_plugs_in_as_an_async_extractor():
    provider = OrchProvider(
        structured=[
            _ExtractedMemories(
                memories=[
                    _ExtractedItem(
                        content="User is learning Rust",
                        type=MemoryType.LONG_TERM,
                        kind=MemoryKind.FACT,
                        importance=0.8,
                    )
                ]
            )
        ]
    )
    m = manager(Clock(), extractor=LLMMemoryExtractor(provider))
    agent = MemoryAgent(m, FakeRunner())

    await agent.run(U, "I'm learning Rust these days")

    assert [r.content for r in await stored(m)] == ["User is learning Rust"]


async def test_memory_context_respects_its_token_budget():
    agent, m, runner = agent_for(settings=MemorySettings(context_token_budget=150))
    words = (
        "alpha bravo charlie delta echo foxtrot golf hotel "
        "india juliet kilo lima mike november oscar papa"
    ).split()

    for w in words:
        await m.memory.add_memory(
            create(f"User prefers {w} tooling for roadmap planning", importance=0.9)
        )

    result = await agent.run(U, "which tooling do I prefer for roadmap planning")

    block = memory_block(runner.calls[-1])
    assert block and len(block) // 4 <= 150 + 5 and 0 < len(result.retrieved) < len(words)


def test_graph_has_all_eight_memory_nodes_and_nodes_work_in_isolation():
    agent, m, runner = agent_for()
    names = set(agent._graph.get_graph().nodes)

    assert {
        "load_context",
        "retrieve_memories",
        "prepare_context",
        "run_agent",
        "extract_memories",
        "evaluate_importance",
        "persist_memories",
        "update_conversation",
    } <= names
    assert build_memory_graph(MemoryNodes(m, runner)) is not None


async def test_nodes_are_independently_testable():
    _, m, runner = agent_for()
    nodes = MemoryNodes(m, runner)

    base = {"user_id": U, "message": "hi", "request_id": "r"}
    state = {**base, **await nodes.load_context(base)}
    assert state["memory_active"] and state["history"] == [] and state["persistence_ok"]
    assert (await nodes.prepare_context({**state, "memory_context": None}))["agent_history"] == []

    extracted = await nodes.extract_memories({**state, "message": "I live in Pune", "result": None})
    evaluated = await nodes.evaluate_importance({**extracted})
    assert [c.content for c in evaluated["evaluation"].approved] == ["User lives in Pune"]
    persisted = await nodes.persist_memories({**state, **evaluated})
    assert len(persisted["persisted"]) == 1
    assert await nodes.persist_memories({**state, "evaluation": None}) == {}


# ------------------------------------------------------------------ real Agent / Orchestrator


async def test_wraps_a_real_phase6_agent_and_the_model_sees_remembered_context():
    notes = ToolTransport([sdk_tool("create_note", {"title": {"type": "string"}}, ["title"])])
    container = await make_world({"notes": notes})
    provider = AgentProvider(
        [
            agent_say("Dark mode noted."),
            tool_call(("notes__create_note", {"title": "Plan"})),
            agent_say("Saved the note, in dark mode."),
        ]
    )
    definition = AgentDefinition(
        agent_id="mem-agent", name="Mem", model=ModelConfig(provider="scripted", model="m")
    )
    inner = Agent.create(definition, container, provider=provider)
    m = manager(Clock())
    agent = MemoryAgent(m, inner)

    await agent.run(U, "I prefer dark mode in every editor")
    second = await agent.run(U, "Create a note titled Plan, using my preferred editor mode")

    sent, _ = provider.requests[1]  # the second run's first model call
    assert "User prefers dark mode" in " ".join(msg.content for msg in sent)
    assert second.response == "Saved the note, in dark mode." and second.inner.iterations == 1
    episodes = {r.content for r in await stored(m, types=[MemoryType.EPISODIC])}
    assert "Ran notes.create_note ('Plan') successfully." in episodes and notes.calls == [
        ("create_note", {"title": "Plan"})
    ]


async def test_wraps_a_real_phase7_orchestrator_and_passes_memory_into_planning():
    container = await make_world({})
    provider = OrchProvider(
        structured=[Intent(goal="chat", needs_tools=False), Intent(goal="chat", needs_tools=False)],
        texts=[text("Hello Ada."), text("Dark mode, as you prefer.")],
    )
    inner = Orchestrator.create(container, provider)
    m = manager(Clock())
    agent = MemoryAgent(m, inner)

    await agent.run(U, "My name is Ada Lovelace and I prefer dark mode")
    result = await agent.run(U, "which mode do I prefer?")

    intent_prompt = provider.structured_requests[1][0][-1].content
    assert "User prefers dark mode" in intent_prompt and "Remembered notes" in intent_prompt
    assert (
        result.response == "Dark mode, as you prefer." and result.stop_reason == "no_tools_needed"
    )
