from __future__ import annotations

import json

import pytest
from support_agent import ScriptedProvider, ToolTransport, call, make_world, say, sdk_tool

from agent.core import Agent, AgentDefinition, ExecutionSettings, trim_messages
from agent.graph import AgentNodes, build_agent_graph
from agent.graph.nodes import NODE_ERROR, NODE_EXECUTE, NODE_FINAL, NODE_LIMIT, make_router
from agent.llm import (
    LLMAuthenticationError,
    LLMRateLimitError,
    Message,
    ModelConfig,
    Role,
    ToolCall,
)
from agent.tools import MCPToolBridge, ToolPermissions


def definition(**kwargs) -> AgentDefinition:
    execution = kwargs.pop("execution", {})
    return AgentDefinition(
        agent_id="helper",
        name="Helper",
        system_instructions="You are Nexus.",
        model=ModelConfig(provider="scripted", model="scripted-1"),
        execution=ExecutionSettings(**execution),
        **kwargs,
    )


async def setup(script, *, definition_=None):
    notes = ToolTransport(
        [
            sdk_tool("create_note", {"title": {"type": "string"}}, ["title"]),
            sdk_tool("search_notes", {"query": {"type": "string"}}),
        ]
    )
    github = ToolTransport([sdk_tool("search_repositories", {"query": {"type": "string"}})])
    container = await make_world({"notes": notes, "github": github})
    provider = ScriptedProvider(script)
    agent = Agent.create(definition_ or definition(), container, provider=provider)
    return agent, provider, container, notes, github


# ------------------------------------------------------------------ flow


async def test_plain_answer_without_tools():
    agent, provider, *_ = await setup([say("Hello!")])

    result = await agent.run("hi", request_id="req-1")

    assert result.response == "Hello!" and result.stop_reason == "completed" and result.succeeded
    assert result.iterations == 0 and result.tool_outcomes == [] and result.request_id == "req-1"
    messages, tools = provider.requests[0]
    assert messages[0].role is Role.SYSTEM and messages[0].content == "You are Nexus."
    assert messages[-1].content == "hi"
    assert {t.name for t in tools} == {
        "notes__create_note",
        "notes__search_notes",
        "github__search_repositories",
    }
    assert [m.role for m in result.new_messages] == [Role.USER, Role.ASSISTANT]


async def test_tool_call_result_is_returned_to_the_model():
    agent, provider, _, notes, _ = await setup(
        [call(("notes__create_note", {"title": "Hi"})), say("Created it.")]
    )

    result = await agent.run("make a note")

    assert result.response == "Created it." and result.iterations == 1
    assert notes.calls == [("create_note", {"title": "Hi"})]
    second_request, _ = provider.requests[1]
    assistant, tool = second_request[-2], second_request[-1]
    assert assistant.role is Role.ASSISTANT and assistant.tool_calls[0].name == "notes__create_note"
    assert (
        tool.role is Role.TOOL and tool.tool_call_id == "c0" and tool.name == "notes__create_note"
    )
    assert json.loads(tool.content)["success"] is True
    assert (result.tool_outcomes[0].server_id, result.tool_outcomes[0].tool_name) == (
        "notes",
        "create_note",
    )


async def test_multiple_tool_calls_in_one_response_across_servers():
    agent, provider, _, notes, github = await setup(
        [
            call(
                ("github__search_repositories", {"query": "mcp"}),
                ("notes__create_note", {"title": "T"}),
            ),
            say("Done."),
        ]
    )

    result = await agent.run("research and note it")

    assert github.calls == [("search_repositories", {"query": "mcp"})]
    assert notes.calls == [("create_note", {"title": "T"})]
    follow_up, _ = provider.requests[1]
    assert [m.role for m in follow_up[-3:]] == [Role.ASSISTANT, Role.TOOL, Role.TOOL]
    assert [m.tool_call_id for m in follow_up[-2:]] == ["c0", "c1"]  # every call answered, in order
    assert result.iterations == 1 and len(result.tool_outcomes) == 2


async def test_multi_step_chain_and_usage_accumulation():
    agent, provider, _, notes, _ = await setup(
        [
            call(("notes__search_notes", {"query": "a"}), tokens=10),
            call(("notes__create_note", {"title": "b"}), tokens=20),
            say("All done", tokens=30),
        ]
    )

    result = await agent.run("go")

    assert result.iterations == 2 and len(provider.requests) == 3
    assert result.usage.prompt_tokens == 60 and result.usage.completion_tokens == 3
    assert [c[0] for c in notes.calls] == ["search_notes", "create_note"]


async def test_iteration_limit_forces_a_final_answer_without_tools():
    always_tools = [call(("notes__search_notes", {"query": "a"}))] * 3
    agent, provider, _, notes, _ = await setup(
        [*always_tools, say("Best effort answer")],
        definition_=definition(execution={"max_iterations": 2}),
    )

    result = await agent.run("loop forever")

    assert result.stop_reason == "max_iterations" and result.response == "Best effort answer"
    assert len(notes.calls) == 2 and len(provider.requests) == 4
    last_messages, last_tools = provider.requests[-1]
    assert last_tools is None and "limit reached" in last_messages[-1].content.lower()
    assert result.succeeded


async def test_tool_call_cap_per_turn_still_answers_every_call():
    agent, provider, _, notes, _ = await setup(
        [
            call(("notes__search_notes", {"query": "1"}), ("notes__search_notes", {"query": "2"})),
            say("ok"),
        ],
        definition_=definition(execution={"max_tool_calls_per_turn": 1}),
    )

    result = await agent.run("two calls")

    assert len(notes.calls) == 1
    codes = [o.error_code for o in result.tool_outcomes]
    assert codes == [None, "TOOL_CALL_LIMIT"]
    assert len([m for m in provider.requests[1][0] if m.role is Role.TOOL]) == 2


# ------------------------------------------------------------------ guard rails


async def test_unauthorized_and_unregistered_tools_are_blocked_but_the_run_continues():
    agent, provider, _, notes, github = await setup(
        [
            call(("github__search_repositories", {"query": "x"}), ("notes__delete_everything", {})),
            say("I could not do that."),
        ],
        definition_=definition(tool_permissions=ToolPermissions(allow=("notes.*",))),
    )

    result = await agent.run("try")

    assert github.calls == [] and notes.calls == []
    assert [o.error_code for o in result.tool_outcomes] == [
        "TOOL_AUTHORIZATION_DENIED",
        "TOOL_NOT_FOUND",
    ]
    offered = {t.name for t in provider.requests[0][1]}
    assert offered == {"notes__create_note", "notes__search_notes"}
    assert result.response == "I could not do that."


async def test_invalid_arguments_are_fed_back_so_the_model_can_correct_itself():
    agent, provider, _, notes, _ = await setup(
        [
            call(("notes__create_note", {"title": 123})),
            call(("notes__create_note", {"title": "fixed"})),
            say("ok"),
        ]
    )

    result = await agent.run("note")

    assert notes.calls == [("create_note", {"title": "fixed"})]
    assert (
        result.tool_outcomes[0].error_code == "INVALID_ARGUMENTS"
        and result.tool_outcomes[1].success
    )


async def test_catalog_changes_between_runs_update_the_offered_tools():
    agent, provider, container, *_ = await setup([say("a"), say("b")])

    await agent.run("first")
    await container.servers.disconnect("github")
    await agent.run("second")

    first = {t.name for t in provider.requests[0][1]}
    second = {t.name for t in provider.requests[1][1]}
    assert "github__search_repositories" in first and "github__search_repositories" not in second


# ------------------------------------------------------------------ provider failures


@pytest.mark.parametrize(
    "error",
    [
        LLMRateLimitError("slow down", provider="scripted"),
        LLMAuthenticationError("bad key", provider="scripted"),
    ],
)
async def test_provider_failures_are_reported_not_raised(error):
    agent, provider, *_ = await setup([error])

    result = await agent.run("hi")

    assert result.stop_reason == "error" and not result.succeeded
    assert result.error["code"] == error.code and result.error["retryable"] is error.retryable
    assert result.response == ""


async def test_provider_failure_mid_workflow_keeps_executed_tool_results():
    agent, _, _, notes, _ = await setup(
        [
            call(("notes__create_note", {"title": "x"})),
            LLMRateLimitError("later", provider="scripted"),
        ]
    )

    result = await agent.run("go")

    assert result.stop_reason == "error" and len(notes.calls) == 1
    assert len(result.tool_outcomes) == 1 and result.error["code"] == "LLM_RATE_LIMITED"


async def test_failure_while_forcing_the_final_answer_is_reported():
    agent, *_ = await setup(
        [call(("notes__search_notes", {"query": "a"})), LLMRateLimitError("x", provider="s")],
        definition_=definition(execution={"max_iterations": 1}),
    )

    assert (await agent.run("go")).stop_reason == "error"


# ------------------------------------------------------------------ context & switching


async def test_history_is_used_and_new_messages_are_returned_for_persistence():
    agent, provider, *_ = await setup([say("Your name is Ada.")])
    history = [
        Message(role=Role.USER, content="I am Ada"),
        Message(role=Role.ASSISTANT, content="Hi Ada"),
    ]

    result = await agent.run("What is my name?", history=history)

    sent, _ = provider.requests[0]
    assert [m.content for m in sent[1:4]] == ["I am Ada", "Hi Ada", "What is my name?"]
    assert [m.content for m in result.new_messages] == ["What is my name?", "Your name is Ada."]
    assert len(history) == 2  # caller's list is not mutated


async def test_context_is_bounded_without_orphaning_tool_messages():
    history = []

    for i in range(30):
        history += [
            Message(role=Role.USER, content=f"q{i}"),
            Message(role=Role.ASSISTANT, tool_calls=[ToolCall(id=f"t{i}", name="n", arguments={})]),
            Message(role=Role.TOOL, tool_call_id=f"t{i}", name="n", content="{}"),
            Message(role=Role.ASSISTANT, content=f"a{i}"),
        ]

    trimmed = trim_messages(history, 10)

    assert len(trimmed) <= 10 and trimmed[0].role is Role.USER
    assert trimmed[-1].content == "a29"
    assert trimmed[0].role is not Role.TOOL

    only_tools = [Message(role=Role.USER, content="u"), *history[1:3] * 20]
    assert trim_messages(only_tools, 4)[0].content == "u"  # latest user request is never lost

    agent, provider, *_ = await setup(
        [say("ok")], definition_=definition(execution={"max_context_messages": 8})
    )
    await agent.run("new", history=history)
    sent, _ = provider.requests[0]
    assert len(sent) <= 9 and sent[0].role is Role.SYSTEM and sent[1].role is Role.USER


async def test_provider_switching_keeps_agent_behaviour():
    agent, _, container, notes, _ = await setup([say("from A")])
    other = ScriptedProvider([call(("notes__create_note", {"title": "t"})), say("from B")])

    result = await agent.with_provider(other).run("hi")

    assert result.response == "from B" and notes.calls and agent.provider is not other


async def test_empty_input_is_rejected():
    agent, *_ = await setup([])

    with pytest.raises(ValueError):
        await agent.run("   ")


async def test_aclose_closes_the_provider():
    agent, provider, *_ = await setup([])
    await agent.aclose()
    assert provider.closed


# ------------------------------------------------------------------ definition & nodes


@pytest.mark.parametrize("agent_id", ["Bad Id", "", "has/slash"])
def test_definition_rejects_invalid_agent_ids(agent_id):
    with pytest.raises(ValueError):
        AgentDefinition(agent_id=agent_id, name="A", model=ModelConfig(provider="x", model="m"))


@pytest.mark.parametrize(
    "settings", [{"max_iterations": 0}, {"max_iterations": 99}, {"tool_timeout_seconds": 0}]
)
def test_execution_settings_bounds(settings):
    with pytest.raises(ValueError):
        ExecutionSettings(**settings)


def test_router_decisions():
    route = make_router(2)

    assert route({"error": {"code": "x"}, "pending_calls": [1], "iterations": 0}) == NODE_ERROR
    assert route({"pending_calls": [], "iterations": 0}) == NODE_FINAL
    assert route({"pending_calls": [1], "iterations": 1}) == NODE_EXECUTE
    assert route({"pending_calls": [1], "iterations": 2}) == NODE_LIMIT


async def test_nodes_are_independently_testable():
    agent, provider, container, notes, _ = await setup([say("x")])
    nodes = AgentNodes(agent.definition, provider, agent.bridge)

    state = await nodes.user_input({"user_input": "hello", "messages": []})
    assert state["messages"][0].content == "hello" and state["iterations"] == 0

    state = {**state, **await nodes.prepare_context(state)}
    assert state["prepared"][0].role is Role.SYSTEM and state["tool_specs"]

    state = {**state, **await nodes.invoke_llm(state)}
    assert state["response"].content == "x"
    assert (await nodes.detect_tool_calls(state))["pending_calls"] == []

    state = {**state, "pending_calls": [ToolCall(id="1", name="notes__search_notes", arguments={})]}
    update = await nodes.execute_tools(state)
    assert update["turn_outcomes"][0].success and notes.calls == [("search_notes", {})]


def test_graph_compiles_with_all_workflow_nodes():
    graph = build_agent_graph(
        AgentNodes(definition(), ScriptedProvider([]), MCPToolBridge(None, None)),  # type: ignore[arg-type]
        max_iterations=3,
    )
    nodes = set(graph.get_graph().nodes)

    assert {
        "user_input",
        "prepare_context",
        "invoke_llm",
        "detect_tool_calls",
        "execute_tools",
        "process_results",
        "final_response",
    } <= nodes
