from __future__ import annotations

import time

import pytest
from support_agent import ToolTransport, make_world, sdk_tool, tool_error
from support_orch import ScriptedProvider, plan, step, text

from agent.llm import LLMRateLimitError, Message, Role
from agent.orchestration import (
    Intent,
    OrchestrationNodes,
    OrchestrationSettings,
    Orchestrator,
    PlanStep,
    StepKind,
    StepState,
    TaskPlan,
    WorkflowStatus,
)
from agent.orchestration.nodes import (
    N_ERROR,
    N_EXECUTE,
    N_FINAL,
    N_REPLAN,
    N_SCHEDULE,
    N_SELECT,
    N_TASKS,
    make_routers,
    overall_status,
)

NEEDS_TOOLS = Intent(goal="do the thing", needs_tools=True)
TASKS = TaskPlan.model_validate({"tasks": [{"id": "t1", "description": "do the thing"}]})


def transports():
    return {
        "notes": ToolTransport(
            [
                sdk_tool(
                    "create_note",
                    {"title": {"type": "string"}, "content": {"type": "string"}},
                    ["title"],
                ),
                sdk_tool("search_notes", {"query": {"type": "string"}}),
            ]
        ),
        "github": ToolTransport(
            [sdk_tool("search_repositories", {"query": {"type": "string"}}, ["query"])]
        ),
        "calendar": ToolTransport(
            [sdk_tool("get_events", {"calendar_id": {"type": "string"}}, ["calendar_id"])]
        ),
    }


async def build(structured, texts=None, *, settings=None, servers=None):
    servers = servers or transports()
    container = await make_world(servers)
    provider = ScriptedProvider(texts=texts or [text("final answer")], structured=structured)
    return Orchestrator.create(container, provider, settings=settings), provider, container, servers


def flow(*selected_plans, intent=NEEDS_TOOLS):
    return [intent, TASKS, *selected_plans]


# ------------------------------------------------------------------ routing & tool selection


async def test_requests_that_need_no_tools_skip_planning_entirely():
    orch, provider, _, servers = await build(
        [Intent(goal="greet", needs_tools=False)], [text("Hello there!")]
    )

    result = await orch.run("hi!", request_id="r-1")

    assert (
        result.response == "Hello there!"
        and result.stop_reason == "no_tools_needed"
        and result.succeeded
    )
    assert result.plan_ids == [] and result.report is None and result.aggregated is None
    assert (
        len(provider.structured_requests) == 1
    )  # intent only: no task planning, no tool selection
    assert all(t.calls == [] for t in servers.values())
    assert provider.text_requests[0][1] is None  # model not offered tools directly


async def test_only_the_needed_tool_is_called_out_of_many():
    p = plan(step("s1", "github__search_repositories", arguments={"query": "mcp"}))
    orch, provider, _, servers = await build(flow(p))

    result = await orch.run("find mcp repositories")

    assert result.stop_reason == "completed" and result.aggregated.servers_used == ["github"]
    assert servers["github"].calls == [("search_repositories", {"query": "mcp"})]
    assert servers["notes"].calls == [] and servers["calendar"].calls == []
    assert result.intent == NEEDS_TOOLS and result.layers == [["s1"]]


async def test_intent_with_no_connected_tools_cannot_need_tools():
    orch, *_ = await build(
        [Intent(goal="x", needs_tools=True)], servers={"notes": transports()["notes"]}
    )
    container = orch.tools._catalog
    await container.remove_server_tools("notes")  # no tools left in the catalog

    result = await orch.run("do something")

    assert result.stop_reason == "no_tools_needed"


async def test_empty_selection_means_answer_directly():
    orch, provider, _, servers = await build(flow(plan()), [text("No tools were required.")])

    result = await orch.run("what is 2+2?")

    assert result.response == "No tools were required." and result.stop_reason == "no_tools_needed"
    assert all(t.calls == [] for t in servers.values())


# ------------------------------------------------------------------ multi-server execution


async def test_cross_server_workflow_runs_independent_steps_in_parallel_then_joins():
    servers = transports()
    servers["github"].delay = servers["calendar"].delay = 0.12
    p = plan(
        step("s1", "github__search_repositories", arguments={"query": "mcp"}),
        step("s2", "calendar__get_events", arguments={"calendar_id": "primary"}),
        step(
            "s3",
            "notes__create_note",
            arguments={
                "title": {"$template": "{s1.tool} + {s2.tool}"},
                "content": {"$template": "{s1.ok}"},
            },
        ),
    )
    orch, provider, _, _ = await build(flow(p), servers=servers)

    started = time.perf_counter()
    result = await orch.run("research and note")
    elapsed = time.perf_counter() - started

    assert result.layers == [["s1", "s2"], ["s3"]]
    assert elapsed < 0.22  # two 0.12s calls overlapped, not 0.24s
    assert servers["notes"].calls == [
        ("create_note", {"title": "search_repositories + get_events", "content": "true"})
    ]
    assert result.aggregated.servers_used == ["calendar", "github", "notes"]
    digest = provider.text_requests[-1][0][-1].content
    assert all(f"## {s}" in digest for s in ("github", "calendar", "notes"))
    assert result.report.request_id == result.request_id


async def test_identical_calls_in_a_plan_run_once():
    p = plan(
        step("s1", "github__search_repositories", arguments={"query": "mcp"}),
        step("s2", "github__search_repositories", arguments={"query": "mcp"}),
    )
    orch, _, _, servers = await build(flow(p))

    result = await orch.run("search twice")

    assert len(servers["github"].calls) == 1 and result.records["s2"].deduplicated


async def test_concurrency_limit_from_settings_is_applied():
    servers = transports()
    servers["notes"].delay = 0.06

    def three():
        return plan(
            *[step(f"s{i}", "notes__search_notes", arguments={"query": str(i)}) for i in range(3)]
        )

    serial, *_ = await build(
        flow(three()), servers=servers, settings=OrchestrationSettings(max_concurrency=1)
    )
    started = time.perf_counter()
    await serial.run("go")
    assert time.perf_counter() - started >= 0.17

    servers2 = transports()
    servers2["notes"].delay = 0.06
    wide, *_ = await build(
        flow(three()), servers=servers2, settings=OrchestrationSettings(max_concurrency=3)
    )
    started = time.perf_counter()
    await wide.run("go")
    assert time.perf_counter() - started < 0.15


async def test_synthesize_steps_use_the_model_and_feed_later_steps():
    p = plan(
        step("s1", "github__search_repositories", arguments={"query": "mcp"}),
        PlanStep(
            id="s2",
            kind=StepKind.SYNTHESIZE,
            instruction="Summarize",
            arguments={"in": {"$ref": "s1"}},
        ),
        step(
            "s3",
            "notes__create_note",
            arguments={"title": "T", "content": {"$ref": "s2", "path": "text"}},
        ),
    )
    orch, provider, _, servers = await build(
        flow(p), [text("model written summary", 7), text("done", 5)]
    )

    result = await orch.run("summarize into a note")

    assert servers["notes"].calls == [
        ("create_note", {"title": "T", "content": "model written summary"})
    ]
    assert result.usage.prompt_tokens == 12  # synthesis + final answer are both counted
    assert "Summarize" in provider.text_requests[0][0][-1].content


async def test_failed_synthesis_is_a_contained_step_failure():
    p = plan(
        PlanStep(id="s1", kind=StepKind.SYNTHESIZE, instruction="x"),
        step("s2", "notes__create_note", deps=["s1"], arguments={"title": "T"}),
    )
    orch, *_ = await build(
        flow(p),
        [LLMRateLimitError("slow", provider="p"), text("could not summarize")],
        settings=OrchestrationSettings(max_replans=0),
    )

    result = await orch.run("go")

    assert (
        result.records["s1"].error_code == "SYNTHESIS_FAILED"
        and result.records["s2"].state is StepState.SKIPPED
    )
    assert result.stop_reason == "failed" and result.response == "could not summarize"


# ------------------------------------------------------------------ plan validation loop


async def test_invalid_plan_is_repaired_with_feedback():
    bad = plan(step("s1", "github__delete_everything"))
    good = plan(step("s1", "github__search_repositories", arguments={"query": "mcp"}))
    orch, provider, _, servers = await build(flow(bad, good))

    result = await orch.run("find repos")

    assert result.stop_reason == "completed" and servers["github"].calls
    feedback = provider.structured_requests[3][0][-1].content
    assert (
        "previous plan was rejected" in feedback
        and "unknown tool 'github__delete_everything'" in feedback
    )


async def test_hopeless_planning_ends_with_a_clear_error_and_runs_nothing():
    bad = plan(step("s1", "github__nope"))
    orch, _, _, servers = await build(
        flow(bad, bad), settings=OrchestrationSettings(max_plan_attempts=2)
    )

    result = await orch.run("find repos")

    assert result.stop_reason == "error" and not result.succeeded and result.response == ""
    assert result.error["code"] == "PLAN_INVALID" and "unknown tool" in result.error["message"]
    assert all(t.calls == [] for t in servers.values())


async def test_schema_violations_in_arguments_are_caught_before_execution():
    bad = plan(step("s1", "github__search_repositories", arguments={}))  # required 'query' missing
    good = plan(step("s1", "github__search_repositories", arguments={"query": "x"}))
    orch, provider, _, servers = await build(flow(bad, good))

    await orch.run("go")

    assert "missing required argument 'query'" in provider.structured_requests[3][0][-1].content
    assert servers["github"].calls == [("search_repositories", {"query": "x"})]


# ------------------------------------------------------------------ partial failure & recovery


def failing_calendar():
    servers = transports()
    servers["calendar"].handlers["get_events"] = lambda args: (
        tool_error("calendar down")
        if args.get("calendar_id") == "primary"
        else __import__("types").SimpleNamespace(
            is_error=False, structured_content={"ok": True, "events": 2}, content=[]
        )
    )
    return servers


async def test_partial_failure_is_reported_honestly_when_recovery_is_disabled():
    p = plan(
        step("s1", "github__search_repositories", arguments={"query": "mcp"}),
        step("s2", "calendar__get_events", arguments={"calendar_id": "primary"}),
        step("s3", "notes__create_note", deps=["s2"], arguments={"title": "T"}),
    )
    orch, provider, _, servers = await build(
        flow(p),
        [text("Found repos, but the calendar failed.")],
        servers=failing_calendar(),
        settings=OrchestrationSettings(max_replans=0),
    )

    result = await orch.run("repos + calendar note")

    assert result.stop_reason == "partial" and not result.succeeded and result.replans == 0
    assert result.report.status is WorkflowStatus.PARTIAL
    assert {k: r.state for k, r in result.records.items()} == {
        "s1": StepState.SUCCEEDED,
        "s2": StepState.FAILED,
        "s3": StepState.SKIPPED,
    }
    assert servers["notes"].calls == []  # the dependent write never happened
    digest = provider.text_requests[-1][0][-1].content
    assert "[failed] s2" in digest and "calendar down" in digest and "[skipped] s3" in digest
    assert result.response == "Found repos, but the calendar failed."


async def test_failed_steps_are_recovered_by_a_replan_that_reuses_completed_work():
    first = plan(
        step("s1", "notes__search_notes", arguments={"query": "plans"}),
        step("s2", "calendar__get_events", arguments={"calendar_id": "primary"}),
        step("s3", "notes__create_note", deps=["s2"], arguments={"title": "T"}),
    )
    recovery = plan(
        step("r1", "calendar__get_events", arguments={"calendar_id": "alt"}),
        step(
            "r2",
            "notes__create_note",
            arguments={
                "title": {"$template": "found {r1.events} events"},
                "content": {"$ref": "s1", "path": "tool"},
            },
        ),
        inputs=["s1"],
    )
    orch, provider, _, servers = await build(flow(first, recovery), servers=failing_calendar())

    result = await orch.run("summarize my calendar into a note")

    assert result.stop_reason == "completed" and result.succeeded and result.replans == 1
    assert len(servers["notes"].calls) == 2  # s1 search (once) + r2 create; never re-ran s1
    assert servers["notes"].calls[1] == (
        "create_note",
        {"title": "found 2 events", "content": "search_notes"},
    )
    assert result.aggregated.failed == {} and result.aggregated.recoveries == 1
    assert set(result.records) == {"s1", "s2", "s3", "r1", "r2"}  # full history kept
    assert result.report.plan_id == result.plan_ids[-1] and len(result.plan_ids) == 2
    recovery_prompt = provider.structured_requests[3][0][-1].content
    assert (
        "RECOVERY" in recovery_prompt
        and "calendar down" in recovery_prompt
        and "- s1:" in recovery_prompt
    )


async def test_recovery_is_bounded_by_max_replans():
    first = plan(step("s1", "calendar__get_events", arguments={"calendar_id": "primary"}))
    again = plan(step("r1", "calendar__get_events", arguments={"calendar_id": "primary"}))
    orch, provider, *_ = await build(flow(first, again), servers=failing_calendar())

    result = await orch.run("calendar")

    assert result.replans == 1 and result.stop_reason == "failed"
    assert len(provider.structured_requests) == 4  # intent, tasks, plan, ONE recovery plan


async def test_recovery_can_decide_nothing_more_can_be_done():
    first = plan(step("s1", "calendar__get_events", arguments={"calendar_id": "primary"}))
    orch, provider, *_ = await build(
        flow(first, plan()), [text("I could not read your calendar.")], servers=failing_calendar()
    )

    result = await orch.run("calendar")

    assert result.stop_reason == "failed" and result.response == "I could not read your calendar."
    assert result.records["s1"].state is StepState.FAILED


async def test_recovery_plans_cannot_reuse_ids_or_unknown_inputs():
    first = plan(step("s1", "calendar__get_events", arguments={"calendar_id": "primary"}))
    reuse = plan(step("s1", "calendar__get_events", arguments={"calendar_id": "alt"}))
    unknown = plan(step("r1", "notes__search_notes", arguments={"query": "x"}), inputs=["ghost"])
    fixed = plan(step("r2", "calendar__get_events", arguments={"calendar_id": "alt"}))
    orch, provider, *_ = await build(
        flow(first, reuse, unknown, fixed),
        servers=failing_calendar(),
        settings=OrchestrationSettings(max_plan_attempts=3),
    )

    result = await orch.run("calendar")

    assert result.stop_reason == "completed"
    seen = " ".join(m.content for req in provider.structured_requests[4:] for m in req[0])
    assert "already used by an earlier step" in seen and "unknown completed step 'ghost'" in seen


async def test_step_timeouts_produce_a_timed_out_partial_result():
    servers = transports()
    servers["github"].delay = 2.0
    p = plan(
        step("s1", "github__search_repositories", arguments={"query": "x"}),
        step("s2", "notes__search_notes", arguments={"query": "y"}),
    )
    orch, *_ = await build(
        flow(p),
        servers=servers,
        settings=OrchestrationSettings(default_tool_timeout_seconds=0.05, max_replans=0),
    )

    started = time.perf_counter()
    result = await orch.run("slow")

    assert time.perf_counter() - started < 1.5
    assert result.records["s1"].state is StepState.TIMED_OUT and result.records["s2"].succeeded
    assert result.stop_reason == "partial"


# ------------------------------------------------------------------ LLM failures


async def test_model_failure_during_intent_is_reported():
    orch, *_ = await build([LLMRateLimitError("slow down", provider="p")])

    result = await orch.run("hello")

    assert result.stop_reason == "error" and result.error["code"] == "LLM_RATE_LIMITED"
    assert result.error["retryable"] is True and result.response == ""


@pytest.mark.parametrize("fail_at", ["tasks", "select"])
async def test_model_failure_during_planning_is_reported(fail_at):
    steps = {
        "tasks": [NEEDS_TOOLS, LLMRateLimitError("x", provider="p")],
        "select": [NEEDS_TOOLS, TASKS, LLMRateLimitError("x", provider="p")],
    }[fail_at]
    orch, _, _, servers = await build(steps)

    result = await orch.run("go")

    assert result.stop_reason == "error" and result.error["code"] == "LLM_RATE_LIMITED"
    assert all(t.calls == [] for t in servers.values())


async def test_work_is_not_lost_when_the_final_summary_call_fails():
    p = plan(step("s1", "github__search_repositories", arguments={"query": "mcp"}))
    orch, *_ = await build(flow(p), [LLMRateLimitError("x", provider="p")])

    result = await orch.run("find repos")

    assert result.stop_reason == "completed" and result.error["code"] == "LLM_RATE_LIMITED"
    assert "[succeeded] s1" in result.response  # raw results are returned instead of nothing


# --------------------------------------------------------- dynamic catalog, discovery, context


async def test_tools_added_or_removed_between_runs_are_respected():
    uses_github = plan(step("s1", "github__search_repositories", arguments={"query": "x"}))
    uses_notes = plan(step("s1", "notes__search_notes", arguments={"query": "x"}))
    orch, provider, container, servers = await build(
        [*flow(uses_github), *flow(uses_github, uses_notes)], [text("one"), text("two")]
    )

    assert (await orch.run("first")).stop_reason == "completed"

    await container.servers.disconnect("github")
    result = await orch.run("second")

    intent_prompt = provider.structured_requests[3][0][-1].content
    assert "github" not in intent_prompt  # overview reflects the disconnect
    assert (
        "unknown tool 'github__search_repositories'"
        in provider.structured_requests[6][0][-1].content
    )
    assert result.stop_reason == "completed" and len(servers["github"].calls) == 1


async def test_large_catalogs_are_narrowed_to_relevant_candidates():
    servers = transports()
    servers["notes"].tools += [sdk_tool(f"extra_tool_{i}") for i in range(10)]
    intent = Intent(goal="search github repositories", needs_tools=True, server_hints=["github"])
    p = plan(step("s1", "github__search_repositories", arguments={"query": "x"}))
    orch, provider, *_ = await build(
        flow(p, intent=intent),
        servers=servers,
        settings=OrchestrationSettings(candidate_tool_limit=3),
    )

    await orch.run("search github repositories")

    listing = provider.structured_requests[2][0][-1].content.split("Available tools")[1]
    offered = [line for line in listing.splitlines() if "(" in line and " - [" in line]
    assert 1 <= len(offered) <= 3 and any("github__search_repositories" in line for line in offered)


async def test_conversation_history_reaches_intent_and_final_answer():
    p = plan(step("s1", "notes__search_notes", arguments={"query": "x"}))
    orch, provider, *_ = await build(flow(p))
    history = [
        Message(role=Role.USER, content="my project is Apollo"),
        Message(role=Role.ASSISTANT, content="Noted"),
    ]

    await orch.run("search my notes", history=history)

    assert "my project is Apollo" in provider.structured_requests[0][0][-1].content
    final_messages = provider.text_requests[-1][0]
    assert [m.content for m in final_messages[1:3]] == ["my project is Apollo", "Noted"]


async def test_each_run_gets_its_own_request_id_and_state():
    p = lambda: plan(step("s1", "notes__search_notes", arguments={"query": "x"}))  # noqa: E731
    orch, *_ = await build([*flow(p()), *flow(p())], [text("a"), text("b")])

    first, second = await orch.run("one"), await orch.run("two")

    assert first.request_id != second.request_id and first.report.request_id == first.request_id
    assert first.report.execution_id != second.report.execution_id


async def test_empty_requests_are_rejected():
    orch, *_ = await build([])

    with pytest.raises(ValueError):
        await orch.run("   ")


# ------------------------------------------------------------------ settings, routing, nodes


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_concurrency": 0},
        {"default_tool_timeout_seconds": 0},
        {"max_plan_attempts": 0},
        {"max_replans": 9},
        {"candidate_tool_limit": 0},
        {"surprise": 1},
    ],
)
def test_settings_validation(kwargs):
    with pytest.raises(ValueError):
        OrchestrationSettings(**kwargs)


def test_routing_decisions():
    after_intent, on_error_else, after_select, after_aggregate = make_routers(
        OrchestrationSettings(max_plan_attempts=2, max_replans=1)
    )
    ok_plan = plan(step("a"))

    assert after_intent({"intent": NEEDS_TOOLS}) == N_TASKS
    assert after_intent({"intent": Intent(goal="g", needs_tools=False)}) == N_FINAL
    assert after_intent({"intent": NEEDS_TOOLS, "error": {"code": "x"}}) == N_ERROR
    assert on_error_else(N_SELECT)({"error": {"code": "x"}}) == N_ERROR
    assert on_error_else(N_SELECT)({}) == N_SELECT

    assert after_select({"plan": ok_plan, "plan_issues": []}) == N_SCHEDULE
    assert after_select({"plan": plan(), "plan_issues": []}) == N_FINAL
    assert after_select({"plan": None, "plan_issues": ["bad"], "plan_attempts": 1}) == N_SELECT
    assert after_select({"plan": None, "plan_issues": ["bad"], "plan_attempts": 2}) == N_ERROR

    class Rep:  # minimal report double
        def __init__(self, failed):
            self.failed_ids = failed

    assert after_aggregate({"report": Rep(["a"]), "replans": 0}) == N_REPLAN
    assert after_aggregate({"report": Rep(["a"]), "replans": 1}) == N_FINAL
    assert after_aggregate({"report": Rep([]), "replans": 0}) == N_FINAL
    assert N_EXECUTE  # exported for graph builders


async def test_nodes_are_independently_testable():
    orch, provider, container, servers = await build([NEEDS_TOOLS])
    nodes = OrchestrationNodes(
        provider, orch.tools, orch._graph and None or _Runner(), orch.settings, "sys"
    )

    state = await nodes.user_request({"user_input": "x", "request_id": "r"})
    assert state["replans"] == 0 and state["records"] == {} and state["error"] is None

    state = {
        **state,
        "user_input": "x",
        "request_id": "r",
        **await nodes.understand_intent({"user_input": "x"}),
    }
    assert state["intent"].needs_tools

    from agent.orchestration import ExecutionScheduler  # noqa: F401

    tools = await nodes.discover_tools({**state, "tasks": TASKS})
    assert len(tools["candidates"]) == 4  # small catalog: everything is a candidate

    retired = await nodes.prepare_replan(
        {
            "report": type(
                "R",
                (),
                {
                    "steps": {
                        "a": type("S", (), {"succeeded": False})(),
                        "b": type("S", (), {"succeeded": True})(),
                    }
                },
            )(),
            "replans": 0,
            "superseded": [],
        }
    )
    assert (
        retired["superseded"] == ["a"] and retired["replans"] == 1 and retired["plan_attempts"] == 0
    )


class _Runner:
    async def execute(self, call, *, timeout_seconds=None):  # pragma: no cover - not used
        raise AssertionError("not used")


def test_overall_status_ignores_nothing_it_is_given():
    from agent.orchestration import StepRecord

    ok_ = StepRecord(step_id="a", state=StepState.SUCCEEDED)
    bad = StepRecord(step_id="b", state=StepState.FAILED)

    assert overall_status({}) is WorkflowStatus.SUCCEEDED
    assert overall_status({"a": ok_}) is WorkflowStatus.SUCCEEDED
    assert overall_status({"a": ok_, "b": bad}) is WorkflowStatus.PARTIAL
    assert overall_status({"b": bad}) is WorkflowStatus.FAILED
