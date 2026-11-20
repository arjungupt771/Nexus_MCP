from __future__ import annotations

import pytest
from support_agent import ToolTransport, make_world, sdk_tool
from support_orch import ScriptedProvider, plan, step

from agent.llm import LLMRateLimitError, Message, Role
from agent.orchestration import (
    AggregatedResult,
    ExecutionPlan,
    Intent,
    IntentAnalyzer,
    PlanStep,
    PlanValidator,
    ResultAggregator,
    StepKind,
    StepRecord,
    StepState,
    TaskPlan,
    TaskPlanner,
    ToolSelector,
    UnifiedToolInterface,
    WorkflowStatus,
)
from agent.orchestration.planner import server_overview
from agent.orchestration.references import ReferenceResolutionError, find_refs, get_path, resolve
from agent.tools import ToolPermissions


async def world():
    notes = ToolTransport(
        [
            sdk_tool(
                "create_note", {"title": {"type": "string"}, "tags": {"type": "array"}}, ["title"]
            ),
            sdk_tool("search_notes", {"query": {"type": "string"}}),
        ]
    )
    github = ToolTransport(
        [sdk_tool("search_repositories", {"query": {"type": "string"}}, ["query"])]
    )
    calendar = ToolTransport(
        [sdk_tool("get_events", {"calendar_id": {"type": "string"}}, ["calendar_id"])]
    )
    return await make_world({"notes": notes, "github": github, "calendar": calendar})


async def interface(container, **kwargs):
    return UnifiedToolInterface(container.catalog, registry=container.registry, **kwargs)


# ------------------------------------------------------------------ 7.1 unified tool access


async def test_unified_view_lists_every_server_with_ownership():
    container = await world()
    tools = await (await interface(container)).list_tools()

    assert {(t.server_id, t.tool_name) for t in tools} >= {
        ("notes", "create_note"),
        ("github", "search_repositories"),
        ("calendar", "get_events"),
    }
    create = next(t for t in tools if t.tool_name == "create_note")
    assert create.spec_name == "notes__create_note" and create.qualified_name == "notes.create_note"
    assert create.server_name == "Notes Server" and create.required == ["title"]
    assert "title*: string" in create.signature()


async def test_grouping_lookup_and_specs():
    unified = await interface(await world())

    grouped = await unified.by_server()
    assert sorted(grouped) == ["calendar", "github", "notes"] and len(grouped["notes"]) == 2
    assert (
        await unified.get("github.search_repositories")
    ).spec_name == "github__search_repositories"
    assert (await unified.get("github__search_repositories")).tool_name == "search_repositories"
    assert await unified.get("nope") is None
    assert {s.name for s in await unified.specs()} >= {"notes__create_note", "calendar__get_events"}


async def test_permissions_filter_the_unified_view():
    container = await world()
    unified = await interface(
        container, permissions=ToolPermissions(allow=("notes.*",), deny=("notes.search_notes",))
    )

    assert [t.qualified_name for t in await unified.list_tools()] == ["notes.create_note"]
    assert await unified.get("github.search_repositories") is None


async def test_view_follows_servers_connecting_and_disconnecting():
    container = await world()
    unified = await interface(container)
    unified.attach(container.events)
    version = unified.version

    await container.servers.disconnect("github")
    assert "github" not in await unified.by_server() and unified.version > version

    await container.servers.connect("github")
    assert "github" in await unified.by_server()

    # A live server changes its tool list.
    transport = container.lifecycle._transports["notes"]
    transport.tools = [sdk_tool("only_one")]
    await container.servers.restart("notes")
    assert [t.tool_name for t in (await unified.by_server())["notes"]] == ["only_one"]


async def test_search_ranks_by_name_description_and_server_metadata():
    container = await world()
    await container.servers.update("github", {"tags": ["repositories", "code"], "description": "x"})
    unified = await interface(container)

    top = await unified.search("search github repositories for mcp")
    assert top[0].qualified_name == "github.search_repositories"
    assert (await unified.search("create a new note"))[0].qualified_name == "notes.create_note"
    assert (await unified.search("what is on my calendar events"))[0].server_id == "calendar"
    assert await unified.search("zzzz qqqq") == []
    assert [t.server_id for t in await unified.search("search", servers={"notes"})] == ["notes"]
    assert len(await unified.search("search note events repositories", limit=2)) == 2
    # Deterministic.
    assert [t.qualified_name for t in await unified.search("search")] == [
        t.qualified_name for t in await unified.search("search")
    ]


async def test_server_overview_mentions_servers_and_tools():
    text = server_overview(await (await interface(await world())).list_tools())
    assert "notes (Notes Server" in text and "create_note" in text and "github" in text
    assert server_overview([]) == "(no servers connected)"


# ------------------------------------------------------------------ 7.2 / 7.4 planning components


async def test_intent_analyzer_sends_overview_history_and_returns_intent():
    unified = await interface(await world())
    expected = Intent(goal="find repos", needs_tools=True, server_hints=["github"])
    provider = ScriptedProvider(structured=[expected])

    intent = await IntentAnalyzer(provider).analyze(
        "find repos",
        await unified.list_tools(),
        [Message(role=Role.USER, content="earlier question")],
    )

    messages, schema = provider.structured_requests[0]
    assert intent == expected and schema is Intent
    prompt = messages[-1].content
    assert "earlier question" in prompt and "github" in prompt and "find repos" in prompt


async def test_task_planner_and_selector_prompts():
    unified = await interface(await world())
    tools = await unified.list_tools()
    intent = Intent(goal="g", needs_tools=True)
    tasks = TaskPlan.model_validate({"tasks": [{"id": "t1", "description": "do it"}]})
    selected = plan(step("s1", "notes__create_note", arguments={"title": "x"}))
    provider = ScriptedProvider(structured=[tasks, selected, selected, selected])

    assert await TaskPlanner(provider).plan("req", intent, tools) == tasks

    selector = ToolSelector(provider)
    await selector.select("req", intent, tasks, tools)
    base = provider.structured_requests[1][0][-1].content
    assert (
        "t1: do it" in base
        and "notes__create_note(title*: string" in base
        and "RECOVERY" not in base
    )

    await selector.select("req", intent, tasks, tools, issues=["step 's1': unknown tool 'x'"])
    assert "previous plan was rejected" in provider.structured_requests[2][0][-1].content

    await selector.select(
        "req", intent, tasks, tools, completed={"a": '{"x":1}'}, failures={"b": "BOOM: bad"}
    )
    recovery = provider.structured_requests[3][0][-1].content
    assert "RECOVERY" in recovery and "- a:" in recovery and "BOOM: bad" in recovery


async def test_planning_errors_propagate_as_llm_errors():
    provider = ScriptedProvider(structured=[LLMRateLimitError("slow", provider="p")])

    with pytest.raises(LLMRateLimitError):
        await IntentAnalyzer(provider).analyze("x", [])


# ------------------------------------------------------------------ plan validation


async def validation_inputs():
    container = await world()
    return {t.qualified_name: t for t in await (await interface(container)).list_tools()}


async def test_valid_plans_pass_and_names_are_canonicalized():
    catalog = await validation_inputs()
    validator = PlanValidator()
    good = plan(
        step("a", "github.search_repositories", arguments={"query": "mcp"}),  # qualified form
        step(
            "b",
            "notes__create_note",
            deps=["a"],
            arguments={"title": {"$ref": "a", "path": "tool"}},
        ),
    )

    assert validator.validate(good, catalog) == []
    validator.canonicalize(good, catalog)
    assert good.steps[0].tool == "github__search_repositories"


@pytest.mark.parametrize(
    ("arguments", "tool", "expected"),
    [
        ({}, "github__search_repositories", "missing required argument 'query'"),
        ({"query": 5}, "github__search_repositories", "argument 'query' is invalid"),
        ({"title": "ok", "tags": "not-a-list"}, "notes__create_note", "argument 'tags' is invalid"),
    ],
)
async def test_argument_problems_are_reported_with_the_step_id(arguments, tool, expected):
    issues = PlanValidator().validate(
        plan(step("s1", tool, arguments=arguments)), await validation_inputs()
    )

    assert len(issues) == 1 and "step 's1'" in issues[0] and expected in issues[0]


async def test_unknown_tools_list_alternatives_and_synthesize_steps_are_skipped():
    p = plan(
        step("s1", "notes__rm_rf"),
        PlanStep(id="s2", kind=StepKind.SYNTHESIZE, instruction="x"),
    )
    issues = PlanValidator().validate(p, await validation_inputs())

    assert (
        len(issues) == 1
        and "unknown tool 'notes__rm_rf'" in issues[0]
        and "Available:" in issues[0]
    )


async def test_reference_valued_arguments_are_not_type_checked_early_but_presence_is():
    catalog = await validation_inputs()
    p = plan(
        step("a"),
        step("b", "github__search_repositories", deps=["a"], arguments={"query": {"$ref": "a"}}),
    )
    assert PlanValidator().validate(p, catalog) == [
        "step 'a': unknown tool 'srv__a'. Available: "
        + ", ".join(sorted({t.spec_name for t in catalog.values()}))
        + "."
    ]
    p2 = plan(
        step("a", "notes__search_notes"), step("b", "github__search_repositories", deps=["a"])
    )
    assert "missing required argument 'query'" in PlanValidator().validate(p2, catalog)[0]


async def test_step_limit_and_additional_properties_false():
    catalog = await validation_inputs()
    many = plan(*[step(f"s{i}", "notes__search_notes") for i in range(5)])
    assert "maximum is 3" in PlanValidator(max_steps=3).validate(many, catalog)[0]

    strict = next(iter(catalog.values())).model_copy(
        update={
            "input_schema": {
                "type": "object",
                "properties": {"a": {}},
                "additionalProperties": False,
            }
        }
    )
    issues = PlanValidator()._check_arguments("s", strict, {"a": 1, "b": 2})
    assert issues == [f"step 's': unknown argument 'b' for {strict.spec_name}."]


# ------------------------------------------------------------------ references


def test_reference_helpers():
    data = {"a": {"items": [{"n": "x"}, {"n": "y"}], "when": "2026-11-17T10:00:00Z"}}

    assert get_path(data["a"], "items.1.n") == "y" and get_path(data, None) is data
    assert resolve({"$ref": "a", "path": "items", "pluck": "n"}, data) == ["x", "y"]
    assert (
        resolve({"$ref": "a", "path": "when", "add_minutes": 45}, data)
        == "2026-11-17T10:45:00+00:00"
    )
    assert resolve({"k": [{"$template": "{a.items.0.n}-{a.items.1.n}"}], "lit": 1}, data) == {
        "k": ["x-y"],
        "lit": 1,
    }
    assert find_refs({"x": [{"$ref": "a"}], "y": {"$template": "{b.q} {c}"}}) == {"a", "b", "c"}


@pytest.mark.parametrize(
    "spec",
    [
        {"$ref": "missing"},
        {"$ref": "a", "path": "items.9"},
        {"$ref": "a", "path": "nope"},
        {"$ref": "a", "path": "items", "pluck": "zzz"},
        {"$ref": "a", "path": "when", "pluck": "n"},
        {"$ref": "a", "path": "items", "add_minutes": 5},
        {"$ref": "a", "wat": 1},
        {"$template": "{ghost.x}"},
    ],
)
def test_bad_references_raise_clear_errors(spec):
    data = {"a": {"items": [{"n": "x"}], "when": "2026-11-17T10:00:00Z"}}

    with pytest.raises(ReferenceResolutionError):
        resolve(spec, data)


# ------------------------------------------------------------------ aggregation


def record(step_id, state, *, server="notes", data=None, **kwargs):
    return StepRecord(
        step_id=step_id, state=state, server_id=server, tool=f"{server}__x", data=data, **kwargs
    )


def test_results_from_different_servers_are_grouped_and_summarized():
    steps = {
        s.id: s
        for s in plan(
            step("a", description="search"),
            step("b", description="write"),
            step("c", description="late"),
            step("d"),
        ).steps
    }
    records = {
        "a": record(
            "a", StepState.SUCCEEDED, server="github", data={"success": True, "data": {"n": 3}}
        ),
        "b": record(
            "b", StepState.SUCCEEDED, server="notes", data={"success": True, "data": {"id": 9}}
        ),
        "c": record("c", StepState.FAILED, server="notes", error_code="X", error_message="broke"),
        "d": record(
            "d", StepState.SKIPPED, server="notes", skip_reason="dependency 'c' did not succeed"
        ),
        "s": StepRecord(
            step_id="s", kind=StepKind.SYNTHESIZE, state=StepState.SUCCEEDED, data={"text": "hi"}
        ),
    }

    result = ResultAggregator().aggregate(steps, records, WorkflowStatus.PARTIAL)

    assert result.servers_used == ["github", "notes"] and "llm" in result.by_server
    assert result.succeeded == ["a", "b", "s"] and result.failed == {"c": "X: broke"}
    assert result.skipped == {"d": "dependency 'c' did not succeed"}
    digest = result.digest()
    assert "## github" in digest and '{"n":3}' in digest  # envelope stripped
    assert "problem: X: broke" in digest and "status: partial" in digest


def test_previews_and_digest_are_bounded():
    big = {"success": True, "data": {"blob": "x" * 5000}}
    result = ResultAggregator(max_preview_chars=200).aggregate(
        {"a": step("a")},
        {"a": record("a", StepState.SUCCEEDED, data=big)},
        WorkflowStatus.SUCCEEDED,
    )

    assert "[truncated]" in result.by_server["notes"][0].data_preview
    many = AggregatedResult(
        status=WorkflowStatus.SUCCEEDED, by_server={"s": result.by_server["notes"] * 50}
    )
    assert len(many.digest(max_chars=500)) <= 520

    with pytest.raises(ValueError):
        ResultAggregator(max_preview_chars=10)


def test_plan_models_serialize():
    p = ExecutionPlan.model_validate_json(plan(step("a"), step("b", deps=["a"])).model_dump_json())
    assert p.layers() == [["a"], ["b"]]
