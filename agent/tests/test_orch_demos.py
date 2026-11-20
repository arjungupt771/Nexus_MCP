"""7.7 cross-server workflows on the REAL GitHub / Calendar / Notes MCP servers."""

from __future__ import annotations

import json

from support_demo import demo_world
from support_orch import ScriptedProvider, text

from agent.orchestration import (
    ExecutionScheduler,
    Intent,
    Orchestrator,
    StepState,
    TaskPlan,
    UnifiedToolInterface,
    WorkflowStatus,
)
from agent.orchestration.examples import (
    calendar_events_to_note,
    github_repositories_to_note,
    issues_to_calendar_followup,
)
from agent.tools import MCPToolBridge
from core.models import ToolCallRequest

WINDOW = ("2026-11-17T08:00:00+00:00", "2026-11-17T18:00:00+00:00")


async def run_plan(world, plan, synth_reply="- summary bullet"):
    async def synthesize(instruction, inputs):
        return synth_reply

    bridge = MCPToolBridge(world.container.catalog, world.container.executor)
    scheduler = ExecutionScheduler(bridge, synthesizer=synthesize)
    return await scheduler.execute(plan, request_id="demo-1")


async def notes_matching(world, query):
    result = await world.container.executor.execute(
        ToolCallRequest(server_id="notes", tool_name="search_notes", arguments={"query": query})
    )
    assert result.success
    return json.dumps(result.data)


async def test_all_three_servers_expose_unified_collision_free_tools(tmp_path):
    world = await demo_world(tmp_path)
    tools = UnifiedToolInterface(world.container.catalog, registry=world.container.registry)
    names = {t.spec_name for t in await tools.list_tools()}

    assert {"github__search_repositories", "calendar__get_events", "notes__create_note"} <= names
    assert len({n.split("__")[0] for n in names}) == 3


async def test_github_repositories_to_note(tmp_path):
    world = await demo_world(tmp_path)

    report = await run_plan(
        world, github_repositories_to_note("mcp server"), "- acme/mcp-hub (900 stars)"
    )

    assert report.status is WorkflowStatus.SUCCEEDED
    endpoint, params = world.github.calls[0]
    assert (
        endpoint == "/search/repositories"
        and params["q"] == "mcp server"
        and params["per_page"] == 3
    )
    # The note really exists in the Notes MCP server, with LLM-written content.
    stored = await notes_matching(world, "GitHub findings")
    assert "acme/mcp-hub (900 stars)" in stored and "github" in stored
    assert (
        report.steps["note"].server_id == "notes" and report.steps["search"].server_id == "github"
    )


async def test_calendar_events_to_note(tmp_path):
    world = await demo_world(tmp_path)
    seen = {}

    async def synthesize(instruction, inputs):
        seen.update(inputs)
        return "Agenda: standup, design review"

    bridge = MCPToolBridge(world.container.catalog, world.container.executor)
    report = await ExecutionScheduler(bridge, synthesizer=synthesize).execute(
        calendar_events_to_note("primary", *WINDOW), request_id="demo-2"
    )

    assert report.status is WorkflowStatus.SUCCEEDED
    assert [e["summary"] for e in seen["events"]] == ["Standup", "Design review"]
    assert "Agenda: standup, design review" in await notes_matching(world, "Calendar summary")


async def test_issues_to_calendar_followup_runs_both_servers_in_parallel_then_joins(tmp_path):
    world = await demo_world(tmp_path)

    report = await run_plan(
        world,
        issues_to_calendar_followup("acme", "mcp-hub", "primary", *WINDOW, duration_minutes=30),
        "- #12 Crash on startup",
    )

    assert report.status is WorkflowStatus.SUCCEEDED
    assert world.github.calls[0][0] == "/repos/acme/mcp-hub/issues"
    ((calendar_id, event),) = world.calendar.created
    # Busy 09:00-10:00 inside the 08:00-18:00 window: the first free slot is 08:00-09:00,
    # so a 30-minute event is placed at its start (end computed with add_minutes).
    assert calendar_id == "primary"
    assert event["start"]["dateTime"] == "2026-11-17T08:00:00+00:00"
    assert event["end"]["dateTime"] == "2026-11-17T08:30:00+00:00"
    assert event["description"] == "- #12 Crash on startup"
    # Independent steps overlapped in time.
    issues, slots = report.steps["issues"], report.steps["slots"]
    assert issues.started_at < slots.finished_at and slots.started_at < issues.finished_at


async def test_partial_failure_across_servers_keeps_independent_branch(tmp_path):
    from mcp_servers.github.client import GitHubAPIError

    world = await demo_world(tmp_path)
    world.github.fail_with = GitHubAPIError("Repository not found", status_code=404)

    report = await run_plan(
        world, issues_to_calendar_followup("acme", "missing", "primary", *WINDOW)
    )

    states = {k: r.state for k, r in report.steps.items()}
    assert states == {
        "issues": StepState.FAILED,  # GitHub reported failure inside a *successful* MCP response
        "slots": StepState.SUCCEEDED,  # independent calendar branch still ran
        "summary": StepState.SKIPPED,
        "event": StepState.SKIPPED,  # never creates a half-baked event
    }
    assert report.status is WorkflowStatus.PARTIAL
    assert report.steps["issues"].error_code == "TOOL_REPORTED_GITHUB_API_ERROR"
    assert "Repository not found" in report.steps["issues"].error_message
    assert world.calendar.created == []


async def test_calendar_failure_prevents_note_creation(tmp_path):
    from mcp_servers.calendar.client import GoogleCalendarAPIError

    world = await demo_world(tmp_path)
    world.calendar.fail_events = GoogleCalendarAPIError("Calendar unavailable", status_code=503)

    report = await run_plan(world, calendar_events_to_note("primary", *WINDOW))

    assert report.steps["events"].state is StepState.FAILED
    assert report.steps["note"].state is StepState.SKIPPED
    assert "Calendar summary" not in await notes_matching(world, "Calendar summary")


async def test_duplicate_identical_search_runs_once_on_a_real_server(tmp_path):
    from agent.orchestration import ExecutionPlan, PlanStep

    world = await demo_world(tmp_path)
    plan = ExecutionPlan(
        steps=[
            PlanStep(id="a", tool="github__search_repositories", arguments={"query": "mcp"}),
            PlanStep(id="b", tool="github__search_repositories", arguments={"query": "mcp"}),
        ]
    )

    report = await run_plan(world, plan)

    assert len(world.github.calls) == 1 and report.steps["b"].deduplicated


async def test_full_orchestrator_workflow_with_a_scripted_model(tmp_path):
    """User request -> intent -> tasks -> selection -> parallel/dependent execution -> answer."""
    world = await demo_world(tmp_path)
    plan = github_repositories_to_note("mcp server")
    provider = ScriptedProvider(
        structured=[
            Intent(
                goal="Research MCP repos and save a note",
                needs_tools=True,
                server_hints=["github", "notes"],
            ),
            TaskPlan.model_validate(
                {
                    "tasks": [
                        {"id": "t1", "description": "search github"},
                        {"id": "t2", "description": "write note", "depends_on": ["t1"]},
                    ]
                }
            ),
            plan,
        ],
        texts=[text("- acme/mcp-hub"), text("I searched GitHub and saved a note.")],
    )
    orchestrator = Orchestrator.create(world.container, provider)

    result = await orchestrator.run(
        "Find top MCP repos on GitHub and save a note", request_id="rq-7"
    )

    assert (
        result.stop_reason == "completed"
        and result.response == "I searched GitHub and saved a note."
    )
    assert result.request_id == "rq-7" and result.report.request_id == "rq-7"
    assert result.layers == [["search"], ["summary"], ["note"]]
    assert result.aggregated.servers_used == ["github", "notes"]
    assert "acme/mcp-hub" in await notes_matching(world, "GitHub findings")
    # The final model call saw the aggregated cross-server results.
    final_prompt = provider.text_requests[-1][0][-1].content
    assert "## github" in final_prompt and "## notes" in final_prompt
