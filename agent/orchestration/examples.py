"""Cross-server workflow plans built on the real registered tool contracts (7.7).

Tool names and argument shapes come from ``mcp_contract_schemas.json``
(github, calendar, notes). Each function returns a plain ``ExecutionPlan``;
run it with ``ExecutionScheduler`` or let the planner produce something similar.
"""

from __future__ import annotations

from agent.orchestration.models import ExecutionPlan, PlanStep, StepKind


def github_repositories_to_note(query: str, *, per_page: int = 3) -> ExecutionPlan:
    """GitHub search -> LLM summary -> Notes note (3 sequential steps, 2 servers)."""
    return ExecutionPlan(
        goal=f"Search GitHub for '{query}' and save the findings as a note",
        steps=[
            PlanStep(
                id="search",
                description=f"Search GitHub repositories for '{query}'",
                tool="github__search_repositories",
                arguments={"query": query, "per_page": per_page, "sort": "stars"},
            ),
            PlanStep(
                id="summary",
                description="Summarize the repositories",
                kind=StepKind.SYNTHESIZE,
                instruction=(
                    "Write a short markdown bullet list: one bullet per repository with its "
                    "full name, star count and purpose."
                ),
                arguments={"repositories": {"$ref": "search", "path": "data.items"}},
            ),
            PlanStep(
                id="note",
                description="Save the findings as a note",
                tool="notes__create_note",
                arguments={
                    "title": f"GitHub findings: {query}",
                    "content": {"$ref": "summary", "path": "text"},
                    "tags": ["github", "research"],
                },
            ),
        ],
    )


def calendar_events_to_note(calendar_id: str, time_min: str, time_max: str) -> ExecutionPlan:
    """Calendar events -> LLM summary -> Notes note."""
    return ExecutionPlan(
        goal="Summarize calendar events into a note",
        steps=[
            PlanStep(
                id="events",
                description="Retrieve calendar events",
                tool="calendar__get_events",
                arguments={
                    "calendar_id": calendar_id,
                    "time_min": time_min,
                    "time_max": time_max,
                    "max_results": 20,
                },
            ),
            PlanStep(
                id="summary",
                description="Summarize the events",
                kind=StepKind.SYNTHESIZE,
                instruction="Summarize these calendar events as a concise agenda, one line each.",
                arguments={"events": {"$ref": "events", "path": "data.items"}},
            ),
            PlanStep(
                id="note",
                description="Save the summary as a note",
                tool="notes__create_note",
                arguments={
                    "title": f"Calendar summary {time_min[:10]} to {time_max[:10]}",
                    "content": {"$ref": "summary", "path": "text"},
                    "tags": ["calendar", "summary"],
                },
            ),
        ],
    )


def issues_to_calendar_followup(
    owner: str,
    repo: str,
    calendar_id: str,
    time_min: str,
    time_max: str,
    *,
    duration_minutes: int = 30,
) -> ExecutionPlan:
    """Open issues + free slots (parallel) -> summary -> follow-up calendar event.

    ``issues`` and ``slots`` are independent and run concurrently on two
    different servers; ``summary`` waits for issues, ``event`` for both.
    """
    return ExecutionPlan(
        goal=f"Schedule a follow-up for open issues in {owner}/{repo}",
        steps=[
            PlanStep(
                id="issues",
                description=f"List open issues in {owner}/{repo}",
                tool="github__list_issues",
                arguments={"owner": owner, "repo": repo, "state": "open", "per_page": 5},
            ),
            PlanStep(
                id="slots",
                description="Find a free slot",
                tool="calendar__find_free_slots",
                arguments={
                    "time_min": time_min,
                    "time_max": time_max,
                    "calendar_ids": [calendar_id],
                    "duration_minutes": duration_minutes,
                    "max_slots": 3,
                },
            ),
            PlanStep(
                id="summary",
                description="Summarize the issues for the event description",
                kind=StepKind.SYNTHESIZE,
                instruction="List the issues as short bullets: number and title.",
                arguments={"issues": {"$ref": "issues", "path": "data"}},
            ),
            PlanStep(
                id="event",
                description="Create the follow-up event",
                tool="calendar__create_event",
                arguments={
                    "calendar_id": calendar_id,
                    "summary": f"Follow up: open issues in {owner}/{repo}",
                    "start": {"$ref": "slots", "path": "data.free_slots.0.start"},
                    "end": {
                        "$ref": "slots",
                        "path": "data.free_slots.0.start",
                        "add_minutes": duration_minutes,
                    },
                    "description": {"$ref": "summary", "path": "text"},
                },
            ),
        ],
    )
