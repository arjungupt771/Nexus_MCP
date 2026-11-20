"""Workflow planning (7.2, 7.4): intent -> tasks -> tool selection.

Everything here is *planning only*: it talks to the LLM and produces plain
data (``Intent``, ``TaskPlan``, ``ExecutionPlan``). Nothing is executed.
"""

from __future__ import annotations

import json
from collections.abc import Mapping

from jsonschema import Draft202012Validator

from agent.llm.models import Message, Role
from agent.llm.provider import LLMProvider
from agent.orchestration.models import ExecutionPlan, Intent, StepKind, TaskPlan
from agent.orchestration.references import find_refs
from agent.orchestration.tools import ToolInfo

INTENT_PROMPT = """You are the intent analyzer of a multi-server tool orchestration system.
Decide what the user wants and whether external tools are REQUIRED.
Set needs_tools=false for greetings, general knowledge, opinions, or anything you can answer
without reading or changing external data. Avoid unnecessary tool use.
Only list server ids that appear in the server overview."""

TASK_PROMPT = """You break a user request into the smallest sensible set of tasks.
Rules: each task is one concrete action or lookup; list depends_on only when a task truly needs
another task's output; independent tasks must NOT depend on each other (they run in parallel);
use ids t1, t2, ...; do not mention specific tools; fewer tasks is better."""

SELECT_PROMPT = """You turn tasks into an executable plan using ONLY the tools listed.
Return JSON for an ExecutionPlan: {"goal": str, "steps": [...]}.
Each step: {"id": "s1", "description": str, "kind": "tool", "tool": "<exact tool name>",
"arguments": {...}, "depends_on": [...]}.
Rules:
- Use exact tool names from the list; never invent tools or arguments.
  Arguments marked * are required.
- Take literal argument values from the user's request. Do not guess values you were not given.
- To use a previous step's output, write {"$ref": "s1", "path": "data.items.0.name"} (dotted path,
  numeric segments index lists; add "pluck": "field" to extract a field from every list item) or
  {"$template": "text with {s1.data.total_count}"} for text. Tool results look like
  {"success": true, "data": ...}. Referenced steps become dependencies automatically.
- To write a summary or message from data, add {"id": "s2", "kind": "synthesize",
  "instruction": "...", "arguments": {"input": {"$ref": ...}}} and then
  {"$ref": "s2", "path": "text"}.
- Steps that do not need each other's output must not depend on each other (they run in parallel).
- Use the fewest steps that fulfil the request. If no tool is needed return {"steps": []}."""


def _messages(system: str, user: str) -> list[Message]:
    return [Message(role=Role.SYSTEM, content=system), Message(role=Role.USER, content=user)]


def server_overview(tools: list[ToolInfo]) -> str:
    """Short per-server summary used for intent analysis."""
    grouped: dict[str, list[ToolInfo]] = {}

    for tool in tools:
        grouped.setdefault(tool.server_id, []).append(tool)

    return (
        "\n".join(
            f"- {server_id} ({infos[0].server_name}"
            f"{', tags: ' + ', '.join(infos[0].tags) if infos[0].tags else ''}): "
            + ", ".join(t.tool_name for t in infos)
            for server_id, infos in sorted(grouped.items())
        )
        or "(no servers connected)"
    )


class IntentAnalyzer:
    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider

    async def analyze(
        self,
        request: str,
        tools: list[ToolInfo],
        history: list[Message] | None = None,
    ) -> Intent:
        context = ""

        if history:
            recent = history[-6:]
            context = (
                "Recent conversation:\n"
                + "\n".join(f"{m.role.value}: {m.content[:300]}" for m in recent if m.content)
                + "\n\n"
            )

        user = f"{context}Server overview:\n{server_overview(tools)}\n\nUser request:\n{request}"
        return await self._provider.generate_structured(_messages(INTENT_PROMPT, user), Intent)


class TaskPlanner:
    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider

    async def plan(self, request: str, intent: Intent, tools: list[ToolInfo]) -> TaskPlan:
        user = (
            f"Goal: {intent.goal}\nServer overview:\n{server_overview(tools)}\n\n"
            f"User request:\n{request}"
        )
        return await self._provider.generate_structured(_messages(TASK_PROMPT, user), TaskPlan)


class PlanValidator:
    """Checks a plan against the *current* tool catalog before anything runs."""

    def __init__(self, *, max_steps: int = 12) -> None:
        self._max_steps = max_steps

    def validate(
        self,
        plan: ExecutionPlan,
        tools: Mapping[str, ToolInfo],
    ) -> list[str]:
        lookup = {
            **{t.spec_name: t for t in tools.values()},
            **{t.qualified_name: t for t in tools.values()},
        }
        issues: list[str] = []

        if len(plan.steps) > self._max_steps:
            issues.append(f"plan has {len(plan.steps)} steps; the maximum is {self._max_steps}.")

        for step in plan.steps:
            if step.kind is StepKind.SYNTHESIZE:
                continue

            info = lookup.get(step.tool or "")

            if info is None:
                names = ", ".join(sorted({t.spec_name for t in lookup.values()})[:20])
                issues.append(f"step '{step.id}': unknown tool '{step.tool}'. Available: {names}.")
                continue

            issues.extend(self._check_arguments(step.id, info, step.arguments))

        return issues

    @staticmethod
    def _check_arguments(step_id: str, info: ToolInfo, arguments: dict) -> list[str]:
        schema = info.input_schema or {}
        properties = schema.get("properties") or {}
        issues = [
            f"step '{step_id}': missing required argument '{name}' for {info.spec_name}."
            for name in info.required
            if name not in arguments
        ]

        if schema.get("additionalProperties") is False:
            issues += [
                f"step '{step_id}': unknown argument '{name}' for {info.spec_name}."
                for name in arguments
                if name not in properties
            ]

        for name, value in arguments.items():
            # Values built from references can only be checked at run time.
            if name not in properties or find_refs(value) or _contains_ref_key(value):
                continue

            error = next(iter(Draft202012Validator(properties[name]).iter_errors(value)), None)

            if error is not None:
                issues.append(
                    f"step '{step_id}': argument '{name}' is invalid: {error.message[:150]}."
                )

        return issues

    @staticmethod
    def canonicalize(plan: ExecutionPlan, tools: Mapping[str, ToolInfo]) -> None:
        """Rewrite tool names to the canonical provider-safe form."""
        lookup = {
            **{t.spec_name: t for t in tools.values()},
            **{t.qualified_name: t for t in tools.values()},
        }

        for step in plan.steps:
            if step.kind is StepKind.TOOL and step.tool in lookup:
                step.tool = lookup[step.tool].spec_name


def _contains_ref_key(value: object) -> bool:
    if isinstance(value, dict):
        return (
            "$ref" in value
            or "$template" in value
            or any(_contains_ref_key(v) for v in value.values())
        )

    if isinstance(value, list):
        return any(_contains_ref_key(v) for v in value)

    return False


class ToolSelector:
    """Binds tasks to concrete tools and arguments (tool selection)."""

    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider

    async def select(
        self,
        request: str,
        intent: Intent,
        tasks: TaskPlan | None,
        candidates: list[ToolInfo],
        *,
        issues: list[str] | None = None,
        completed: Mapping[str, str] | None = None,
        failures: Mapping[str, str] | None = None,
    ) -> ExecutionPlan:
        sections = [
            f"User request:\n{request}",
            f"Goal: {intent.goal}",
            "Tasks:\n"
            + "\n".join(
                f"- {t.id}: {t.description}"
                + (f" (after {', '.join(t.depends_on)})" if t.depends_on else "")
                for t in (tasks.tasks if tasks else [])
            ),
            "Available tools (* = required argument):\n"
            + "\n".join(t.signature() for t in candidates),
        ]

        if completed or failures:
            sections.append(
                "This is a RECOVERY plan. Plan only the work still needed, using NEW unique "
                "step ids "
                "(not those below).\nCompleted steps (reference them via $ref and list their "
                "ids in "
                '"inputs"):\n'
                + "\n".join(f"- {sid}: {summary}" for sid, summary in (completed or {}).items())
                + "\nFailed steps:\n"
                + "\n".join(f"- {sid}: {error}" for sid, error in (failures or {}).items())
                + '\nTry a different approach for failures, or return {"steps": []} if they cannot '
                "be recovered."
            )

        if issues:
            sections.append(
                "Your previous plan was rejected. Fix these problems:\n"
                + "\n".join(f"- {i}" for i in issues)
            )

        return await self._provider.generate_structured(
            _messages(SELECT_PROMPT, "\n\n".join(sections)), ExecutionPlan
        )


def describe_record_for_prompt(data: object, limit: int = 300) -> str:
    text = json.dumps(data, default=str, ensure_ascii=False, separators=(",", ":"))
    return text if len(text) <= limit else text[:limit] + "..."
