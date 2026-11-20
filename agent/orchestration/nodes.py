"""LangGraph nodes for multi-server orchestration (7.5).

Each node is an ``async (state) -> partial state`` method so it can be tested
alone. Planning nodes only plan; ``execute_plan`` is the only node that runs
tools, and it delegates to ``ExecutionScheduler``.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agent.llm.errors import LLMError
from agent.llm.models import Message, Role, TokenUsage
from agent.llm.provider import LLMProvider
from agent.orchestration.aggregator import ResultAggregator
from agent.orchestration.models import (
    ExecutionPlan,
    StepRecord,
    StepState,
    WorkflowStatus,
)
from agent.orchestration.planner import (
    IntentAnalyzer,
    PlanValidator,
    TaskPlanner,
    ToolSelector,
    describe_record_for_prompt,
)
from agent.orchestration.scheduler import ExecutionScheduler, ToolRunner
from agent.orchestration.state import OrchestrationState
from agent.orchestration.tools import ToolInfo, UnifiedToolInterface

N_USER_REQUEST = "user_request"
N_INTENT = "understand_intent"
N_TASKS = "plan_tasks"
N_DISCOVER = "discover_tools"
N_SELECT = "select_tools"
N_SCHEDULE = "schedule_execution"
N_EXECUTE = "execute_plan"
N_AGGREGATE = "aggregate_results"
N_REPLAN = "prepare_replan"
N_FINAL = "final_response"
N_ERROR = "handle_error"

SYNTH_SYSTEM = (
    "You write concise, factual text from the data you are given. Output only the requested text."
)
FINAL_SYSTEM_SUFFIX = (
    "\n\nWhen tool results are provided, answer using ONLY those results. If any step failed "
    "or was "
    "skipped, say plainly what was not done and why. Never claim an action succeeded unless its "
    "step is marked succeeded."
)


class OrchestrationSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_concurrency: int = Field(default=4, ge=1, le=32)
    default_tool_timeout_seconds: float = Field(default=30.0, gt=0, le=600)
    tool_timeouts: dict[str, float] = Field(default_factory=dict)
    max_plan_steps: int = Field(default=12, ge=1, le=50)
    # LLM attempts to produce a *valid* plan (repair rounds included).
    max_plan_attempts: int = Field(default=2, ge=1, le=5)
    # Recovery rounds after step failures.
    max_replans: int = Field(default=1, ge=0, le=3)
    fail_fast: bool = False
    dedupe: bool = True
    candidate_tool_limit: int = Field(default=20, ge=1, le=200)
    retry_delay_seconds: float = Field(default=0.2, ge=0)


def _failed_records(records: dict[str, StepRecord], skip: set[str]) -> dict[str, StepRecord]:
    return {k: v for k, v in records.items() if k not in skip}


def overall_status(records: dict[str, StepRecord]) -> WorkflowStatus:
    states = [r.state for r in records.values()]
    succeeded = sum(s is StepState.SUCCEEDED for s in states)

    if not states or succeeded == len(states):
        return WorkflowStatus.SUCCEEDED

    return WorkflowStatus.FAILED if succeeded == 0 else WorkflowStatus.PARTIAL


class OrchestrationNodes:
    def __init__(
        self,
        provider: LLMProvider,
        tools: UnifiedToolInterface,
        runner: ToolRunner,
        settings: OrchestrationSettings,
        system_instructions: str,
    ) -> None:
        self._provider = provider
        self._tools = tools
        self._runner = runner
        self._settings = settings
        self._instructions = system_instructions
        self._intent = IntentAnalyzer(provider)
        self._tasks = TaskPlanner(provider)
        self._selector = ToolSelector(provider)
        self._validator = PlanValidator(max_steps=settings.max_plan_steps)
        self._aggregator = ResultAggregator()

    @staticmethod
    def _llm_error(exc: LLMError) -> dict[str, Any]:
        return {"error": exc.to_dict(), "stop_reason": "error"}

    # 1 ------------------------------------------------------------------
    async def user_request(self, state: OrchestrationState) -> OrchestrationState:
        return {
            "history": state.get("history", []),
            "plan_ids": [],
            "plan_issues": [],
            "plan_attempts": 0,
            "steps": {},
            "records": {},
            "superseded": [],
            "replans": 0,
            "usage": TokenUsage(),
            "error": None,
            "report": None,
            "plan": None,
            "aggregated": None,
        }

    # 2 ------------------------------------------------------------------
    async def understand_intent(self, state: OrchestrationState) -> OrchestrationState:
        try:
            tools = await self._tools.list_tools()
            intent = await self._intent.analyze(state["user_input"], tools, state.get("history"))
        except LLMError as exc:
            return self._llm_error(exc)

        # No connected tools means tools cannot be needed.
        if not tools:
            intent = intent.model_copy(update={"needs_tools": False})

        return {"intent": intent}

    # 3 ------------------------------------------------------------------
    async def plan_tasks(self, state: OrchestrationState) -> OrchestrationState:
        try:
            tasks = await self._tasks.plan(
                state["user_input"], state["intent"], await self._tools.list_tools()
            )
        except LLMError as exc:
            return self._llm_error(exc)

        return {"tasks": tasks}

    # 4 ------------------------------------------------------------------
    async def discover_tools(self, state: OrchestrationState) -> OrchestrationState:
        """Retrieve candidate tools; narrows large catalogs, keeps small ones whole."""
        all_tools = await self._tools.list_tools()
        limit = self._settings.candidate_tool_limit

        if len(all_tools) <= limit:
            return {"candidates": all_tools}

        intent = state["intent"]
        query = " ".join(
            [state["user_input"], intent.goal, *(t.description for t in state["tasks"].tasks)]
        )
        chosen: dict[str, ToolInfo] = {
            t.qualified_name: t for t in await self._tools.search(query, limit=limit)
        }

        # Tools of servers the intent named are always worth considering.
        for hint in intent.server_hints:
            for tool in await self._tools.search(query, limit=5, servers={hint}):
                chosen.setdefault(tool.qualified_name, tool)

        return {"candidates": list(chosen.values())[:limit] or all_tools[:limit]}

    # 5 ------------------------------------------------------------------
    async def select_tools(self, state: OrchestrationState) -> OrchestrationState:
        attempts = state.get("plan_attempts", 0) + 1
        records = state.get("records", {})
        replanning = state.get("replans", 0) > 0
        completed = (
            {sid: describe_record_for_prompt(r.data) for sid, r in records.items() if r.succeeded}
            if replanning
            else None
        )
        failures = (
            {
                sid: f"{r.error_code}: {r.error_message or r.skip_reason}"
                for sid, r in records.items()
                if not r.succeeded
            }
            if replanning
            else None
        )

        try:
            plan = await self._selector.select(
                state["user_input"],
                state["intent"],
                state.get("tasks"),
                state.get("candidates", []),
                issues=state.get("plan_issues") or None,
                completed=completed,
                failures=failures,
            )
        except LLMError as exc:
            return self._llm_error(exc)

        catalog = {t.qualified_name: t for t in await self._tools.list_tools()}
        issues = self._validator.validate(plan, catalog)
        done = {sid for sid, r in records.items() if r.succeeded}
        issues += [
            f"'inputs' lists unknown completed step '{i}'." for i in plan.inputs if i not in done
        ]
        issues += [
            f"step id '{s.id}' is already used by an earlier step."
            for s in plan.steps
            if s.id in records
        ]

        if issues:
            return {"plan": None, "plan_issues": issues, "plan_attempts": attempts}

        self._validator.canonicalize(plan, catalog)
        return {"plan": plan, "plan_issues": [], "plan_attempts": attempts}

    # 6 ------------------------------------------------------------------
    async def schedule_execution(self, state: OrchestrationState) -> OrchestrationState:
        plan = state["plan"]
        assert plan is not None
        return {
            "layers": plan.layers(),  # parallel waves, for observability
            "plan_ids": [*state.get("plan_ids", []), plan.plan_id],
        }

    # 7 ------------------------------------------------------------------
    async def execute_plan(self, state: OrchestrationState) -> OrchestrationState:
        plan = state["plan"]
        assert plan is not None
        synth_usage: list[TokenUsage] = []

        async def synthesize(instruction: str, inputs: dict[str, Any]) -> str:
            payload = json.dumps(inputs, default=str, ensure_ascii=False)[:12000]
            response = await self._provider.generate(
                [
                    Message(role=Role.SYSTEM, content=SYNTH_SYSTEM),
                    Message(role=Role.USER, content=f"{instruction}\n\nDATA:\n{payload}"),
                ]
            )
            synth_usage.append(response.usage)

            if not response.content.strip():
                raise ValueError("empty synthesis")

            return response.content.strip()

        s = self._settings
        scheduler = ExecutionScheduler(
            self._runner,
            max_concurrency=s.max_concurrency,
            default_timeout_seconds=s.default_tool_timeout_seconds,
            tool_timeouts=s.tool_timeouts,
            synthesizer=synthesize,
            dedupe=s.dedupe,
            retry_delay_seconds=s.retry_delay_seconds,
        )
        prior = {k: v for k, v in state.get("records", {}).items() if v.succeeded}
        report = await scheduler.execute(
            plan, request_id=state["request_id"], prior=prior, fail_fast=s.fail_fast
        )
        usage = state["usage"]

        for item in synth_usage:
            usage = usage + item

        return {
            "report": report,
            "records": {**state.get("records", {}), **report.steps},
            "steps": {**state.get("steps", {}), **{st.id: st for st in plan.steps}},
            "usage": usage,
        }

    # 8 ------------------------------------------------------------------
    async def aggregate_results(self, state: OrchestrationState) -> OrchestrationState:
        records = _failed_records(state["records"], set(state.get("superseded", [])))
        aggregated = self._aggregator.aggregate(state["steps"], records, overall_status(records))
        aggregated.recoveries = state.get("replans", 0)
        return {"aggregated": aggregated}

    async def prepare_replan(self, state: OrchestrationState) -> OrchestrationState:
        """Mark failed/skipped steps as superseded and reset the repair counter."""
        report = state["report"]
        assert report is not None
        retired = [i for i, r in report.steps.items() if not r.succeeded]
        return {
            "superseded": [*state.get("superseded", []), *retired],
            "replans": state.get("replans", 0) + 1,
            "plan_attempts": 0,
            "plan_issues": [],
            "plan": None,
        }

    # 9 ------------------------------------------------------------------
    async def final_response(self, state: OrchestrationState) -> OrchestrationState:
        aggregated = state.get("aggregated")
        history = [
            m
            for m in state.get("history", [])
            if m.role in (Role.USER, Role.ASSISTANT) and m.content
        ]
        user = state["user_input"]

        if aggregated is not None:
            user = f"{user}\n\n--- Tool results ---\n{aggregated.digest()}"

        messages = [
            Message(role=Role.SYSTEM, content=self._instructions + FINAL_SYSTEM_SUFFIX),
            *history[-10:],
            Message(role=Role.USER, content=user),
        ]
        status = aggregated.status if aggregated else None
        reason = {
            None: "no_tools_needed",
            WorkflowStatus.SUCCEEDED: "completed",
            WorkflowStatus.PARTIAL: "partial",
            WorkflowStatus.FAILED: "failed",
        }[status]

        try:
            response = await self._provider.generate(messages, tools=None)
        except LLMError as exc:
            # The work is done; do not lose it because the summary call failed.
            fallback = aggregated.digest() if aggregated else ""
            return {
                "final_response": fallback,
                "error": exc.to_dict(),
                "stop_reason": reason if aggregated else "error",
            }

        return {
            "final_response": response.content,
            "usage": state["usage"] + response.usage,
            "stop_reason": reason,
        }

    async def handle_error(self, state: OrchestrationState) -> OrchestrationState:
        error = dict(state.get("error") or {})
        issues = state.get("plan_issues")

        if not error and issues:
            error = {
                "code": "PLAN_INVALID",
                "message": "Could not build a valid plan: " + "; ".join(issues[:3]),
                "retryable": False,
            }

        aggregated = state.get("aggregated")
        return {
            "error": error,
            "stop_reason": "error",
            "final_response": aggregated.digest() if aggregated else "",
        }


# ---------------------------------------------------------------- routing


def make_routers(settings: OrchestrationSettings):
    def after_intent(state: OrchestrationState) -> str:
        if state.get("error"):
            return N_ERROR

        return N_TASKS if state["intent"].needs_tools else N_FINAL

    def on_error_else(next_node: str):
        def route(state: OrchestrationState) -> str:
            return N_ERROR if state.get("error") else next_node

        return route

    def after_select(state: OrchestrationState) -> str:
        if state.get("error"):
            return N_ERROR

        if state.get("plan_issues"):
            # Controlled repair loop: bounded by max_plan_attempts.
            return N_SELECT if state["plan_attempts"] < settings.max_plan_attempts else N_ERROR

        plan: ExecutionPlan | None = state.get("plan")
        return N_SCHEDULE if plan and plan.steps else N_FINAL

    def after_aggregate(state: OrchestrationState) -> str:
        report = state["report"]
        failed = bool(report and report.failed_ids)
        return N_REPLAN if failed and state["replans"] < settings.max_replans else N_FINAL

    return after_intent, on_error_else, after_select, after_aggregate
