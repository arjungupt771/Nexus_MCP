"""Dependency-aware execution of an ``ExecutionPlan`` (7.3, 7.6).

* Steps run as soon as every dependency has succeeded; independent steps run
  in parallel, bounded by ``max_concurrency``.
* A failed step skips only the steps that (transitively) depend on it;
  unrelated branches still complete, giving a *partial* result.
* Identical tool calls (same tool + resolved arguments) execute once and share
  the result, including calls still in flight.
* Every step has its own timeout, attempt counter, execution id and state.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, datetime
from typing import Any, Protocol

from agent.llm.models import ToolCall
from agent.orchestration.models import (
    TERMINAL_FAILURES,
    ExecutionPlan,
    ExecutionReport,
    PlanStep,
    StepKind,
    StepRecord,
    StepState,
)
from agent.orchestration.references import ReferenceResolutionError, resolve
from agent.tools.bridge import ToolOutcome

logger = logging.getLogger(__name__)

Synthesizer = Callable[[str, dict[str, Any]], Awaitable[str]]
UpdateCallback = Callable[[StepRecord], Awaitable[None] | None]
_TIMEOUT_MARGIN = 0.25


class ToolRunner(Protocol):
    async def execute(
        self, call: ToolCall, *, timeout_seconds: float | None = None
    ) -> ToolOutcome: ...


def _now() -> datetime:
    return datetime.now(UTC)


def dedupe_key(tool: str, arguments: dict[str, Any]) -> str:
    return f"{tool}|{json.dumps(arguments, sort_keys=True, default=str)}"


class ExecutionScheduler:
    def __init__(
        self,
        runner: ToolRunner,
        *,
        max_concurrency: int = 4,
        default_timeout_seconds: float = 30.0,
        tool_timeouts: Mapping[str, float] | None = None,
        synthesizer: Synthesizer | None = None,
        dedupe: bool = True,
        retry_delay_seconds: float = 0.2,
        sleep: Callable[[float], Awaitable[None]] | None = None,
        on_update: UpdateCallback | None = None,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")

        if default_timeout_seconds <= 0:
            raise ValueError("default_timeout_seconds must be positive")

        self._runner = runner
        self._max_concurrency = max_concurrency
        self._default_timeout = default_timeout_seconds
        self._tool_timeouts = dict(tool_timeouts or {})
        self._synthesizer = synthesizer
        self._dedupe = dedupe
        self._retry_delay = retry_delay_seconds
        self._sleep = sleep or asyncio.sleep
        self._on_update = on_update

    # ------------------------------------------------------------------ public

    async def execute(
        self,
        plan: ExecutionPlan,
        *,
        request_id: str,
        prior: Mapping[str, StepRecord] | None = None,
        fail_fast: bool = False,
    ) -> ExecutionReport:
        """Run ``plan``. ``prior`` supplies already-succeeded steps (replanning)."""
        prior = {k: v for k, v in (prior or {}).items() if v.succeeded}
        clash = sorted(set(prior) & {s.id for s in plan.steps})

        if clash:
            raise ValueError(f"plan reuses ids of completed steps: {', '.join(clash)}")

        run = _Run(self, plan, request_id, prior, fail_fast)
        return await run.execute()


class _Run:
    """State of a single ``execute`` call (keeps the scheduler reusable)."""

    def __init__(
        self,
        scheduler: ExecutionScheduler,
        plan: ExecutionPlan,
        request_id: str,
        prior: Mapping[str, StepRecord],
        fail_fast: bool,
    ) -> None:
        self.s = scheduler
        self.plan = plan
        self.fail_fast = fail_fast
        self.report = ExecutionReport(request_id=request_id, plan_id=plan.plan_id)
        self.results: dict[str, Any] = {k: v.data for k, v in prior.items()}
        self.satisfied: set[str] = set(prior)
        self.failed: set[str] = set()
        self.semaphore = asyncio.Semaphore(scheduler._max_concurrency)
        # dedupe key -> future resolving to the finished StepRecord of the first run
        self.inflight: dict[str, asyncio.Future[StepRecord]] = {}

        for step in plan.steps:
            self.report.steps[step.id] = StepRecord(step_id=step.id, kind=step.kind, tool=step.tool)

        for record in prior.values():
            if record.dedupe_key:
                done: asyncio.Future[StepRecord] = asyncio.get_running_loop().create_future()
                done.set_result(record)
                self.inflight[record.dedupe_key] = done

    async def _notify(self, record: StepRecord) -> None:
        if self.s._on_update is None:
            return

        try:
            outcome = self.s._on_update(record.model_copy(deep=True))

            if asyncio.iscoroutine(outcome):
                await outcome
        except Exception:
            logger.exception("Step update callback failed")

    async def execute(self) -> ExecutionReport:
        pending = {step.id: step for step in self.plan.steps}
        running: dict[asyncio.Task[None], str] = {}

        try:
            while pending or running:
                await self._skip_blocked(pending)

                for step_id in [
                    sid for sid, st in pending.items() if set(st.depends_on) <= self.satisfied
                ]:
                    step = pending.pop(step_id)
                    task = asyncio.create_task(self._run_step(step), name=f"step-{step_id}")
                    running[task] = step_id

                if not running:
                    break

                done, _ = await asyncio.wait(running, return_when=asyncio.FIRST_COMPLETED)

                for task in done:
                    step_id = running.pop(task)
                    task.result()  # _run_step never raises except cancellation
                    record = self.report.steps[step_id]

                    if record.succeeded:
                        self.satisfied.add(step_id)
                    else:
                        self.failed.add(step_id)
        except asyncio.CancelledError:
            for task in running:
                task.cancel()

            await asyncio.gather(*running, return_exceptions=True)

            for record in self.report.steps.values():
                if record.state in (StepState.PENDING, StepState.RUNNING):
                    record.state = StepState.CANCELLED
                    record.finished_at = _now()

            self.report.finalize(cancelled=True)
            raise

        self.report.finalize()
        return self.report

    async def _skip_blocked(self, pending: dict[str, PlanStep]) -> None:
        """Skip steps that can never run; cascades to a fixpoint."""
        changed = True

        while changed:
            changed = False

            for step_id, step in list(pending.items()):
                blocker = next((d for d in step.depends_on if d in self.failed), None)
                reason = None

                if blocker is not None:
                    reason = f"dependency '{blocker}' did not succeed"
                elif self.fail_fast and self.failed:
                    reason = "skipped after an earlier failure (fail_fast)"

                if reason is None:
                    continue

                record = self.report.steps[step_id]
                record.state = StepState.SKIPPED
                record.skip_reason = reason
                record.finished_at = _now()
                del pending[step_id]
                self.failed.add(step_id)
                changed = True
                await self._notify(record)

    # ---------------------------------------------------------------- one step

    async def _run_step(self, step: PlanStep) -> None:
        record = self.report.steps[step.id]

        async with self.semaphore:
            if self.fail_fast and self.failed:
                # Queued behind the concurrency limit while another step failed.
                record.state = StepState.SKIPPED
                record.skip_reason = "skipped after an earlier failure (fail_fast)"
                record.finished_at = _now()
                self.failed.add(step.id)
                await self._notify(record)
                return

            record.state = StepState.RUNNING
            record.started_at = _now()
            started = time.perf_counter()
            await self._notify(record)

            try:
                arguments = resolve(step.arguments, self.results)
            except ReferenceResolutionError as exc:
                self._fail(record, StepState.FAILED, "REFERENCE_UNRESOLVED", str(exc))
            else:
                record.resolved_arguments = arguments

                if step.kind is StepKind.SYNTHESIZE:
                    await self._synthesize(step, record, arguments)
                else:
                    await self._call_tool(step, record, arguments)

            record.finished_at = _now()
            record.duration_ms = (time.perf_counter() - started) * 1000

            # Bookkeeping happens before the slot is released so the next
            # queued step already sees this outcome.
            if record.succeeded:
                self.results[step.id] = record.data
                self.satisfied.add(step.id)
            else:
                self.failed.add(step.id)

        await self._notify(record)

    @staticmethod
    def _fail(
        record: StepRecord,
        state: StepState,
        code: str,
        message: str,
        *,
        retryable: bool = False,
    ) -> None:
        record.state = state
        record.error_code = code
        record.error_message = message[:500]
        record.retryable = retryable

    async def _synthesize(self, step: PlanStep, record: StepRecord, inputs: dict[str, Any]) -> None:
        if self.s._synthesizer is None:
            self._fail(
                record,
                StepState.FAILED,
                "SYNTHESIZER_UNAVAILABLE",
                "No LLM synthesizer configured.",
            )
            return

        record.attempts = 1
        timeout = step.timeout_seconds or self.s._default_timeout

        try:
            async with asyncio.timeout(timeout):
                text = await self.s._synthesizer(step.instruction or "", inputs)
        except TimeoutError:
            self._fail(
                record,
                StepState.TIMED_OUT,
                "STEP_TIMEOUT",
                f"Synthesis exceeded {timeout}s.",
                retryable=True,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._fail(
                record,
                StepState.FAILED,
                "SYNTHESIS_FAILED",
                f"Synthesis failed ({type(exc).__name__}).",
            )
        else:
            record.state = StepState.SUCCEEDED
            record.data = {"text": text}

    async def _call_tool(
        self, step: PlanStep, record: StepRecord, arguments: dict[str, Any]
    ) -> None:
        assert step.tool is not None
        key = dedupe_key(step.tool, arguments)
        record.dedupe_key = key
        use_dedupe = self.s._dedupe and not step.allow_duplicate

        if use_dedupe and key in self.inflight:
            source = await asyncio.shield(self.inflight[key])
            self._copy_result(source, record)
            return

        future: asyncio.Future[StepRecord] | None = None

        if use_dedupe:
            future = asyncio.get_running_loop().create_future()
            self.inflight[key] = future

        try:
            await self._attempts(step, record, arguments)
        finally:
            if future is not None and not future.done():
                future.set_result(record)

    @staticmethod
    def _copy_result(source: StepRecord, record: StepRecord) -> None:
        record.deduplicated = True
        record.state = source.state
        record.data = source.data
        record.server_id, record.tool_name = source.server_id, source.tool_name
        record.error_code, record.error_message = source.error_code, source.error_message
        record.retryable = source.retryable
        record.tool_request_id = source.tool_request_id

    async def _attempts(
        self, step: PlanStep, record: StepRecord, arguments: dict[str, Any]
    ) -> None:
        assert step.tool is not None
        timeout = (
            step.timeout_seconds or self.s._tool_timeouts.get(step.tool) or self.s._default_timeout
        )

        for attempt in range(1, step.retries + 2):
            record.attempts = attempt
            record.error_code = record.error_message = None

            try:
                async with asyncio.timeout(timeout + _TIMEOUT_MARGIN):
                    outcome = await self.s._runner.execute(
                        ToolCall(id=record.execution_id, name=step.tool, arguments=arguments),
                        timeout_seconds=timeout,
                    )
            except TimeoutError:
                self._fail(
                    record,
                    StepState.TIMED_OUT,
                    "STEP_TIMEOUT",
                    f"Step exceeded {timeout}s.",
                    retryable=True,
                )
            else:
                self._interpret(record, outcome)

            if record.succeeded or not record.retryable or attempt > step.retries:
                return

            await self.s._sleep(self.s._retry_delay * attempt)

    def _interpret(self, record: StepRecord, outcome: ToolOutcome) -> None:
        record.server_id, record.tool_name = outcome.server_id, outcome.tool_name
        record.tool_request_id = outcome.request_id

        if not outcome.success:
            timed_out = "TIMEOUT" in (outcome.error_code or "").upper()
            self._fail(
                record,
                StepState.TIMED_OUT if timed_out else StepState.FAILED,
                outcome.error_code or "TOOL_FAILED",
                _error_text(outcome.content),
                retryable=outcome.retryable,
            )
            return

        data = outcome.data

        # Nexus MCP servers report domain failures inside a *successful* MCP
        # response: {"success": false, "error": {...}}. Treat that as a failure.
        if isinstance(data, dict) and data.get("success") is False:
            error = data.get("error")
            error = error if isinstance(error, dict) else {}
            self._fail(
                record,
                StepState.FAILED,
                f"TOOL_REPORTED_{str(error.get('type', 'error')).upper()}",
                str(error.get("message", "The tool reported a failure.")),
                retryable=bool(error.get("retryable", False)),
            )
            return

        record.state = StepState.SUCCEEDED
        record.data = data


def _error_text(content: str) -> str:
    try:
        return str(json.loads(content)["error"]["message"])
    except (ValueError, KeyError, TypeError):
        return content[:300]


__all__ = ["ExecutionScheduler", "Synthesizer", "ToolRunner", "dedupe_key", "TERMINAL_FAILURES"]
