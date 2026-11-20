from __future__ import annotations

import asyncio
import time

import pytest
from support_orch import FakeRunner, domain_fail, fail, ok, plan, step

from agent.orchestration.models import (
    ExecutionPlan,
    PlanStep,
    StepState,
    WorkflowStatus,
    topological_layers,
)
from agent.orchestration.scheduler import ExecutionScheduler

REQ = "req-1"


def scheduler(runner, **kwargs):
    async def no_sleep(_):
        return None

    kwargs.setdefault("sleep", no_sleep)
    return ExecutionScheduler(runner, **kwargs)


def ref(step_id, path=None):
    return {"$ref": step_id, **({"path": path} if path else {})}


# ------------------------------------------------------------------ ordering & parallelism


async def test_sequential_chain_passes_results_between_steps():
    runner = FakeRunner()
    runner.behaviors["srv__a"] = lambda args: ok({"value": 41})
    runner.behaviors["srv__b"] = lambda args: ok({"value": args["n"] + 1})
    p = plan(step("a"), step("b", deps=["a"], arguments={"n": ref("a", "data.value")}))

    report = await scheduler(runner).execute(p, request_id=REQ)

    assert runner.calls == [("srv__a", {}), ("srv__b", {"n": 41})]
    assert report.status is WorkflowStatus.SUCCEEDED
    assert report.steps["b"].data == {"success": True, "data": {"value": 42}}


async def test_independent_steps_run_in_parallel():
    runner = FakeRunner()
    runner.delays = {"srv__a": 0.1, "srv__b": 0.1, "srv__c": 0.1}
    started = time.perf_counter()

    report = await scheduler(runner).execute(plan(step("a"), step("b"), step("c")), request_id=REQ)

    assert runner.peak == 3
    assert time.perf_counter() - started < 0.25  # not 0.3s sequential
    assert report.status is WorkflowStatus.SUCCEEDED


async def test_diamond_dependencies_are_respected():
    runner = FakeRunner()
    runner.delays = {"srv__b": 0.05, "srv__c": 0.02}
    p = plan(step("a"), step("b", deps=["a"]), step("c", deps=["a"]), step("d", deps=["b", "c"]))

    await scheduler(runner).execute(p, request_id=REQ)

    assert runner.finished[0] == "srv__a" and runner.finished[-1] == "srv__d"
    assert set(runner.finished[1:3]) == {"srv__b", "srv__c"}
    assert p.layers() == [["a"], ["b", "c"], ["d"]]


async def test_mixed_independent_and_dependent_work_overlaps():
    runner = FakeRunner()
    runner.delays = {"srv__slow": 0.15, "srv__a": 0.02, "srv__b": 0.02}
    p = plan(step("slow"), step("a"), step("b", deps=["a"]))

    await scheduler(runner).execute(p, request_id=REQ)

    # The a->b chain finishes while the slow independent step is still running.
    assert runner.finished == ["srv__a", "srv__b", "srv__slow"]


@pytest.mark.parametrize("limit", [1, 2, 3])
async def test_concurrency_limit_is_enforced(limit):
    runner = FakeRunner()
    runner.delays = {f"srv__s{i}": 0.03 for i in range(6)}
    p = plan(*[step(f"s{i}") for i in range(6)])

    report = await scheduler(runner, max_concurrency=limit).execute(p, request_id=REQ)

    assert runner.peak == limit and report.status is WorkflowStatus.SUCCEEDED


# ------------------------------------------------------------------ failures


async def test_partial_failure_skips_only_dependents():
    runner = FakeRunner()
    runner.behaviors["srv__a"] = lambda args: fail("TOOL_EXECUTION_FAILED", message="upstream down")
    p = plan(
        step("a"),
        step("b", deps=["a"]),
        step("c", deps=["b"]),  # transitive
        step("ok1"),
        step("ok2", deps=["ok1"]),
    )

    report = await scheduler(runner).execute(p, request_id=REQ)

    states = {k: r.state for k, r in report.steps.items()}
    assert states == {
        "a": StepState.FAILED,
        "b": StepState.SKIPPED,
        "c": StepState.SKIPPED,
        "ok1": StepState.SUCCEEDED,
        "ok2": StepState.SUCCEEDED,
    }
    assert report.status is WorkflowStatus.PARTIAL
    assert "dependency 'a'" in report.steps["b"].skip_reason
    assert "dependency 'b'" in report.steps["c"].skip_reason
    assert report.steps["a"].error_message == "upstream down"
    assert runner.count("srv__b") == 0 and runner.count("srv__c") == 0
    assert report.failed_ids == ["a"] and report.has_failures


async def test_all_failed_and_all_succeeded_statuses():
    runner = FakeRunner()
    runner.behaviors["srv__a"] = lambda args: fail()
    failed = await scheduler(runner).execute(plan(step("a"), step("b", deps=["a"])), request_id=REQ)
    assert failed.status is WorkflowStatus.FAILED

    empty = await scheduler(FakeRunner()).execute(plan(), request_id=REQ)
    assert empty.status is WorkflowStatus.SUCCEEDED and empty.steps == {}


async def test_fail_fast_skips_everything_pending():
    runner = FakeRunner()
    runner.behaviors["srv__a"] = lambda args: fail()
    p = plan(step("a"), step("b"), step("c"))

    report = await scheduler(runner, max_concurrency=1).execute(p, request_id=REQ, fail_fast=True)

    assert report.steps["a"].state is StepState.FAILED
    assert report.steps["b"].state is report.steps["c"].state is StepState.SKIPPED
    assert "fail_fast" in report.steps["b"].skip_reason


async def test_domain_failures_inside_successful_responses_are_failures():
    runner = FakeRunner()
    runner.behaviors["srv__a"] = lambda args: domain_fail(
        "Repository not found", "github_api_error"
    )

    report = await scheduler(runner).execute(plan(step("a"), step("b", deps=["a"])), request_id=REQ)

    record = report.steps["a"]
    assert (
        record.state is StepState.FAILED and record.error_code == "TOOL_REPORTED_GITHUB_API_ERROR"
    )
    assert (
        record.error_message == "Repository not found"
        and report.steps["b"].state is StepState.SKIPPED
    )


async def test_unresolvable_references_fail_the_step_without_calling_the_tool():
    runner = FakeRunner()
    runner.behaviors["srv__a"] = lambda args: ok({"items": []})
    p = plan(step("a"), step("b", deps=["a"], arguments={"x": ref("a", "data.items.0.name")}))

    report = await scheduler(runner).execute(p, request_id=REQ)

    assert report.steps["b"].error_code == "REFERENCE_UNRESOLVED" and runner.count("srv__b") == 0
    assert "data.items.0.name" in report.steps["b"].error_message


# ------------------------------------------------------------------ timeouts & retries


async def test_step_timeout_is_enforced_and_marked():
    runner = FakeRunner()
    runner.delays = {"srv__slow": 5}
    p = plan(step("slow", timeout_seconds=0.05), step("after", deps=["slow"]), step("fine"))

    started = time.perf_counter()
    report = await scheduler(runner).execute(p, request_id=REQ)

    assert time.perf_counter() - started < 1.5
    assert report.steps["slow"].state is StepState.TIMED_OUT
    assert report.steps["slow"].error_code == "STEP_TIMEOUT" and report.steps["slow"].retryable
    assert report.steps["after"].state is StepState.SKIPPED
    assert report.steps["fine"].succeeded and report.status is WorkflowStatus.PARTIAL
    assert report.failed_ids == ["slow"]


async def test_timeout_precedence_step_then_tool_then_default():
    runner = FakeRunner()
    p = plan(step("a", timeout_seconds=1.5), step("b"), step("c"))

    await scheduler(runner, default_timeout_seconds=7, tool_timeouts={"srv__b": 3}).execute(
        p, request_id=REQ
    )

    assert dict(zip([c[0] for c in runner.calls], runner.timeouts, strict=True)) == {
        "srv__a": 1.5,
        "srv__b": 3,
        "srv__c": 7,
    }


async def test_executor_reported_timeouts_map_to_timed_out():
    runner = FakeRunner()
    runner.behaviors["srv__a"] = lambda args: fail("TOOL_TIMEOUT", retryable=True)

    report = await scheduler(runner).execute(plan(step("a")), request_id=REQ)

    assert report.steps["a"].state is StepState.TIMED_OUT


async def test_retryable_failures_are_retried_then_succeed():
    runner = FakeRunner()
    runner.sequence("srv__a", fail(retryable=True), fail(retryable=True), ok({"v": 1}))
    sleeps: list[float] = []

    async def record_sleep(delay):
        sleeps.append(delay)

    report = await scheduler(runner, sleep=record_sleep, retry_delay_seconds=0.5).execute(
        plan(step("a", retries=2)), request_id=REQ
    )

    assert report.steps["a"].succeeded and report.steps["a"].attempts == 3
    assert sleeps == [0.5, 1.0] and runner.count("srv__a") == 3


async def test_non_retryable_failures_and_exhausted_retries_stop():
    runner = FakeRunner()
    runner.sequence("srv__a", fail(retryable=False), ok({}))
    first = await scheduler(runner).execute(plan(step("a", retries=3)), request_id=REQ)
    assert first.steps["a"].state is StepState.FAILED and first.steps["a"].attempts == 1

    runner = FakeRunner()
    runner.behaviors["srv__a"] = lambda args: fail(retryable=True)
    second = await scheduler(runner).execute(plan(step("a", retries=1)), request_id=REQ)
    assert second.steps["a"].attempts == 2 and runner.count("srv__a") == 2


# ------------------------------------------------------------------ duplicate prevention


async def test_identical_calls_execute_once_and_share_the_result():
    runner = FakeRunner()
    runner.behaviors["srv__search"] = lambda args: ok({"hits": 3})
    p = plan(
        step("a", "srv__search", arguments={"q": "mcp"}),
        step("b", "srv__search", arguments={"q": "mcp"}),
    )

    report = await scheduler(runner).execute(p, request_id=REQ)

    assert runner.count("srv__search") == 1
    assert report.steps["b"].deduplicated and not report.steps["a"].deduplicated
    assert report.steps["b"].data == report.steps["a"].data and report.steps["b"].succeeded


async def test_duplicate_prevention_covers_calls_still_in_flight():
    runner = FakeRunner()
    runner.delays = {"srv__search": 0.05}
    p = plan(*[step(f"s{i}", "srv__search", arguments={"q": "x"}) for i in range(4)])

    report = await scheduler(runner, max_concurrency=4).execute(p, request_id=REQ)

    assert runner.count("srv__search") == 1
    assert sum(r.deduplicated for r in report.steps.values()) == 3


async def test_different_arguments_are_not_duplicates_and_key_order_is_irrelevant():
    runner = FakeRunner()
    p = plan(
        step("a", "srv__t", arguments={"q": "1", "n": 2}),
        step("b", "srv__t", arguments={"q": "2"}),
        step("c", "srv__t", arguments={"n": 2, "q": "1"}),
    )

    await scheduler(runner).execute(p, request_id=REQ)

    assert runner.count("srv__t") == 2


async def test_duplicates_can_be_allowed_or_disabled():
    p = plan(
        step("a", "srv__t", arguments={"q": "1"}, allow_duplicate=True),
        step("b", "srv__t", arguments={"q": "1"}, allow_duplicate=True),
    )
    runner = FakeRunner()
    await scheduler(runner).execute(p, request_id=REQ)
    assert runner.count("srv__t") == 2

    runner = FakeRunner()
    p2 = plan(step("a", "srv__t"), step("b", "srv__t"))
    await scheduler(runner, dedupe=False).execute(p2, request_id=REQ)
    assert runner.count("srv__t") == 2


async def test_a_failed_call_is_not_repeated_for_its_duplicates():
    runner = FakeRunner()
    runner.behaviors["srv__t"] = lambda args: fail(message="nope")

    report = await scheduler(runner).execute(
        plan(step("a", "srv__t"), step("b", "srv__t")), request_id=REQ
    )

    assert runner.count("srv__t") == 1
    assert report.steps["b"].state is StepState.FAILED and report.steps["b"].error_message == "nope"


# ------------------------------------------------------------------ synthesize steps


async def test_synthesize_steps_receive_resolved_inputs():
    runner = FakeRunner()
    runner.behaviors["srv__a"] = lambda args: ok({"items": ["x", "y"]})
    seen = {}

    async def synth(instruction, inputs):
        seen.update(instruction=instruction, inputs=inputs)
        return "summary text"

    p = plan(
        step("a"),
        PlanStep(
            id="s",
            kind="synthesize",
            instruction="Summarize",
            arguments={"items": ref("a", "data.items")},
        ),
        step("w", deps=["s"], arguments={"body": ref("s", "text")}),
    )

    report = await scheduler(runner, synthesizer=synth).execute(p, request_id=REQ)

    assert seen == {"instruction": "Summarize", "inputs": {"items": ["x", "y"]}}
    assert report.steps["s"].data == {"text": "summary text"}
    assert runner.calls[-1] == ("srv__w", {"body": "summary text"})


async def test_synthesize_failures_are_contained():
    async def broken(instruction, inputs):
        raise RuntimeError("llm down")

    p = plan(PlanStep(id="s", kind="synthesize", instruction="x"), step("w", deps=["s"]))
    report = await scheduler(FakeRunner(), synthesizer=broken).execute(p, request_id=REQ)
    assert (
        report.steps["s"].error_code == "SYNTHESIS_FAILED"
        and report.steps["w"].state is StepState.SKIPPED
    )

    report = await scheduler(FakeRunner()).execute(
        plan(PlanStep(id="s", kind="synthesize", instruction="x")), request_id=REQ
    )
    assert report.steps["s"].error_code == "SYNTHESIZER_UNAVAILABLE"


# ------------------------------------------------------------------ identity, state, replanning


async def test_ids_are_preserved_and_unique_per_step():
    runner = FakeRunner()
    report = await scheduler(runner).execute(plan(step("a"), step("b")), request_id="request-xyz")

    assert report.request_id == "request-xyz" and report.execution_id
    ids = {r.execution_id for r in report.steps.values()}
    assert len(ids) == 2 and report.finished_at and report.duration_ms >= 0
    a = report.steps["a"]
    assert (
        a.tool_request_id == "req" and a.resolved_arguments == {} and a.started_at <= a.finished_at
    )


async def test_state_transitions_are_reported_in_order():
    runner = FakeRunner()
    runner.behaviors["srv__a"] = lambda args: fail()
    events: list[tuple[str, StepState]] = []

    def on_update(record):
        events.append((record.step_id, record.state))

    p = plan(step("a"), step("b", deps=["a"]))
    await scheduler(runner, on_update=on_update).execute(p, request_id=REQ)

    assert events == [
        ("a", StepState.RUNNING),
        ("b", StepState.SKIPPED),
        ("a", StepState.FAILED),
    ] or events == [("a", StepState.RUNNING), ("a", StepState.FAILED), ("b", StepState.SKIPPED)]


async def test_callback_errors_do_not_break_execution():
    def broken(record):
        raise RuntimeError("ui crashed")

    report = await scheduler(FakeRunner(), on_update=broken).execute(
        plan(step("a")), request_id=REQ
    )

    assert report.status is WorkflowStatus.SUCCEEDED


async def test_cancellation_stops_running_steps_and_does_not_leak_tasks():
    runner = FakeRunner()
    runner.delays = {"srv__a": 10, "srv__b": 10}
    task = asyncio.create_task(
        scheduler(runner).execute(plan(step("a"), step("b")), request_id=REQ)
    )
    await asyncio.sleep(0.05)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert runner.cancelled == 2 and runner.active == 0


async def test_prior_results_support_replanning_without_repeating_work():
    runner = FakeRunner()
    runner.behaviors["srv__a"] = lambda args: ok({"id": 7})
    first = await scheduler(runner).execute(plan(step("a", arguments={"q": "x"})), request_id=REQ)

    follow_up = plan(
        step("b", arguments={"id": ref("a", "data.id")}),
        step("again", "srv__a", arguments={"q": "x"}),  # identical to completed call
        inputs=["a"],
    )
    report = await scheduler(runner).execute(follow_up, request_id=REQ, prior=first.steps)

    assert runner.calls[-1] == ("srv__b", {"id": 7}) and runner.count("srv__a") == 1
    assert report.steps["again"].deduplicated and report.status is WorkflowStatus.SUCCEEDED


async def test_plans_cannot_reuse_completed_ids():
    runner = FakeRunner()
    first = await scheduler(runner).execute(plan(step("a")), request_id=REQ)

    with pytest.raises(ValueError, match="reuses ids"):
        await scheduler(runner).execute(plan(step("a")), request_id=REQ, prior=first.steps)


# ------------------------------------------------------------------ settings & plan validation


def test_scheduler_validates_settings():
    with pytest.raises(ValueError):
        ExecutionScheduler(FakeRunner(), max_concurrency=0)

    with pytest.raises(ValueError):
        ExecutionScheduler(FakeRunner(), default_timeout_seconds=0)


@pytest.mark.parametrize(
    "steps",
    [
        [step("a", deps=["b"]), step("b", deps=["a"])],
        [step("a"), step("a")],
        [step("a", deps=["ghost"])],
        [step("a", arguments={"x": ref("ghost")})],
    ],
)
def test_invalid_plans_are_rejected(steps):
    with pytest.raises(ValueError):
        ExecutionPlan(steps=steps)


def test_step_validation():
    for bad in (
        {"id": "1bad", "tool": "x"},
        {"id": "a"},
        {"id": "a", "kind": "synthesize"},
        {"id": "a", "tool": "x", "depends_on": ["a"]},
        {"id": "a", "tool": "x", "retries": 9},
    ):
        with pytest.raises(ValueError):
            PlanStep(**bad)


def test_references_become_dependencies_and_templates_are_scanned():
    p = plan(step("a"), step("b", arguments={"t": {"$template": "x {a.data.n}"}}))
    assert p.step("b").depends_on == ["a"]

    assert topological_layers({"x": [], "y": ["x"]}) == [["x"], ["y"]]
    with pytest.raises(ValueError, match="cycle"):
        topological_layers({"x": ["y"], "y": ["x"]})
