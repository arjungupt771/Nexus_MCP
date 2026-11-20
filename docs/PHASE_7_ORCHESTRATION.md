# Phase 7 — Multi-server Agent Orchestration

Code: `agent/orchestration/`. Reuses the Phase 5 registry/catalog/events, the Phase 6 providers and
`MCPToolBridge`, and the existing `ToolExecutorService`. Planning and execution are separate modules.

| Module | Role |
|---|---|
| `tools.py` `UnifiedToolInterface` | Always-current, permission-filtered view of every server's tools; ownership, grouping, keyword search (uses server tags/capabilities), change counter via events |
| `models.py` | `Intent`, `TaskPlan`, `ExecutionPlan`/`PlanStep` (validated dependency graph), `StepRecord`, `ExecutionReport` |
| `references.py` | `$ref` / `$template` / `pluck` / `add_minutes` to pass one step's output into another |
| `planner.py` | `IntentAnalyzer`, `TaskPlanner`, `ToolSelector`, `PlanValidator` (LLM planning only, nothing executes) |
| `scheduler.py` `ExecutionScheduler` | Dependency-aware parallel execution, concurrency limit, timeouts, retries, dedupe, partial failure |
| `aggregator.py` | Groups results by server; text digest for the final answer |
| `nodes.py`, `graph.py`, `orchestrator.py` | LangGraph workflow and public `Orchestrator.run()` |
| `examples.py` | The three cross-server workflows on the real GitHub / Calendar / Notes contracts |

## Graph
`user_request -> understand_intent -> (no tools) final_response`, otherwise
`plan_tasks -> discover_tools -> select_tools -> schedule_execution -> execute_plan -> aggregate_results -> final_response`.
Loops: `select_tools` repeats with validator feedback (bounded by `max_plan_attempts`); failed steps trigger
`prepare_replan -> select_tools` (bounded by `max_replans`). Any model failure routes to `handle_error`.

## Execution semantics
* A step runs when all dependencies succeeded; independent steps run concurrently up to `max_concurrency`.
* A failed/timed-out step skips only its (transitive) dependents; other branches finish -> `partial` status.
* Domain failures that MCP servers return inside a successful response (`{"success": false, ...}`) count as failures.
* Identical tool+arguments calls run once per workflow (including in-flight and across recovery rounds); `allow_duplicate` opts out.
* Timeout precedence: step > per-tool setting > default. Retries only for retryable errors (`retries` per step).
* Every step has its own `execution_id`; the workflow keeps the `request_id`.
* Recovery plans may reference completed steps via `inputs`, never re-run them, and must use fresh step ids.

## Usage
```python
orchestrator = Orchestrator.create(container, provider, settings=OrchestrationSettings(max_concurrency=4))
result = await orchestrator.run("Find top MCP repos on GitHub and save a note")
result.response, result.report.steps, result.aggregated.servers_used
```
Fixed plans can be run directly: `ExecutionScheduler(bridge, synthesizer=...).execute(plan, request_id=...)`
(see `examples.py`).

## Out of scope (per brief)
Persistent memory, human-in-the-loop approval, dashboard.
