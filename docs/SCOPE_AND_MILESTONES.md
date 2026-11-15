# Nexus MCP — Scope and Milestones

## MVP Scope

The first functional MVP must include:

- GitHub MCP server.
- Google Calendar MCP server.
- Notes MCP server.
- MCP client engine.
- Dynamic tool discovery.
- Gemini and Groq provider adapters.
- LangGraph agent.
- At least two cross-server workflows.
- Basic persistent execution state.
- Human approval for protected operations.
- Basic structured logging.
- Automated tests.

## Complete Release Scope

The full release additionally includes:

- Dynamic server registry.
- Full workflow orchestration.
- Persistent memory.
- Failure recovery.
- Idempotency and reconciliation.
- Security and authorization.
- OpenTelemetry tracing.
- Evaluation framework.
- Benchmarking.
- React dashboard.
- Integration marketplace.
- Complete documentation.

## Out of Scope

- Autonomous financial transactions.
- Unrestricted operating-system access.
- Arbitrary code execution.
- Unrestricted remote MCP connections.
- Paid infrastructure dependency.
- Production multi-tenant SaaS operations.

## Milestones

| Milestone | Completion condition |
|---|---|
| M0 | Requirements baseline completed |
| M1 | Architecture and repository established |
| M2 | Three independent MCP servers functional |
| M3 | MCP client and registry operational |
| M4 | Agent executes multi-server workflows |
| M5 | State, approvals and recovery functional |
| M6 | Security and observability implemented |
| M7 | Evaluation and benchmarks completed |
| M8 | Frontend and integration management completed |
| M9 | Final testing and release completed |

## Development Policy

Work sequentially through milestones, but allow earlier components to be revisited when integration testing exposes defects.

Every milestone requires implementation evidence, passing tests and updated documentation.

**Status:** Initial scope baseline.