# Nexus MCP — Use Cases

## UC-01: Repository Analysis
**Actor:** User  
**Servers:** GitHub, Notes

1. User requests repository analysis.
2. Agent identifies the repository.
3. Agent retrieves repository details and open issues.
4. Agent categorizes issues.
5. Agent generates a summary.
6. Agent saves the summary in Notes.
7. Agent reports the result.

**Expected outcome:** Repository analysis is saved and accessible.

## UC-02: Intelligent Scheduling
**Actor:** User  
**Server:** Calendar

1. User requests a meeting.
2. Agent retrieves calendar availability.
3. Agent identifies suitable time slots.
4. Agent presents a proposed schedule.
5. User approves the selected slot.
6. Agent creates the event.
7. Agent verifies the event.

**Expected outcome:** An approved calendar event is created.

## UC-03: Development Planning
**Actor:** User  
**Servers:** GitHub, Notes

1. Retrieve repository issues.
2. Prioritize issues.
3. Generate a development plan.
4. Save the plan in Notes.
5. Return a summary.

## UC-04: Cross-Application Coordination
**Actor:** User  
**Servers:** GitHub, Calendar, Notes

1. Analyze repository issues.
2. Generate a development plan.
3. Retrieve calendar availability.
4. Propose a task schedule.
5. Obtain user approval.
6. Create approved calendar events.
7. Save the final development plan.
8. Verify all completed actions.

## UC-05: Failure Recovery
**Actor:** System

1. Begin a multi-server workflow.
2. Simulate an external server failure.
3. Persist completed execution steps.
4. Classify the failure.
5. Recover or resume the workflow.
6. Reconcile uncertain external operations.
7. Prevent duplicate side effects.

## UC-06: Authorization Enforcement
**Actor:** User

1. Request a protected operation.
2. Agent prepares the operation.
3. Backend evaluates permissions.
4. Approval request is generated.
5. User approves or rejects.
6. Backend verifies the approval.
7. Execute only if authorized.
8. Record the decision.

## UC-07: Dynamic Server Registration
**Actor:** Administrator/User

1. Submit a compatible MCP server configuration.
2. Validate its connection and trust status.
3. Initialize the MCP connection.
4. Discover capabilities and tools.
5. Apply permissions.
6. Register approved capabilities.
7. Make authorized tools available to the agent.

## UC-08: LLM Provider Switching
**Actor:** User/Administrator

1. Select Gemini or Groq.
2. Validate provider configuration.
3. Initialize the configured model.
4. Verify supported capabilities.
5. Execute a test request.
6. Continue agent operations with the selected provider.

## UC-09: Execution Inspection
**Actor:** User

1. Open execution history.
2. Select an execution.
3. Inspect planning and tool calls.
4. Inspect results, errors and timing.
5. Review approvals and recovery attempts.

## UC-10: Persistent Context
**Actor:** User

1. Start a conversation.
2. Execute a task.
3. Persist relevant conversation and execution state.
4. Restart or reconnect to the application.
5. Restore authorized context.
6. Continue the conversation.

**Status:** Baseline use cases defined.