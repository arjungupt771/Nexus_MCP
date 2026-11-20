"""Independently testable LangGraph nodes.

Each node is an async method ``state -> partial state update``. Routing is
done by ``route_after_detection`` so graph topology stays declarative.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from agent.core.context import prepare_context
from agent.core.definition import AgentDefinition
from agent.graph.state import AgentState
from agent.llm.errors import LLMError
from agent.llm.models import Message, Role, TokenUsage, ToolCall
from agent.llm.provider import LLMProvider
from agent.tools.bridge import MCPToolBridge, ToolOutcome

NODE_USER_INPUT = "user_input"
NODE_PREPARE = "prepare_context"
NODE_INVOKE = "invoke_llm"
NODE_DETECT = "detect_tool_calls"
NODE_EXECUTE = "execute_tools"
NODE_PROCESS = "process_results"
NODE_FINAL = "final_response"
NODE_LIMIT = "limit_reached"
NODE_ERROR = "handle_error"

LIMIT_NOTICE = (
    "Tool-call limit reached. Do not call any more tools; answer now using the "
    "information already gathered, and mention anything you could not complete."
)


class AgentNodes:
    def __init__(
        self,
        definition: AgentDefinition,
        provider: LLMProvider,
        bridge: MCPToolBridge,
    ) -> None:
        self._definition = definition
        self._provider = provider
        self._bridge = bridge

    # 1. user input ------------------------------------------------------
    async def user_input(self, state: AgentState) -> AgentState:
        turn = Message(role=Role.USER, content=state["user_input"])
        return {
            "messages": [*state.get("messages", []), turn],
            "new_messages": [turn],
            "outcomes": [],
            "iterations": 0,
            "usage": TokenUsage(),
            "error": None,
        }

    # 2. context preparation --------------------------------------------
    async def prepare_context(self, state: AgentState) -> AgentState:
        specs = await self._bridge.specs()  # fresh each loop: catalog may change
        return {
            "tool_specs": specs,
            "prepared": prepare_context(
                self._definition.system_instructions,
                state["messages"],
                self._definition.execution.max_context_messages,
            ),
        }

    # 3. LLM invocation --------------------------------------------------
    async def invoke_llm(self, state: AgentState) -> AgentState:
        try:
            response = await self._provider.generate(
                state["prepared"], tools=state.get("tool_specs") or None
            )
        except LLMError as exc:
            return {"response": None, "error": exc.to_dict(), "stop_reason": "error"}

        return {"response": response, "usage": state["usage"] + response.usage}

    # 4. tool-call detection ---------------------------------------------
    async def detect_tool_calls(self, state: AgentState) -> AgentState:
        response = state.get("response")
        return {"pending_calls": list(response.tool_calls) if response else []}

    # 5. tool execution handoff -----------------------------------------
    async def execute_tools(self, state: AgentState) -> AgentState:
        limit = self._definition.execution.max_tool_calls_per_turn
        outcomes: list[ToolOutcome] = []

        for index, call in enumerate(state["pending_calls"]):
            if index >= limit:
                outcomes.append(
                    self._refused(call, "TOOL_CALL_LIMIT", "Too many tool calls in one turn.")
                )
                continue

            try:
                outcomes.append(await self._bridge.execute(call))
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # bridge contract says it never raises; stay safe
                outcomes.append(
                    self._refused(
                        call, "INTERNAL_ERROR", f"Tool bridge failed ({type(exc).__name__})."
                    )
                )

        return {"outcomes": [*state.get("outcomes", []), *outcomes], "turn_outcomes": outcomes}

    @staticmethod
    def _refused(call: ToolCall, code: str, message: str) -> ToolOutcome:
        return ToolOutcome(
            call_id=call.id,
            spec_name=call.name,
            success=False,
            error_code=code,
            arguments=call.arguments,
            content=json.dumps({"success": False, "error": {"code": code, "message": message}}),
        )

    # 6. tool-result processing -----------------------------------------
    async def process_results(self, state: AgentState) -> AgentState:
        response = state["response"]
        assert response is not None
        turn_outcomes: list[ToolOutcome] = state["turn_outcomes"]
        added = [
            response.to_message(),
            *(
                Message(role=Role.TOOL, tool_call_id=o.call_id, name=o.spec_name, content=o.content)
                for o in turn_outcomes
            ),
        ]
        return {
            "messages": [*state["messages"], *added],
            "new_messages": [*state["new_messages"], *added],
            "iterations": state["iterations"] + 1,
            "pending_calls": [],
        }

    # 7. final response --------------------------------------------------
    async def final_response(self, state: AgentState) -> AgentState:
        response = state["response"]
        assert response is not None
        message = response.to_message()
        return {
            "final_response": response.content,
            "messages": [*state["messages"], message],
            "new_messages": [*state["new_messages"], message],
            "stop_reason": "completed",
        }

    async def limit_reached(self, state: AgentState) -> AgentState:
        """Iteration budget exhausted: ask for a best-effort answer without tools."""
        notice = Message(role=Role.USER, content=LIMIT_NOTICE)

        try:
            response = await self._provider.generate([*state["prepared"], notice], tools=None)
        except LLMError as exc:
            return {"error": exc.to_dict(), "stop_reason": "error", "final_response": ""}

        message = response.to_message().model_copy(update={"tool_calls": []})
        return {
            "final_response": response.content,
            "usage": state["usage"] + response.usage,
            "messages": [*state["messages"], message],
            "new_messages": [*state["new_messages"], message],
            "stop_reason": "max_iterations",
        }

    async def handle_error(self, state: AgentState) -> AgentState:
        error = state.get("error") or {}
        return {
            "final_response": "",
            "stop_reason": "error",
            "error": {**error, "message": error.get("message", "The model provider failed.")},
        }


def make_router(max_iterations: int) -> Any:
    def route(state: AgentState) -> str:
        if state.get("error"):
            return NODE_ERROR

        if not state.get("pending_calls"):
            return NODE_FINAL

        return NODE_LIMIT if state["iterations"] >= max_iterations else NODE_EXECUTE

    return route
