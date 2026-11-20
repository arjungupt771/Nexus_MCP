"""The Agent: identity + provider + tool bridge, executed as a LangGraph."""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from agent.core.definition import AgentDefinition
from agent.graph.builder import build_agent_graph
from agent.graph.nodes import AgentNodes
from agent.graph.state import AgentState
from agent.llm.factory import ProviderRegistry, default_registry
from agent.llm.models import Message, TokenUsage
from agent.llm.provider import LLMProvider
from agent.tools.bridge import MCPToolBridge, ToolOutcome

if TYPE_CHECKING:
    from core.container import NexusContainer


class AgentResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    agent_id: str
    response: str
    # Messages produced by this run, ready to append to a conversation history.
    new_messages: list[Message] = Field(default_factory=list)
    tool_outcomes: list[ToolOutcome] = Field(default_factory=list)
    usage: TokenUsage = Field(default_factory=TokenUsage)
    iterations: int = 0
    stop_reason: str  # completed | max_iterations | error
    error: dict[str, Any] | None = None

    @property
    def succeeded(self) -> bool:
        return self.stop_reason != "error"


class Agent:
    def __init__(
        self,
        definition: AgentDefinition,
        provider: LLMProvider,
        bridge: MCPToolBridge,
    ) -> None:
        self.definition = definition
        self.provider = provider
        self.bridge = bridge
        self._nodes = AgentNodes(definition, provider, bridge)
        self._graph = build_agent_graph(
            self._nodes, max_iterations=definition.execution.max_iterations
        )

    @classmethod
    def create(
        cls,
        definition: AgentDefinition,
        container: NexusContainer,
        *,
        provider: LLMProvider | None = None,
        registry: ProviderRegistry | None = None,
        env: Mapping[str, str] | None = None,
    ) -> Agent:
        """Wire an agent to the Nexus container's catalog and executor."""
        provider = provider or (registry or default_registry()).create(definition.model, env=env)
        bridge = MCPToolBridge(
            container.catalog,
            container.executor,
            definition.tool_permissions,
            timeout_seconds=definition.execution.tool_timeout_seconds,
            max_result_chars=definition.execution.max_tool_result_chars,
        )
        return cls(definition, provider, bridge)

    def with_provider(self, provider: LLMProvider) -> Agent:
        """Return the same agent running on a different provider."""
        return Agent(self.definition, provider, self.bridge)

    async def run(
        self,
        user_input: str,
        *,
        history: list[Message] | None = None,
        request_id: str | None = None,
    ) -> AgentResult:
        if not user_input or not user_input.strip():
            raise ValueError("user_input cannot be empty.")

        initial: AgentState = {
            "request_id": request_id or uuid.uuid4().hex,
            "user_input": user_input.strip(),
            "messages": list(history or []),
        }
        final = await self._graph.ainvoke(
            initial,
            config={"recursion_limit": self.definition.execution.max_iterations * 4 + 10},
        )
        return AgentResult(
            request_id=final["request_id"],
            agent_id=self.definition.agent_id,
            response=final.get("final_response", ""),
            new_messages=final.get("new_messages", []),
            tool_outcomes=final.get("outcomes", []),
            usage=final.get("usage", TokenUsage()),
            iterations=final.get("iterations", 0),
            stop_reason=final.get("stop_reason", "error"),
            error=final.get("error"),
        )

    async def aclose(self) -> None:
        await self.provider.aclose()
