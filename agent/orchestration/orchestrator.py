"""Public entry point for multi-server orchestration."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from agent.llm.models import Message, TokenUsage
from agent.llm.provider import LLMProvider
from agent.orchestration.aggregator import AggregatedResult
from agent.orchestration.graph import build_orchestration_graph
from agent.orchestration.models import ExecutionReport, Intent, StepRecord
from agent.orchestration.nodes import OrchestrationNodes, OrchestrationSettings
from agent.orchestration.state import OrchestrationState
from agent.orchestration.tools import UnifiedToolInterface
from agent.tools.bridge import MCPToolBridge, ToolPermissions

if TYPE_CHECKING:
    from core.container import NexusContainer


class OrchestrationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    response: str
    stop_reason: str  # completed | partial | failed | no_tools_needed | error
    intent: Intent | None = None
    plan_ids: list[str] = Field(default_factory=list)
    layers: list[list[str]] = Field(default_factory=list)
    report: ExecutionReport | None = None
    records: dict[str, StepRecord] = Field(default_factory=dict)
    aggregated: AggregatedResult | None = None
    replans: int = 0
    # Text generations only; structured planning calls do not report usage.
    usage: TokenUsage = Field(default_factory=TokenUsage)
    error: dict[str, Any] | None = None

    @property
    def succeeded(self) -> bool:
        return self.stop_reason in ("completed", "no_tools_needed")


class Orchestrator:
    def __init__(
        self,
        provider: LLMProvider,
        tools: UnifiedToolInterface,
        runner: MCPToolBridge,
        settings: OrchestrationSettings | None = None,
        *,
        system_instructions: str = (
            "You are Nexus, an assistant that can use tools from several servers."
        ),
    ) -> None:
        self.provider = provider
        self.tools = tools
        self.settings = settings or OrchestrationSettings()
        nodes = OrchestrationNodes(provider, tools, runner, self.settings, system_instructions)
        self._graph = build_orchestration_graph(nodes, self.settings)

    @classmethod
    def create(
        cls,
        container: NexusContainer,
        provider: LLMProvider,
        *,
        permissions: ToolPermissions | None = None,
        settings: OrchestrationSettings | None = None,
        system_instructions: str | None = None,
    ) -> Orchestrator:
        """Wire to a ``NexusContainer``: shared catalog, registry, executor, events."""
        settings = settings or OrchestrationSettings()
        tools = UnifiedToolInterface(container.catalog, permissions, container.registry)
        tools.attach(container.events)
        bridge = MCPToolBridge(
            container.catalog,
            container.executor,
            permissions,
            timeout_seconds=settings.default_tool_timeout_seconds,
        )
        kwargs = {"system_instructions": system_instructions} if system_instructions else {}
        return cls(provider, tools, bridge, settings, **kwargs)

    async def run(
        self,
        request: str,
        *,
        history: list[Message] | None = None,
        request_id: str | None = None,
    ) -> OrchestrationResult:
        if not request or not request.strip():
            raise ValueError("request cannot be empty.")

        final = await self._graph.ainvoke(
            OrchestrationState(
                request_id=request_id or uuid.uuid4().hex,
                user_input=request.strip(),
                history=list(history or []),
            ),
            config={"recursion_limit": 40 + 12 * self.settings.max_replans},
        )
        return OrchestrationResult(
            request_id=final["request_id"],
            response=final.get("final_response", ""),
            stop_reason=final.get("stop_reason", "error"),
            intent=final.get("intent"),
            plan_ids=final.get("plan_ids", []),
            layers=final.get("layers", []),
            report=final.get("report"),
            records=final.get("records", {}),
            aggregated=final.get("aggregated"),
            replans=final.get("replans", 0),
            usage=final.get("usage", TokenUsage()),
            error=final.get("error"),
        )
