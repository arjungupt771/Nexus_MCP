"""Unified, always-current view of every connected server's tools (7.1).

Reads the Phase 5 catalog on every call, so servers that connect, disconnect
or change their tool lists are reflected immediately. Naming and permissions
are shared with ``MCPToolBridge`` so a name offered here always resolves there.
"""

from __future__ import annotations

import re
from collections.abc import Collection
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agent.llm.models import ToolSpec
from agent.tools.bridge import ToolPermissions, build_bindings
from core.events import EventBus, RegistryEvent
from core.interfaces import ServerRegistry, ToolCatalog
from core.models import ServerDescriptor, ToolDefinition

_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an and the to of in on for with from by at as is are be it this that my me i "
    "please can you could would then also all any into out up get make".split()
)


def tokenize(text: str) -> set[str]:
    """Lowercase word tokens with light plural stripping (deterministic)."""
    tokens = set()

    for raw in _TOKEN.findall(text.lower()):
        if raw in _STOPWORDS or len(raw) < 2:
            continue

        tokens.add(raw[:-1] if len(raw) > 3 and raw.endswith("s") else raw)

    return tokens


class ToolInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    spec_name: str
    qualified_name: str
    server_id: str
    server_name: str
    tool_name: str
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)

    @property
    def required(self) -> list[str]:
        return list(self.input_schema.get("required", []))

    def signature(self) -> str:
        """Compact one-line description for prompts: ``name(arg*: type, ...)``."""
        required = set(self.required)
        args = ", ".join(
            f"{name}{'*' if name in required else ''}: {spec.get('type', 'any')}"
            for name, spec in (self.input_schema.get("properties") or {}).items()
        )
        return f"{self.spec_name}({args}) - [{self.server_name}] {self.description}".strip()


class UnifiedToolInterface:
    def __init__(
        self,
        catalog: ToolCatalog,
        permissions: ToolPermissions | None = None,
        registry: ServerRegistry | None = None,
    ) -> None:
        self._catalog = catalog
        self._permissions = permissions or ToolPermissions()
        self._registry = registry
        self.version = 0

    def attach(self, events: EventBus) -> None:
        """Count registry/tool changes so callers can detect catalog drift."""

        def _bump(event: RegistryEvent) -> None:
            self.version += 1

        events.subscribe(_bump)

    async def _servers(self) -> dict[str, ServerDescriptor]:
        if self._registry is None:
            return {}

        return {s.server_id: s for s in await self._registry.list_servers()}

    async def list_tools(self, servers: Collection[str] | None = None) -> list[ToolInfo]:
        tools: list[ToolDefinition] = await self._catalog.list_tools()
        bindings = build_bindings(tools)
        by_qualified = {t.qualified_name: t for t in tools}
        descriptors = await self._servers()
        infos: list[ToolInfo] = []

        for spec_name, binding in bindings.items():
            if not self._permissions.allows(binding.qualified_name):
                continue

            if servers is not None and binding.server_id not in servers:
                continue

            tool = by_qualified[binding.qualified_name]
            descriptor = descriptors.get(binding.server_id)
            infos.append(
                ToolInfo(
                    spec_name=spec_name,
                    qualified_name=binding.qualified_name,
                    server_id=binding.server_id,
                    server_name=binding.server_name,
                    tool_name=binding.tool_name,
                    description=tool.description,
                    input_schema=tool.input_schema,
                    tags=sorted(
                        {
                            *(descriptor.tags if descriptor else []),
                            *(descriptor.capabilities if descriptor else []),
                        }
                    ),
                )
            )

        return sorted(infos, key=lambda i: i.qualified_name)

    async def by_server(self) -> dict[str, list[ToolInfo]]:
        grouped: dict[str, list[ToolInfo]] = {}

        for info in await self.list_tools():
            grouped.setdefault(info.server_id, []).append(info)

        return grouped

    async def get(self, name: str) -> ToolInfo | None:
        """Look up by provider-safe name or qualified ``server.tool`` name."""
        return next(
            (t for t in await self.list_tools() if name in (t.spec_name, t.qualified_name)), None
        )

    async def specs(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                name=t.spec_name,
                description=f"[{t.server_name}] {t.description}".strip(),
                parameters=t.input_schema,
            )
            for t in await self.list_tools()
        ]

    async def search(
        self,
        query: str,
        *,
        limit: int = 15,
        servers: Collection[str] | None = None,
    ) -> list[ToolInfo]:
        """Keyword relevance across tool names, descriptions and server metadata."""
        query_tokens = tokenize(query)
        scored: list[tuple[float, str, ToolInfo]] = []

        for tool in await self.list_tools(servers):
            name_tokens = tokenize(tool.tool_name.replace("_", " "))
            description_tokens = tokenize(tool.description)
            server_tokens = tokenize(" ".join([tool.server_name, tool.server_id, *tool.tags]))
            score = (
                3.0 * len(query_tokens & name_tokens)
                + 2.0 * len(query_tokens & description_tokens)
                + 1.5 * len(query_tokens & server_tokens)
            )

            if score > 0:
                scored.append((-score, tool.qualified_name, tool))

        return [tool for _, _, tool in sorted(scored)[:limit]]
