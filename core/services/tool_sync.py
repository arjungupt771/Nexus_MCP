"""Tool catalog synchronization with change detection and event emission."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from core.events import EventBus, NullEventBus, RegistryEvent, RegistryEventType
from core.exceptions import NexusError, ServerNotFoundError
from core.interfaces import MutableServerRegistry, ToolCatalog
from core.interfaces.transport import MCPTransport
from core.models import ServerCapabilities, ToolCatalogDiff, ToolDefinition
from core.services.tool_discovery import ToolDiscoveryService

logger = logging.getLogger(__name__)


def capabilities_from_transport(transport: Any) -> ServerCapabilities | None:
    """Best-effort extraction of MCP capabilities from a transport.

    Transports expose ``server_capabilities`` (an SDK ServerCapabilities or a
    mapping) once connected; transports that do not are simply skipped.
    """
    raw = getattr(transport, "server_capabilities", None)

    if raw is None:
        return None

    def _get(name: str) -> Any:
        return raw.get(name) if isinstance(raw, dict) else getattr(raw, name, None)

    tools = _get("tools")
    list_changed = (
        tools.get("list_changed") or tools.get("listChanged")
        if isinstance(tools, dict)
        else getattr(tools, "list_changed", None)
    )

    return ServerCapabilities(
        tools=tools is not None,
        tools_list_changed=bool(list_changed),
        resources=_get("resources") is not None,
        prompts=_get("prompts") is not None,
        logging=_get("logging") is not None,
    )


class ToolSyncService:
    """Drop-in replacement for ``ToolDiscoveryService`` that also:

    * computes added / removed / changed tools against the previous snapshot,
    * records discovery success or failure on the server descriptor,
    * records observed protocol capabilities,
    * emits ``tools.updated`` and ``tools.discovery_failed`` events.

    It exposes the same ``discover`` signature, so ``ServerLifecycleManager``
    uses it without modification. A failed discovery never touches the
    existing catalog snapshot (``ToolDiscoveryService`` only upserts after a
    complete successful pass).
    """

    def __init__(
        self,
        catalog: ToolCatalog,
        registry: MutableServerRegistry | None = None,
        events: EventBus | None = None,
        discovery: ToolDiscoveryService | None = None,
    ) -> None:
        self._catalog = catalog
        self._registry = registry
        self._events = events or NullEventBus()
        self._discovery = discovery or ToolDiscoveryService(catalog)

    async def discover(
        self,
        server_id: str,
        server_name: str,
        transport: MCPTransport,
    ) -> list[ToolDefinition]:
        previous = await self._catalog.list_tools(server_id)

        try:
            current = await self._discovery.discover(
                server_id=server_id,
                server_name=server_name,
                transport=transport,
            )
        except Exception as exc:
            await self._record_failure(server_id, exc)
            raise

        diff = ToolCatalogDiff.compute(server_id, previous, current)
        await self._record_success(server_id, len(current), transport)

        if diff.has_changes:
            await self._events.publish(
                RegistryEvent(
                    type=RegistryEventType.TOOLS_UPDATED,
                    server_id=server_id,
                    data={**diff.model_dump(), "tool_count": len(current)},
                )
            )

        return current

    async def _record_success(self, server_id: str, count: int, transport: MCPTransport) -> None:
        if self._registry is None:
            return

        changes: dict[str, Any] = {
            "tool_count": count,
            "last_discovery_at": datetime.now(UTC),
            "last_discovery_error": None,
            "discovery_failure_count": 0,
        }
        capabilities = capabilities_from_transport(transport)

        if capabilities is not None:
            changes["protocol_capabilities"] = capabilities

        try:
            await self._registry.update_fields(server_id, **changes)
        except ServerNotFoundError:
            pass

    async def _record_failure(self, server_id: str, exc: Exception) -> None:
        code = exc.code if isinstance(exc, NexusError) else type(exc).__name__

        if self._registry is not None:
            try:
                current = await self._registry.get(server_id)
                failures = (current.discovery_failure_count if current else 0) + 1
                await self._registry.update_fields(
                    server_id,
                    last_discovery_error=str(code),
                    discovery_failure_count=failures,
                )
            except ServerNotFoundError:
                pass

        await self._events.publish(
            RegistryEvent(
                type=RegistryEventType.DISCOVERY_FAILED,
                server_id=server_id,
                data={"error_code": str(code)},
            )
        )
