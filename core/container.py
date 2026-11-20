
from __future__ import annotations

import os
from collections.abc import Mapping

from core.events import EventBus, InMemoryEventBus
from core.interfaces import (
    MCPServerConnector,
    MutableServerRegistry,
    ServerConfigStore,
    ServerRegistry,
    ToolCatalog,
)
from core.models.server_report import RestoreReport
from core.registry import (
    InMemoryServerConfigStore,
    InMemoryServerRegistry,
    InMemoryToolCatalog,
)
from core.services import ServerLifecycleManager
from core.services.configured_connector import ConfiguredServerConnector
from core.services.server_manager import ServerManagementService
from core.services.tool_discovery import ToolDiscoveryService
from core.services.tool_executor import ToolExecutorService
from core.services.tool_sync import ToolSyncService


class NexusContainer:
    """Composition root for Nexus MCP core dependencies.

    Phase 0-4 usage (``NexusContainer(connector)``) is unchanged. Phase 5
    adds an event bus, a persistent configuration store and the
    ``servers`` management service. When no ``connector`` is supplied, one
    is built from stored configuration (stdio / Streamable HTTP).
    """

    def __init__(
        self,
        connector: MCPServerConnector | None = None,
        registry: ServerRegistry | None = None,
        catalog: ToolCatalog | None = None,
        discovery: ToolDiscoveryService | None = None,
        *,
        events: EventBus | None = None,
        config_store: ServerConfigStore | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        self.registry = registry or InMemoryServerRegistry()
        self.catalog = catalog or InMemoryToolCatalog()
        self.events = events or InMemoryEventBus()
        self.config_store = config_store or InMemoryServerConfigStore()

        if connector is None:
            connector = ConfiguredServerConnector(
                self.config_store,
                environ=environ if environ is not None else os.environ,
            )

        if discovery is None:
            # Synchronizes the catalog, tracks discovery state and emits
            # tool change events. Same `discover` contract as Phase 4.
            discovery = ToolSyncService(  # type: ignore[assignment]
                catalog=self.catalog,
                registry=(
                    self.registry if isinstance(self.registry, MutableServerRegistry) else None
                ),
                events=self.events,
            )

        self.lifecycle = ServerLifecycleManager(
            registry=self.registry,
            connector=connector,
            catalog=self.catalog,
            discovery=discovery,
            events=self.events,
        )
        self.executor = ToolExecutorService(
            catalog=self.catalog,
            lifecycle=self.lifecycle,
        )

        if isinstance(connector, ConfiguredServerConnector):
            connector.on_tools_list_changed = self.lifecycle.notify_tools_changed

        self.servers: ServerManagementService | None = None

        if isinstance(self.registry, MutableServerRegistry):
            self.servers = ServerManagementService(
                registry=self.registry,
                config_store=self.config_store,
                lifecycle=self.lifecycle,
                catalog=self.catalog,
                events=self.events,
            )

        self._closed = False

    async def initialize(self, *, restore: bool = True, auto_connect: bool = True) -> RestoreReport:
        """Prepare persistence and restore registered servers."""
        await self.config_store.initialize()

        if restore and self.servers is not None:
            return await self.servers.restore(auto_connect=auto_connect)

        return RestoreReport()

    async def close(self) -> None:
        """Disconnect active MCP connections and release resources."""
        if self._closed:
            return

        try:
            await self.lifecycle.close_all()
        finally:
            await self.config_store.close()
            self._closed = True

    async def __aenter__(self) -> NexusContainer:
        if self._closed:
            raise RuntimeError("NexusContainer is already closed.")

        return self

    async def __aexit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> None:
        await self.close()

    def __repr__(self) -> str:
        return "NexusContainer()"
