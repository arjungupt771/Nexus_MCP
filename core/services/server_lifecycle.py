
from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from core.events import EventBus, NullEventBus, RegistryEvent, RegistryEventType
from core.exceptions import (
    ConfigurationError,
    DiscoveryError,
    NexusError,
    ServerConnectionError,
    ServerDisabledError,
    ServerNotFoundError,
)
from core.interfaces import (
    MCPServerConnector,
    MCPTransport,
    MutableServerRegistry,
    ServerRegistry,
    ToolCatalog,
)
from core.models import ServerDescriptor, ServerStatus
from core.models.tool import ToolDefinition
from core.services.tool_discovery import ToolDiscoveryService

logger = logging.getLogger(__name__)


class ServerLifecycleManager:
    """Coordinates MCP server registration, connection and discovery."""

    def __init__(
        self,
        registry: ServerRegistry,
        connector: MCPServerConnector,
        catalog: ToolCatalog,
        discovery: ToolDiscoveryService | None = None,
        events: EventBus | None = None,
    ) -> None:
        self._events = events or NullEventBus()
        self._refresh_tasks: set[asyncio.Task[None]] = set()
        self._refresh_pending: set[str] = set()
        self._registry = registry
        self._connector = connector
        self._catalog = catalog
        self._discovery = discovery or ToolDiscoveryService(catalog)

        self._transports: dict[str, MCPTransport] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._locks_guard = asyncio.Lock()
        self._active_calls: dict[str, int] = {}
        self._activity_conditions: dict[str, asyncio.Condition] = {}

    async def _emit(
        self,
        event_type: RegistryEventType,
        server_id: str,
        **data: object,
    ) -> None:
        await self._events.publish(
            RegistryEvent(type=event_type, server_id=server_id, data=dict(data))
        )

    async def _lock_for(self, server_id: str) -> asyncio.Lock:
        """Return the shared lock for a server."""
        async with self._locks_guard:
            if server_id not in self._locks:
                self._locks[server_id] = asyncio.Lock()

            return self._locks[server_id]

    async def _condition_for(
        self,
        server_id: str,
    ) -> asyncio.Condition:
        async with self._locks_guard:
            if server_id not in self._activity_conditions:
                self._activity_conditions[server_id] = asyncio.Condition()

            return self._activity_conditions[server_id]

    @asynccontextmanager
    async def use_transport(
        self,
        server_id: str,
    ) -> AsyncIterator[MCPTransport]:
        """Acquire a transport lease for the duration of an operation."""
        lock = await self._lock_for(server_id)
        condition = await self._condition_for(server_id)
        transport: MCPTransport | None = None
        acquired = False

        try:
            async with lock:
                transport = self._transports.get(server_id)

                if transport is None:
                    raise ServerConnectionError(
                        f"Server '{server_id}' is not connected.",
                        details={"server_id": server_id},
                    )

                async with condition:
                    self._active_calls[server_id] = (
                        self._active_calls.get(server_id, 0) + 1
                    )
                    acquired = True

            yield transport

        finally:
            if acquired:
                async with condition:
                    active_calls = self._active_calls[server_id] - 1

                    if active_calls == 0:
                        self._active_calls.pop(server_id)
                    else:
                        self._active_calls[server_id] = active_calls

                    condition.notify_all()

    async def register(
        self,
        server: ServerDescriptor,
    ) -> ServerDescriptor:
        """Register a server without establishing a connection."""
        registered = await self._registry.register(server)
        await self._emit(
            RegistryEventType.SERVER_REGISTERED,
            registered.server_id,
            name=registered.name,
            transport=registered.transport.value,
            enabled=registered.enabled,
        )
        return registered

    async def get(
        self,
        server_id: str,
    ) -> ServerDescriptor:
        """Retrieve a registered server or raise if it does not exist."""
        server = await self._registry.get(server_id)

        if server is None:
            raise ServerNotFoundError(
                f"Server '{server_id}' is not registered.",
                details={"server_id": server_id},
            )

        return server

    async def list_servers(self) -> list[ServerDescriptor]:
        """Return all registered MCP servers."""
        return await self._registry.list_servers()

    async def connect(self, server_id: str) -> ServerDescriptor:
        """Connect to an MCP server and discover its tools."""
        lock = await self._lock_for(server_id)

        async with lock:
            server = await self.get(server_id)

            if server_id in self._transports:
                return server

            if not server.enabled:
                raise ServerDisabledError(
                    f"Server '{server_id}' is disabled.",
                    details={"server_id": server_id},
                )

            transport: MCPTransport | None = None

            try:
                await self._registry.update_status(
                    server_id,
                    ServerStatus.CONNECTING,
                )
                await self._emit(RegistryEventType.SERVER_CONNECTING, server_id)

                transport = await self._connector.connect(server)

                await self._discovery.discover(
                    server_id=server_id,
                    server_name=server.name,
                    transport=transport,
                )

                connected_server = await self._registry.update_status(
                    server_id,
                    ServerStatus.CONNECTED,
                )

                # Publish the transport only after successful discovery
                # and registry status update.
                self._transports[server_id] = transport
                await self._emit(
                    RegistryEventType.SERVER_CONNECTED,
                    server_id,
                    tool_count=connected_server.tool_count,
                )

                return connected_server

            except asyncio.CancelledError:
                if transport is not None:
                    try:
                        await transport.disconnect()
                    except Exception:
                        pass

                self._transports.pop(server_id, None)

                try:
                    await self._registry.update_status(
                        server_id,
                        ServerStatus.ERROR,
                    )
                except Exception:
                    pass

                raise

            except Exception as exc:
                if transport is not None:
                    try:
                        await transport.disconnect()
                    except Exception:
                        pass

                self._transports.pop(server_id, None)

                try:
                    await self._registry.update_status(
                        server_id,
                        ServerStatus.ERROR,
                    )
                except Exception:
                    pass

                await self._emit(
                    RegistryEventType.SERVER_ERROR,
                    server_id,
                    error_code=exc.code if isinstance(exc, NexusError) else type(exc).__name__,
                )

                if isinstance(exc, (ServerConnectionError, DiscoveryError, ConfigurationError)):
                    raise

                raise ServerConnectionError(
                    f"Failed to connect to server '{server_id}'.",
                    details={
                        "server_id": server_id,
                        "cause_type": type(exc).__name__,
                    },
                ) from exc

    async def refresh_tools(
        self,
        server_id: str,
    ) -> list[ToolDefinition]:
        """Refresh the discovered tools for a connected server."""
        lock = await self._lock_for(server_id)

        async with lock:
            transport = self._transports.get(server_id)

            if transport is None:
                raise ServerConnectionError(
                    f"Server '{server_id}' is not connected.",
                    details={"server_id": server_id},
                )

            server = await self._registry.get(server_id)

            if server is None:
                raise ServerNotFoundError(
                    f"Server '{server_id}' is not registered.",
                    details={"server_id": server_id},
                )

            return await self._discovery.discover(
                server_id=server_id,
                server_name=server.name,
                transport=transport,
            )

    async def _disconnect_locked(
        self,
        server_id: str,
    ) -> ServerDescriptor:
        """Disconnect a server while its lifecycle lock is held."""
        condition = await self._condition_for(server_id)

        async with condition:
            await condition.wait_for(
                lambda: self._active_calls.get(server_id, 0) == 0
            )

        transport = self._transports.get(server_id)
        removed_tools = sorted(t.name for t in await self._catalog.list_tools(server_id))

        if transport is not None:
            try:
                await transport.disconnect()

            except Exception as exc:
                await self._catalog.remove_server_tools(server_id)
                await self._emit_tools_removed(server_id, removed_tools)

                await self._registry.update_status(
                    server_id,
                    ServerStatus.ERROR,
                )

                raise ServerConnectionError(
                    f"Could not disconnect MCP server '{server_id}'.",
                    details={
                        "server_id": server_id,
                        "cause_type": type(exc).__name__,
                    },
                ) from exc

            self._transports.pop(server_id, None)

        await self._catalog.remove_server_tools(server_id)
        await self._emit_tools_removed(server_id, removed_tools)

        disconnected = await self._registry.update_status(
            server_id,
            ServerStatus.DISCONNECTED,
        )

        if transport is not None:
            await self._emit(RegistryEventType.SERVER_DISCONNECTED, server_id)

        if isinstance(self._registry, MutableServerRegistry):
            return await self._registry.update_fields(server_id, tool_count=0)

        return disconnected

    async def _emit_tools_removed(self, server_id: str, removed: list[str]) -> None:
        if removed:
            await self._emit(
                RegistryEventType.TOOLS_UPDATED,
                server_id,
                added=[],
                removed=removed,
                changed=[],
                tool_count=0,
            )

    async def disconnect(
        self,
        server_id: str,
    ) -> ServerDescriptor:
        """Disconnect a server and remove its discovered tools."""
        lock = await self._lock_for(server_id)

        async with lock:
            await self.get(server_id)

            return await self._disconnect_locked(server_id)

    async def unregister(
        self,
        server_id: str,
    ) -> ServerDescriptor:
        """Disconnect and remove a registered server."""
        lock = await self._lock_for(server_id)

        async with lock:
            await self.get(server_id)

            await self._disconnect_locked(server_id)

            # Keep the lock in the lock registry. Removing it here could
            # allow concurrent operations to create a second lock.
            removed = await self._registry.unregister(server_id)
            await self._emit(RegistryEventType.SERVER_UNREGISTERED, server_id)
            return removed

    def notify_tools_changed(self, server_id: str) -> None:
        """Schedule a catalog refresh after a tools/list_changed notification.

        Safe to call from any MCP notification callback: refreshes are
        coalesced per server and run in the background.
        """
        if server_id in self._refresh_pending:
            return

        self._refresh_pending.add(server_id)
        task = asyncio.get_running_loop().create_task(self._refresh_after_notification(server_id))
        self._refresh_tasks.add(task)
        task.add_done_callback(self._refresh_tasks.discard)

    async def _refresh_after_notification(self, server_id: str) -> None:
        try:
            # Allow the notification burst to settle, then clear the flag
            # so a change arriving mid-refresh schedules another refresh.
            await asyncio.sleep(0)
            self._refresh_pending.discard(server_id)
            await self.refresh_tools(server_id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Tool refresh after list_changed failed for '%s': %s",
                server_id,
                type(exc).__name__,
            )
        finally:
            self._refresh_pending.discard(server_id)

    async def is_connected(self, server_id: str) -> bool:
        """Return whether this manager currently holds a transport."""
        lock = await self._lock_for(server_id)

        async with lock:
            return server_id in self._transports

    async def get_transport(self, server_id: str) -> MCPTransport:
        """Return the active transport for a connected server."""
        lock = await self._lock_for(server_id)

        async with lock:
            transport = self._transports.get(server_id)

            if transport is None:
                raise ServerConnectionError(
                    f"Server '{server_id}' is not connected.",
                    details={"server_id": server_id},
                )

            return transport

    async def close_all(self) -> None:
        """Disconnect all currently registered connections."""
        for task in list(self._refresh_tasks):
            task.cancel()

        servers = await self._registry.list_servers()

        failures: list[str] = []

        for server in servers:
            try:
                await self.disconnect(server.server_id)

            except Exception:
                failures.append(server.server_id)

        if failures:
            raise ServerConnectionError(
                "One or more MCP servers could not be disconnected.",
                details={"failed_server_ids": failures},
            )

    def __repr__(self) -> str:
        return "ServerLifecycleManager()"
