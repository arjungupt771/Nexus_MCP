"""Dynamic MCP server management.

``ServerManagementService`` is the single entry point for runtime server
administration. It composes the existing building blocks instead of
replacing them:

* ``ServerRegistry``          - runtime descriptors / status
* ``ServerConfigStore``       - persisted configuration (no secrets)
* ``ServerLifecycleManager``  - connections, discovery, catalog cleanup
* ``EventBus``                - change notifications

Mutating operations on the same server are serialized by a per-server lock
(always acquired *before* the lifecycle lock), so concurrent callers cannot
interleave a restart with a configuration update.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from pydantic import ValidationError

from core.events import EventBus, NullEventBus, RegistryEvent, RegistryEventType
from core.exceptions import (
    NexusError,
    ServerAlreadyRegisteredError,
    ServerConnectionError,
    ServerDisabledError,
    ServerNotFoundError,
)
from core.interfaces import MutableServerRegistry, ServerConfigStore, ToolCatalog
from core.models import ServerDescriptor
from core.models.server_config import (
    ServerRecord,
    ServerRegistration,
    ServerUpdate,
    sanitize_validation_error,
)
from core.models.server_report import RestoreReport, ServerStatusReport
from core.services.server_lifecycle import ServerLifecycleManager

logger = logging.getLogger(__name__)

_DESCRIPTOR_FIELDS = ("name", "description", "tags", "metadata", "capabilities")


class ServerManagementService:
    def __init__(
        self,
        registry: MutableServerRegistry,
        config_store: ServerConfigStore,
        lifecycle: ServerLifecycleManager,
        catalog: ToolCatalog,
        events: EventBus | None = None,
        *,
        restore_concurrency: int = 4,
    ) -> None:
        if restore_concurrency < 1:
            raise ValueError("restore_concurrency must be at least 1")

        self._registry = registry
        self._store = config_store
        self._lifecycle = lifecycle
        self._catalog = catalog
        self._events = events or NullEventBus()
        self._restore_concurrency = restore_concurrency
        self._locks: dict[str, asyncio.Lock] = {}

    def _lock(self, server_id: str) -> asyncio.Lock:
        # No await between check and set, so this is race-free on one loop.
        return self._locks.setdefault(server_id, asyncio.Lock())

    async def _emit(self, event_type: RegistryEventType, server_id: str, **data: Any) -> None:
        await self._events.publish(RegistryEvent(type=event_type, server_id=server_id, data=data))

    async def _require_record(self, server_id: str) -> ServerRecord:
        record = await self._store.get(server_id)

        if record is None:
            raise ServerNotFoundError(
                f"Server '{server_id}' is not registered.",
                details={"server_id": server_id},
            )

        return record

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    async def register(
        self,
        registration: ServerRegistration | Mapping[str, Any],
    ) -> ServerStatusReport:
        """Validate, persist and register a server (without connecting)."""
        if not isinstance(registration, ServerRegistration):
            registration = ServerRegistration.from_payload(registration)

        server_id = registration.server_id

        async with self._lock(server_id):
            if await self._registry.get(server_id) is not None:
                raise ServerAlreadyRegisteredError(
                    f"Server '{server_id}' is already registered.",
                    details={"server_id": server_id},
                )

            await self._store.add(ServerRecord(registration=registration))

            try:
                await self._lifecycle.register(registration.to_descriptor())
            except BaseException:
                # Keep store and registry consistent.
                await self._store.delete(server_id)
                raise

        return await self.status(server_id)

    async def update(
        self,
        server_id: str,
        changes: ServerUpdate | Mapping[str, Any],
        *,
        restart: bool = False,
    ) -> ServerStatusReport:
        """Update metadata and/or connection config of a registered server.

        Metadata changes apply immediately. Connection-config changes on a
        *connected* server never disturb the live connection: they are
        persisted and flagged ``config_pending_restart`` until the next
        restart, unless ``restart=True`` which restarts now and rolls back
        to the previous configuration if the new one fails to connect.
        """
        if not isinstance(changes, ServerUpdate):
            try:
                changes = ServerUpdate.model_validate(dict(changes))
            except ValidationError as exc:
                raise sanitize_validation_error(exc) from None

        async with self._lock(server_id):
            old_record = await self._require_record(server_id)
            old = old_record.registration
            descriptor = await self._require_descriptor(server_id)

            merged = old.model_dump()
            given = changes.model_dump(exclude_unset=True)

            for key, value in given.items():
                if value is not None:
                    merged[key] = value

            try:
                new = ServerRegistration.model_validate(merged)
            except ValidationError as exc:
                raise sanitize_validation_error(exc) from None

            connection_changed = new.config != old.config
            changed_fields = sorted(
                key
                for key in (*_DESCRIPTOR_FIELDS, "config")
                if getattr(new, key) != getattr(old, key)
            )

            if not changed_fields:
                return await self.status(server_id)

            connected = await self._lifecycle.is_connected(server_id)
            new_record = ServerRecord(
                registration=new,
                created_at=old_record.created_at,
                updated_at=datetime.now(UTC),
            )
            await self._store.update(new_record)

            try:
                fields: dict[str, Any] = {
                    key: getattr(new, key) for key in _DESCRIPTOR_FIELDS if key in changed_fields
                }

                if connection_changed:
                    if connected:
                        fields["config_pending_restart"] = True
                    else:
                        fields["transport"] = new.config.transport
                        fields["config_pending_restart"] = False

                await self._registry.update_fields(server_id, **fields)
            except BaseException:
                await self._store.update(old_record)
                raise

            if "name" in changed_fields and connected:
                # Tool entries carry the server name; refresh them.
                await self._lifecycle.refresh_tools(server_id)

            await self._emit(
                RegistryEventType.SERVER_UPDATED,
                server_id,
                changed=changed_fields,
                config_changed=connection_changed,
                pending_restart=bool(connection_changed and connected),
            )

            if restart and connection_changed and descriptor.enabled:
                await self._restart_with_rollback(server_id, old_record, was_connected=connected)

        return await self.status(server_id)

    async def _restart_with_rollback(
        self,
        server_id: str,
        previous: ServerRecord,
        *,
        was_connected: bool,
    ) -> None:
        try:
            await self._restart_locked(server_id)
        except Exception as exc:
            # Revert persisted config and descriptor, then try to bring the
            # previous configuration back so the caller is not left dark.
            await self._store.update(previous)
            old = previous.registration
            await self._registry.update_fields(
                server_id,
                transport=old.config.transport,
                config_pending_restart=False,
            )

            if was_connected:
                try:
                    await self._lifecycle.connect(server_id)
                except Exception:
                    logger.warning("Rollback reconnect failed for '%s'.", server_id)

            raise ServerConnectionError(
                f"Server '{server_id}' could not be restarted with the new configuration; "
                "the previous configuration was restored.",
                details={
                    "server_id": server_id,
                    "rolled_back": True,
                    "cause": exc.code if isinstance(exc, NexusError) else type(exc).__name__,
                },
            ) from exc

    async def unregister(self, server_id: str) -> ServerStatusReport:
        """Disconnect, remove runtime state and delete persisted config."""
        async with self._lock(server_id):
            report = await self.status(server_id)
            await self._lifecycle.unregister(server_id)
            await self._store.delete(server_id)

        return report

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    async def connect(self, server_id: str) -> ServerStatusReport:
        async with self._lock(server_id):
            await self._connect_locked(server_id)

        return await self.status(server_id)

    async def _connect_locked(self, server_id: str) -> None:
        descriptor = await self._require_descriptor(server_id)

        if descriptor.config_pending_restart and not await self._lifecycle.is_connected(server_id):
            record = await self._require_record(server_id)
            await self._registry.update_fields(
                server_id,
                transport=record.registration.config.transport,
                config_pending_restart=False,
            )

        await self._lifecycle.connect(server_id)

    async def disconnect(self, server_id: str) -> ServerStatusReport:
        async with self._lock(server_id):
            await self._require_descriptor(server_id)
            await self._lifecycle.disconnect(server_id)

        return await self.status(server_id)

    async def restart(self, server_id: str) -> ServerStatusReport:
        async with self._lock(server_id):
            descriptor = await self._require_descriptor(server_id)

            if not descriptor.enabled:
                raise ServerDisabledError(
                    f"Server '{server_id}' is disabled.",
                    details={"server_id": server_id},
                )

            await self._restart_locked(server_id)

        return await self.status(server_id)

    async def _restart_locked(self, server_id: str) -> None:
        if await self._lifecycle.is_connected(server_id):
            await self._lifecycle.disconnect(server_id)

        await self._connect_locked(server_id)

    async def enable(self, server_id: str, *, connect: bool = False) -> ServerStatusReport:
        async with self._lock(server_id):
            await self._set_enabled(server_id, True)

            if connect:
                await self._connect_locked(server_id)

        return await self.status(server_id)

    async def disable(self, server_id: str) -> ServerStatusReport:
        """Disable a server: disconnects it and blocks further connections."""
        async with self._lock(server_id):
            await self._require_descriptor(server_id)

            if await self._lifecycle.is_connected(server_id):
                await self._lifecycle.disconnect(server_id)

            await self._set_enabled(server_id, False)

        return await self.status(server_id)

    async def _set_enabled(self, server_id: str, enabled: bool) -> None:
        old_record = await self._require_record(server_id)
        descriptor = await self._require_descriptor(server_id)

        if descriptor.enabled == enabled and old_record.registration.enabled == enabled:
            return

        new_registration = old_record.registration.model_copy(update={"enabled": enabled})
        await self._store.update(
            ServerRecord(
                registration=new_registration,
                created_at=old_record.created_at,
                updated_at=datetime.now(UTC),
            )
        )

        try:
            await self._registry.update_fields(server_id, enabled=enabled)
        except BaseException:
            await self._store.update(old_record)
            raise

        await self._emit(
            RegistryEventType.SERVER_ENABLED if enabled else RegistryEventType.SERVER_DISABLED,
            server_id,
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    async def _require_descriptor(self, server_id: str) -> ServerDescriptor:
        descriptor = await self._registry.get(server_id)

        if descriptor is None:
            raise ServerNotFoundError(
                f"Server '{server_id}' is not registered.",
                details={"server_id": server_id},
            )

        return descriptor

    async def _build_report(
        self,
        descriptor: ServerDescriptor,
        record: ServerRecord | None,
    ) -> ServerStatusReport:
        tools = await self._catalog.list_tools(descriptor.server_id)

        return ServerStatusReport(
            **descriptor.model_dump(exclude={"version"}),
            connected=await self._lifecycle.is_connected(descriptor.server_id),
            tools=sorted(tool.name for tool in tools),
            config=record.registration.config.public_view() if record else {},
        )

    async def status(self, server_id: str) -> ServerStatusReport:
        descriptor = await self._require_descriptor(server_id)
        return await self._build_report(descriptor, await self._store.get(server_id))

    async def list_servers(
        self,
        *,
        enabled: bool | None = None,
        tag: str | None = None,
    ) -> list[ServerStatusReport]:
        records = {r.registration.server_id: r for r in await self._store.list_records()}
        reports: list[ServerStatusReport] = []

        for descriptor in sorted(await self._registry.list_servers(), key=lambda d: d.server_id):
            if enabled is not None and descriptor.enabled != enabled:
                continue

            if tag is not None and tag.strip().lower() not in descriptor.tags:
                continue

            reports.append(await self._build_report(descriptor, records.get(descriptor.server_id)))

        return reports

    # ------------------------------------------------------------------
    # Startup restoration
    # ------------------------------------------------------------------

    async def restore(self, *, auto_connect: bool = True) -> RestoreReport:
        """Load persisted servers into the registry and reconnect enabled ones.

        A server that fails to connect is reported (and left in ERROR
        status) but never prevents the others - or Nexus itself - from
        starting.
        """
        await self._store.initialize()
        report = RestoreReport()
        to_connect: list[str] = []

        for record in await self._store.list_records():
            registration = record.registration
            server_id = registration.server_id

            async with self._lock(server_id):
                if await self._registry.get(server_id) is not None:
                    report.already_registered.append(server_id)
                    continue

                await self._lifecycle.register(registration.to_descriptor())

            report.restored.append(server_id)

            if not registration.enabled:
                report.skipped_disabled.append(server_id)
            elif auto_connect:
                to_connect.append(server_id)

        semaphore = asyncio.Semaphore(self._restore_concurrency)

        async def _connect(server_id: str) -> None:
            async with semaphore:
                try:
                    await self.connect(server_id)
                    report.connected.append(server_id)
                except Exception as exc:
                    code = exc.code if isinstance(exc, NexusError) else type(exc).__name__
                    report.failed[server_id] = str(code)
                    logger.warning("Restore could not connect '%s': %s", server_id, code)

        await asyncio.gather(*(_connect(server_id) for server_id in to_connect))
        report.connected.sort()
        return report

    def __repr__(self) -> str:
        return "ServerManagementService()"
