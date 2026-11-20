"""MCPServerConnector that builds transports from persisted configuration."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Mapping

from core.exceptions import ConfigurationError
from core.interfaces import MCPTransport, ServerConfigStore
from core.models import ServerDescriptor
from core.models.server_config import HttpConnectionConfig, StdioConnectionConfig
from core.transports import StdioMCPTransport, StreamableHTTPMCPTransport

logger = logging.getLogger(__name__)


class ConfiguredServerConnector:
    """Open stdio or Streamable HTTP connections for registered servers.

    The configuration is read from the store at connect time, so a restart
    always picks up the latest persisted settings, while an already-open
    transport is never affected by later edits. Secret references
    (``${VAR}``) are resolved from ``environ`` here and nowhere else.
    """

    def __init__(
        self,
        config_store: ServerConfigStore,
        *,
        environ: Mapping[str, str] | None = None,
        on_tools_list_changed: Callable[[str], None] | None = None,
    ) -> None:
        self._store = config_store
        self._environ = environ
        self.on_tools_list_changed = on_tools_list_changed

    def _env(self) -> Mapping[str, str]:
        return self._environ if self._environ is not None else os.environ

    async def connect(self, server: ServerDescriptor) -> MCPTransport:
        record = await self._store.get(server.server_id)

        if record is None:
            raise ConfigurationError(
                f"No stored configuration for server '{server.server_id}'.",
                details={"server_id": server.server_id},
            )

        config = record.registration.config
        listener = self.on_tools_list_changed
        callback = (lambda sid=server.server_id: listener(sid)) if listener else None

        transport: MCPTransport

        if isinstance(config, StdioConnectionConfig):
            transport = StdioMCPTransport(
                config.to_runtime(self._env()),
                on_tools_list_changed=callback,
            )
        elif isinstance(config, HttpConnectionConfig):
            transport = StreamableHTTPMCPTransport(
                config.to_runtime(self._env()),
                on_tools_list_changed=callback,
            )
        else:  # pragma: no cover - discriminated union is exhaustive
            raise ConfigurationError("Unsupported transport configuration.")

        await transport.connect()
        return transport
