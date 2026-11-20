"""MCP stdio transport implementation using MCP SDK 2.x."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from mcp import Client, StdioServerParameters

from core.transports.sdk_session import MCPSDKSession, build_tools_changed_handler
from core.transports.stdio_config import StdioServerConfig


class StdioMCPTransport:
    """Connect to an MCP server launched as a local subprocess."""

    def __init__(
        self,
        config: StdioServerConfig,
        *,
        on_tools_list_changed: Callable[[], None] | None = None,
    ) -> None:
        self.config = config
        self._on_tools_list_changed = on_tools_list_changed

        self._session = MCPSDKSession(client_factory=self._create_client)

    @property
    def connected(self) -> bool:
        return self._session.connected

    @property
    def server_capabilities(self) -> Any | None:
        """Capabilities the server declared during initialization."""
        return self._session.server_capabilities

    def _message_handler(self) -> Any | None:
        if self._on_tools_list_changed is None:
            return None

        return build_tools_changed_handler(self._on_tools_list_changed)

    def _create_client(self) -> Client:
        """Create a new SDK client for each connection lifecycle."""

        parameters = StdioServerParameters(
            command=self.config.command,
            args=list(self.config.args),
            env=dict(self.config.env) if self.config.env else None,
            cwd=self.config.cwd,
            encoding=self.config.encoding,
            encoding_error_handler=self.config.encoding_error_handler,
        )

        return Client(
            parameters,
            raise_exceptions=True,
            read_timeout_seconds=self.config.read_timeout_seconds,
            message_handler=self._message_handler(),
            mode="auto",
        )

    async def connect(self) -> None:
        """Launch the subprocess and establish the MCP session."""
        await self._session.connect()

    async def disconnect(self) -> None:
        """Close the MCP session and terminate its subprocess."""
        await self._session.disconnect()

    async def list_tools(self, *, cursor: str | None = None) -> Any:
        """Retrieve a page of tools exposed by the MCP server."""
        return await self._session.list_tools(cursor=cursor)

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        *,
        read_timeout_seconds: float | None = None,
    ) -> Any:
        """Invoke a tool exposed by the connected MCP server."""
        return await self._session.call_tool(
            name,
            arguments,
            read_timeout_seconds=read_timeout_seconds,
        )

    async def __aenter__(self) -> StdioMCPTransport:
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        await self.disconnect()
