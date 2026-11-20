"""MCP Streamable HTTP transport using the official MCP SDK."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

from mcp import Client
from mcp.client.streamable_http import (
    httpx2,
    streamable_http_client,
)

from core.transports.http_config import HTTPServerConfig
from core.transports.sdk_session import MCPSDKSession, build_tools_changed_handler


class StreamableHTTPMCPTransport:
    """Connect to a remote MCP server over Streamable HTTP."""

    def __init__(
        self,
        config: HTTPServerConfig,
        *,
        on_tools_list_changed: Callable[[], None] | None = None,
    ) -> None:
        self.config = config
        self._on_tools_list_changed = on_tools_list_changed

        self._session = MCPSDKSession(
            client_factory=self._create_client,
        )

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

    @asynccontextmanager
    async def _open_transport(self) -> AsyncIterator[Any]:
        """Create and manage the SDK-compatible HTTP client."""

        timeout = httpx2.Timeout(
            timeout=self.config.request_timeout_seconds,
            connect=self.config.connect_timeout_seconds,
            read=self.config.read_timeout_seconds,
        )

        async with httpx2.AsyncClient(
            headers=dict(self.config.headers),
            timeout=timeout,
            verify=self.config.verify_tls,
            follow_redirects=False,
        ) as http_client:
            async with streamable_http_client(
                self.config.url,
                http_client=http_client,
                terminate_on_close=self.config.terminate_on_close,
                max_sse_event_size=self.config.max_sse_event_size,
            ) as streams:
                yield streams

    def _create_client(self) -> Client:
        """Create a fresh SDK Client for each connection lifecycle."""

        return Client(
            self._open_transport(),
            raise_exceptions=True,
            read_timeout_seconds=self.config.read_timeout_seconds,
            message_handler=self._message_handler(),
            mode="auto",
        )

    async def connect(self) -> None:
        """Establish the remote MCP session."""
        await self._session.connect()

    async def disconnect(self) -> None:
        """Close the MCP session and HTTP resources."""
        await self._session.disconnect()

    async def list_tools(
        self,
        *,
        cursor: str | None = None,
    ) -> Any:
        """Retrieve a page of remote MCP tools."""
        return await self._session.list_tools(cursor=cursor)

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        *,
        read_timeout_seconds: float | None = None,
    ) -> Any:
        """Invoke a remote MCP tool."""
        return await self._session.call_tool(
            name,
            arguments,
            read_timeout_seconds=read_timeout_seconds,
        )

    async def __aenter__(self) -> StreamableHTTPMCPTransport:
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        await self.disconnect()
