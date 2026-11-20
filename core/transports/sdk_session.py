"""Lifecycle adapter for the official MCP Python SDK Client.

This module owns SDK Client creation and its async context-manager lifecycle.
It deliberately returns raw SDK results; Nexus normalization belongs to
subsequent discovery and invocation layers.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import AsyncExitStack
from typing import Any

from mcp import Client

ClientFactory = Callable[[], Client]


def build_tools_changed_handler(
    callback: Callable[[], None],
) -> Callable[[Any], Awaitable[None]]:
    """Create an SDK ``message_handler`` that reports tools/list_changed.

    The callback must be cheap and non-blocking (it typically schedules a
    refresh). Errors raised by the callback never reach the SDK session.
    """
    from mcp import types

    async def _handler(message: Any) -> None:
        notification = getattr(message, "root", message)

        if isinstance(notification, types.ToolListChangedNotification):
            try:
                callback()
            except Exception:
                pass

    return _handler


class MCPSDKSession:
    """Manage one active MCP SDK Client session."""

    def __init__(self, client_factory: ClientFactory) -> None:
        self._client_factory = client_factory
        self._client: Client | None = None
        self._owner: asyncio.Task[None] | None = None
        self._stop: asyncio.Event | None = None
        self._lock = asyncio.Lock()

    @property
    def connected(self) -> bool:
        """Whether a successfully initialized SDK client is active."""
        return self._client is not None

    @property
    def server_capabilities(self) -> Any | None:
        """Capabilities negotiated during initialization, if connected."""
        if self._client is None:
            return None

        try:
            return self._client.server_capabilities
        except Exception:
            return None

    @property
    def client(self) -> Client:
        """Return the active SDK client."""
        if self._client is None:
            raise RuntimeError("MCP session is not connected")

        return self._client

    @staticmethod
    async def _close_stack(
        exit_stack: AsyncExitStack,
    ) -> None:
        """Close SDK resources in their owning asyncio task."""
        await exit_stack.aclose()

    async def connect(self) -> None:
        """Create a fresh SDK Client and complete its handshake.

        The SDK's async context (anyio cancel scopes, subprocess/HTTP task
        groups) must be entered and exited by the *same* asyncio task. A
        dedicated owner task therefore holds the client for the whole
        session, so ``connect``, ``disconnect`` and restarts may safely be
        invoked from different tasks (startup restore, request handlers,
        background refresh).
        """
        async with self._lock:
            if self._client is not None:
                return

            loop = asyncio.get_running_loop()
            ready: asyncio.Future[Client] = loop.create_future()
            stop = asyncio.Event()
            owner = loop.create_task(
                self._own_session(ready, stop),
                name="nexus-mcp-session",
            )

            try:
                await ready
            except BaseException:
                # Handshake failed (the owner already cleaned up) or the
                # caller was cancelled mid-handshake: make sure the owner
                # is gone, then surface the original error.
                stop.set()
                owner.cancel()
                await asyncio.gather(owner, return_exceptions=True)
                raise

            # Publish state only after successful SDK initialization.
            self._owner = owner
            self._stop = stop

    async def _own_session(
        self,
        ready: asyncio.Future[Client],
        stop: asyncio.Event,
    ) -> None:
        exit_stack = AsyncExitStack()

        try:
            client = self._client_factory()
            entered = await exit_stack.enter_async_context(client)
        except BaseException as exc:
            # Preserve the original connection/handshake failure.
            try:
                await self._close_stack(exit_stack)
            except BaseException:
                pass

            if isinstance(exc, asyncio.CancelledError):
                ready.cancel()
                raise

            if not ready.done():
                ready.set_exception(exc)

            return

        if ready.done():  # caller gave up while we were connecting
            await self._close_stack(exit_stack)
            return

        self._client = entered
        ready.set_result(entered)

        cancelled = False

        try:
            await stop.wait()
        except asyncio.CancelledError:
            # Loop shutdown / owner cancelled: treat it as a close request and
            # tear the SDK context down *without* a pending exception, which
            # otherwise can send contextlib into an endless context-fix loop.
            cancelled = True
            task = asyncio.current_task()

            if task is not None:
                task.uncancel()

        # Clear state before cleanup so a failed close cannot leave the
        # object reporting an active session.
        self._client = None
        await self._close_stack(exit_stack)

        if cancelled:
            raise asyncio.CancelledError

    async def disconnect(self) -> None:
        """Close the active SDK session. Safe to call repeatedly."""
        async with self._lock:
            owner, stop = self._owner, self._stop
            self._client = None
            self._owner = None
            self._stop = None

            if owner is None or stop is None:
                return

            stop.set()
            # Shield: if this caller is cancelled, cleanup still completes.
            await asyncio.shield(owner)

    async def list_tools(
        self,
        *,
        cursor: str | None = None,
    ) -> Any:
        """Delegate paginated tool discovery to the active SDK client."""
        return await self.client.list_tools(cursor=cursor)

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        *,
        read_timeout_seconds: float | None = None,
    ) -> Any:
        """Delegate a tool call to the active SDK client."""
        return await self.client.call_tool(
            name,
            arguments,
            read_timeout_seconds=read_timeout_seconds,
        )

    async def __aenter__(self) -> MCPSDKSession:
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        await self.disconnect()
