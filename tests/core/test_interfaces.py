
from __future__ import annotations

from uuid import uuid4

import pytest

from core.interfaces import (
    MCPServerConnector,
    MCPTransport,
    ServerRegistry,
    ToolCatalog,
    ToolExecutor,
)
from core.models import (
    ServerDescriptor,
    ServerStatus,
    ToolCallRequest,
    ToolDefinition,
    ToolExecutionResult,
)


class FakeTransport:
    async def connect(self) -> None:
        pass

    async def disconnect(self) -> None:
        pass

    async def list_tools(self) -> list[ToolDefinition]:
        return []

    async def call_tool(
        self,
        request: ToolCallRequest,
    ) -> ToolExecutionResult:
        return ToolExecutionResult(
            request_id=request.request_id,
            server_id=request.server_id,
            tool_name=request.tool_name,
            success=True,
            data={"message": "success"},
            duration_ms=1.0,
        )


class FakeConnector:
    async def connect(
        self,
        server: ServerDescriptor,
    ) -> MCPTransport:
        return FakeTransport()


class FakeRegistry:
    def __init__(self) -> None:
        self.servers: dict[str, ServerDescriptor] = {}

    async def register(
        self,
        server: ServerDescriptor,
    ) -> ServerDescriptor:
        self.servers[server.server_id] = server
        return server

    async def update_status(
        self,
        server_id: str,
        status: ServerStatus,
        ) -> ServerDescriptor:
        server = self.servers.get(server_id)
        if server is None:
            raise KeyError(server_id)
        updated = server.model_copy(update={"status": status})
        self.servers[server_id] = updated

        return updated.model_copy(deep=True)

    async def unregister(
        self,
        server_id: str,
    ) -> ServerDescriptor:
        return self.servers.pop(server_id)

    async def get(
        self,
        server_id: str,
    ) -> ServerDescriptor | None:
        return self.servers.get(server_id)

    async def list_servers(self) -> list[ServerDescriptor]:
        return list(self.servers.values())


class FakeToolCatalog:
    def __init__(self) -> None:
        self.tools: dict[str, ToolDefinition] = {}

    async def upsert_tools(
        self,
        server_id: str,
        tools: list[ToolDefinition],
    ) -> list[ToolDefinition]:
        for tool in tools:
            self.tools[tool.qualified_name] = tool
        return tools

    async def get_tool(
        self,
        qualified_name: str,
    ) -> ToolDefinition | None:
        return self.tools.get(qualified_name)

    async def list_tools(
        self,
        server_id: str | None = None,
    ) -> list[ToolDefinition]:
        tools = list(self.tools.values())

        if server_id is not None:
            tools = [
                tool for tool in tools
                if tool.server_id == server_id
            ]

        return tools

    async def remove_server_tools(
        self,
        server_id: str,
    ) -> int:
        keys = [
            key for key, tool in self.tools.items()
            if tool.server_id == server_id
        ]

        for key in keys:
            del self.tools[key]

        return len(keys)


class FakeExecutor:
    async def execute(
        self,
        request: ToolCallRequest,
    ) -> ToolExecutionResult:
        return ToolExecutionResult(
            request_id=request.request_id,
            server_id=request.server_id,
            tool_name=request.tool_name,
            success=True,
            data={"executed": True},
            duration_ms=1.0,
        )


def create_server() -> ServerDescriptor:
    return ServerDescriptor(
        server_id="test-server",
        name="Test MCP Server",
        transport="stdio",
    )


def create_tool() -> ToolDefinition:
    return ToolDefinition(
        server_id="test-server",
        server_name="Test MCP Server",
        name="test_tool",
        description="A test tool",
        input_schema={
            "type": "object",
            "properties": {},
            "required": [],
        },
    )


def create_request() -> ToolCallRequest:
    return ToolCallRequest(
        request_id=uuid4(),
        server_id="test-server",
        tool_name="test_tool",
        arguments={},
    )


def test_fake_transport_satisfies_protocol() -> None:
    assert isinstance(FakeTransport(), MCPTransport)


def test_fake_connector_satisfies_protocol() -> None:
    assert isinstance(FakeConnector(), MCPServerConnector)


def test_fake_registry_satisfies_protocol() -> None:
    assert isinstance(FakeRegistry(), ServerRegistry)


def test_fake_tool_catalog_satisfies_protocol() -> None:
    assert isinstance(FakeToolCatalog(), ToolCatalog)


def test_fake_executor_satisfies_protocol() -> None:
    assert isinstance(FakeExecutor(), ToolExecutor)


@pytest.mark.asyncio
async def test_transport_lifecycle_and_discovery() -> None:
    transport = FakeTransport()

    await transport.connect()

    tools = await transport.list_tools()

    assert tools == []

    await transport.disconnect()


@pytest.mark.asyncio
async def test_transport_tool_invocation() -> None:
    transport = FakeTransport()
    request = create_request()

    result = await transport.call_tool(request)

    assert result.success is True
    assert result.request_id == request.request_id
    assert result.server_id == request.server_id


@pytest.mark.asyncio
async def test_connector_returns_transport() -> None:
    connector = FakeConnector()

    transport = await connector.connect(create_server())

    assert isinstance(transport, MCPTransport)


@pytest.mark.asyncio
async def test_registry_lifecycle() -> None:
    registry = FakeRegistry()
    server = create_server()

    registered = await registry.register(server)

    assert registered.server_id == server.server_id

    retrieved = await registry.get(server.server_id)

    assert retrieved == server

    servers = await registry.list_servers()

    assert len(servers) == 1

    removed = await registry.unregister(server.server_id)

    assert removed == server
    assert await registry.get(server.server_id) is None


@pytest.mark.asyncio
async def test_tool_catalog_lifecycle() -> None:
    catalog = FakeToolCatalog()
    tool = create_tool()

    inserted = await catalog.upsert_tools(
        "test-server",
        [tool],
    )

    assert len(inserted) == 1

    retrieved = await catalog.get_tool(tool.qualified_name)

    assert retrieved == tool

    listed = await catalog.list_tools("test-server")

    assert len(listed) == 1
    assert listed[0] == tool

    removed = await catalog.remove_server_tools("test-server")

    assert removed == 1
    assert await catalog.get_tool(tool.qualified_name) is None


@pytest.mark.asyncio
async def test_tool_catalog_can_list_all_servers() -> None:
    catalog = FakeToolCatalog()

    first = create_tool()

    second = ToolDefinition(
        server_id="another-server",
        server_name="Another Server",
        name="another_tool",
        description="Another test tool",
        input_schema={
            "type": "object",
            "properties": {},
            "required": [],
        },
    )

    await catalog.upsert_tools("test-server", [first])
    await catalog.upsert_tools("another-server", [second])

    all_tools = await catalog.list_tools()

    assert len(all_tools) == 2


@pytest.mark.asyncio
async def test_executor_returns_normalized_result() -> None:
    executor = FakeExecutor()
    request = create_request()

    result = await executor.execute(request)

    assert isinstance(result, ToolExecutionResult)
    assert result.success is True
    assert result.request_id == request.request_id
    assert result.data == {"executed": True}
