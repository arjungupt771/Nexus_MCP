
from __future__ import annotations

import asyncio

import pytest

from core.exceptions import (
    ServerAlreadyRegisteredError,
    ServerNotFoundError,
)
from core.interfaces import ServerRegistry, ToolCatalog
from core.models import ServerDescriptor, ToolDefinition
from core.registry import (
    InMemoryServerRegistry,
    InMemoryToolCatalog,
)


def make_server(
    server_id: str = "github",
) -> ServerDescriptor:
    return ServerDescriptor(
        server_id=server_id,
        name=f"{server_id.title()} MCP Server",
        transport="stdio",
    )


def make_tool(
    server_id: str = "github",
    name: str = "search_repositories",
) -> ToolDefinition:
    return ToolDefinition(
        server_id=server_id,
        server_name=f"{server_id.title()} MCP Server",
        name=name,
        description="Test tool",
        input_schema={
            "type": "object",
            "properties": {},
            "required": [],
        },
    )


# ---------------------------------------------------------
# Server registry tests
# ---------------------------------------------------------


def test_registry_implements_protocol() -> None:
    assert isinstance(InMemoryServerRegistry(), ServerRegistry)


@pytest.mark.asyncio
async def test_register_and_retrieve_server() -> None:
    registry = InMemoryServerRegistry()
    server = make_server()

    registered = await registry.register(server)
    retrieved = await registry.get("github")

    assert registered.server_id == "github"
    assert retrieved == server


@pytest.mark.asyncio
async def test_duplicate_registration_is_rejected() -> None:
    registry = InMemoryServerRegistry()

    await registry.register(make_server())

    with pytest.raises(ServerAlreadyRegisteredError):
        await registry.register(make_server())


@pytest.mark.asyncio
async def test_unknown_server_returns_none() -> None:
    registry = InMemoryServerRegistry()

    assert await registry.get("unknown") is None


@pytest.mark.asyncio
async def test_unregister_server() -> None:
    registry = InMemoryServerRegistry()
    server = make_server()

    await registry.register(server)

    removed = await registry.unregister("github")

    assert removed.server_id == "github"
    assert await registry.get("github") is None


@pytest.mark.asyncio
async def test_unregister_unknown_server_raises() -> None:
    registry = InMemoryServerRegistry()

    with pytest.raises(ServerNotFoundError):
        await registry.unregister("unknown")


@pytest.mark.asyncio
async def test_list_servers() -> None:
    registry = InMemoryServerRegistry()

    await registry.register(make_server("github"))
    await registry.register(make_server("notes"))

    servers = await registry.list_servers()

    assert len(servers) == 2
    assert {server.server_id for server in servers} == {
        "github",
        "notes",
    }


@pytest.mark.asyncio
async def test_registry_contains_and_count() -> None:
    registry = InMemoryServerRegistry()

    await registry.register(make_server())

    assert await registry.contains("github")
    assert not await registry.contains("unknown")
    assert await registry.count() == 1


@pytest.mark.asyncio
async def test_registry_clear() -> None:
    registry = InMemoryServerRegistry()

    await registry.register(make_server("github"))
    await registry.register(make_server("notes"))

    await registry.clear()

    assert await registry.count() == 0


@pytest.mark.asyncio
async def test_registry_returns_independent_copies() -> None:
    registry = InMemoryServerRegistry()

    original = make_server()
    await registry.register(original)

    retrieved = await registry.get("github")
    assert retrieved is not None

    retrieved.name = "Modified Name"

    stored = await registry.get("github")

    assert stored is not None
    assert stored.name == "Github MCP Server"


@pytest.mark.asyncio
async def test_concurrent_server_registration() -> None:
    registry = InMemoryServerRegistry()

    results = await asyncio.gather(
        registry.register(make_server("github")),
        registry.register(make_server("notes")),
        registry.register(make_server("calendar")),
    )

    assert len(results) == 3
    assert await registry.count() == 3


# ---------------------------------------------------------
# Tool catalog tests
# ---------------------------------------------------------


def test_catalog_implements_protocol() -> None:
    assert isinstance(InMemoryToolCatalog(), ToolCatalog)


@pytest.mark.asyncio
async def test_insert_and_retrieve_tool() -> None:
    catalog = InMemoryToolCatalog()
    tool = make_tool()

    await catalog.upsert_tools("github", [tool])

    retrieved = await catalog.get_tool(
        "github.search_repositories"
    )

    assert retrieved == tool


@pytest.mark.asyncio
async def test_tool_names_are_server_qualified() -> None:
    catalog = InMemoryToolCatalog()

    github_tool = make_tool("github")
    notes_tool = make_tool("notes")

    await catalog.upsert_tools("github", [github_tool])
    await catalog.upsert_tools("notes", [notes_tool])

    assert await catalog.count() == 2

    github_result = await catalog.get_tool(
        "github.search_repositories"
    )
    notes_result = await catalog.get_tool(
        "notes.search_repositories"
    )

    assert github_result is not None
    assert notes_result is not None
    assert github_result.server_id == "github"
    assert notes_result.server_id == "notes"


@pytest.mark.asyncio
async def test_list_tools_by_server() -> None:
    catalog = InMemoryToolCatalog()

    await catalog.upsert_tools(
        "github",
        [
            make_tool("github", "search_repositories"),
            make_tool("github", "list_issues"),
        ],
    )

    await catalog.upsert_tools(
        "notes",
        [make_tool("notes", "create_note")],
    )

    github_tools = await catalog.list_tools("github")

    assert len(github_tools) == 2
    assert all(tool.server_id == "github" for tool in github_tools)


@pytest.mark.asyncio
async def test_list_all_tools() -> None:
    catalog = InMemoryToolCatalog()

    await catalog.upsert_tools("github", [make_tool("github")])
    await catalog.upsert_tools("notes", [make_tool("notes")])

    assert len(await catalog.list_tools()) == 2


@pytest.mark.asyncio
async def test_upsert_replaces_server_snapshot() -> None:
    catalog = InMemoryToolCatalog()

    await catalog.upsert_tools(
        "github",
        [
            make_tool("github", "search_repositories"),
            make_tool("github", "list_issues"),
        ],
    )

    await catalog.upsert_tools(
        "github",
        [make_tool("github", "get_repository_details")],
    )

    tools = await catalog.list_tools("github")

    assert len(tools) == 1
    assert tools[0].name == "get_repository_details"


@pytest.mark.asyncio
async def test_upsert_does_not_remove_other_servers() -> None:
    catalog = InMemoryToolCatalog()

    await catalog.upsert_tools("github", [make_tool("github")])
    await catalog.upsert_tools("notes", [make_tool("notes")])

    await catalog.upsert_tools(
        "github",
        [make_tool("github", "list_issues")],
    )

    assert await catalog.count() == 2
    assert await catalog.get_tool(
        "notes.search_repositories"
    ) is not None


@pytest.mark.asyncio
async def test_wrong_server_ownership_is_rejected() -> None:
    catalog = InMemoryToolCatalog()

    with pytest.raises(ValueError, match="belongs to server"):
        await catalog.upsert_tools(
            "github",
            [make_tool("notes")],
        )

    assert await catalog.count() == 0


@pytest.mark.asyncio
async def test_duplicate_tools_in_one_update_are_rejected() -> None:
    catalog = InMemoryToolCatalog()

    with pytest.raises(ValueError, match="Duplicate tool"):
        await catalog.upsert_tools(
            "github",
            [
                make_tool("github"),
                make_tool("github"),
            ],
        )

    assert await catalog.count() == 0


@pytest.mark.asyncio
async def test_remove_server_tools() -> None:
    catalog = InMemoryToolCatalog()

    await catalog.upsert_tools(
        "github",
        [
            make_tool("github", "search_repositories"),
            make_tool("github", "list_issues"),
        ],
    )

    await catalog.upsert_tools(
        "notes",
        [make_tool("notes", "create_note")],
    )

    removed = await catalog.remove_server_tools("github")

    assert removed == 2
    assert await catalog.count() == 1


@pytest.mark.asyncio
async def test_remove_unknown_server_tools() -> None:
    catalog = InMemoryToolCatalog()

    assert await catalog.remove_server_tools("unknown") == 0


@pytest.mark.asyncio
async def test_catalog_returns_independent_copies() -> None:
    catalog = InMemoryToolCatalog()

    await catalog.upsert_tools("github", [make_tool()])

    retrieved = await catalog.get_tool(
        "github.search_repositories"
    )

    assert retrieved is not None
    retrieved.description = "Modified description"

    stored = await catalog.get_tool(
        "github.search_repositories"
    )

    assert stored is not None
    assert stored.description == "Test tool"


@pytest.mark.asyncio
async def test_catalog_clear() -> None:
    catalog = InMemoryToolCatalog()

    await catalog.upsert_tools("github", [make_tool()])

    await catalog.clear()

    assert await catalog.count() == 0


@pytest.mark.asyncio
async def test_concurrent_tool_updates() -> None:
    catalog = InMemoryToolCatalog()

    await asyncio.gather(
        catalog.upsert_tools(
            "github",
            [make_tool("github")],
        ),
        catalog.upsert_tools(
            "notes",
            [make_tool("notes")],
        ),
        catalog.upsert_tools(
            "calendar",
            [make_tool("calendar")],
        ),
    )

    assert await catalog.count() == 3
