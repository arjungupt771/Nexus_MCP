from __future__ import annotations

from types import SimpleNamespace

import pytest
from phase5_support import FakeTransport, registration, sdk_tool

from core.events import InMemoryEventBus
from core.models import ToolCatalogDiff, ToolDefinition
from core.registry import InMemoryServerRegistry, InMemoryToolCatalog
from core.services.tool_sync import ToolSyncService, capabilities_from_transport


def tool(name: str, description: str = "d", props: dict | None = None) -> ToolDefinition:
    return ToolDefinition(
        server_id="s",
        server_name="S",
        name=name,
        description=description,
        input_schema={"type": "object", "properties": props or {}},
    )


def test_diff_detects_added_removed_and_changed():
    previous = [tool("keep"), tool("gone"), tool("edit", "old"), tool("schema")]
    current = [
        tool("keep"),
        tool("new"),
        tool("edit", "new description"),
        tool("schema", props={"q": {"type": "string"}}),
    ]
    diff = ToolCatalogDiff.compute("s", previous, current)

    assert diff.added == ["new"]
    assert diff.removed == ["gone"]
    assert diff.changed == ["edit", "schema"]
    assert diff.has_changes


def test_identical_snapshots_have_no_changes():
    assert not ToolCatalogDiff.compute("s", [tool("a")], [tool("a")]).has_changes


def test_capabilities_from_mapping_and_object_and_none():
    caps = capabilities_from_transport(
        SimpleNamespace(server_capabilities={"tools": {"listChanged": True}, "prompts": {}})
    )
    assert caps.tools and caps.tools_list_changed and caps.prompts and not caps.resources

    obj = SimpleNamespace(
        tools=SimpleNamespace(list_changed=False), resources=object(), prompts=None, logging=None
    )
    caps = capabilities_from_transport(SimpleNamespace(server_capabilities=obj))
    assert caps.tools and not caps.tools_list_changed and caps.resources

    assert capabilities_from_transport(SimpleNamespace()) is None


@pytest.fixture
async def parts():
    catalog = InMemoryToolCatalog()
    registry = InMemoryServerRegistry()
    events = InMemoryEventBus()
    await registry.register(registration("s").to_descriptor())
    sync = ToolSyncService(catalog, registry, events)
    return catalog, registry, events, sync


async def test_first_sync_adds_tools_and_records_state(parts):
    catalog, registry, events, sync = parts
    transport = FakeTransport([sdk_tool("a"), sdk_tool("b")])

    await sync.discover("s", "S", transport)

    descriptor = await registry.get("s")
    assert descriptor.tool_count == 2
    assert descriptor.last_discovery_at is not None
    assert descriptor.last_discovery_error is None
    assert descriptor.protocol_capabilities.tools_list_changed is True

    (event,) = events.recent()
    assert event.type.value == "tools.updated"
    assert event.data["added"] == ["a", "b"]


async def test_resync_handles_add_remove_and_schema_change(parts):
    catalog, _, events, sync = parts
    transport = FakeTransport([sdk_tool("a"), sdk_tool("b"), sdk_tool("c")])
    await sync.discover("s", "S", transport)

    transport.tools = [
        sdk_tool("a"),
        sdk_tool("b", properties={"x": {"type": "string"}}),
        sdk_tool("d"),
    ]
    await sync.discover("s", "S", transport)

    last = events.recent()[-1]
    assert (last.data["added"], last.data["removed"], last.data["changed"]) == (["d"], ["c"], ["b"])
    assert sorted(t.name for t in await catalog.list_tools("s")) == ["a", "b", "d"]
    assert (await catalog.get_tool("s.b")).input_schema["properties"] == {"x": {"type": "string"}}


async def test_unchanged_resync_emits_no_event(parts):
    _, _, events, sync = parts
    transport = FakeTransport([sdk_tool("a")])
    await sync.discover("s", "S", transport)
    await sync.discover("s", "S", transport)

    assert len(events.recent()) == 1


async def test_failed_discovery_is_tracked_and_keeps_existing_tools(parts):
    catalog, registry, events, sync = parts
    transport = FakeTransport([sdk_tool("a")])
    await sync.discover("s", "S", transport)

    transport.fail_list = True

    for expected_failures in (1, 2):
        with pytest.raises(Exception, match="discovery failed"):
            await sync.discover("s", "S", transport)

        descriptor = await registry.get("s")
        assert descriptor.discovery_failure_count == expected_failures
        assert descriptor.last_discovery_error == "TOOL_DISCOVERY_FAILED"

    assert [t.name for t in await catalog.list_tools("s")] == ["a"]
    assert events.recent()[-1].type.value == "tools.discovery_failed"

    transport.fail_list = False
    await sync.discover("s", "S", transport)
    descriptor = await registry.get("s")
    assert descriptor.discovery_failure_count == 0
    assert descriptor.last_discovery_error is None


async def test_sync_for_unregistered_server_still_updates_catalog():
    catalog = InMemoryToolCatalog()
    sync = ToolSyncService(catalog, InMemoryServerRegistry(), InMemoryEventBus())
    await sync.discover("s", "S", FakeTransport([sdk_tool("a")]))

    assert len(await catalog.list_tools("s")) == 1
