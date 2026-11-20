from __future__ import annotations

import asyncio
import json

import pytest
from phase5_support import (
    FakeConnector,
    event_types,
    http_payload,
    make_container,
    sdk_tool,
    stdio_payload,
)

from core.exceptions import (
    ConfigurationError,
    RegistryPersistenceError,
    ServerAlreadyRegisteredError,
    ServerConnectionError,
    ServerDisabledError,
    ServerNotFoundError,
)
from core.models import ServerStatus
from core.registry import InMemoryServerConfigStore, SQLiteServerConfigStore

# --------------------------------------------------------------------- registration


async def test_register_persists_and_exposes_metadata_without_connecting():
    container = make_container()
    report = await container.servers.register(stdio_payload("notes", metadata={"owner": "me"}))

    assert report.status is ServerStatus.DISCONNECTED
    assert report.connected is False
    assert (report.tags, report.capabilities) == (["test"], ["notes"])
    assert report.metadata == {"owner": "me"}
    assert report.transport.value == "stdio"
    assert (await container.config_store.get("notes")) is not None
    assert event_types(container) == ["server.registered"]


async def test_duplicate_registration_is_rejected_and_changes_nothing():
    container = make_container()
    await container.servers.register(stdio_payload("notes", description="original"))

    with pytest.raises(ServerAlreadyRegisteredError):
        await container.servers.register(stdio_payload("notes", description="intruder"))

    assert (await container.servers.status("notes")).description == "original"
    assert len(await container.config_store.list_records()) == 1


async def test_duplicate_detection_is_case_insensitive_via_normalization():
    container = make_container()
    await container.servers.register(stdio_payload("Notes"))

    with pytest.raises(ServerAlreadyRegisteredError):
        await container.servers.register(stdio_payload("NOTES"))


async def test_invalid_configuration_is_rejected_before_any_state_changes():
    container = make_container()

    with pytest.raises(ConfigurationError):
        bad = {"transport": "stdio", "command": ""}
        await container.servers.register(stdio_payload(config=bad))

    assert await container.servers.list_servers() == []
    assert await container.config_store.list_records() == []


async def test_failed_persistence_leaves_registry_untouched():
    class BrokenStore(InMemoryServerConfigStore):
        async def add(self, record):
            raise RegistryPersistenceError("disk full")

    container = make_container(config_store=BrokenStore())

    with pytest.raises(RegistryPersistenceError):
        await container.servers.register(stdio_payload())

    assert await container.registry.list_servers() == []
    assert event_types(container) == []


async def test_http_servers_can_be_registered():
    container = make_container()
    report = await container.servers.register(http_payload())

    assert report.transport.value == "streamable_http"
    assert report.config["headers"] == {"Authorization": "Bearer ${REMOTE_TOKEN}"}


async def test_concurrent_registration_of_same_id_has_one_winner():
    container = make_container()
    results = await asyncio.gather(
        *(container.servers.register(stdio_payload("race")) for _ in range(15)),
        return_exceptions=True,
    )

    assert sum(not isinstance(r, Exception) for r in results) == 1
    errors = [r for r in results if isinstance(r, Exception)]
    assert all(isinstance(r, ServerAlreadyRegisteredError) for r in errors)
    assert len(await container.config_store.list_records()) == 1


async def test_concurrent_registration_of_distinct_ids_all_succeed():
    container = make_container()
    await asyncio.gather(*(container.servers.register(stdio_payload(f"s{i}")) for i in range(20)))

    assert len(await container.servers.list_servers()) == 20
    assert len(await container.config_store.list_records()) == 20


# ------------------------------------------------------------ connect / tools / capabilities


async def test_connect_synchronizes_tools_and_tracks_capabilities():
    connector = FakeConnector()
    connector.transport_for("notes").tools = [sdk_tool("create_note"), sdk_tool("get_note")]
    container = make_container(connector)
    await container.servers.register(stdio_payload("notes"))

    report = await container.servers.connect("notes")

    assert report.connected and report.status is ServerStatus.CONNECTED
    assert report.tools == ["create_note", "get_note"] and report.tool_count == 2
    assert report.last_discovery_at is not None and report.last_discovery_error is None
    assert report.protocol_capabilities.tools and report.protocol_capabilities.tools_list_changed
    assert report.transport.value == "stdio"
    # Ownership is preserved in the catalog.
    tool = await container.catalog.get_tool("notes.create_note")
    assert tool.server_id == "notes" and tool.server_name == "Notes Server"


async def test_event_sequence_for_connect_and_disconnect():
    container = make_container()
    await container.servers.register(stdio_payload("notes"))
    await container.servers.connect("notes")
    await container.servers.disconnect("notes")

    assert event_types(container) == [
        "server.registered",
        "server.connecting",
        "tools.updated",
        "server.connected",
        "tools.updated",
        "server.disconnected",
    ]
    removed = container.events.recent()[-2]
    assert removed.data["removed"] == ["create_note"]


async def test_disconnect_removes_stale_tools_and_resets_count():
    container = make_container()
    await container.servers.register(stdio_payload("notes"))
    await container.servers.connect("notes")
    report = await container.servers.disconnect("notes")

    assert report.tools == [] and report.tool_count == 0 and not report.connected
    assert await container.catalog.list_tools("notes") == []


async def test_connection_failure_sets_error_status_and_emits_event():
    connector = FakeConnector()
    connector.fail_for.add("notes")
    container = make_container(connector)
    await container.servers.register(stdio_payload("notes"))

    with pytest.raises(ServerConnectionError):
        await container.servers.connect("notes")

    report = await container.servers.status("notes")
    assert report.status is ServerStatus.ERROR and not report.connected
    assert "server.error" in event_types(container)
    assert await container.catalog.list_tools("notes") == []


async def test_discovery_failure_is_tracked_and_leaves_no_tools():
    connector = FakeConnector()
    connector.transport_for("notes").fail_list = True
    container = make_container(connector)
    await container.servers.register(stdio_payload("notes"))

    with pytest.raises(Exception, match="discovery failed"):
        await container.servers.connect("notes")

    report = await container.servers.status("notes")
    assert report.status is ServerStatus.ERROR
    assert report.last_discovery_error == "TOOL_DISCOVERY_FAILED"
    assert report.discovery_failure_count == 1
    assert report.tools == []
    assert {"tools.discovery_failed", "server.error"} <= set(event_types(container))


async def test_concurrent_connects_open_one_connection():
    connector = FakeConnector()
    connector.connect_delay = 0.02
    container = make_container(connector)
    await container.servers.register(stdio_payload("notes"))

    await asyncio.gather(*(container.servers.connect("notes") for _ in range(10)))

    assert connector.connects == ["notes"]


async def test_tools_from_multiple_servers_stay_owned_by_their_server():
    connector = FakeConnector()
    connector.transport_for("a").tools = [sdk_tool("search")]
    connector.transport_for("b").tools = [sdk_tool("search")]
    container = make_container(connector)

    for server_id in ("a", "b"):
        await container.servers.register(stdio_payload(server_id))
        await container.servers.connect(server_id)

    names = sorted(t.qualified_name for t in await container.catalog.list_tools())
    assert names == ["a.search", "b.search"]

    await container.servers.disconnect("a")
    assert [t.qualified_name for t in await container.catalog.list_tools()] == ["b.search"]


async def test_tools_list_changed_notification_refreshes_the_catalog():
    connector = FakeConnector()
    container = make_container(connector)
    await container.servers.register(stdio_payload("notes"))
    await container.servers.connect("notes")

    connector.transports["notes"].tools = [sdk_tool("create_note"), sdk_tool("delete_note")]
    # Several notifications in a burst are coalesced into one refresh.
    for _ in range(5):
        container.lifecycle.notify_tools_changed("notes")

    for _ in range(50):
        if len(await container.catalog.list_tools("notes")) == 2:
            break
        await asyncio.sleep(0.01)

    assert sorted(t.name for t in await container.catalog.list_tools("notes")) == [
        "create_note",
        "delete_note",
    ]
    assert (await container.servers.status("notes")).tool_count == 2
    assert connector.transports["notes"].list_calls <= 3  # connect + coalesced refresh(es)
    assert container.events.recent()[-1].data["added"] == ["delete_note"]


async def test_notification_for_disconnected_server_is_harmless():
    container = make_container()
    await container.servers.register(stdio_payload("notes"))
    container.lifecycle.notify_tools_changed("notes")
    await asyncio.sleep(0.05)

    assert await container.catalog.list_tools("notes") == []


# --------------------------------------------------------------------- updates


async def test_metadata_update_applies_immediately_without_touching_the_connection():
    connector = FakeConnector()
    container = make_container(connector)
    await container.servers.register(stdio_payload("notes"))
    await container.servers.connect("notes")
    transport = connector.transports["notes"]

    report = await container.servers.update(
        "notes", {"description": "new", "tags": ["x", "y"], "capabilities": ["z"]}
    )

    assert (report.description, report.tags, report.capabilities) == ("new", ["x", "y"], ["z"])
    assert report.connected and transport.disconnect_calls == 0
    assert connector.connects == ["notes"]
    stored = (await container.config_store.get("notes")).registration
    assert stored.description == "new"
    assert container.events.recent()[-1].data["changed"] == ["capabilities", "description", "tags"]


async def test_noop_update_emits_nothing():
    container = make_container()
    await container.servers.register(stdio_payload("notes"))
    before = len(container.events.recent())

    await container.servers.update("notes", {"description": "Test server"})

    assert len(container.events.recent()) == before


async def test_rename_of_connected_server_refreshes_tool_ownership_name():
    container = make_container()
    await container.servers.register(stdio_payload("notes"))
    await container.servers.connect("notes")

    await container.servers.update("notes", {"name": "Renamed"})

    assert (await container.catalog.get_tool("notes.create_note")).server_name == "Renamed"


async def test_config_change_on_connected_server_does_not_corrupt_active_connection():
    connector = FakeConnector()
    container = make_container(connector)
    await container.servers.register(stdio_payload("notes"))
    await container.servers.connect("notes")
    live = connector.transports["notes"]

    report = await container.servers.update(
        "notes",
        {"config": {"transport": "stdio", "command": "python", "args": ["-m", "other"]}},
    )

    assert report.config_pending_restart is True
    assert report.connected and report.status is ServerStatus.CONNECTED
    assert live.connected and live.disconnect_calls == 0
    assert connector.connects == ["notes"]
    assert report.config["args"] == ["-m", "other"]  # persisted
    assert [t.name for t in await container.catalog.list_tools("notes")] == ["create_note"]


async def test_pending_config_is_applied_on_next_restart():
    connector = FakeConnector()
    container = make_container(connector)
    await container.servers.register(stdio_payload("notes"))
    await container.servers.connect("notes")
    await container.servers.update(
        "notes",
        {
            "config": {
                "transport": "streamable_http",
                "url": "https://example.com/mcp",
            }
        },
    )

    report = await container.servers.restart("notes")

    assert report.config_pending_restart is False
    assert report.transport.value == "streamable_http"
    assert connector.connects == ["notes", "notes"]


async def test_update_with_restart_applies_new_config_immediately():
    connector = FakeConnector()
    container = make_container(connector)
    await container.servers.register(stdio_payload("notes"))
    await container.servers.connect("notes")

    report = await container.servers.update(
        "notes",
        {"config": {"transport": "stdio", "command": "python", "args": ["-m", "v2"]}},
        restart=True,
    )

    assert report.connected and not report.config_pending_restart
    assert connector.connects == ["notes", "notes"]


async def test_failed_restart_rolls_back_config_and_reconnects_previous():
    connector = FakeConnector()
    container = make_container(connector)
    await container.servers.register(stdio_payload("notes"))
    await container.servers.connect("notes")
    original = (await container.config_store.get("notes")).registration.config

    real_connect = connector.connect
    attempts = {"n": 0}

    async def flaky(server):
        attempts["n"] += 1

        if attempts["n"] == 1:  # the restart with the NEW config fails
            raise RuntimeError("bad new config")

        return await real_connect(server)

    connector.connect = flaky

    with pytest.raises(ServerConnectionError) as exc_info:
        await container.servers.update(
            "notes",
            {"config": {"transport": "stdio", "command": "python", "args": ["-m", "broken"]}},
            restart=True,
        )

    assert exc_info.value.details["rolled_back"] is True
    assert (await container.config_store.get("notes")).registration.config == original
    report = await container.servers.status("notes")
    assert report.connected and report.status is ServerStatus.CONNECTED
    assert report.config_pending_restart is False


async def test_invalid_update_is_rejected_and_nothing_changes():
    container = make_container()
    await container.servers.register(stdio_payload("notes"))

    with pytest.raises(ConfigurationError):
        await container.servers.update(
            "notes",
            {"config": {"transport": "stdio", "command": "x", "env": {"API_KEY": "plain"}}},
        )

    with pytest.raises(ConfigurationError):
        await container.servers.update("notes", {"server_id": "other"})

    stored = (await container.config_store.get("notes")).registration
    assert stored.config.command == "python" and stored.config.env == {}


async def test_update_of_unknown_server_fails():
    container = make_container()

    with pytest.raises(ServerNotFoundError):
        await container.servers.update("ghost", {"description": "x"})


async def test_failed_store_write_during_update_changes_nothing():
    container = make_container()
    await container.servers.register(stdio_payload("notes"))

    async def boom(record):
        raise RegistryPersistenceError("disk full")

    container.config_store.update = boom

    with pytest.raises(RegistryPersistenceError):
        await container.servers.update("notes", {"description": "changed"})

    assert (await container.servers.status("notes")).description == "Test server"


async def test_concurrent_updates_are_serialized_and_consistent():
    container = make_container()
    await container.servers.register(stdio_payload("notes"))

    await asyncio.gather(
        *(container.servers.update("notes", {"description": f"d{i}"}) for i in range(20))
    )

    stored = (await container.config_store.get("notes")).registration.description
    assert (await container.servers.status("notes")).description == stored


# --------------------------------------------------------- enable / disable / restart / unregister


async def test_disable_disconnects_blocks_connect_and_persists():
    container = make_container()
    await container.servers.register(stdio_payload("notes"))
    await container.servers.connect("notes")

    report = await container.servers.disable("notes")

    assert report.enabled is False and not report.connected and report.tools == []
    assert (await container.config_store.get("notes")).registration.enabled is False
    assert "server.disabled" in event_types(container)

    with pytest.raises(ServerDisabledError):
        await container.servers.connect("notes")

    with pytest.raises(ServerDisabledError):
        await container.servers.restart("notes")


async def test_enable_restores_connectability_and_can_connect():
    container = make_container()
    await container.servers.register(stdio_payload("notes", enabled=False))

    report = await container.servers.enable("notes", connect=True)

    assert report.enabled and report.connected
    assert "server.enabled" in event_types(container)


async def test_restart_reconnects_and_refreshes_tools():
    connector = FakeConnector()
    container = make_container(connector)
    await container.servers.register(stdio_payload("notes"))
    await container.servers.connect("notes")
    first = connector.transports["notes"]
    connector.transports["notes"].tools = [sdk_tool("create_note"), sdk_tool("extra")]

    report = await container.servers.restart("notes")

    assert first.disconnect_calls == 1 and connector.transports["notes"] is not first
    assert report.tools == ["create_note", "extra"]


async def test_unregister_removes_everything_and_emits_event():
    container = make_container()
    await container.servers.register(stdio_payload("notes"))
    await container.servers.connect("notes")

    await container.servers.unregister("notes")

    assert await container.servers.list_servers() == []
    assert await container.catalog.list_tools("notes") == []
    assert await container.config_store.get("notes") is None
    assert event_types(container)[-1] == "server.unregistered"

    with pytest.raises(ServerNotFoundError):
        await container.servers.status("notes")

    # The id is reusable afterwards.
    await container.servers.register(stdio_payload("notes"))


async def test_list_servers_filters_by_enabled_state_and_tag():
    container = make_container()
    await container.servers.register(stdio_payload("a", tags=["code"]))
    await container.servers.register(stdio_payload("b", tags=["calendar"], enabled=False))

    assert [s.server_id for s in await container.servers.list_servers()] == ["a", "b"]
    assert [s.server_id for s in await container.servers.list_servers(enabled=True)] == ["a"]
    assert [s.server_id for s in await container.servers.list_servers(tag="Calendar")] == ["b"]


# --------------------------------------------------------------------- restoration


async def test_registered_servers_are_restored_after_restart(tmp_path):
    url = f"sqlite:///{tmp_path / 'nexus.db'}"
    connector = FakeConnector()

    first = make_container(connector, config_store=SQLiteServerConfigStore(url))
    await first.initialize()
    await first.servers.register(stdio_payload("notes", description="persisted"))
    await first.servers.register(stdio_payload("off", enabled=False))
    await first.servers.connect("notes")
    await first.close()

    second = make_container(FakeConnector(), config_store=SQLiteServerConfigStore(url))
    report = await second.initialize()

    assert sorted(report.restored) == ["notes", "off"]
    assert report.connected == ["notes"] and report.skipped_disabled == ["off"]
    assert report.failed == {}
    status = await second.servers.status("notes")
    assert status.connected and status.description == "persisted"
    assert status.tools == ["create_note"]
    assert (await second.servers.status("off")).connected is False
    await second.close()


async def test_restore_survives_a_server_that_cannot_connect():
    store = InMemoryServerConfigStore()
    seed = make_container(config_store=store)
    for server_id in ("good", "bad"):
        await seed.servers.register(stdio_payload(server_id))

    connector = FakeConnector()
    connector.fail_for.add("bad")
    fresh = make_container(connector, config_store=store)
    report = await fresh.initialize()

    assert report.connected == ["good"]
    assert report.failed == {"bad": "SERVER_CONNECTION_FAILED"}
    assert (await fresh.servers.status("bad")).status is ServerStatus.ERROR


async def test_restore_is_idempotent():
    store = InMemoryServerConfigStore()
    container = make_container(config_store=store)
    await container.servers.register(stdio_payload("notes"))

    report = await container.initialize()

    assert report.already_registered == ["notes"] and report.restored == []


async def test_restore_without_auto_connect_only_registers():
    store = InMemoryServerConfigStore()
    await make_container(config_store=store).servers.register(stdio_payload("notes"))

    fresh = make_container(config_store=store)
    report = await fresh.initialize(auto_connect=False)

    assert report.restored == ["notes"] and report.connected == []
    assert (await fresh.servers.status("notes")).connected is False


# --------------------------------------------------------------------- credentials


async def test_credentials_never_appear_in_reports_events_or_storage(monkeypatch, tmp_path):
    from core.services import configured_connector as module

    secret = "s3cr3t-runtime-value-9f2c"
    captured: dict[str, object] = {}

    class SpyTransport:
        server_capabilities = {"tools": {}}

        def __init__(self, config, *, on_tools_list_changed=None):
            captured["config"] = config

        async def connect(self):
            return None

        async def disconnect(self):
            return None

        async def list_tools(self, *, cursor=None):
            from types import SimpleNamespace

            return SimpleNamespace(tools=[sdk_tool("t")], next_cursor=None)

    monkeypatch.setattr(module, "StdioMCPTransport", SpyTransport)
    db = tmp_path / "creds.db"
    from core.container import NexusContainer

    container = NexusContainer(
        config_store=SQLiteServerConfigStore(f"sqlite:///{db}"),
        environ={"GITHUB_TOKEN": secret},
    )
    await container.servers.register(
        stdio_payload(
            "github",
            config={
                "transport": "stdio",
                "command": "python",
                "env": {"GITHUB_TOKEN": "${GITHUB_TOKEN}"},
            },
        )
    )
    report = await container.servers.connect("github")

    # The runtime transport received the resolved value...
    assert captured["config"].env["GITHUB_TOKEN"] == secret
    # ...but it appears nowhere observable.
    observable = [
        report.model_dump_json(),
        json.dumps([e.model_dump(mode="json") for e in container.events.recent()]),
        repr(container),
        db.read_bytes().decode("latin-1"),
    ]
    assert all(secret not in text for text in observable)
    await container.close()


async def test_missing_secret_surfaces_a_clear_non_leaky_error():
    from core.container import NexusContainer

    container = NexusContainer(environ={})
    await container.servers.register(
        stdio_payload(
            "github",
            config={
                "transport": "stdio",
                "command": "python",
                "env": {"GITHUB_TOKEN": "${GITHUB_TOKEN}"},
            },
        )
    )

    with pytest.raises(ConfigurationError) as exc_info:
        await container.servers.connect("github")

    assert exc_info.value.details == {"variable": "GITHUB_TOKEN"}
    assert (await container.servers.status("github")).status is ServerStatus.ERROR
