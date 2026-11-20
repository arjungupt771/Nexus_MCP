from __future__ import annotations

import json

import pytest
from support_agent import ToolTransport, make_world, sdk_tool, tool_error

from agent.llm import ToolCall
from agent.tools import MCPToolBridge, ToolPermissions
from agent.tools.bridge import MAX_NAME_LENGTH, build_bindings
from core.models import ToolDefinition


def tdef(server: str, name: str) -> ToolDefinition:
    return ToolDefinition(
        server_id=server,
        server_name=server,
        name=name,
        input_schema={"type": "object", "properties": {}},
    )


async def world(**handlers):
    notes = ToolTransport(
        [
            sdk_tool("create_note", {"title": {"type": "string"}}, ["title"]),
            sdk_tool("delete_note", {"note_id": {"type": "integer"}}, ["note_id"]),
            sdk_tool("search", {"q": {"type": "string"}}),
        ],
        handlers,
    )
    github = ToolTransport([sdk_tool("search", {"q": {"type": "string"}})])
    container = await make_world({"notes": notes, "github": github})
    return container, notes, github


def bridge_for(container, **kwargs):
    permissions = kwargs.pop("permissions", None)
    return MCPToolBridge(container.catalog, container.executor, permissions, **kwargs)


def call(name, **arguments):
    return ToolCall(id="c1", name=name, arguments=arguments)


async def test_specs_expose_all_servers_with_collision_free_identity():
    container, *_ = await world()
    specs = {s.name: s for s in await bridge_for(container).specs()}

    # Same tool name on two servers stays distinct and server-owned.
    assert {"notes__search", "github__search", "notes__create_note"} <= set(specs)
    assert specs["notes__create_note"].description.startswith("[Notes Server]")
    assert specs["notes__create_note"].parameters["required"] == ["title"]


async def test_specs_only_include_permitted_tools():
    container, *_ = await world()
    permissions = ToolPermissions(allow=("notes.*",), deny=("notes.delete_note",))

    names = {s.name for s in await bridge_for(container, permissions=permissions).specs()}

    assert names == {"notes__create_note", "notes__search"}


def test_permission_patterns():
    p = ToolPermissions(allow=("github.*", "notes.search"), deny=("github.delete_*",))

    assert p.allows("github.search") and p.allows("notes.search")
    assert not p.allows("github.delete_repo") and not p.allows("notes.create_note")
    assert ToolPermissions().allows("any.thing")
    assert not ToolPermissions(allow=()).allows("any.thing")


def test_names_are_sanitized_truncated_and_unique():
    bindings = build_bindings(
        [
            tdef("srv", "my.tool name"),
            tdef("srv", "a" * 100),
            tdef("a", "b_c"),  # a__b_c
            tdef(
                "a_", "_b_c"
            ),  # collides after sanitization? a___b_c differs; keep distinct anyway
            tdef("x", "p.q"),
            tdef("x", "p_q"),  # sanitizes to the same name as p.q -> must disambiguate
        ]
    )

    assert len(bindings) == 6
    assert all(
        len(n) <= MAX_NAME_LENGTH and n.replace("_", "").replace("-", "").isalnum()
        for n in bindings
    )
    assert {b.qualified_name for b in bindings.values()} >= {"x.p.q", "x.p_q"}


async def test_execute_routes_to_the_owning_server_and_returns_json():
    container, notes, github = await world()
    outcome = await bridge_for(container).execute(call("notes__create_note", title="Hi"))

    assert outcome.success and (outcome.server_id, outcome.tool_name) == ("notes", "create_note")
    assert json.loads(outcome.content) == {
        "success": True,
        "data": {"ok": True, "tool": "create_note"},
    }
    assert notes.calls == [("create_note", {"title": "Hi"})] and github.calls == []
    assert outcome.request_id and outcome.duration_ms >= 0


async def test_same_tool_name_on_different_servers_hits_the_right_one():
    container, notes, github = await world()
    bridge = bridge_for(container)

    await bridge.execute(call("github__search", q="mcp"))

    assert github.calls == [("search", {"q": "mcp"})] and notes.calls == []


async def test_qualified_names_are_accepted():
    container, notes, _ = await world()

    assert (await bridge_for(container).execute(call("notes.search", q="x"))).success
    assert notes.calls


async def test_unregistered_tools_are_never_executed():
    container, notes, github = await world()
    outcome = await bridge_for(container).execute(call("notes__rm_rf"))

    assert not outcome.success and outcome.error_code == "TOOL_NOT_FOUND"
    assert notes.calls == github.calls == []


async def test_unauthorized_tools_are_never_executed():
    container, notes, _ = await world()
    bridge = bridge_for(container, permissions=ToolPermissions(deny=("notes.delete_note",)))

    outcome = await bridge.execute(call("notes__delete_note", note_id=1))

    assert outcome.error_code == "TOOL_AUTHORIZATION_DENIED" and notes.calls == []
    assert (outcome.server_id, outcome.tool_name) == ("notes", "delete_note")


@pytest.mark.parametrize("arguments", [{}, {"title": 5}])
async def test_invalid_arguments_are_rejected_before_execution(arguments):
    container, notes, _ = await world()
    outcome = await bridge_for(container).execute(
        ToolCall(id="c", name="notes__create_note", arguments=arguments)
    )

    assert outcome.error_code == "INVALID_ARGUMENTS" and notes.calls == []
    assert "title" in json.loads(outcome.content)["error"]["message"]


async def test_tool_level_errors_become_readable_outcomes_not_exceptions():
    container, *_ = await world(create_note=lambda a: tool_error("disk full"))
    outcome = await bridge_for(container).execute(call("notes__create_note", title="x"))

    assert not outcome.success and outcome.error_code
    assert "disk full" in json.loads(outcome.content)["error"]["message"]


async def test_tool_timeouts_are_enforced():
    container, notes, _ = await world()
    notes.delay = 1.0

    outcome = await bridge_for(container, timeout_seconds=0.05).execute(
        call("notes__search", q="x")
    )

    assert not outcome.success and "TIMEOUT" in (outcome.error_code or "").upper()


async def test_large_results_are_truncated():
    container, *_ = await world(
        search=lambda a: __import__("types").SimpleNamespace(
            is_error=False, structured_content={"blob": "x" * 5000}, content=[]
        )
    )
    outcome = await bridge_for(container, max_result_chars=500).execute(
        call("notes__search", q="x")
    )

    payload = json.loads(outcome.content)
    assert outcome.success and payload["truncated"] is True and len(outcome.content) < 800


async def test_catalog_changes_are_reflected_without_rebuilding_the_bridge():
    container, notes, _ = await world()
    bridge = bridge_for(container)
    before = {s.name for s in await bridge.specs()}
    assert "notes__create_note" in before

    notes.tools = [sdk_tool("only_tool")]
    await container.servers.restart("notes")

    after = {s.name for s in await bridge.specs()}
    assert "notes__create_note" not in after and "notes__only_tool" in after
    stale = await bridge.execute(call("notes__create_note", title="x"))
    assert stale.error_code == "TOOL_NOT_FOUND"

    await container.servers.disconnect("github")
    assert not {n for n in {s.name for s in await bridge.specs()} if n.startswith("github")}


def test_bridge_validates_its_settings():
    with pytest.raises(ValueError):
        MCPToolBridge(None, None, timeout_seconds=0)  # type: ignore[arg-type]
