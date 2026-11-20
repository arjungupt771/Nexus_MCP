from __future__ import annotations

import asyncio
import os
import sys
from contextlib import AsyncExitStack
from pathlib import Path

import pytest
from mcp import Client, StdioServerParameters

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SERVER_MODULES = {
    "notes": ("mcp_servers.notes.main", 10),
    "github": ("mcp_servers.github.main", 10),
    "calendar": ("mcp_servers.calendar.main", 8),
}


def build_server_parameters(module: str) -> StdioServerParameters:
    env = os.environ.copy()

    existing_pythonpath = env.get("PYTHONPATH", "")

    env["PYTHONPATH"] = os.pathsep.join(
        part for part in [str(PROJECT_ROOT), existing_pythonpath] if part
    )

    # Prevent accidental use of real credentials.
    env.pop("GITHUB_TOKEN", None)
    env.pop("GOOGLE_ACCESS_TOKEN", None)

    return StdioServerParameters(
        command=sys.executable,
        args=["-m", module],
        env=env,
    )


async def connect_all_servers(stack: AsyncExitStack):
    clients = {}

    for server_name, (module, _) in SERVER_MODULES.items():
        params = build_server_parameters(module)

        client = await stack.enter_async_context(Client(params, raise_exceptions=True))

        clients[server_name] = client

    return clients


def get_payload(result) -> dict:
    payload = result.structured_content

    assert isinstance(payload, dict)
    return payload


def assert_success(result):
    assert result.is_error is not True

    payload = get_payload(result)

    assert payload.get("success") is True
    assert "data" in payload

    return payload


def assert_application_error(result):
    assert result.is_error is not True

    payload = get_payload(result)

    assert payload.get("success") is False
    assert isinstance(payload.get("error"), dict)
    assert payload["error"].get("message")

    return payload


# =========================================================
# 1. Combined discovery and server-qualified tool registry
# =========================================================


@pytest.mark.asyncio
async def test_combined_discovery_builds_unique_qualified_tools():
    async with AsyncExitStack() as stack:
        clients = await connect_all_servers(stack)

        qualified_tools = {}

        for server_name, client in clients.items():
            result = await client.list_tools()

            expected_count = SERVER_MODULES[server_name][1]

            assert len(result.tools) == expected_count

            for tool in result.tools:
                qualified_name = f"{server_name}.{tool.name}"

                assert qualified_name not in qualified_tools

                qualified_tools[qualified_name] = {
                    "server": server_name,
                    "tool_name": tool.name,
                    "input_schema": tool.input_schema,
                }

        assert len(qualified_tools) == 28

        assert "notes.create_note" in qualified_tools
        assert "github.search_repositories" in qualified_tools
        assert "calendar.list_calendars" in qualified_tools

        assert len(qualified_tools) == len(set(qualified_tools))


# =========================================================
# 2. Cross-server invocation in one connected workflow
# =========================================================


@pytest.mark.asyncio
async def test_cross_server_invocation_workflow():
    async with AsyncExitStack() as stack:
        clients = await connect_all_servers(stack)

        # Notes: safe read-only operation.
        notes_result = await clients["notes"].call_tool(
            "list_notes",
            {},
        )

        notes_payload = assert_success(notes_result)

        # GitHub: invalid identifier rejected before HTTP request.
        github_result = await clients["github"].call_tool(
            "get_repository_details",
            {
                "owner": "..",
                "repo": "invalid-repository",
            },
        )

        github_payload = assert_application_error(github_result)

        # Calendar: invalid pagination rejected before API request.
        calendar_result = await clients["calendar"].call_tool(
            "list_calendars",
            {"max_results": 0},
        )

        calendar_payload = assert_application_error(calendar_result)

        assert notes_payload["success"] is True
        assert github_payload["success"] is False
        assert calendar_payload["success"] is False


# =========================================================
# 3. Concurrent invocation across independent servers
# =========================================================


@pytest.mark.asyncio
async def test_concurrent_cross_server_invocations():
    async with AsyncExitStack() as stack:
        clients = await connect_all_servers(stack)

        notes_task = clients["notes"].call_tool(
            "list_notes",
            {},
        )

        github_task = clients["github"].call_tool(
            "get_repository_details",
            {
                "owner": "..",
                "repo": "invalid-repository",
            },
        )

        calendar_task = clients["calendar"].call_tool(
            "list_calendars",
            {"max_results": 0},
        )

        results = await asyncio.gather(
            notes_task,
            github_task,
            calendar_task,
        )

        notes_payload = assert_success(results[0])
        github_payload = assert_application_error(results[1])
        calendar_payload = assert_application_error(results[2])

        assert notes_payload["success"] is True
        assert github_payload["success"] is False
        assert calendar_payload["success"] is False


# =========================================================
# 4. Failure isolation across connected servers
# =========================================================


@pytest.mark.asyncio
async def test_failure_in_one_server_does_not_interrupt_others():
    async with AsyncExitStack() as stack:
        clients = await connect_all_servers(stack)

        # Trigger a controlled application-level error in GitHub.
        github_result = await clients["github"].call_tool(
            "get_repository_details",
            {
                "owner": "..",
                "repo": "invalid-repository",
            },
        )

        assert_application_error(github_result)

        # Notes must remain usable.
        notes_result = await clients["notes"].call_tool(
            "list_notes",
            {},
        )

        assert_success(notes_result)

        # Calendar must remain usable too.
        calendar_result = await clients["calendar"].call_tool(
            "list_calendars",
            {"max_results": 0},
        )

        assert_application_error(calendar_result)

        # GitHub's connection must also remain usable after its error.
        github_recovery = await clients["github"].call_tool(
            "get_repository_details",
            {
                "owner": "..",
                "repo": "another-invalid-repository",
            },
        )

        assert_application_error(github_recovery)

        # Confirm that all three sessions still respond to discovery.
        for client in clients.values():
            discovery = await client.list_tools()
            assert discovery.tools
