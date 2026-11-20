from __future__ import annotations

import pytest
from mcp import Client

from mcp_servers.calendar.server import create_calendar_server
from mcp_servers.github.server import create_github_server
from mcp_servers.notes.server import create_notes_server

SERVER_CASES = [
    pytest.param(create_notes_server, 10, id="notes"),
    pytest.param(create_github_server, 10, id="github"),
    pytest.param(create_calendar_server, 8, id="calendar"),
]


class MockGitHubClient:
    """Deterministic GitHub API client for contract tests."""

    def get(self, path, params=None):
        if "search/repositories" in path:
            return {"total_count": 0, "items": []}

        return {}

    def post(self, *args, **kwargs):
        raise AssertionError("Unexpected GitHub POST")

    def patch(self, *args, **kwargs):
        raise AssertionError("Unexpected GitHub PATCH")

    def delete(self, *args, **kwargs):
        raise AssertionError("Unexpected GitHub DELETE")


class MockCalendarClient:
    """Deterministic Calendar API client for contract tests."""

    def list_calendars(self, max_results=100, page_token=None):
        return {"items": [], "nextPageToken": None}

    def close(self):
        pass


def extract_payload(result):
    payload = result.structured_content

    assert isinstance(payload, dict)
    return payload


def assert_success_contract(payload):
    assert payload.get("success") is True
    assert "data" in payload
    assert "error" not in payload


def assert_error_contract(payload):
    assert payload.get("success") is False
    assert isinstance(payload.get("error"), dict)
    assert payload["error"].get("message")


@pytest.mark.parametrize("server_factory, expected_tool_count", SERVER_CASES)
@pytest.mark.asyncio
async def test_discovered_tools_have_inspectable_invocation_schemas(
    server_factory,
    expected_tool_count: int,
):
    server = server_factory()

    async with Client(server, raise_exceptions=True) as client:
        result = await client.list_tools()

    assert len(result.tools) == expected_tool_count

    for tool in result.tools:
        schema = tool.input_schema

        assert isinstance(schema, dict)
        assert schema.get("type") == "object"
        assert isinstance(schema.get("properties", {}), dict)

        for field in schema.get("required", []):
            assert field in schema["properties"]


# ---------------------------------------------------------
# Successful invocation contracts
# ---------------------------------------------------------


@pytest.mark.asyncio
async def test_notes_list_notes_success_contract():
    server = create_notes_server()

    async with Client(server, raise_exceptions=True) as client:
        result = await client.call_tool("list_notes", {})

    payload = extract_payload(result)
    assert_success_contract(payload)


@pytest.mark.asyncio
async def test_github_search_repositories_success_contract():
    server = create_github_server(client=MockGitHubClient())

    async with Client(server, raise_exceptions=True) as client:
        result = await client.call_tool(
            "search_repositories",
            {"query": "nexus-mcp"},
        )

    payload = extract_payload(result)
    assert_success_contract(payload)


@pytest.mark.asyncio
async def test_calendar_list_calendars_success_contract():
    server = create_calendar_server(client=MockCalendarClient())

    async with Client(server, raise_exceptions=True) as client:
        result = await client.call_tool("list_calendars", {})

    payload = extract_payload(result)
    assert_success_contract(payload)


# ---------------------------------------------------------
# Application-level error contracts
# ---------------------------------------------------------


@pytest.mark.asyncio
async def test_notes_validation_error_contract():
    server = create_notes_server()

    async with Client(server, raise_exceptions=True) as client:
        result = await client.call_tool(
            "create_note",
            {
                "title": "",
                "content": "Contract validation test",
            },
        )

    payload = extract_payload(result)
    assert_error_contract(payload)


@pytest.mark.asyncio
async def test_github_validation_error_contract():
    server = create_github_server(client=MockGitHubClient())

    async with Client(server, raise_exceptions=True) as client:
        result = await client.call_tool(
            "get_repository_details",
            {
                "owner": "..",
                "repo": "invalid-repository",
            },
        )

    payload = extract_payload(result)
    assert_error_contract(payload)


@pytest.mark.asyncio
async def test_calendar_validation_error_contract():
    server = create_calendar_server(client=MockCalendarClient())

    async with Client(server, raise_exceptions=True) as client:
        result = await client.call_tool(
            "list_calendars",
            {"max_results": 0},
        )

    payload = extract_payload(result)
    assert_error_contract(payload)


# ---------------------------------------------------------
# Protocol-level error contracts
# ---------------------------------------------------------


@pytest.mark.asyncio
async def test_unknown_tool_is_rejected():
    server = create_notes_server()

    async with Client(server, raise_exceptions=True) as client:
        result = await client.call_tool(
            "this_tool_does_not_exist",
            {},
        )

    assert result.is_error is True


@pytest.mark.asyncio
async def test_invalid_argument_type_is_rejected():
    server = create_notes_server()

    async with Client(server, raise_exceptions=True) as client:
        result = await client.call_tool(
            "list_notes",
            {"limit": "not-an-integer"},
        )

    assert result.is_error is True
