from __future__ import annotations

import pytest

from mcp import Client

from mcp_servers.notes.server import create_notes_server
from mcp_servers.github.server import create_github_server
from mcp_servers.github.service import GitHubService
from mcp_servers.calendar.server import create_calendar_server
from mcp_servers.calendar.service import GoogleCalendarService


SECRET_MARKER = "SECRET_TOKEN_DO_NOT_LEAK"


class MockGitHubClient:
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
    def list_calendars(self, max_results=100, page_token=None):
        return {"items": [], "nextPageToken": None}

    def close(self):
        pass


def get_payload(result):
    payload = result.structured_content
    assert isinstance(payload, dict)
    return payload


def assert_error_contract(payload):
    assert payload.get("success") is False
    assert isinstance(payload.get("error"), dict)
    assert isinstance(payload["error"].get("message"), str)
    assert payload["error"]["message"].strip()


def assert_success_contract(payload):
    assert payload.get("success") is True
    assert "data" in payload


# =========================================================
# 1. Validation error contracts
# =========================================================


@pytest.mark.asyncio
async def test_notes_validation_error_is_structured():
    server = create_notes_server()

    async with Client(server, raise_exceptions=True) as client:
        result = await client.call_tool(
            "create_note",
            {
                "title": "",
                "content": "Invalid note test",
            },
        )

    assert_error_contract(get_payload(result))


@pytest.mark.asyncio
async def test_github_validation_error_is_structured():
    server = create_github_server(client=MockGitHubClient())

    async with Client(server, raise_exceptions=True) as client:
        result = await client.call_tool(
            "get_repository_details",
            {
                "owner": "..",
                "repo": "invalid-repository",
            },
        )

    assert_error_contract(get_payload(result))


@pytest.mark.asyncio
async def test_calendar_validation_error_is_structured():
    server = create_calendar_server(client=MockCalendarClient())

    async with Client(server, raise_exceptions=True) as client:
        result = await client.call_tool(
            "list_calendars",
            {"max_results": 0},
        )

    assert_error_contract(get_payload(result))


# =========================================================
# 2. Internal exception isolation
# =========================================================


@pytest.mark.asyncio
async def test_github_internal_exception_is_sanitized(monkeypatch):
    def raise_internal_error(self, *args, **kwargs):
        raise RuntimeError(f"Internal failure: {SECRET_MARKER}")

    monkeypatch.setattr(
        GitHubService,
        "search_repositories",
        raise_internal_error,
    )

    server = create_github_server(client=MockGitHubClient())

    async with Client(server, raise_exceptions=True) as client:
        result = await client.call_tool(
            "search_repositories",
            {"query": "nexus-mcp"},
        )

    payload = get_payload(result)

    assert_error_contract(payload)
    assert SECRET_MARKER not in str(payload)


@pytest.mark.asyncio
async def test_calendar_internal_exception_is_sanitized(monkeypatch):
    def raise_internal_error(self, *args, **kwargs):
        raise RuntimeError(f"Internal failure: {SECRET_MARKER}")

    monkeypatch.setattr(
        GoogleCalendarService,
        "list_calendars",
        raise_internal_error,
    )

    server = create_calendar_server(client=MockCalendarClient())

    async with Client(server, raise_exceptions=True) as client:
        result = await client.call_tool(
            "list_calendars",
            {},
        )

    payload = get_payload(result)

    assert_error_contract(payload)
    assert SECRET_MARKER not in str(payload)


# =========================================================
# 3. Recovery after application-level failures
# =========================================================


@pytest.mark.asyncio
async def test_notes_server_recovers_after_validation_error():
    server = create_notes_server()

    async with Client(server, raise_exceptions=True) as client:
        failed_result = await client.call_tool(
            "create_note",
            {
                "title": "",
                "content": "Invalid note",
            },
        )

        assert_error_contract(get_payload(failed_result))

        recovery_result = await client.call_tool(
            "list_notes",
            {},
        )

    assert_success_contract(get_payload(recovery_result))


@pytest.mark.asyncio
async def test_github_server_recovers_after_validation_error():
    server = create_github_server(client=MockGitHubClient())

    async with Client(server, raise_exceptions=True) as client:
        failed_result = await client.call_tool(
            "get_repository_details",
            {
                "owner": "..",
                "repo": "invalid-repository",
            },
        )

        assert_error_contract(get_payload(failed_result))

        recovery_result = await client.call_tool(
            "search_repositories",
            {"query": "nexus-mcp"},
        )

    assert_success_contract(get_payload(recovery_result))


@pytest.mark.asyncio
async def test_calendar_server_recovers_after_validation_error():
    server = create_calendar_server(client=MockCalendarClient())

    async with Client(server, raise_exceptions=True) as client:
        failed_result = await client.call_tool(
            "list_calendars",
            {"max_results": 0},
        )

        assert_error_contract(get_payload(failed_result))

        recovery_result = await client.call_tool(
            "list_calendars",
            {},
        )

    assert_success_contract(get_payload(recovery_result))
