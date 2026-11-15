
from __future__ import annotations

from typing import Any

import pytest
from mcp import Client

from mcp_servers.calendar.client import (
    GoogleCalendarAPIError,
)
from mcp_servers.calendar.server import create_calendar_server
from mcp_servers.calendar.service import (
    CalendarValidationError,
)


class MockCalendarClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def list_calendars(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("list_calendars", kwargs))
        return {"items": [{"id": "primary"}]}

    def list_events(self, calendar_id: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(
            ("list_events", {"calendar_id": calendar_id, **kwargs})
        )
        return {"items": [{"id": "event-1"}]}

    def get_event(self, calendar_id: str, event_id: str) -> dict[str, Any]:
        self.calls.append(
            ("get_event", {"calendar_id": calendar_id, "event_id": event_id})
        )
        return {"id": event_id, "summary": "Test event"}

    def create_event(
        self,
        calendar_id: str,
        event: dict[str, Any],
    ) -> dict[str, Any]:
        self.calls.append(
            ("create_event", {"calendar_id": calendar_id, "event": event})
        )
        return {"id": "created-event", **event}

    def update_event(
        self,
        calendar_id: str,
        event_id: str,
        event: dict[str, Any],
    ) -> dict[str, Any]:
        self.calls.append(
            (
                "update_event",
                {
                    "calendar_id": calendar_id,
                    "event_id": event_id,
                    "event": event,
                },
            )
        )
        return {"id": event_id, **event}

    def delete_event(self, calendar_id: str, event_id: str) -> None:
        self.calls.append(
            ("delete_event", {"calendar_id": calendar_id, "event_id": event_id})
        )

    def query_freebusy(
        self,
        *,
        time_min: str,
        time_max: str,
        calendar_ids: list[str],
        time_zone: str | None = None,
    ) -> dict[str, Any]:
        self.calls.append(
            (
                "query_freebusy",
                {
                    "time_min": time_min,
                    "time_max": time_max,
                    "calendar_ids": calendar_ids,
                    "time_zone": time_zone,
                },
            )
        )
        return {
            "calendars": {
                calendar_id: {"busy": []}
                for calendar_id in calendar_ids
            }
        }


@pytest.fixture
def mock_client() -> MockCalendarClient:
    return MockCalendarClient()


@pytest.fixture
def calendar_server(mock_client):
    return create_calendar_server(mock_client)


async def get_data(result: Any) -> dict[str, Any]:
    content = result.structured_content
    assert isinstance(content, dict)
    return content


@pytest.mark.asyncio
async def test_server_discovers_eight_tools(calendar_server: Any) -> None:
    async with Client(calendar_server, raise_exceptions=True) as mcp_client:
        result = await mcp_client.list_tools()

    tool_names = {tool.name for tool in result.tools}

    assert tool_names == {
        "list_calendars",
        "get_events",
        "get_event_details",
        "create_event",
        "update_event",
        "delete_event",
        "check_availability",
        "find_free_slots",
    }


@pytest.mark.asyncio
async def test_list_calendars_tool(
    calendar_server: Any,
    mock_client: MockCalendarClient,
) -> None:
    async with Client(calendar_server, raise_exceptions=True) as client:
        result = await client.call_tool(
            "list_calendars",
            {"max_results": 25},
        )

    data = await get_data(result)

    assert data["success"] is True
    assert data["data"]["items"][0]["id"] == "primary"
    assert mock_client.calls[0][0] == "list_calendars"


@pytest.mark.asyncio
async def test_get_events_tool(
    calendar_server: Any,
    mock_client: MockCalendarClient,
) -> None:
    async with Client(calendar_server, raise_exceptions=True) as client:
        result = await client.call_tool(
            "get_events",
            {
                "calendar_id": "primary",
                "time_min": "2026-11-16T09:00:00Z",
                "time_max": "2026-11-17T09:00:00Z",
            },
        )

    data = await get_data(result)

    assert data["success"] is True
    assert data["data"]["items"][0]["id"] == "event-1"
    assert mock_client.calls[0][0] == "list_events"


@pytest.mark.asyncio
async def test_get_event_details_tool(
    calendar_server: Any,
) -> None:
    async with Client(calendar_server, raise_exceptions=True) as client:
        result = await client.call_tool(
            "get_event_details",
            {
                "calendar_id": "primary",
                "event_id": "event-1",
            },
        )

    data = await get_data(result)

    assert data["success"] is True
    assert data["data"]["summary"] == "Test event"


@pytest.mark.asyncio
async def test_create_event_tool(
    calendar_server: Any,
    mock_client: MockCalendarClient,
) -> None:
    async with Client(calendar_server, raise_exceptions=True) as client:
        result = await client.call_tool(
            "create_event",
            {
                "calendar_id": "primary",
                "summary": "Nexus planning",
                "start": "2026-11-16T10:00:00+05:30",
                "end": "2026-11-16T11:00:00+05:30",
                "description": "Planning session",
                "location": "Delhi",
                "time_zone": "Asia/Kolkata",
            },
        )

    data = await get_data(result)

    assert data["success"] is True
    assert data["data"]["id"] == "created-event"

    call = mock_client.calls[0]
    assert call[0] == "create_event"
    assert call[1]["event"]["summary"] == "Nexus planning"
    assert call[1]["event"]["start"]["timeZone"] == "Asia/Kolkata"


@pytest.mark.asyncio
async def test_update_event_tool(
    calendar_server: Any,
) -> None:
    async with Client(calendar_server, raise_exceptions=True) as client:
        result = await client.call_tool(
            "update_event",
            {
                "calendar_id": "primary",
                "event_id": "event-1",
                "summary": "Updated meeting",
            },
        )

    data = await get_data(result)

    assert data["success"] is True
    assert data["data"]["summary"] == "Updated meeting"


@pytest.mark.asyncio
async def test_delete_event_tool(
    calendar_server: Any,
) -> None:
    async with Client(calendar_server, raise_exceptions=True) as client:
        result = await client.call_tool(
            "delete_event",
            {
                "calendar_id": "primary",
                "event_id": "event-1",
            },
        )

    data = await get_data(result)

    assert data["success"] is True
    assert data["data"]["deleted"] is True


@pytest.mark.asyncio
async def test_check_availability_tool(
    calendar_server: Any,
) -> None:
    async with Client(calendar_server, raise_exceptions=True) as client:
        result = await client.call_tool(
            "check_availability",
            {
                "time_min": "2026-11-16T09:00:00Z",
                "time_max": "2026-11-16T18:00:00Z",
                "calendar_ids": ["primary"],
                "time_zone": "Asia/Kolkata",
            },
        )

    data = await get_data(result)

    assert data["success"] is True
    assert "calendars" in data["data"]


@pytest.mark.asyncio
async def test_find_free_slots_tool(
    calendar_server: Any,
) -> None:
    async with Client(calendar_server, raise_exceptions=True) as client:
        result = await client.call_tool(
            "find_free_slots",
            {
                "time_min": "2026-11-16T09:00:00Z",
                "time_max": "2026-11-16T18:00:00Z",
                "calendar_ids": ["primary"],
                "duration_minutes": 30,
                "time_zone": "Asia/Kolkata",
            },
        )

    data = await get_data(result)

    assert data["success"] is True
    assert data["data"]["total_slots"] == 1


@pytest.mark.asyncio
async def test_validation_error_returns_safe_response() -> None:
    class InvalidCalendarClient(MockCalendarClient):
        def list_calendars(self, **kwargs: Any) -> dict[str, Any]:
            raise CalendarValidationError("Invalid test input.")

    server = create_calendar_server(InvalidCalendarClient())

    async with Client(server, raise_exceptions=True) as client:
        result = await client.call_tool(
            "list_calendars",
            {"max_results": 0},
        )

    data = await get_data(result)

    assert data["success"] is False
    assert data["error"]["type"] == "validation_error"


@pytest.mark.asyncio
async def test_google_api_error_returns_safe_response() -> None:
    class FailingCalendarClient(MockCalendarClient):
        def list_calendars(self, **kwargs: Any) -> dict[str, Any]:
            raise GoogleCalendarAPIError(
                "Google authentication failed.",
                status_code=401,
            )

    server = create_calendar_server(FailingCalendarClient())

    async with Client(server, raise_exceptions=True) as client:
        result = await client.call_tool("list_calendars", {})

    data = await get_data(result)

    assert data["success"] is False
    assert data["error"]["type"] == "google_calendar_api_error"
    assert data["error"]["status_code"] == 401


@pytest.mark.asyncio
async def test_internal_error_does_not_expose_exception_details() -> None:
    class BrokenCalendarClient(MockCalendarClient):
        def list_calendars(self, **kwargs: Any) -> dict[str, Any]:
            raise RuntimeError("secret-token-should-not-leak")

    server = create_calendar_server(BrokenCalendarClient())

    async with Client(server, raise_exceptions=True) as client:
        result = await client.call_tool("list_calendars", {})

    data = await get_data(result)

    assert data["success"] is False
    assert data["error"]["type"] == "internal_error"
    assert "secret-token-should-not-leak" not in str(data)
