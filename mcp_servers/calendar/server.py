
from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from mcp.server import MCPServer

from mcp_servers.calendar.client import (
    GoogleCalendarAPIError,
    GoogleCalendarClient,
)
from mcp_servers.calendar.service import (
    CalendarValidationError,
    GoogleCalendarService,
)

T = TypeVar("T")


def _execute(operation: Callable[[], T]) -> dict[str, Any]:
    """Execute a service operation and return a safe response envelope."""
    try:
        return {
            "success": True,
            "data": operation(),
        }

    except CalendarValidationError as exc:
        return {
            "success": False,
            "error": {
                "type": "validation_error",
                "message": str(exc),
            },
        }

    except GoogleCalendarAPIError as exc:
        return {
            "success": False,
            "error": {
                "type": "google_calendar_api_error",
                "message": exc.message,
                "status_code": exc.status_code,
            },
        }

    except Exception:
        # Never expose internal exception messages or credentials to the model.
        return {
            "success": False,
            "error": {
                "type": "internal_error",
                "message": "An internal Calendar server error occurred.",
            },
        }


def create_calendar_server(
    client: GoogleCalendarClient | None = None,
) -> MCPServer:
    """Create the Google Calendar MCP server."""
    calendar_client = client or GoogleCalendarClient()
    service = GoogleCalendarService(calendar_client)

    server = MCPServer("Nexus Google Calendar MCP Server")

    @server.tool()
    def list_calendars(
        max_results: int = 100,
        page_token: str | None = None,
    ) -> dict[str, Any]:
        """List calendars accessible to the authenticated Google account."""
        return _execute(
            lambda: service.list_calendars(
                max_results=max_results,
                page_token=page_token,
            )
        )

    @server.tool()
    def get_events(
        calendar_id: str,
        time_min: str,
        time_max: str,
        max_results: int = 50,
        page_token: str | None = None,
    ) -> dict[str, Any]:
        """Retrieve events from a calendar within a timezone-aware time range."""
        return _execute(
            lambda: service.get_events(
                calendar_id,
                time_min,
                time_max,
                max_results=max_results,
                page_token=page_token,
            )
        )

    @server.tool()
    def get_event_details(
        calendar_id: str,
        event_id: str,
    ) -> dict[str, Any]:
        """Retrieve the details of one calendar event."""
        return _execute(
            lambda: service.get_event_details(
                calendar_id,
                event_id,
            )
        )

    @server.tool()
    def create_event(
        calendar_id: str,
        summary: str,
        start: str,
        end: str,
        description: str | None = None,
        location: str | None = None,
        time_zone: str | None = None,
    ) -> dict[str, Any]:
        """Create a calendar event with optional description and location."""
        return _execute(
            lambda: service.create_event(
                calendar_id,
                summary,
                start,
                end,
                description=description,
                location=location,
                time_zone=time_zone,
            )
        )

    @server.tool()
    def update_event(
        calendar_id: str,
        event_id: str,
        summary: str | None = None,
        description: str | None = None,
        location: str | None = None,
        start: str | None = None,
        end: str | None = None,
        time_zone: str | None = None,
    ) -> dict[str, Any]:
        """Update selected fields of an existing calendar event."""
        return _execute(
            lambda: service.update_event(
                calendar_id,
                event_id,
                summary=summary,
                description=description,
                location=location,
                start=start,
                end=end,
                time_zone=time_zone,
            )
        )

    @server.tool()
    def delete_event(
        calendar_id: str,
        event_id: str,
    ) -> dict[str, Any]:
        """Delete a calendar event."""
        return _execute(
            lambda: service.delete_event(
                calendar_id,
                event_id,
            )
        )

    @server.tool()
    def check_availability(
        time_min: str,
        time_max: str,
        calendar_ids: list[str],
        time_zone: str | None = None,
    ) -> dict[str, Any]:
        """Check busy periods across one or more calendars."""
        return _execute(
            lambda: service.check_availability(
                time_min,
                time_max,
                calendar_ids,
                time_zone=time_zone,
            )
        )

    @server.tool()
    def find_free_slots(
        time_min: str,
        time_max: str,
        calendar_ids: list[str],
        duration_minutes: int,
        time_zone: str | None = None,
        max_slots: int = 20,
    ) -> dict[str, Any]:
        """Find free meeting slots across one or more calendars."""
        return _execute(
            lambda: service.find_free_slots(
                time_min,
                time_max,
                calendar_ids,
                duration_minutes,
                time_zone=time_zone,
                max_slots=max_slots,
            )
        )

    return server
