
from __future__ import annotations

from typing import Any

import httpx
import pytest

from mcp_servers.calendar.client import (
    GOOGLE_CALENDAR_API_URL,
    GoogleCalendarAPIError,
    GoogleCalendarClient,
)


def make_client(
    handler: Any,
    *,
    access_token: str | None = "test-access-token",
    token_provider: Any = None,
) -> GoogleCalendarClient:
    return GoogleCalendarClient(
        access_token=access_token,
        token_provider=token_provider,
        transport=httpx.MockTransport(handler),
    )


def json_response(
    request: httpx.Request,
    payload: dict[str, Any],
    status_code: int = 200,
) -> httpx.Response:
    return httpx.Response(status_code, json=payload, request=request)


def test_client_uses_google_calendar_base_url() -> None:
    assert GOOGLE_CALENDAR_API_URL == (
        "https://www.googleapis.com/calendar/v3"
    )


def test_client_sends_authorization_and_default_headers() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == (
            "Bearer test-access-token"
        )
        assert request.headers["Accept"] == "application/json"
        assert request.headers["User-Agent"] == (
            "Nexus-MCP-Calendar-Server"
        )
        return json_response(request, {"ok": True})

    with make_client(handler) as client:
        result = client.get("/users/me/calendarList")

    assert result == {"ok": True}


def test_token_provider_takes_precedence_over_static_token() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == (
            "Bearer provider-token"
        )
        return json_response(request, {"ok": True})

    with make_client(
        handler,
        access_token="static-token",
        token_provider=lambda: "provider-token",
    ) as client:
        assert client.get("/test") == {"ok": True}


def test_client_uses_environment_token_when_no_static_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOOGLE_ACCESS_TOKEN", "environment-token")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == (
            "Bearer environment-token"
        )
        return json_response(request, {"ok": True})

    with make_client(handler, access_token=None) as client:
        assert client.get("/test") == {"ok": True}


def test_client_can_make_request_without_token() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "Authorization" not in request.headers
        return json_response(request, {"ok": True})

    with make_client(handler, access_token=None) as client:
        assert client.get("/public-test") == {"ok": True}


def test_get_request_passes_query_parameters() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/calendar/v3/test"
        assert request.url.params["pageToken"] == "next-page"
        assert request.url.params["maxResults"] == "25"
        return json_response(request, {"items": []})

    with make_client(handler) as client:
        result = client.get(
            "/test",
            params={"pageToken": "next-page", "maxResults": 25},
        )

    assert result == {"items": []}


def test_post_request_sends_json_body() -> None:
    payload = {"summary": "Nexus planning", "description": "Test event"}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.read() == httpx.Request(
            "POST", "https://example.com", json=payload
        ).read()
        return json_response(request, {"id": "event-123"})

    with make_client(handler) as client:
        result = client.post("/test", json=payload)

    assert result == {"id": "event-123"}


def test_patch_request_sends_json_body() -> None:
    payload = {"summary": "Updated meeting"}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "PATCH"
        assert request.read() == httpx.Request(
            "PATCH", "https://example.com", json=payload
        ).read()
        return json_response(request, {"updated": True})

    with make_client(handler) as client:
        result = client.patch("/test", json=payload)

    assert result == {"updated": True}


def test_delete_request_returns_none_for_204() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "DELETE"
        return httpx.Response(204, request=request)

    with make_client(handler) as client:
        result = client.delete("/test")

    assert result is None


def test_empty_success_response_returns_none() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"", request=request)

    with make_client(handler) as client:
        result = client.get("/test")

    assert result is None


def test_list_calendars_builds_expected_request() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/calendar/v3/users/me/calendarList"
        assert request.url.params["maxResults"] == "25"
        assert request.url.params["pageToken"] == "page-2"
        return json_response(request, {"items": []})

    with make_client(handler) as client:
        result = client.list_calendars(
            max_results=25,
            page_token="page-2",
        )

    assert result == {"items": []}


def test_list_events_builds_expected_request() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/calendar/v3/calendars/primary/events"
        assert request.url.params["maxResults"] == "30"
        assert request.url.params["singleEvents"] == "true"
        assert request.url.params["orderBy"] == "startTime"
        assert request.url.params["timeMin"] == (
            "2026-11-16T09:00:00Z"
        )
        assert request.url.params["timeMax"] == (
            "2026-11-17T09:00:00Z"
        )
        assert request.url.params["pageToken"] == "page-3"
        return json_response(request, {"items": []})

    with make_client(handler) as client:
        result = client.list_events(
            "primary",
            time_min="2026-11-16T09:00:00Z",
            time_max="2026-11-17T09:00:00Z",
            max_results=30,
            page_token="page-3",
        )

    assert result == {"items": []}


def test_calendar_and_event_ids_are_url_encoded() -> None:
    expected_path = (
        b"/calendar/v3/calendars/team%2Fa%2Bb%40example.com"
        b"/events/event%2Fone%2Btwo"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.raw_path == expected_path
        return json_response(request, {"id": "event/one+two"})

    with make_client(handler) as client:
        result = client.get_event(
            "team/a+b@example.com",
            "event/one+two",
        )

    assert result["id"] == "event/one+two"


def test_create_event_sends_event_payload() -> None:
    event = {
        "summary": "Project review",
        "start": {"dateTime": "2026-11-16T10:00:00+05:30"},
        "end": {"dateTime": "2026-11-16T11:00:00+05:30"},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/calendar/v3/calendars/primary/events"
        assert request.read() == httpx.Request(
            "POST", "https://example.com", json=event
        ).read()
        return json_response(request, {"id": "created-event"})

    with make_client(handler) as client:
        result = client.create_event("primary", event)

    assert result["id"] == "created-event"


def test_update_event_uses_patch() -> None:
    event = {"summary": "Revised review"}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "PATCH"
        assert request.url.path == (
            "/calendar/v3/calendars/primary/events/event-123"
        )
        return json_response(request, {"id": "event-123"})

    with make_client(handler) as client:
        result = client.update_event("primary", "event-123", event)

    assert result["id"] == "event-123"


def test_delete_event_uses_delete() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "DELETE"
        assert request.url.path == (
            "/calendar/v3/calendars/primary/events/event-123"
        )
        return httpx.Response(204, request=request)

    with make_client(handler) as client:
        result = client.delete_event("primary", "event-123")

    assert result is None


def test_freebusy_request_contains_calendars_and_timezone() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/calendar/v3/freeBusy"

        import json

        payload = json.loads(request.read())

        assert payload == {
            "timeMin": "2026-11-16T09:00:00Z",
            "timeMax": "2026-11-16T18:00:00Z",
            "items": [
                {"id": "primary"},
                {"id": "team@example.com"},
            ],
            "timeZone": "Asia/Kolkata",
        }

        return json_response(
            request,
            {"calendars": {"primary": {"busy": []}}},
        )

    with make_client(handler) as client:
        result = client.query_freebusy(
            time_min="2026-11-16T09:00:00Z",
            time_max="2026-11-16T18:00:00Z",
            calendar_ids=["primary", "team@example.com"],
            time_zone="Asia/Kolkata",
        )

    assert "calendars" in result


def test_freebusy_omits_timezone_when_not_provided() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        import json

        payload = json.loads(request.read())
        assert "timeZone" not in payload
        return json_response(request, {"calendars": {}})

    with make_client(handler) as client:
        result = client.query_freebusy(
            time_min="2026-11-16T09:00:00Z",
            time_max="2026-11-16T18:00:00Z",
            calendar_ids=["primary"],
        )

    assert result == {"calendars": {}}


@pytest.mark.parametrize(
    ("status_code", "expected_message"),
    [
        (401, "Google authentication failed."),
        (403, "Google Calendar access denied."),
        (404, "Calendar or event not found"),
        (429, "Google Calendar API rate limit exceeded."),
        (500, "Internal server error"),
        (503, "Service unavailable"),
    ],
)
def test_http_errors_are_normalized(
    status_code: int,
    expected_message: str,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return json_response(
            request,
            {
                "error": {
                    "message": "Internal server error"
                    if status_code == 500
                    else (
                        "Service unavailable"
                        if status_code == 503
                        else "Provider error"
                    ),
                    "errors": [{"reason": "test"}],
                }
            },
            status_code=status_code,
        )

    with make_client(handler) as client:
        with pytest.raises(GoogleCalendarAPIError) as exc_info:
            client.get("/test")

    error = exc_info.value
    assert error.status_code == status_code
    assert expected_message in error.message
    assert error.to_dict()["type"] == "google_calendar_api_error"
    assert error.details == [{"reason": "test"}]


def test_non_json_http_error_is_handled_safely() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            502,
            text="Bad gateway",
            request=request,
        )

    with make_client(handler) as client:
        with pytest.raises(GoogleCalendarAPIError) as exc_info:
            client.get("/test")

    assert exc_info.value.status_code == 502
    assert exc_info.value.details is None


def test_invalid_json_success_response_is_normalized() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text="not-json",
            request=request,
        )

    with make_client(handler) as client:
        with pytest.raises(GoogleCalendarAPIError) as exc_info:
            client.get("/test")

    assert exc_info.value.status_code == 200
    assert exc_info.value.message == (
        "Google Calendar returned invalid JSON."
    )


def test_timeout_is_normalized() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout(
            "Sensitive transport detail",
            request=request,
        )

    with make_client(handler) as client:
        with pytest.raises(GoogleCalendarAPIError) as exc_info:
            client.get("/test")

    assert exc_info.value.message == (
        "Google Calendar API request timed out."
    )
    assert exc_info.value.status_code is None
    assert "Sensitive transport detail" not in str(exc_info.value)


def test_network_error_is_normalized() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(
            "Sensitive network detail",
            request=request,
        )

    with make_client(handler) as client:
        with pytest.raises(GoogleCalendarAPIError) as exc_info:
            client.get("/test")

    assert exc_info.value.message == (
        "Could not connect to Google Calendar API."
    )
    assert exc_info.value.status_code is None
    assert "Sensitive network detail" not in str(exc_info.value)


def test_error_response_does_not_include_access_token() -> None:
    token = "super-secret-test-token"

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == f"Bearer {token}"
        return json_response(
            request,
            {"error": {"message": "Permission denied"}},
            status_code=403,
        )

    with make_client(handler, access_token=token) as client:
        with pytest.raises(GoogleCalendarAPIError) as exc_info:
            client.get("/test")

    error_output = str(exc_info.value) + str(exc_info.value.to_dict())
    assert token not in error_output
