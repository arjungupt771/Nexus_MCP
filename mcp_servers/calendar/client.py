from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any
from urllib.parse import quote

import httpx

GOOGLE_CALENDAR_API_URL = "https://www.googleapis.com/calendar/v3"


class GoogleCalendarAPIError(Exception):
    """Normalized Google Calendar API error."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        details: Any = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "google_calendar_api_error",
            "message": self.message,
            "status_code": self.status_code,
            "details": self.details,
        }


class GoogleCalendarClient:
    """HTTP client for the Google Calendar REST API."""

    def __init__(
        self,
        access_token: str | None = None,
        *,
        token_provider: Callable[[], str | None] | None = None,
        timeout: float = 15.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._access_token = access_token
        self._token_provider = token_provider

        self._http = httpx.Client(
            base_url=GOOGLE_CALENDAR_API_URL,
            headers={
                "Accept": "application/json",
                "User-Agent": "Nexus-MCP-Calendar-Server",
            },
            timeout=httpx.Timeout(timeout),
            transport=transport,
        )

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> GoogleCalendarClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _get_access_token(self) -> str | None:
        if self._token_provider:
            return self._token_provider()

        return self._access_token or os.getenv("GOOGLE_ACCESS_TOKEN")

    @staticmethod
    def _calendar_path(calendar_id: str) -> str:
        encoded_id = quote(calendar_id, safe="")
        return f"/calendars/{encoded_id}"

    def request(
        self,
        method: str,
        endpoint: str,
        *,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
    ) -> Any:
        headers = {}
        token = self._get_access_token()

        if token:
            headers["Authorization"] = f"Bearer {token}"

        try:
            response = self._http.request(
                method,
                endpoint,
                params=params,
                json=json,
                headers=headers,
            )

        except httpx.TimeoutException as exc:
            raise GoogleCalendarAPIError("Google Calendar API request timed out.") from exc

        except httpx.RequestError as exc:
            raise GoogleCalendarAPIError("Could not connect to Google Calendar API.") from exc

        if response.is_error:
            try:
                payload = response.json()
            except ValueError:
                payload = {}

            error = payload.get("error", {})
            message = error.get(
                "message",
                "Google Calendar API request failed.",
            )

            if response.status_code == 401:
                message = (
                    "Google authentication failed. Check OAuth credentials and token validity."
                )

            elif response.status_code == 403:
                message = (
                    "Google Calendar access denied. Check OAuth scopes and calendar permissions."
                )

            elif response.status_code == 404:
                message = "Calendar or event not found, or access is unavailable."

            elif response.status_code == 429:
                message = "Google Calendar API rate limit exceeded."

            raise GoogleCalendarAPIError(
                message,
                status_code=response.status_code,
                details=error.get("errors"),
            )

        if response.status_code == 204 or not response.content:
            return None

        try:
            return response.json()
        except ValueError as exc:
            raise GoogleCalendarAPIError(
                "Google Calendar returned invalid JSON.",
                status_code=response.status_code,
            ) from exc

    def get(
        self,
        endpoint: str,
        *,
        params: dict[str, Any] | None = None,
    ) -> Any:
        return self.request(
            "GET",
            endpoint,
            params=params,
        )

    def post(
        self,
        endpoint: str,
        *,
        json: dict[str, Any],
    ) -> Any:
        return self.request(
            "POST",
            endpoint,
            json=json,
        )

    def patch(
        self,
        endpoint: str,
        *,
        json: dict[str, Any],
    ) -> Any:
        return self.request(
            "PATCH",
            endpoint,
            json=json,
        )

    def delete(self, endpoint: str) -> None:
        self.request("DELETE", endpoint)

    def list_calendars(
        self,
        *,
        max_results: int = 100,
        page_token: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "maxResults": max_results,
        }

        if page_token:
            params["pageToken"] = page_token

        return self.get(
            "/users/me/calendarList",
            params=params,
        )

    def list_events(
        self,
        calendar_id: str,
        *,
        time_min: str | None = None,
        time_max: str | None = None,
        max_results: int = 50,
        page_token: str | None = None,
        single_events: bool = True,
        order_by: str = "startTime",
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "maxResults": max_results,
            "singleEvents": single_events,
            "orderBy": order_by,
        }

        if time_min:
            params["timeMin"] = time_min

        if time_max:
            params["timeMax"] = time_max

        if page_token:
            params["pageToken"] = page_token

        return self.get(
            f"{self._calendar_path(calendar_id)}/events",
            params=params,
        )

    def get_event(
        self,
        calendar_id: str,
        event_id: str,
    ) -> dict[str, Any]:
        return self.get(f"{self._calendar_path(calendar_id)}/events/{quote(event_id, safe='')}")

    def create_event(
        self,
        calendar_id: str,
        event: dict[str, Any],
    ) -> dict[str, Any]:
        return self.post(
            f"{self._calendar_path(calendar_id)}/events",
            json=event,
        )

    def update_event(
        self,
        calendar_id: str,
        event_id: str,
        event: dict[str, Any],
    ) -> dict[str, Any]:
        return self.patch(
            f"{self._calendar_path(calendar_id)}/events/{quote(event_id, safe='')}",
            json=event,
        )

    def delete_event(
        self,
        calendar_id: str,
        event_id: str,
    ) -> None:
        self.delete(f"{self._calendar_path(calendar_id)}/events/{quote(event_id, safe='')}")

    def query_freebusy(
        self,
        *,
        time_min: str,
        time_max: str,
        calendar_ids: list[str],
        time_zone: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "timeMin": time_min,
            "timeMax": time_max,
            "items": [{"id": calendar_id} for calendar_id in calendar_ids],
        }

        if time_zone:
            payload["timeZone"] = time_zone

        return self.post(
            "/freeBusy",
            json=payload,
        )
