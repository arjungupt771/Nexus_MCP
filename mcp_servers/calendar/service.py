from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from mcp_servers.calendar.client import (
    GoogleCalendarClient,
)


class CalendarValidationError(ValueError):
    """Raised when Calendar tool arguments are invalid."""


class GoogleCalendarService:
    """Business logic for Google Calendar operations."""

    MAX_CALENDARS = 20
    MAX_RESULTS = 250
    MAX_EVENT_DURATION_MINUTES = 10080

    def __init__(self, client: GoogleCalendarClient) -> None:
        self.client = client

    @staticmethod
    def _validate_identifier(
        value: str,
        field: str,
    ) -> str:
        value = value.strip()

        if not value or len(value) > 1024 or any(ord(char) < 32 for char in value):
            raise CalendarValidationError(f"Invalid {field}.")

        return value

    @staticmethod
    def _parse_datetime(
        value: str,
        field: str,
    ) -> datetime:
        if not isinstance(value, str) or not value.strip():
            raise CalendarValidationError(f"{field} must be a valid ISO 8601 datetime.")

        try:
            parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError as exc:
            raise CalendarValidationError(f"{field} must be a valid ISO 8601 datetime.") from exc

        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise CalendarValidationError(f"{field} must include a timezone offset.")

        return parsed

    @staticmethod
    def _validate_timezone(
        time_zone: str | None,
    ) -> str | None:
        if time_zone is None:
            return None

        try:
            ZoneInfo(time_zone)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise CalendarValidationError("Invalid IANA timezone identifier.") from exc

        return time_zone

    @classmethod
    def _validate_time_range(
        cls,
        time_min: str,
        time_max: str,
    ) -> tuple[datetime, datetime]:
        start = cls._parse_datetime(time_min, "time_min")
        end = cls._parse_datetime(time_max, "time_max")

        if end <= start:
            raise CalendarValidationError("time_max must be later than time_min.")

        return start, end

    @classmethod
    def _validate_calendar_ids(
        cls,
        calendar_ids: list[str],
    ) -> list[str]:
        if not calendar_ids:
            raise CalendarValidationError("At least one calendar ID is required.")

        if len(calendar_ids) > cls.MAX_CALENDARS:
            raise CalendarValidationError(f"At most {cls.MAX_CALENDARS} calendars are allowed.")

        validated = [
            cls._validate_identifier(calendar_id, "calendar ID") for calendar_id in calendar_ids
        ]

        if len(set(validated)) != len(validated):
            raise CalendarValidationError("Duplicate calendar IDs are not allowed.")

        return validated

    @staticmethod
    def _validate_max_results(max_results: int) -> None:
        if not 1 <= max_results <= 250:
            raise CalendarValidationError("max_results must be between 1 and 250.")

    @classmethod
    def _validate_event_times(
        cls,
        start: str,
        end: str,
    ) -> tuple[datetime, datetime]:
        start_dt = cls._parse_datetime(start, "start")
        end_dt = cls._parse_datetime(end, "end")

        if end_dt <= start_dt:
            raise CalendarValidationError("Event end must be later than event start.")

        duration = end_dt - start_dt

        if duration > timedelta(minutes=cls.MAX_EVENT_DURATION_MINUTES):
            raise CalendarValidationError("Event duration cannot exceed seven days.")

        return start_dt, end_dt

    @classmethod
    def _validate_event_summary(cls, summary: str) -> str:
        if not isinstance(summary, str) or not summary.strip():
            raise CalendarValidationError("Event summary cannot be empty.")

        summary = summary.strip()

        if len(summary) > 1024:
            raise CalendarValidationError("Event summary cannot exceed 1024 characters.")

        return summary

    @staticmethod
    def _validate_event_text(
        value: str,
        field: str,
        max_length: int = 8192,
    ) -> str:
        if not isinstance(value, str):
            raise CalendarValidationError(f"{field} must be a string.")

        if len(value) > max_length:
            raise CalendarValidationError(f"{field} cannot exceed {max_length} characters.")

        return value

    def list_calendars(
        self,
        *,
        max_results: int = 100,
        page_token: str | None = None,
    ) -> dict[str, Any]:
        self._validate_max_results(max_results)

        return self.client.list_calendars(
            max_results=max_results,
            page_token=page_token,
        )

    def get_events(
        self,
        calendar_id: str,
        time_min: str,
        time_max: str,
        *,
        max_results: int = 50,
        page_token: str | None = None,
    ) -> dict[str, Any]:
        calendar_id = self._validate_identifier(
            calendar_id,
            "calendar ID",
        )

        self._validate_time_range(time_min, time_max)
        self._validate_max_results(max_results)

        return self.client.list_events(
            calendar_id,
            time_min=time_min,
            time_max=time_max,
            max_results=max_results,
            page_token=page_token,
        )

    def get_event_details(
        self,
        calendar_id: str,
        event_id: str,
    ) -> dict[str, Any]:
        calendar_id = self._validate_identifier(
            calendar_id,
            "calendar ID",
        )
        event_id = self._validate_identifier(
            event_id,
            "event ID",
        )

        return self.client.get_event(
            calendar_id,
            event_id,
        )

    def create_event(
        self,
        calendar_id: str,
        summary: str,
        start: str,
        end: str,
        *,
        description: str | None = None,
        location: str | None = None,
        time_zone: str | None = None,
    ) -> dict[str, Any]:
        calendar_id = self._validate_identifier(
            calendar_id,
            "calendar ID",
        )

        summary = self._validate_event_summary(summary)
        self._validate_event_times(start, end)
        time_zone = self._validate_timezone(time_zone)

        event: dict[str, Any] = {
            "summary": summary,
            "start": {"dateTime": start},
            "end": {"dateTime": end},
        }

        if time_zone:
            event["start"]["timeZone"] = time_zone
            event["end"]["timeZone"] = time_zone

        if description is not None:
            event["description"] = self._validate_event_text(
                description,
                "description",
            )

        if location is not None:
            event["location"] = self._validate_event_text(
                location,
                "location",
                max_length=2048,
            )

        return self.client.create_event(
            calendar_id,
            event,
        )

    def update_event(
        self,
        calendar_id: str,
        event_id: str,
        *,
        summary: str | None = None,
        description: str | None = None,
        location: str | None = None,
        start: str | None = None,
        end: str | None = None,
        time_zone: str | None = None,
    ) -> dict[str, Any]:
        calendar_id = self._validate_identifier(
            calendar_id,
            "calendar ID",
        )
        event_id = self._validate_identifier(
            event_id,
            "event ID",
        )

        if not any(
            value is not None
            for value in (
                summary,
                description,
                location,
                start,
                end,
            )
        ):
            raise CalendarValidationError("Provide at least one field to update.")

        if (start is None) != (end is None):
            raise CalendarValidationError(
                "Both start and end must be provided when changing event time."
            )

        event: dict[str, Any] = {}

        if summary is not None:
            event["summary"] = self._validate_event_summary(summary)

        if description is not None:
            event["description"] = self._validate_event_text(
                description,
                "description",
            )

        if location is not None:
            event["location"] = self._validate_event_text(
                location,
                "location",
                max_length=2048,
            )

        if start is not None and end is not None:
            self._validate_event_times(start, end)
            time_zone = self._validate_timezone(time_zone)

            event["start"] = {"dateTime": start}
            event["end"] = {"dateTime": end}

            if time_zone:
                event["start"]["timeZone"] = time_zone
                event["end"]["timeZone"] = time_zone

        elif time_zone is not None:
            raise CalendarValidationError(
                "time_zone can only be supplied when updating start and end."
            )

        return self.client.update_event(
            calendar_id,
            event_id,
            event,
        )

    def delete_event(
        self,
        calendar_id: str,
        event_id: str,
    ) -> dict[str, Any]:
        calendar_id = self._validate_identifier(
            calendar_id,
            "calendar ID",
        )
        event_id = self._validate_identifier(
            event_id,
            "event ID",
        )

        self.client.delete_event(calendar_id, event_id)

        return {
            "deleted": True,
            "calendar_id": calendar_id,
            "event_id": event_id,
        }

    def check_availability(
        self,
        time_min: str,
        time_max: str,
        calendar_ids: list[str],
        *,
        time_zone: str | None = None,
    ) -> dict[str, Any]:
        self._validate_time_range(time_min, time_max)
        calendar_ids = self._validate_calendar_ids(calendar_ids)
        time_zone = self._validate_timezone(time_zone)

        result = self.client.query_freebusy(
            time_min=time_min,
            time_max=time_max,
            calendar_ids=calendar_ids,
            time_zone=time_zone,
        )

        return {
            "time_min": time_min,
            "time_max": time_max,
            "calendars": result.get("calendars", {}),
            "time_zone": time_zone,
        }

    def find_free_slots(
        self,
        time_min: str,
        time_max: str,
        calendar_ids: list[str],
        duration_minutes: int,
        *,
        time_zone: str | None = None,
        max_slots: int = 20,
    ) -> dict[str, Any]:
        window_start, window_end = self._validate_time_range(
            time_min,
            time_max,
        )

        calendar_ids = self._validate_calendar_ids(calendar_ids)
        time_zone = self._validate_timezone(time_zone)

        if not 1 <= duration_minutes <= 1440:
            raise CalendarValidationError("duration_minutes must be between 1 and 1440.")

        if not 1 <= max_slots <= 100:
            raise CalendarValidationError("max_slots must be between 1 and 100.")

        freebusy = self.client.query_freebusy(
            time_min=time_min,
            time_max=time_max,
            calendar_ids=calendar_ids,
            time_zone=time_zone,
        )

        calendars = freebusy.get("calendars", {})

        busy_intervals: list[tuple[datetime, datetime]] = []

        for calendar_id in calendar_ids:
            calendar_data = calendars.get(calendar_id, {})

            if calendar_data.get("errors"):
                raise CalendarValidationError(
                    f"Could not retrieve availability for calendar {calendar_id}."
                )

            for interval in calendar_data.get("busy", []):
                busy_start = self._parse_datetime(
                    interval["start"],
                    "busy interval start",
                )
                busy_end = self._parse_datetime(
                    interval["end"],
                    "busy interval end",
                )

                if busy_end <= busy_start:
                    continue

                # Clip busy periods to the requested window.
                clipped_start = max(busy_start, window_start)
                clipped_end = min(busy_end, window_end)

                if clipped_start < clipped_end:
                    busy_intervals.append((clipped_start, clipped_end))

        # Merge overlapping and adjacent busy intervals.
        busy_intervals.sort(key=lambda item: item[0])

        merged: list[tuple[datetime, datetime]] = []

        for current_start, current_end in busy_intervals:
            if not merged or current_start > merged[-1][1]:
                merged.append((current_start, current_end))
            else:
                previous_start, previous_end = merged[-1]

                merged[-1] = (
                    previous_start,
                    max(previous_end, current_end),
                )

        # Calculate the gaps between merged busy intervals.
        free_intervals: list[tuple[datetime, datetime]] = []
        cursor = window_start

        for busy_start, busy_end in merged:
            if busy_start > cursor:
                free_intervals.append((cursor, busy_start))

            cursor = max(cursor, busy_end)

        if cursor < window_end:
            free_intervals.append((cursor, window_end))

        required_duration = timedelta(minutes=duration_minutes)

        slots = []

        for free_start, free_end in free_intervals:
            if free_end - free_start < required_duration:
                continue

            slots.append(
                {
                    "start": free_start.astimezone(window_start.tzinfo).isoformat(),
                    "end": free_end.astimezone(window_start.tzinfo).isoformat(),
                    "duration_minutes": int((free_end - free_start).total_seconds() // 60),
                }
            )

            if len(slots) >= max_slots:
                break

        return {
            "time_min": time_min,
            "time_max": time_max,
            "duration_requested_minutes": duration_minutes,
            "time_zone": time_zone,
            "free_slots": slots,
            "total_slots": len(slots),
        }
