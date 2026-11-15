from unittest.mock import Mock

import pytest

from mcp_servers.calendar.service import (
    CalendarValidationError,
    GoogleCalendarService,
)


@pytest.fixture
def client():
    return Mock()


@pytest.fixture
def service(client):
    return GoogleCalendarService(client)


# ---------------------------------------------------------
# Calendar listing and event retrieval
# ---------------------------------------------------------


def test_list_calendars(service, client):
    client.list_calendars.return_value = {"items": [{"id": "primary"}]}

    result = service.list_calendars(max_results=50)

    assert result["items"][0]["id"] == "primary"
    client.list_calendars.assert_called_once_with(
        max_results=50,
        page_token=None,
    )


def test_get_events(service, client):
    client.list_events.return_value = {"items": [{"id": "event-1"}]}

    result = service.get_events(
        "primary",
        "2026-11-15T09:00:00+05:30",
        "2026-11-15T17:00:00+05:30",
    )

    assert result["items"][0]["id"] == "event-1"
    client.list_events.assert_called_once()


def test_get_event_details(service, client):
    client.get_event.return_value = {
        "id": "event-1",
        "summary": "Team meeting",
    }

    result = service.get_event_details("primary", "event-1")

    assert result["summary"] == "Team meeting"
    client.get_event.assert_called_once_with(
        "primary",
        "event-1",
    )


# ---------------------------------------------------------
# Event creation and updates
# ---------------------------------------------------------


def test_create_event(service, client):
    client.create_event.return_value = {
        "id": "event-123",
        "summary": "Team meeting",
    }

    result = service.create_event(
        "primary",
        "Team meeting",
        "2026-11-16T10:00:00+05:30",
        "2026-11-16T11:00:00+05:30",
    )

    assert result["id"] == "event-123"

    client.create_event.assert_called_once_with(
        "primary",
        {
            "summary": "Team meeting",
            "start": {"dateTime": "2026-11-16T10:00:00+05:30"},
            "end": {"dateTime": "2026-11-16T11:00:00+05:30"},
        },
    )


def test_create_event_with_optional_fields(service, client):
    client.create_event.return_value = {"id": "event-123"}

    service.create_event(
        "primary",
        "Project review",
        "2026-11-16T10:00:00+05:30",
        "2026-11-16T11:00:00+05:30",
        description="Quarterly review",
        location="Conference Room",
        time_zone="Asia/Kolkata",
    )

    payload = client.create_event.call_args.args[1]

    assert payload["description"] == "Quarterly review"
    assert payload["location"] == "Conference Room"
    assert payload["start"]["timeZone"] == "Asia/Kolkata"
    assert payload["end"]["timeZone"] == "Asia/Kolkata"


def test_update_event(service, client):
    client.update_event.return_value = {
        "id": "event-123",
        "summary": "Updated meeting",
    }

    service.update_event(
        "primary",
        "event-123",
        summary="Updated meeting",
    )

    client.update_event.assert_called_once_with(
        "primary",
        "event-123",
        {"summary": "Updated meeting"},
    )


def test_update_event_time(service, client):
    client.update_event.return_value = {"id": "event-123"}

    service.update_event(
        "primary",
        "event-123",
        start="2026-11-16T11:00:00+05:30",
        end="2026-11-16T12:00:00+05:30",
        time_zone="Asia/Kolkata",
    )

    payload = client.update_event.call_args.args[2]

    assert payload["start"]["timeZone"] == "Asia/Kolkata"
    assert payload["end"]["timeZone"] == "Asia/Kolkata"


def test_delete_event(service, client):
    client.delete_event.return_value = None

    result = service.delete_event("primary", "event-123")

    assert result == {
        "deleted": True,
        "calendar_id": "primary",
        "event_id": "event-123",
    }

    client.delete_event.assert_called_once_with(
        "primary",
        "event-123",
    )


# ---------------------------------------------------------
# Datetime and event validation
# ---------------------------------------------------------


@pytest.mark.parametrize(
    ("time_min", "time_max"),
    [
        ("invalid", "2026-11-16T12:00:00+05:30"),
        ("2026-11-16T10:00:00", "2026-11-16T12:00:00+05:30"),
        ("2026-11-16T10:00:00+05:30", "invalid"),
        ("2026-11-16T10:00:00+05:30", "2026-11-16T10:00:00+05:30"),
        ("2026-11-16T12:00:00+05:30", "2026-11-16T10:00:00+05:30"),
    ],
)
def test_rejects_invalid_time_ranges(
    service,
    time_min,
    time_max,
):
    with pytest.raises(CalendarValidationError):
        service.get_events("primary", time_min, time_max)


def test_rejects_event_without_timezone(service):
    with pytest.raises(CalendarValidationError):
        service.create_event(
            "primary",
            "Meeting",
            "2026-11-16T10:00:00",
            "2026-11-16T11:00:00",
        )


def test_rejects_event_with_end_before_start(service):
    with pytest.raises(CalendarValidationError):
        service.create_event(
            "primary",
            "Meeting",
            "2026-11-16T12:00:00+05:30",
            "2026-11-16T10:00:00+05:30",
        )


def test_rejects_empty_event_summary(service):
    with pytest.raises(CalendarValidationError):
        service.create_event(
            "primary",
            " ",
            "2026-11-16T10:00:00+05:30",
            "2026-11-16T11:00:00+05:30",
        )


def test_rejects_event_longer_than_seven_days(service):
    with pytest.raises(CalendarValidationError):
        service.create_event(
            "primary",
            "Long event",
            "2026-11-01T10:00:00+05:30",
            "2026-11-10T10:00:00+05:30",
        )


def test_rejects_invalid_timezone(service):
    with pytest.raises(CalendarValidationError):
        service.create_event(
            "primary",
            "Meeting",
            "2026-11-16T10:00:00+05:30",
            "2026-11-16T11:00:00+05:30",
            time_zone="Invalid/Timezone",
        )


def test_update_requires_both_start_and_end(service):
    with pytest.raises(CalendarValidationError):
        service.update_event(
            "primary",
            "event-123",
            start="2026-11-16T10:00:00+05:30",
        )


def test_update_rejects_empty_update(service):
    with pytest.raises(CalendarValidationError):
        service.update_event(
            "primary",
            "event-123",
        )


def test_update_rejects_timezone_without_time_changes(service):
    with pytest.raises(CalendarValidationError):
        service.update_event(
            "primary",
            "event-123",
            summary="New summary",
            time_zone="Asia/Kolkata",
        )


# ---------------------------------------------------------
# Availability validation
# ---------------------------------------------------------


def test_check_availability(service, client):
    client.query_freebusy.return_value = {
        "calendars": {
            "primary": {
                "busy": [],
            }
        }
    }

    result = service.check_availability(
        "2026-11-16T09:00:00+05:30",
        "2026-11-16T17:00:00+05:30",
        ["primary"],
        time_zone="Asia/Kolkata",
    )

    assert "primary" in result["calendars"]
    client.query_freebusy.assert_called_once()


def test_rejects_empty_calendar_list(service):
    with pytest.raises(CalendarValidationError):
        service.check_availability(
            "2026-11-16T09:00:00+05:30",
            "2026-11-16T17:00:00+05:30",
            [],
        )


def test_rejects_duplicate_calendar_ids(service):
    with pytest.raises(CalendarValidationError):
        service.check_availability(
            "2026-11-16T09:00:00+05:30",
            "2026-11-16T17:00:00+05:30",
            ["primary", "primary"],
        )


def test_rejects_too_many_calendars(service):
    with pytest.raises(CalendarValidationError):
        service.check_availability(
            "2026-11-16T09:00:00+05:30",
            "2026-11-16T17:00:00+05:30",
            [f"calendar-{i}" for i in range(21)],
        )


def test_rejects_invalid_max_results(service):
    with pytest.raises(CalendarValidationError):
        service.list_calendars(max_results=251)


# ---------------------------------------------------------
# Free-slot calculation
# ---------------------------------------------------------


def configure_busy_periods(client, calendars):
    client.query_freebusy.return_value = {
        "calendars": calendars,
    }


def test_find_free_slots_without_busy_events(service, client):
    configure_busy_periods(
        client,
        {"primary": {"busy": []}},
    )

    result = service.find_free_slots(
        "2026-11-16T09:00:00+00:00",
        "2026-11-16T17:00:00+00:00",
        ["primary"],
        duration_minutes=60,
    )

    assert result["total_slots"] == 1
    assert result["free_slots"][0]["duration_minutes"] == 480


def test_find_free_slots_with_one_busy_interval(service, client):
    configure_busy_periods(
        client,
        {
            "primary": {
                "busy": [
                    {
                        "start": "2026-11-16T12:00:00+00:00",
                        "end": "2026-11-16T13:00:00+00:00",
                    }
                ]
            }
        },
    )

    result = service.find_free_slots(
        "2026-11-16T09:00:00+00:00",
        "2026-11-16T17:00:00+00:00",
        ["primary"],
        duration_minutes=60,
    )

    slots = result["free_slots"]

    assert len(slots) == 2
    assert slots[0]["duration_minutes"] == 180
    assert slots[1]["duration_minutes"] == 240


def test_merges_overlapping_busy_intervals(service, client):
    configure_busy_periods(
        client,
        {
            "primary": {
                "busy": [
                    {
                        "start": "2026-11-16T10:00:00+00:00",
                        "end": "2026-11-16T11:30:00+00:00",
                    },
                    {
                        "start": "2026-11-16T11:00:00+00:00",
                        "end": "2026-11-16T12:00:00+00:00",
                    },
                ]
            }
        },
    )

    result = service.find_free_slots(
        "2026-11-16T09:00:00+00:00",
        "2026-11-16T17:00:00+00:00",
        ["primary"],
        duration_minutes=30,
    )

    slots = result["free_slots"]

    assert len(slots) == 2
    assert slots[0]["end"] == "2026-11-16T10:00:00+00:00"
    assert slots[1]["start"] == "2026-11-16T12:00:00+00:00"


def test_merges_adjacent_busy_intervals(service, client):
    configure_busy_periods(
        client,
        {
            "primary": {
                "busy": [
                    {
                        "start": "2026-11-16T10:00:00+00:00",
                        "end": "2026-11-16T11:00:00+00:00",
                    },
                    {
                        "start": "2026-11-16T11:00:00+00:00",
                        "end": "2026-11-16T12:00:00+00:00",
                    },
                ]
            }
        },
    )

    result = service.find_free_slots(
        "2026-11-16T09:00:00+00:00",
        "2026-11-16T17:00:00+00:00",
        ["primary"],
        duration_minutes=30,
    )

    assert result["total_slots"] == 2
    assert result["free_slots"][0]["end"] == ("2026-11-16T10:00:00+00:00")
    assert result["free_slots"][1]["start"] == ("2026-11-16T12:00:00+00:00")


def test_merges_busy_intervals_across_calendars(service, client):
    configure_busy_periods(
        client,
        {
            "primary": {
                "busy": [
                    {
                        "start": "2026-11-16T10:00:00+00:00",
                        "end": "2026-11-16T11:00:00+00:00",
                    }
                ]
            },
            "work": {
                "busy": [
                    {
                        "start": "2026-11-16T10:30:00+00:00",
                        "end": "2026-11-16T12:00:00+00:00",
                    }
                ]
            },
        },
    )

    result = service.find_free_slots(
        "2026-11-16T09:00:00+00:00",
        "2026-11-16T17:00:00+00:00",
        ["primary", "work"],
        duration_minutes=30,
    )

    assert result["total_slots"] == 2
    assert result["free_slots"][0]["end"] == ("2026-11-16T10:00:00+00:00")
    assert result["free_slots"][1]["start"] == ("2026-11-16T12:00:00+00:00")


def test_clips_busy_intervals_to_requested_window(service, client):
    configure_busy_periods(
        client,
        {
            "primary": {
                "busy": [
                    {
                        "start": "2026-11-16T08:00:00+00:00",
                        "end": "2026-11-16T10:00:00+00:00",
                    }
                ]
            }
        },
    )

    result = service.find_free_slots(
        "2026-11-16T09:00:00+00:00",
        "2026-11-16T17:00:00+00:00",
        ["primary"],
        duration_minutes=60,
    )

    assert result["total_slots"] == 1
    assert result["free_slots"][0]["start"] == ("2026-11-16T10:00:00+00:00")


def test_filters_slots_shorter_than_requested_duration(service, client):
    configure_busy_periods(
        client,
        {
            "primary": {
                "busy": [
                    {
                        "start": "2026-11-16T09:30:00+00:00",
                        "end": "2026-11-16T16:30:00+00:00",
                    }
                ]
            }
        },
    )

    result = service.find_free_slots(
        "2026-11-16T09:00:00+00:00",
        "2026-11-16T17:00:00+00:00",
        ["primary"],
        duration_minutes=60,
    )

    assert result["total_slots"] == 0


def test_limits_number_of_returned_slots(service, client):
    configure_busy_periods(
        client,
        {"primary": {"busy": []}},
    )

    result = service.find_free_slots(
        "2026-11-16T09:00:00+00:00",
        "2026-11-16T17:00:00+00:00",
        ["primary"],
        duration_minutes=30,
        max_slots=1,
    )

    assert result["total_slots"] == 1


def test_rejects_invalid_meeting_duration(service):
    with pytest.raises(CalendarValidationError):
        service.find_free_slots(
            "2026-11-16T09:00:00+00:00",
            "2026-11-16T17:00:00+00:00",
            ["primary"],
            duration_minutes=0,
        )


def test_rejects_invalid_max_slots(service):
    with pytest.raises(CalendarValidationError):
        service.find_free_slots(
            "2026-11-16T09:00:00+00:00",
            "2026-11-16T17:00:00+00:00",
            ["primary"],
            duration_minutes=30,
            max_slots=101,
        )


def test_rejects_calendar_specific_freebusy_errors(service, client):
    configure_busy_periods(
        client,
        {
            "primary": {
                "errors": [{"reason": "notFound"}],
                "busy": [],
            }
        },
    )

    with pytest.raises(CalendarValidationError):
        service.find_free_slots(
            "2026-11-16T09:00:00+00:00",
            "2026-11-16T17:00:00+00:00",
            ["primary"],
            duration_minutes=30,
        )


def test_free_slot_calculation_handles_different_offsets(
    service,
    client,
):
    configure_busy_periods(
        client,
        {
            "primary": {
                "busy": [
                    {
                        "start": "2026-11-16T10:00:00+00:00",
                        "end": "2026-11-16T11:00:00+00:00",
                    }
                ]
            }
        },
    )

    result = service.find_free_slots(
        "2026-11-16T14:30:00+05:30",
        "2026-11-16T18:30:00+05:30",
        ["primary"],
        duration_minutes=30,
    )

    assert result["total_slots"] == 2
    assert result["free_slots"][0]["start"] == ("2026-11-16T14:30:00+05:30")
    assert result["free_slots"][0]["end"] == ("2026-11-16T15:30:00+05:30")
