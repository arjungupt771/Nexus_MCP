from __future__ import annotations

import asyncio

import pytest

from core.events import InMemoryEventBus, NullEventBus, RegistryEvent, RegistryEventType


def make_event(server_id: str = "a", kind=RegistryEventType.SERVER_REGISTERED) -> RegistryEvent:
    return RegistryEvent(type=kind, server_id=server_id)


async def test_publish_assigns_increasing_sequence_numbers():
    bus = InMemoryEventBus()
    first = await bus.publish(make_event())
    second = await bus.publish(make_event())

    assert (first.sequence, second.sequence) == (1, 2)


async def test_sync_and_async_handlers_receive_events():
    bus = InMemoryEventBus()
    seen: list[str] = []

    async def async_handler(event: RegistryEvent) -> None:
        seen.append(f"async:{event.server_id}")

    bus.subscribe(lambda event: seen.append(f"sync:{event.server_id}"))
    bus.subscribe(async_handler)
    await bus.publish(make_event("x"))

    assert seen == ["sync:x", "async:x"]


async def test_failing_handler_does_not_break_publisher_or_other_handlers():
    bus = InMemoryEventBus()
    seen: list[int] = []

    def broken(_: RegistryEvent) -> None:
        raise RuntimeError("handler exploded")

    bus.subscribe(broken)
    bus.subscribe(lambda e: seen.append(e.sequence))

    await bus.publish(make_event())
    assert seen == [1]


async def test_unsubscribe_stops_delivery():
    bus = InMemoryEventBus()
    seen: list[int] = []
    unsubscribe = bus.subscribe(lambda e: seen.append(e.sequence))

    await bus.publish(make_event())
    unsubscribe()
    await bus.publish(make_event())

    assert seen == [1]


async def test_stream_delivers_live_events_and_cleans_up():
    bus = InMemoryEventBus()

    async with bus.stream() as events:
        await bus.publish(make_event("s1"))
        received = await asyncio.wait_for(anext(events), timeout=1)
        assert received.server_id == "s1"

    assert bus._queues == []


async def test_stream_drops_oldest_when_consumer_is_slow():
    bus = InMemoryEventBus()

    async with bus.stream(max_queue=2) as events:
        for _ in range(4):
            await bus.publish(make_event())

        first = await asyncio.wait_for(anext(events), timeout=1)
        assert first.sequence == 3


async def test_recent_supports_resume_and_history_is_bounded():
    bus = InMemoryEventBus(history_size=3)

    for _ in range(5):
        await bus.publish(make_event())

    assert [e.sequence for e in bus.recent()] == [3, 4, 5]
    assert [e.sequence for e in bus.recent(since=4)] == [5]


def test_history_size_must_be_positive():
    with pytest.raises(ValueError):
        InMemoryEventBus(history_size=0)


async def test_null_bus_accepts_events():
    bus = NullEventBus()
    event = await bus.publish(make_event())
    assert event.server_id == "a"
    bus.subscribe(lambda e: None)()


def test_events_are_json_serializable():
    event = RegistryEvent(
        type=RegistryEventType.TOOLS_UPDATED, server_id="a", data={"added": ["t"]}
    )
    assert "tools.updated" in event.model_dump_json()
