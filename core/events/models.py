"""Event models for registry and tool-catalog changes."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class RegistryEventType(StrEnum):
    SERVER_REGISTERED = "server.registered"
    SERVER_UPDATED = "server.updated"
    SERVER_UNREGISTERED = "server.unregistered"
    SERVER_ENABLED = "server.enabled"
    SERVER_DISABLED = "server.disabled"
    SERVER_CONNECTING = "server.connecting"
    SERVER_CONNECTED = "server.connected"
    SERVER_DISCONNECTED = "server.disconnected"
    SERVER_ERROR = "server.error"
    TOOLS_UPDATED = "tools.updated"
    DISCOVERY_FAILED = "tools.discovery_failed"


class RegistryEvent(BaseModel):
    """A JSON-serializable change notification.

    ``sequence`` is assigned by the bus and is strictly increasing, so a
    future dashboard can resume a stream with ``recent(since=sequence)``.
    ``data`` never contains configuration values or credentials.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: UUID = Field(default_factory=uuid4)
    sequence: int = 0
    type: RegistryEventType
    server_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    data: dict[str, Any] = Field(default_factory=dict)
