"""Read models returned by server management operations."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from core.models.server import ServerCapabilities, ServerStatus, TransportType


class ServerStatusReport(BaseModel):
    """Full, credential-free view of a registered server."""

    model_config = ConfigDict(extra="forbid")

    server_id: str
    name: str
    description: str = ""
    transport: TransportType
    status: ServerStatus
    enabled: bool
    connected: bool
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)
    capabilities: list[str] = Field(default_factory=list)
    protocol_capabilities: ServerCapabilities = Field(default_factory=ServerCapabilities)
    tool_count: int = 0
    tools: list[str] = Field(default_factory=list)
    last_discovery_at: datetime | None = None
    last_discovery_error: str | None = None
    discovery_failure_count: int = 0
    config_pending_restart: bool = False
    # Public view only: literal env/header values are masked.
    config: dict[str, Any] = Field(default_factory=dict)


class RestoreReport(BaseModel):
    """Outcome of restoring persisted servers at startup."""

    model_config = ConfigDict(extra="forbid")

    restored: list[str] = Field(default_factory=list)
    connected: list[str] = Field(default_factory=list)
    skipped_disabled: list[str] = Field(default_factory=list)
    already_registered: list[str] = Field(default_factory=list)
    # server_id -> error code; a failed connection never aborts startup.
    failed: dict[str, str] = Field(default_factory=dict)
