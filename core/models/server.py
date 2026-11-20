from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TransportType(StrEnum):
    STDIO = "stdio"
    STREAMABLE_HTTP = "streamable_http"


class ServerStatus(StrEnum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


def normalize_server_id(value: str) -> str:
    """Normalize and validate a server identifier."""
    value = value.strip().lower()

    if not value:
        raise ValueError("Server ID cannot be empty.")

    allowed = set("abcdefghijklmnopqrstuvwxyz0123456789-_")

    if any(char not in allowed for char in value):
        raise ValueError(
            "Server ID may contain only lowercase letters, numbers, hyphens and underscores."
        )

    return value


class ServerCapabilities(BaseModel):
    """Protocol-level capabilities a server declared during initialization."""

    model_config = ConfigDict(extra="forbid")

    tools: bool = False
    tools_list_changed: bool = False
    resources: bool = False
    prompts: bool = False
    logging: bool = False


class ServerDescriptor(BaseModel):
    """Runtime-independent identity and status of an MCP server.

    Phase 5 adds descriptive metadata, enabled state and discovery tracking.
    Every new field has a default, so Phase 0-4 call sites are unaffected.
    """

    model_config = ConfigDict(extra="forbid")

    server_id: str = Field(min_length=1, max_length=100)
    name: str = Field(min_length=1, max_length=150)
    transport: TransportType
    status: ServerStatus = ServerStatus.DISCONNECTED
    version: str | None = None

    description: str = Field(default="", max_length=1000)
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)
    # Capability labels declared by the operator (e.g. "repository-search").
    capabilities: list[str] = Field(default_factory=list)
    # Capabilities observed from the MCP initialize handshake.
    protocol_capabilities: ServerCapabilities = Field(default_factory=ServerCapabilities)
    enabled: bool = True
    tool_count: int = Field(default=0, ge=0)
    last_discovery_at: datetime | None = None
    last_discovery_error: str | None = None
    discovery_failure_count: int = Field(default=0, ge=0)
    # True when a persisted config change has not yet been applied to the
    # active connection.
    config_pending_restart: bool = False

    @field_validator("server_id")
    @classmethod
    def validate_server_id(cls, value: str) -> str:
        return normalize_server_id(value)

    @field_validator("tags", "capabilities")
    @classmethod
    def validate_labels(cls, value: list[str]) -> list[str]:
        cleaned: list[str] = []

        for item in value:
            item = item.strip().lower()

            if not item:
                raise ValueError("Labels cannot be empty.")

            if item not in cleaned:
                cleaned.append(item)

        return cleaned

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("Server name cannot be empty.")

        return value
