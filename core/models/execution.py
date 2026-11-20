from __future__ import annotations

from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)


class ErrorCategory(StrEnum):
    VALIDATION = "validation"
    CONNECTION = "connection"
    DISCOVERY = "discovery"
    EXECUTION = "execution"
    TIMEOUT = "timeout"
    AUTHORIZATION = "authorization"
    PROTOCOL = "protocol"
    INTERNAL = "internal"


class ToolCallRequest(BaseModel):
    """Normalized request to execute an MCP tool."""

    model_config = ConfigDict(extra="forbid")

    request_id: UUID = Field(default_factory=uuid4)
    server_id: str = Field(min_length=1)
    tool_name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)
    timeout_seconds: float = Field(default=30.0, gt=0, le=600)

    @property
    def qualified_name(self) -> str:
        return f"{self.server_id}.{self.tool_name}"


class ExecutionError(BaseModel):
    """Safe, structured representation of an execution failure."""

    model_config = ConfigDict(extra="forbid")

    code: str = Field(min_length=1, max_length=100)
    message: str = Field(min_length=1, max_length=1000)
    category: ErrorCategory
    retryable: bool = False
    details: dict[str, Any] | None = None


class ToolExecutionResult(BaseModel):
    """Normalized result of an MCP tool invocation."""

    model_config = ConfigDict(extra="forbid")

    request_id: UUID
    server_id: str = Field(min_length=1)
    tool_name: str = Field(min_length=1)

    success: bool
    data: Any = None
    error: ExecutionError | None = None

    duration_ms: float = Field(default=0.0, ge=0)
    is_protocol_error: bool = False

    @model_validator(mode="after")
    def validate_result_state(self):
        if self.success and self.error is not None:
            raise ValueError("Successful execution cannot contain an error.")

        if not self.success and self.error is None:
            raise ValueError("Failed execution must contain an error.")

        if self.success and self.is_protocol_error:
            raise ValueError("Successful execution cannot be a protocol error.")

        return self
