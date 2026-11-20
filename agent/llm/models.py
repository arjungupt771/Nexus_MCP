"""Provider-independent LLM request/response models."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ToolSpec(BaseModel):
    """A tool described to the model (JSON Schema parameters)."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=lambda: {"type": "object", "properties": {}})


class ToolCall(BaseModel):
    """A tool invocation requested by the model."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    # For Role.TOOL messages: which call this answers and which tool produced it.
    tool_call_id: str | None = None
    name: str | None = None


class FinishReason(StrEnum):
    STOP = "stop"
    TOOL_CALLS = "tool_calls"
    LENGTH = "length"
    CONTENT_FILTER = "content_filter"
    OTHER = "other"


class TokenUsage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )

    @classmethod
    def of(cls, prompt: int | None, completion: int | None, total: int | None = None) -> TokenUsage:
        prompt, completion = prompt or 0, completion or 0
        return cls(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=total if total else prompt + completion,
        )


class LLMResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: TokenUsage = Field(default_factory=TokenUsage)
    finish_reason: FinishReason = FinishReason.STOP
    model: str = ""
    provider: str = ""
    latency_ms: float = 0.0

    def to_message(self) -> Message:
        return Message(role=Role.ASSISTANT, content=self.content, tool_calls=self.tool_calls)


class StreamChunk(BaseModel):
    """Incremental output. Tool calls and usage arrive once, on the final chunk."""

    model_config = ConfigDict(extra="forbid")

    delta: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: TokenUsage | None = None
    finish_reason: FinishReason | None = None


class ModelInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str
    model: str
    supports_tools: bool = True
    supports_streaming: bool = True
    supports_structured_output: bool = True
    context_window: int | None = None


class RetryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_retries: int = Field(default=3, ge=0, le=10)
    initial_delay_seconds: float = Field(default=0.5, ge=0)
    max_delay_seconds: float = Field(default=8.0, gt=0)
    multiplier: float = Field(default=2.0, ge=1)
    jitter: bool = True


class StructuredOutputSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Extra model calls allowed to repair malformed JSON / schema violations.
    max_repair_attempts: int = Field(default=1, ge=0, le=3)


class ModelConfig(BaseModel):
    """Validated model selection and generation settings.

    API keys are *not* part of this model; adapters receive them from the
    environment so configs can be logged and persisted safely.
    """

    model_config = ConfigDict(extra="forbid")

    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=1024, ge=1, le=200_000)
    request_timeout_seconds: float = Field(default=60.0, gt=0, le=600)
    retry: RetryConfig = Field(default_factory=RetryConfig)
    structured_output: StructuredOutputSettings = Field(default_factory=StructuredOutputSettings)
    base_url: str | None = None

    @field_validator("provider")
    @classmethod
    def _normalize_provider(cls, value: str) -> str:
        return value.strip().lower()

    @field_validator("base_url")
    @classmethod
    def _validate_base_url(cls, value: str | None) -> str | None:
        if value is None:
            return None

        value = value.strip().rstrip("/")

        if (
            not value.startswith(("http://", "https://"))
            or "@" in value.split("//", 1)[1].split("/")[0]
        ):
            raise ValueError("base_url must be an http(s) URL without embedded credentials.")

        return value


class ProviderCredentials(BaseModel):
    """Holds a secret without ever printing it."""

    model_config = ConfigDict(extra="forbid")

    api_key: SecretStr | None = None
