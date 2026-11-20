"""Agent identity, instructions, model settings and permissions."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from agent.llm.models import ModelConfig
from agent.tools.bridge import ToolPermissions


class ExecutionSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Model <-> tool round trips before the agent must answer.
    max_iterations: int = Field(default=5, ge=1, le=25)
    max_tool_calls_per_turn: int = Field(default=8, ge=1, le=32)
    tool_timeout_seconds: float = Field(default=30.0, gt=0, le=600)
    max_context_messages: int = Field(default=40, ge=4, le=500)
    max_tool_result_chars: int = Field(default=8000, ge=100, le=100_000)


class AgentDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent_id: str = Field(min_length=1, max_length=64)
    name: str = Field(min_length=1, max_length=150)
    system_instructions: str = "You are a helpful assistant."
    model: ModelConfig
    tool_permissions: ToolPermissions = Field(default_factory=ToolPermissions)
    execution: ExecutionSettings = Field(default_factory=ExecutionSettings)

    @field_validator("agent_id")
    @classmethod
    def _slug(cls, value: str) -> str:
        value = value.strip().lower()

        if not value or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in value):
            raise ValueError("agent_id may contain only lowercase letters, digits, '-' and '_'.")

        return value
