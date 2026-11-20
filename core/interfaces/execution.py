
from __future__ import annotations

from typing import Protocol, runtime_checkable

from core.models import ToolCallRequest, ToolExecutionResult


@runtime_checkable
class ToolExecutor(Protocol):
    """Contract for executing normalized MCP tool requests."""

    async def execute(
        self,
        request: ToolCallRequest,
    ) -> ToolExecutionResult:
        """Execute a tool request and return its result."""
        ...
