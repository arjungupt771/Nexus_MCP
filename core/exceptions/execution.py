from __future__ import annotations

from typing import ClassVar

from core.exceptions.base import NexusError
from core.models.execution import ErrorCategory


class ExecutionErrorBase(NexusError):
    error_code: ClassVar[str] = "EXECUTION_ERROR"
    category: ClassVar[ErrorCategory] = ErrorCategory.EXECUTION


class ToolExecutionError(ExecutionErrorBase):
    error_code = "TOOL_EXECUTION_FAILED"


class ToolTimeoutError(ExecutionErrorBase):
    error_code = "TOOL_TIMEOUT"
    category = ErrorCategory.TIMEOUT
    default_retryable = True
