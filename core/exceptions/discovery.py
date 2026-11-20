from __future__ import annotations

from typing import ClassVar

from core.exceptions.base import NexusError
from core.models.execution import ErrorCategory


class DiscoveryError(NexusError):
    error_code: ClassVar[str] = "TOOL_DISCOVERY_FAILED"
    category: ClassVar[ErrorCategory] = ErrorCategory.DISCOVERY
    default_retryable: ClassVar[bool] = True


class ToolNotFoundError(DiscoveryError):
    error_code = "TOOL_NOT_FOUND"
    category = ErrorCategory.VALIDATION
    default_retryable = False
