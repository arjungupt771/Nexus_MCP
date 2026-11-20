from __future__ import annotations

from typing import ClassVar

from core.exceptions.base import NexusError
from core.models.execution import ErrorCategory


class AuthorizationError(NexusError):
    error_code: ClassVar[str] = "AUTHORIZATION_ERROR"
    category: ClassVar[ErrorCategory] = ErrorCategory.AUTHORIZATION


class ToolAuthorizationError(AuthorizationError):
    error_code = "TOOL_AUTHORIZATION_DENIED"
    default_retryable = False
