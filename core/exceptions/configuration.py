from __future__ import annotations

from typing import ClassVar

from core.exceptions.base import NexusError
from core.models.execution import ErrorCategory


class ConfigurationError(NexusError):
    error_code: ClassVar[str] = "CONFIGURATION_ERROR"
    category: ClassVar[ErrorCategory] = ErrorCategory.VALIDATION
    default_retryable: ClassVar[bool] = False
