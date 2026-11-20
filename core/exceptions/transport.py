from __future__ import annotations

from typing import ClassVar

from core.exceptions.base import NexusError
from core.models.execution import ErrorCategory


class TransportError(NexusError):
    error_code: ClassVar[str] = "TRANSPORT_ERROR"
    category: ClassVar[ErrorCategory] = ErrorCategory.CONNECTION
    default_retryable: ClassVar[bool] = True
