from __future__ import annotations

from typing import Any, ClassVar

from core.models.execution import ErrorCategory, ExecutionError

SENSITIVE_KEYS = {
    "token",
    "access_token",
    "refresh_token",
    "password",
    "secret",
    "api_key",
    "apikey",
    "authorization",
    "credential",
    "credentials",
}


def _sanitize_details(value: Any) -> Any:
    """Remove values stored under sensitive keys from error details."""

    if isinstance(value, dict):
        sanitized = {}

        for key, item in value.items():
            if str(key).lower() in SENSITIVE_KEYS:
                sanitized[key] = "[REDACTED]"
            else:
                sanitized[key] = _sanitize_details(item)

        return sanitized

    if isinstance(value, list):
        return [_sanitize_details(item) for item in value]

    if isinstance(value, tuple):
        return tuple(_sanitize_details(item) for item in value)

    return value


class NexusError(Exception):
    """Base class for all application-level Nexus errors."""

    error_code: ClassVar[str] = "NEXUS_ERROR"
    category: ClassVar[ErrorCategory] = ErrorCategory.INTERNAL
    default_retryable: ClassVar[bool] = False

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        retryable: bool | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        safe_message = message.strip()

        if not safe_message:
            raise ValueError("Nexus error message cannot be empty.")

        self.code = code or self.error_code
        self.message = safe_message
        self.retryable = self.default_retryable if retryable is None else retryable
        self.details = _sanitize_details(details) if details is not None else None

        super().__init__(self.message)

    def to_execution_error(self) -> ExecutionError:
        """Convert this exception into the shared domain error model."""

        return ExecutionError(
            code=self.code,
            message=self.message,
            category=self.category,
            retryable=self.retryable,
            details=self.details,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a safe, serializable representation."""

        return {
            "code": self.code,
            "message": self.message,
            "category": self.category.value,
            "retryable": self.retryable,
            "details": self.details,
        }
