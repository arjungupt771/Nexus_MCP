"""Provider-independent LLM errors.

Every provider adapter converts transport/API failures into these types so
the agent core never depends on a particular SDK or HTTP library.
"""

from __future__ import annotations

from typing import Any, ClassVar


class LLMError(Exception):
    """Base class for normalized LLM provider errors."""

    code: ClassVar[str] = "LLM_ERROR"
    default_retryable: ClassVar[bool] = False

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        retryable: bool | None = None,
        status_code: int | None = None,
        retry_after: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.provider = provider
        self.retryable = self.default_retryable if retryable is None else retryable
        self.status_code = status_code
        self.retry_after = retry_after
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "provider": self.provider,
            "retryable": self.retryable,
            "status_code": self.status_code,
        }


class LLMConfigurationError(LLMError):
    code = "LLM_CONFIGURATION_ERROR"


class LLMAuthenticationError(LLMError):
    code = "LLM_AUTHENTICATION_FAILED"


class LLMInvalidRequestError(LLMError):
    code = "LLM_INVALID_REQUEST"


class LLMRateLimitError(LLMError):
    code = "LLM_RATE_LIMITED"
    default_retryable = True


class LLMTimeoutError(LLMError):
    code = "LLM_TIMEOUT"
    default_retryable = True


class LLMConnectionError(LLMError):
    code = "LLM_CONNECTION_FAILED"
    default_retryable = True


class LLMProviderUnavailableError(LLMError):
    code = "LLM_PROVIDER_UNAVAILABLE"
    default_retryable = True


class LLMResponseError(LLMError):
    """The provider answered, but the payload was unusable."""

    code = "LLM_INVALID_RESPONSE"


class LLMStructuredOutputError(LLMError):
    code = "LLM_STRUCTURED_OUTPUT_INVALID"
