import pytest

from core.exceptions import (
    AuthorizationError,
    ConfigurationError,
    DiscoveryError,
    NexusError,
    ServerAlreadyRegisteredError,
    ServerConnectionError,
    ServerNotFoundError,
    ToolAuthorizationError,
    ToolExecutionError,
    ToolNotFoundError,
    ToolTimeoutError,
    TransportError,
)
from core.models.execution import ErrorCategory

# =========================================================
# Inheritance and identity
# =========================================================


@pytest.mark.parametrize(
    "exception_class",
    [
        ConfigurationError,
        ServerNotFoundError,
        ServerAlreadyRegisteredError,
        ServerConnectionError,
        TransportError,
        DiscoveryError,
        ToolNotFoundError,
        ToolExecutionError,
        ToolTimeoutError,
        AuthorizationError,
        ToolAuthorizationError,
    ],
)
def test_all_exceptions_inherit_from_nexus_error(exception_class):
    error = exception_class("Test failure")

    assert isinstance(error, NexusError)
    assert isinstance(error, Exception)


# =========================================================
# Stable error codes and categories
# =========================================================


@pytest.mark.parametrize(
    "exception_class, expected_code, expected_category",
    [
        (
            ConfigurationError,
            "CONFIGURATION_ERROR",
            ErrorCategory.VALIDATION,
        ),
        (
            ServerNotFoundError,
            "SERVER_NOT_FOUND",
            ErrorCategory.VALIDATION,
        ),
        (
            ServerAlreadyRegisteredError,
            "SERVER_ALREADY_REGISTERED",
            ErrorCategory.VALIDATION,
        ),
        (
            ServerConnectionError,
            "SERVER_CONNECTION_FAILED",
            ErrorCategory.CONNECTION,
        ),
        (
            TransportError,
            "TRANSPORT_ERROR",
            ErrorCategory.CONNECTION,
        ),
        (
            DiscoveryError,
            "TOOL_DISCOVERY_FAILED",
            ErrorCategory.DISCOVERY,
        ),
        (
            ToolNotFoundError,
            "TOOL_NOT_FOUND",
            ErrorCategory.VALIDATION,
        ),
        (
            ToolExecutionError,
            "TOOL_EXECUTION_FAILED",
            ErrorCategory.EXECUTION,
        ),
        (
            ToolTimeoutError,
            "TOOL_TIMEOUT",
            ErrorCategory.TIMEOUT,
        ),
        (
            ToolAuthorizationError,
            "TOOL_AUTHORIZATION_DENIED",
            ErrorCategory.AUTHORIZATION,
        ),
    ],
)
def test_error_codes_and_categories(
    exception_class,
    expected_code,
    expected_category,
):
    error = exception_class("Test failure")

    assert error.code == expected_code
    assert error.category == expected_category


# =========================================================
# Retryability
# =========================================================


@pytest.mark.parametrize(
    "exception_class, expected_retryable",
    [
        (ConfigurationError, False),
        (ServerNotFoundError, False),
        (ServerAlreadyRegisteredError, False),
        (ServerConnectionError, True),
        (TransportError, True),
        (DiscoveryError, True),
        (ToolNotFoundError, False),
        (ToolExecutionError, False),
        (ToolTimeoutError, True),
        (ToolAuthorizationError, False),
    ],
)
def test_default_retryability(exception_class, expected_retryable):
    error = exception_class("Test failure")

    assert error.retryable is expected_retryable


def test_retryability_can_be_overridden():
    error = ToolExecutionError(
        "Temporary execution failure",
        retryable=True,
    )

    assert error.retryable is True


# =========================================================
# Error construction and validation
# =========================================================


def test_empty_error_message_rejected():
    with pytest.raises(ValueError):
        NexusError("   ")


def test_error_string_contains_message():
    error = ToolExecutionError("Execution failed")

    assert str(error) == "Execution failed"


def test_custom_error_code():
    error = NexusError(
        "Custom failure",
        code="CUSTOM_ERROR",
    )

    assert error.code == "CUSTOM_ERROR"


# =========================================================
# Secret redaction
# =========================================================


def test_sensitive_details_are_redacted():
    error = NexusError(
        "Authentication failed",
        details={
            "server_id": "github",
            "access_token": "secret-access-token",
            "password": "secret-password",
            "nested": {
                "api_key": "secret-api-key",
                "status": "failed",
            },
        },
    )

    assert error.details == {
        "server_id": "github",
        "access_token": "[REDACTED]",
        "password": "[REDACTED]",
        "nested": {
            "api_key": "[REDACTED]",
            "status": "failed",
        },
    }


def test_sensitive_details_do_not_leak_into_execution_error():
    error = ToolExecutionError(
        "Execution failed",
        details={"authorization": "Bearer secret-token"},
    )

    execution_error = error.to_execution_error()

    assert execution_error.details["authorization"] == "[REDACTED]"
    assert "secret-token" not in str(execution_error.details)


# =========================================================
# Serialization and domain conversion
# =========================================================


def test_exception_to_dict():
    error = ToolTimeoutError(
        "Operation timed out",
        details={"server_id": "github"},
    )

    result = error.to_dict()

    assert result == {
        "code": "TOOL_TIMEOUT",
        "message": "Operation timed out",
        "category": "timeout",
        "retryable": True,
        "details": {"server_id": "github"},
    }


def test_exception_converts_to_execution_error():
    error = ToolTimeoutError(
        "Execution timed out",
        retryable=True,
    )

    result = error.to_execution_error()

    assert result.code == "TOOL_TIMEOUT"
    assert result.message == "Execution timed out"
    assert result.category == ErrorCategory.TIMEOUT
    assert result.retryable is True


def test_exception_without_details_converts_correctly():
    error = ServerNotFoundError("Server does not exist")

    result = error.to_execution_error()

    assert result.category == ErrorCategory.VALIDATION
    assert result.details is None
