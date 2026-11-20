
"""Map execution failures into safe Nexus error categories."""

from __future__ import annotations

import re

from core.models.execution import ErrorCategory, ExecutionError


class ExecutionErrorMapper:
    """Convert exceptions into sanitized execution errors."""

    @staticmethod
    def map_exception(exc: Exception) -> ExecutionError:
        name = type(exc).__name__.lower()

        if isinstance(exc, TimeoutError):
            return ExecutionError(
                code="TOOL_TIMEOUT",
                message="The tool execution exceeded its timeout.",
                category=ErrorCategory.TIMEOUT,
                retryable=True,
                details={"cause_type": type(exc).__name__},
            )

        if isinstance(exc, PermissionError) or any(
            token in name
            for token in ("authorization", "authentication", "forbidden")
        ):
            return ExecutionError(
                code="TOOL_AUTHORIZATION_FAILED",
                message="The tool request was not authorized.",
                category=ErrorCategory.AUTHORIZATION,
                retryable=False,
                details={"cause_type": type(exc).__name__},
            )

        if "discovery" in name:
            return ExecutionError(
                code="MCP_DISCOVERY_FAILED",
                message="MCP tool discovery failed.",
                category=ErrorCategory.DISCOVERY,
                retryable=False,
                details={"cause_type": type(exc).__name__},
            )

        if "validation" in name:
            return ExecutionError(
                code="TOOL_VALIDATION_FAILED",
                message="Tool validation failed.",
                category=ErrorCategory.VALIDATION,
                retryable=False,
                details={"cause_type": type(exc).__name__},
            )

        if any(
            token in name
            for token in ("protocol", "jsonrpc", "parseerror", "mcpresponse")
        ):
            return ExecutionError(
                code="MCP_PROTOCOL_ERROR",
                message="The MCP server returned a protocol-level failure.",
                category=ErrorCategory.PROTOCOL,
                retryable=False,
                details={"cause_type": type(exc).__name__},
            )

        if isinstance(exc, (ConnectionError, OSError)) or any(
            token in name
            for token in ("connection", "transport", "network")
        ):
            return ExecutionError(
                code="MCP_CONNECTION_FAILED",
                message="The MCP server connection failed.",
                category=ErrorCategory.CONNECTION,
                retryable=True,
                details={"cause_type": type(exc).__name__},
            )

        return ExecutionError(
            code="TOOL_EXECUTION_FAILED",
            message="The tool could not be executed.",
            category=ErrorCategory.EXECUTION,
            retryable=False,
            details={"cause_type": type(exc).__name__},
        )

    @staticmethod
    def _sanitize_message(message: str) -> str:
        """Remove common credential patterns from external error text."""

        message = re.sub(
            r"(?i)\bBearer\s+\S+",
            "Bearer [REDACTED]",
            message,
        )

        message = re.sub(
            r"(?i)(api[_-]?key|access[_-]?token|password|secret)"
            r"(\s*[:=]\s*)[^\s,;]+",
            r"\1\2[REDACTED]",
            message,
        )

        return message[:1000]

    @classmethod
    def validation_error(cls, message: str) -> ExecutionError:
        return ExecutionError(
            code="TOOL_ARGUMENT_VALIDATION_FAILED",
            message=cls._sanitize_message(message),
            category=ErrorCategory.VALIDATION,
            retryable=False,
        )

    @classmethod
    def tool_error(cls, message: str) -> ExecutionError:
        return ExecutionError(
            code="MCP_TOOL_ERROR",
            message=(
                cls._sanitize_message(message)
                or "The MCP tool reported an error."
            ),
            category=ErrorCategory.EXECUTION,
            retryable=False,
        )
