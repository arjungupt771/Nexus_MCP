
"""Unified MCP tool invocation engine."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from core.interfaces.tool import ToolCatalog
from core.models.execution import (
    ErrorCategory,
    ExecutionError,
    ToolCallRequest,
    ToolExecutionResult,
)
from core.services.argument_validator import ArgumentValidator
from core.services.execution_error_mapper import ExecutionErrorMapper
from core.services.server_lifecycle import ServerLifecycleManager


class MCPResponseProtocolError(Exception):
    """Raised when an MCP server returns a malformed response."""


class ToolExecutorService:
    """Resolve, validate, route and execute MCP tool requests."""

    def __init__(
        self,
        catalog: ToolCatalog,
        lifecycle: ServerLifecycleManager,
        validator: ArgumentValidator | None = None,
        error_mapper: ExecutionErrorMapper | None = None,
    ) -> None:
        self._catalog = catalog
        self._lifecycle = lifecycle
        self._validator = validator or ArgumentValidator()
        self._error_mapper = error_mapper or ExecutionErrorMapper()

    async def execute(
        self,
        request: ToolCallRequest,
    ) -> ToolExecutionResult:
        """Execute a qualified MCP tool request."""

        started = time.perf_counter()

        def result(
            *,
            success: bool,
            data: Any = None,
            error: ExecutionError | None = None,
            is_protocol_error: bool = False,
        ) -> ToolExecutionResult:
            return ToolExecutionResult(
                request_id=request.request_id,
                server_id=request.server_id,
                tool_name=request.tool_name,
                success=success,
                data=data,
                error=error,
                duration_ms=max(
                    0.0,
                    (time.perf_counter() - started) * 1000,
                ),
                is_protocol_error=is_protocol_error,
            )

        def failure(exc: Exception) -> ToolExecutionResult:
            mapped = self._error_mapper.map_exception(exc)

            return result(
                success=False,
                error=mapped,
                is_protocol_error=(
                    mapped.category == ErrorCategory.PROTOCOL
                ),
            )

        # 1. Resolve the tool from the Nexus catalog.
        try:
            tool = await self._catalog.get_tool(
                request.qualified_name
            )

        except Exception:
            return result(
                success=False,
                error=ExecutionError(
                    code="TOOL_CATALOG_ERROR",
                    message="The tool catalog could not be queried.",
                    category=ErrorCategory.INTERNAL,
                    retryable=True,
                ),
            )

        if tool is None:
            return result(
                success=False,
                error=ExecutionError(
                    code="TOOL_NOT_FOUND",
                    message="The requested tool was not found.",
                    category=ErrorCategory.DISCOVERY,
                    retryable=False,
                    details={
                        "server_id": request.server_id,
                        "tool_name": request.tool_name,
                    },
                ),
            )

        # 2. Validate arguments before accessing the server.
        try:
            self._validator.validate(
                tool,
                request.arguments,
            )

        except ValueError as exc:
            return result(
                success=False,
                error=self._error_mapper.validation_error(
                    str(exc)
                ),
            )

        except Exception as exc:
            return failure(exc)

        # 3. Acquire a transport lease and execute with a bounded timeout.
        try:
            async with self._lifecycle.use_transport(
                request.server_id
            ) as transport:
                async with asyncio.timeout(request.timeout_seconds):
                    sdk_result = await transport.call_tool(
                        request.tool_name,
                        request.arguments,
                        read_timeout_seconds=request.timeout_seconds,
                    )

                # 4. Validate the MCP response structure.
                if sdk_result is None:
                    raise MCPResponseProtocolError(
                        "MCP server returned an empty response."
                    )

                is_error = getattr(sdk_result, "is_error", None)

                if not isinstance(is_error, bool):
                    raise MCPResponseProtocolError(
                        "MCP server returned an invalid response status."
                    )

                # 5. Handle MCP tool-level errors separately from
                # protocol-level failures.
                if is_error:
                    message = self._extract_error_message(sdk_result)

                    return result(
                        success=False,
                        error=self._error_mapper.tool_error(message),
                    )

                # 6. Normalize the successful response.
                data = self._extract_result_data(sdk_result)

                return result(
                    success=True,
                    data=data,
                )

        except asyncio.CancelledError:
            raise

        except TimeoutError:
            return failure(TimeoutError())

        except Exception as exc:
            return failure(exc)

    @staticmethod
    def _extract_result_data(sdk_result: Any) -> Any:
        """Extract structured content or serialize MCP content blocks."""

        structured = getattr(
            sdk_result,
            "structured_content",
            None,
        )

        if structured is not None:
            return structured

        content = getattr(sdk_result, "content", None)

        if content is None:
            return None

        if not isinstance(content, (list, tuple)):
            raise MCPResponseProtocolError(
                "MCP server returned malformed content."
            )

        normalized = []

        for block in content:
            if hasattr(block, "model_dump"):
                normalized.append(
                    block.model_dump(mode="json")
                )

            elif isinstance(block, dict):
                normalized.append(block)

            else:
                raise MCPResponseProtocolError(
                    "MCP server returned an unsupported content block."
                )

        return normalized

    @staticmethod
    def _extract_error_message(sdk_result: Any) -> str:
        """Extract a bounded MCP tool-level error message."""

        messages: list[str] = []

        for block in getattr(sdk_result, "content", None) or []:
            text = getattr(block, "text", None)

            if isinstance(text, str) and text.strip():
                messages.append(text.strip())

        if messages:
            return "; ".join(messages)[:1000]

        return "The MCP tool reported an execution error."
