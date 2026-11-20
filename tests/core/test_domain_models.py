from uuid import uuid4

import pytest
from pydantic import ValidationError

from core.models import (
    ErrorCategory,
    ExecutionError,
    ServerDescriptor,
    ServerStatus,
    ToolCallRequest,
    ToolDefinition,
    ToolExecutionResult,
    TransportType,
)

# =========================================================
# Server models
# =========================================================


def test_server_descriptor_creation():
    server = ServerDescriptor(
        server_id="notes",
        name="Notes MCP Server",
        transport=TransportType.STDIO,
    )

    assert server.server_id == "notes"
    assert server.status == ServerStatus.DISCONNECTED


def test_server_id_normalization():
    server = ServerDescriptor(
        server_id=" GitHub ",
        name="GitHub Server",
        transport=TransportType.STDIO,
    )

    assert server.server_id == "github"


@pytest.mark.parametrize(
    "server_id",
    ["", "invalid id", "server@123", "../server"],
)
def test_invalid_server_id(server_id):
    with pytest.raises(ValidationError):
        ServerDescriptor(
            server_id=server_id,
            name="Invalid Server",
            transport=TransportType.STDIO,
        )


def test_empty_server_name_rejected():
    with pytest.raises(ValidationError):
        ServerDescriptor(
            server_id="notes",
            name="   ",
            transport=TransportType.STDIO,
        )


# =========================================================
# Tool models
# =========================================================


def test_tool_definition_creation():
    tool = ToolDefinition(
        server_id="github",
        server_name="GitHub MCP",
        name="search_repositories",
        description="Search repositories.",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    )

    assert tool.qualified_name == "github.search_repositories"


def test_tool_definition_without_required_arguments():
    tool = ToolDefinition(
        server_id="notes",
        server_name="Notes MCP",
        name="list_notes",
        input_schema={
            "type": "object",
            "properties": {},
        },
    )

    assert tool.qualified_name == "notes.list_notes"


def test_invalid_tool_schema_type_rejected():
    with pytest.raises(ValidationError):
        ToolDefinition(
            server_id="notes",
            server_name="Notes MCP",
            name="invalid_tool",
            input_schema={"type": "string"},
        )


def test_invalid_tool_schema_properties_rejected():
    with pytest.raises(ValidationError):
        ToolDefinition(
            server_id="notes",
            server_name="Notes MCP",
            name="invalid_tool",
            input_schema={
                "type": "object",
                "properties": [],
            },
        )


def test_missing_required_tool_property_rejected():
    with pytest.raises(ValidationError):
        ToolDefinition(
            server_id="notes",
            server_name="Notes MCP",
            name="invalid_tool",
            input_schema={
                "type": "object",
                "properties": {},
                "required": ["note_id"],
            },
        )


# =========================================================
# Tool request models
# =========================================================


def test_tool_call_request_creation():
    request = ToolCallRequest(
        server_id="github",
        tool_name="search_repositories",
        arguments={"query": "nexus"},
    )

    assert request.qualified_name == "github.search_repositories"
    assert request.timeout_seconds == 30.0
    assert isinstance(request.request_id, type(uuid4()))


def test_tool_request_default_arguments():
    request = ToolCallRequest(
        server_id="notes",
        tool_name="list_notes",
    )

    assert request.arguments == {}


@pytest.mark.parametrize("timeout", [0, -1, 601])
def test_invalid_tool_timeout_rejected(timeout):
    with pytest.raises(ValidationError):
        ToolCallRequest(
            server_id="notes",
            tool_name="list_notes",
            timeout_seconds=timeout,
        )


def test_extra_tool_request_fields_rejected():
    with pytest.raises(ValidationError):
        ToolCallRequest(
            server_id="notes",
            tool_name="list_notes",
            unexpected_field=True,
        )


# =========================================================
# Execution error models
# =========================================================


def test_execution_error_creation():
    error = ExecutionError(
        code="SERVER_TIMEOUT",
        message="The MCP server did not respond.",
        category=ErrorCategory.TIMEOUT,
        retryable=True,
    )

    assert error.category == ErrorCategory.TIMEOUT
    assert error.retryable is True


def test_execution_error_defaults():
    error = ExecutionError(
        code="INVALID_ARGUMENT",
        message="Invalid argument.",
        category=ErrorCategory.VALIDATION,
    )

    assert error.retryable is False
    assert error.details is None


# =========================================================
# Execution result models
# =========================================================


def test_successful_execution_result():
    request_id = uuid4()

    result = ToolExecutionResult(
        request_id=request_id,
        server_id="notes",
        tool_name="list_notes",
        success=True,
        data={"items": []},
        duration_ms=12.5,
    )

    assert result.success is True
    assert result.error is None
    assert result.data == {"items": []}


def test_failed_execution_result():
    result = ToolExecutionResult(
        request_id=uuid4(),
        server_id="github",
        tool_name="search_repositories",
        success=False,
        error=ExecutionError(
            code="GITHUB_TIMEOUT",
            message="GitHub request timed out.",
            category=ErrorCategory.TIMEOUT,
            retryable=True,
        ),
    )

    assert result.success is False
    assert result.error.category == ErrorCategory.TIMEOUT


def test_success_cannot_contain_error():
    with pytest.raises(ValidationError):
        ToolExecutionResult(
            request_id=uuid4(),
            server_id="notes",
            tool_name="list_notes",
            success=True,
            error=ExecutionError(
                code="ERROR",
                message="Invalid state.",
                category=ErrorCategory.INTERNAL,
            ),
        )


def test_failure_must_contain_error():
    with pytest.raises(ValidationError):
        ToolExecutionResult(
            request_id=uuid4(),
            server_id="notes",
            tool_name="list_notes",
            success=False,
        )


def test_success_cannot_be_protocol_error():
    with pytest.raises(ValidationError):
        ToolExecutionResult(
            request_id=uuid4(),
            server_id="notes",
            tool_name="list_notes",
            success=True,
            is_protocol_error=True,
        )


def test_negative_duration_rejected():
    with pytest.raises(ValidationError):
        ToolExecutionResult(
            request_id=uuid4(),
            server_id="notes",
            tool_name="list_notes",
            success=True,
            duration_ms=-1,
        )
