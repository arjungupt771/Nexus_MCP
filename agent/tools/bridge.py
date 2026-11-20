"""Bridge between Nexus MCP tools and model tool-calling.

* Converts catalog ``ToolDefinition`` objects into provider-safe ``ToolSpec``s.
* Maps model tool calls back to ``server_id`` + ``tool_name`` (identity kept).
* Blocks unregistered and unauthorized tools *before* anything is executed.
* Validates arguments, then delegates to the existing ``ToolExecutor``.
* Never raises for tool failures: errors come back as results the model can read.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from fnmatch import fnmatchcase
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agent.llm.models import ToolCall, ToolSpec
from core.interfaces import ToolCatalog, ToolExecutor
from core.models import ToolCallRequest, ToolDefinition
from core.services.argument_validator import ArgumentValidator

_INVALID_CHARS = re.compile(r"[^A-Za-z0-9_-]")
MAX_NAME_LENGTH = 64


class ToolPermissions(BaseModel):
    """Which tools an agent may see and call.

    Patterns match qualified names (``server.tool``) with shell-style
    wildcards, e.g. ``notes.*`` or ``github.search_repositories``.
    Deny rules always win. The default allows everything registered.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    allow: tuple[str, ...] = ("*",)
    deny: tuple[str, ...] = ()

    def allows(self, qualified_name: str) -> bool:
        if any(fnmatchcase(qualified_name, pattern) for pattern in self.deny):
            return False

        return any(fnmatchcase(qualified_name, pattern) for pattern in self.allow)


@dataclass(frozen=True, slots=True)
class ToolBinding:
    """Identity of a tool as exposed to the model."""

    spec_name: str
    server_id: str
    tool_name: str
    server_name: str

    @property
    def qualified_name(self) -> str:
        return f"{self.server_id}.{self.tool_name}"


class ToolOutcome(BaseModel):
    """Result of one tool call, ready to hand back to the model."""

    model_config = ConfigDict(extra="forbid")

    call_id: str
    spec_name: str
    server_id: str | None = None
    tool_name: str | None = None
    success: bool
    content: str
    error_code: str | None = None
    retryable: bool = False
    duration_ms: float = 0.0
    request_id: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    # Structured tool result (pre-truncation); lets orchestration reference it.
    data: Any = None


def _safe_name(raw: str) -> str:
    name = _INVALID_CHARS.sub("_", raw)

    if len(name) > MAX_NAME_LENGTH:
        digest = hashlib.sha1(raw.encode()).hexdigest()[:8]
        name = f"{name[: MAX_NAME_LENGTH - 9]}_{digest}"

    return name


def build_bindings(tools: list[ToolDefinition]) -> dict[str, ToolBinding]:
    """Deterministically map tools to unique, provider-safe names."""
    bindings: dict[str, ToolBinding] = {}

    for tool in sorted(tools, key=lambda t: t.qualified_name):
        name = _safe_name(f"{tool.server_id}__{tool.name}")

        if name in bindings:  # sanitization collision: disambiguate with a hash
            digest = hashlib.sha1(tool.qualified_name.encode()).hexdigest()[:8]
            name = f"{name[: MAX_NAME_LENGTH - 9]}_{digest}"

        bindings[name] = ToolBinding(name, tool.server_id, tool.name, tool.server_name)

    return bindings


def _failure_content(code: str, message: str, retryable: bool = False) -> str:
    return json.dumps(
        {"success": False, "error": {"code": code, "message": message, "retryable": retryable}}
    )


class MCPToolBridge:
    def __init__(
        self,
        catalog: ToolCatalog,
        executor: ToolExecutor,
        permissions: ToolPermissions | None = None,
        *,
        validator: ArgumentValidator | None = None,
        timeout_seconds: float = 30.0,
        max_result_chars: int = 8000,
    ) -> None:
        if timeout_seconds <= 0 or max_result_chars < 100:
            raise ValueError("timeout_seconds must be > 0 and max_result_chars >= 100")

        self._catalog = catalog
        self._executor = executor
        self._permissions = permissions or ToolPermissions()
        self._validator = validator or ArgumentValidator()
        self._timeout = timeout_seconds
        self._max_chars = max_result_chars

    async def _snapshot(self) -> tuple[dict[str, ToolBinding], dict[str, ToolDefinition]]:
        tools = await self._catalog.list_tools()
        return build_bindings(tools), {t.qualified_name: t for t in tools}

    async def specs(self) -> list[ToolSpec]:
        """Tool descriptions for the model: current catalog, permitted tools only."""
        bindings, tools = await self._snapshot()
        specs = []

        for name, binding in bindings.items():
            if not self._permissions.allows(binding.qualified_name):
                continue

            tool = tools[binding.qualified_name]
            description = f"[{tool.server_name}] {tool.description}".strip()
            specs.append(ToolSpec(name=name, description=description, parameters=tool.input_schema))

        return specs

    @property
    def permissions(self) -> ToolPermissions:
        return self._permissions

    async def execute(
        self,
        call: ToolCall,
        *,
        timeout_seconds: float | None = None,
    ) -> ToolOutcome:
        bindings, tools = await self._snapshot()
        binding = bindings.get(call.name) or next(
            (b for b in bindings.values() if b.qualified_name == call.name), None
        )
        base = {"call_id": call.id, "spec_name": call.name, "arguments": call.arguments}

        if binding is None:
            return ToolOutcome(
                **base,
                success=False,
                error_code="TOOL_NOT_FOUND",
                content=_failure_content(
                    "TOOL_NOT_FOUND", f"Tool '{call.name}' is not registered."
                ),
            )

        identity = {"server_id": binding.server_id, "tool_name": binding.tool_name}

        if not self._permissions.allows(binding.qualified_name):
            return ToolOutcome(
                **base,
                **identity,
                success=False,
                error_code="TOOL_AUTHORIZATION_DENIED",
                content=_failure_content(
                    "TOOL_AUTHORIZATION_DENIED", f"This agent may not use '{call.name}'."
                ),
            )

        try:
            self._validator.validate(tools[binding.qualified_name], call.arguments)
        except ValueError as exc:
            return ToolOutcome(
                **base,
                **identity,
                success=False,
                error_code="INVALID_ARGUMENTS",
                content=_failure_content("INVALID_ARGUMENTS", str(exc)[:500]),
            )

        request = ToolCallRequest(
            server_id=binding.server_id,
            tool_name=binding.tool_name,
            arguments=call.arguments,
            timeout_seconds=timeout_seconds or self._timeout,
        )
        result = await self._executor.execute(request)
        common = {
            **base,
            **identity,
            "duration_ms": result.duration_ms,
            "request_id": str(result.request_id),
        }

        if result.success:
            return ToolOutcome(
                **common, success=True, content=self._encode(result.data), data=result.data
            )

        error = result.error
        assert error is not None  # guaranteed by ToolExecutionResult validation
        return ToolOutcome(
            **common,
            success=False,
            error_code=error.code,
            retryable=error.retryable,
            content=_failure_content(error.code, error.message, error.retryable),
        )

    def _encode(self, data: Any) -> str:
        text = json.dumps({"success": True, "data": data}, default=str)

        if len(text) <= self._max_chars:
            return text

        preview = text[: self._max_chars - 120]
        return json.dumps({"success": True, "truncated": True, "data_preview": preview})
