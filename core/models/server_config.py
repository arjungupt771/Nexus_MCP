"""Persistable MCP server configuration.

Design rule: persisted configuration never contains secrets. Values for
sensitive environment variables / headers must be *references* of the form
``${ENV_VAR}`` that are resolved from the process environment only when a
connection is opened.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from core.exceptions import ConfigurationError
from core.models.server import (
    ServerDescriptor,
    TransportType,
    normalize_server_id,
)
from core.transports.http_config import HTTPServerConfig
from core.transports.stdio_config import StdioServerConfig

_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_SENSITIVE_KEY = re.compile(
    r"(token|secret|passw(or)?d|passwd|api[-_]?key|apikey|auth|credential|"
    r"bearer|cookie|private[-_]?key|signature)",
    re.IGNORECASE,
)
_SECRET_VALUE = re.compile(
    r"(ghp_|gho_|ghs_|github_pat_|sk-[A-Za-z0-9]|xox[abpr]-|AIza|ya29\.|"
    r"eyJ[A-Za-z0-9_-]{10,}\.|-----BEGIN [A-Z ]*PRIVATE KEY)"
)
_AUTH_PREFIX = re.compile(r"^(bearer|basic|token)(\s+|$)", re.IGNORECASE)
_SENSITIVE_QUERY = re.compile(r"[?&](token|key|api_?key|secret|password|auth)[^=&]*=", re.I)
_PLACEHOLDER = "placeholder"


def _placeholder(value: str) -> str:
    return _REF.sub(_PLACEHOLDER, value)


def has_reference(value: str) -> bool:
    return _REF.search(value) is not None


def _is_safe_value(key: str, value: str) -> bool:
    """Return True if a value may be persisted for the given key."""
    without_refs = _REF.sub("", value)

    if _SECRET_VALUE.search(without_refs):
        return False

    if _SENSITIVE_KEY.search(key):
        # Sensitive keys must be fully reference-driven.
        remainder = _AUTH_PREFIX.sub("", without_refs.strip()).strip()
        return has_reference(value) and remainder == ""

    return not _AUTH_PREFIX.match(without_refs.strip()) or has_reference(value)


def _check_mapping(label: str, mapping: Mapping[str, str]) -> None:
    unsafe = sorted(k for k, v in mapping.items() if not _is_safe_value(k, v))

    if unsafe:
        # Names only: values are never echoed.
        raise ValueError(
            f"{label} {unsafe} must not contain literal secrets; "
            "use an environment reference such as ${ENV_VAR}."
        )


def resolve_references(value: str, environ: Mapping[str, str]) -> str:
    """Substitute ``${VAR}`` references from ``environ``."""

    def _sub(match: re.Match[str]) -> str:
        name = match.group(1)

        if name not in environ:
            raise ConfigurationError(
                f"Environment variable '{name}' referenced by server configuration is not set.",
                details={"variable": name},
            )

        return environ[name]

    return _REF.sub(_sub, value)


class StdioConnectionConfig(BaseModel):
    """Persistable configuration for a stdio MCP server."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    transport: Literal[TransportType.STDIO] = TransportType.STDIO
    command: str
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    cwd: str | None = None
    read_timeout_seconds: float = 30.0

    @model_validator(mode="after")
    def _validate(self) -> StdioConnectionConfig:
        _check_mapping("Environment variables", self.env)
        self._build(_placeholder)  # reuse runtime validation without secrets
        return self

    def to_runtime(self, environ: Mapping[str, str]) -> StdioServerConfig:
        """Resolve references from ``environ`` into a runtime config."""
        return self._build(lambda v: resolve_references(v, environ))

    def _build(self, convert: Callable[[str], str]) -> StdioServerConfig:
        env = {k: convert(v) for k, v in self.env.items()}
        return StdioServerConfig(
            command=self.command,
            args=tuple(self.args),
            env=env,
            cwd=self.cwd,
            read_timeout_seconds=self.read_timeout_seconds,
        )

    def public_view(self) -> dict[str, Any]:
        return {
            "transport": self.transport.value,
            "command": self.command,
            "args": list(self.args),
            "env": {k: _mask(v) for k, v in self.env.items()},
            "cwd": self.cwd,
            "read_timeout_seconds": self.read_timeout_seconds,
        }


class HttpConnectionConfig(BaseModel):
    """Persistable configuration for a Streamable HTTP MCP server."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    transport: Literal[TransportType.STREAMABLE_HTTP] = TransportType.STREAMABLE_HTTP
    url: str
    headers: dict[str, str] = Field(default_factory=dict)
    request_timeout_seconds: float = 30.0
    connect_timeout_seconds: float = 10.0
    read_timeout_seconds: float = 30.0
    verify_tls: bool = True

    @model_validator(mode="after")
    def _validate(self) -> HttpConnectionConfig:
        _check_mapping("HTTP headers", self.headers)

        if _SENSITIVE_QUERY.search(self.url):
            raise ValueError(
                "MCP endpoint URL must not carry credentials in its query string; "
                "use a header with an environment reference instead."
            )

        self._build(_placeholder)
        return self

    def to_runtime(self, environ: Mapping[str, str]) -> HTTPServerConfig:
        """Resolve references from ``environ`` into a runtime config."""
        return self._build(lambda v: resolve_references(v, environ))

    def _build(self, convert: Callable[[str], str]) -> HTTPServerConfig:
        headers = {k: convert(v) for k, v in self.headers.items()}
        return HTTPServerConfig(
            url=self.url,
            headers=headers,
            request_timeout_seconds=self.request_timeout_seconds,
            connect_timeout_seconds=self.connect_timeout_seconds,
            read_timeout_seconds=self.read_timeout_seconds,
            verify_tls=self.verify_tls,
        )

    def public_view(self) -> dict[str, Any]:
        return {
            "transport": self.transport.value,
            "url": self.url,
            "headers": {k: _mask(v) for k, v in self.headers.items()},
            "request_timeout_seconds": self.request_timeout_seconds,
            "connect_timeout_seconds": self.connect_timeout_seconds,
            "read_timeout_seconds": self.read_timeout_seconds,
            "verify_tls": self.verify_tls,
        }


def _mask(value: str) -> str:
    """References are safe to show; literals are masked."""
    return value if has_reference(value) else "***"


ConnectionConfig = Annotated[
    StdioConnectionConfig | HttpConnectionConfig,
    Field(discriminator="transport"),
]


class ServerRegistration(BaseModel):
    """Everything needed to register (and later restore) an MCP server."""

    model_config = ConfigDict(extra="forbid")

    server_id: str = Field(min_length=1, max_length=100)
    name: str = Field(min_length=1, max_length=150)
    description: str = Field(default="", max_length=1000)
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)
    capabilities: list[str] = Field(default_factory=list)
    enabled: bool = True
    config: ConnectionConfig

    @field_validator("server_id")
    @classmethod
    def _server_id(cls, value: str) -> str:
        return normalize_server_id(value)

    @field_validator("tags", "capabilities")
    @classmethod
    def _labels(cls, value: list[str]) -> list[str]:
        return ServerDescriptor.validate_labels(value)

    @field_validator("name")
    @classmethod
    def _name(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("Server name cannot be empty.")

        return value

    @model_validator(mode="after")
    def _validate_descriptor(self) -> ServerRegistration:
        # Reuses ServerDescriptor validation for tags / capabilities.
        self.to_descriptor()
        _check_mapping("Metadata", self.metadata)
        return self

    @property
    def transport(self) -> TransportType:
        return self.config.transport

    def to_descriptor(self) -> ServerDescriptor:
        return ServerDescriptor(
            server_id=self.server_id,
            name=self.name,
            transport=self.config.transport,
            description=self.description,
            tags=list(self.tags),
            metadata=dict(self.metadata),
            capabilities=list(self.capabilities),
            enabled=self.enabled,
        )

    def public_view(self) -> dict[str, Any]:
        return {
            "server_id": self.server_id,
            "name": self.name,
            "description": self.description,
            "tags": list(self.tags),
            "metadata": dict(self.metadata),
            "capabilities": list(self.capabilities),
            "enabled": self.enabled,
            "config": self.config.public_view(),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> ServerRegistration:
        """Validate untrusted input, raising a sanitized ConfigurationError."""
        try:
            return cls.model_validate(dict(payload))
        except ValidationError as exc:
            raise sanitize_validation_error(exc) from None


class ServerUpdate(BaseModel):
    """Partial update of a registered server. Unset fields are unchanged."""

    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    description: str | None = None
    tags: list[str] | None = None
    metadata: dict[str, str] | None = None
    capabilities: list[str] | None = None
    config: ConnectionConfig | None = None


class ServerRecord(BaseModel):
    """A persisted registration with bookkeeping timestamps."""

    model_config = ConfigDict(extra="forbid")

    registration: ServerRegistration
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


def sanitize_validation_error(exc: ValidationError) -> ConfigurationError:
    """Convert a pydantic error without echoing any input values."""
    issues = [
        {"field": ".".join(str(p) for p in e["loc"]), "message": e["msg"]}
        for e in exc.errors(include_input=False, include_url=False, include_context=False)
    ]
    return ConfigurationError(
        "Invalid server configuration.",
        details={"issues": issues},
    )
