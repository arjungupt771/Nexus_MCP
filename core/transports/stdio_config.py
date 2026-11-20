"""Configuration for an MCP server using stdio transport."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class StdioServerConfig:
    """Configuration required to launch an MCP server over stdio."""

    command: str
    args: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | Path | None = None
    encoding: str = "utf-8"
    encoding_error_handler: str = "strict"
    read_timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        if not self.command or not self.command.strip():
            raise ValueError("Stdio command cannot be empty")

        if any("\x00" in arg for arg in self.args):
            raise ValueError("Stdio arguments cannot contain NUL bytes")

        if "\x00" in self.command:
            raise ValueError("Stdio command cannot contain NUL bytes")

        for key, value in self.env.items():
            if not key or "=" in key or "\x00" in key:
                raise ValueError(f"Invalid environment variable name: {key!r}")

            if "\x00" in value:
                raise ValueError(f"Environment variable {key!r} contains a NUL byte")

        if self.read_timeout_seconds <= 0:
            raise ValueError("Read timeout must be greater than zero")

        if self.encoding_error_handler not in {"strict", "ignore", "replace"}:
            raise ValueError("Unsupported encoding error handler")
