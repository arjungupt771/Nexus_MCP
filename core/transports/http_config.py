"""Configuration for remote MCP Streamable HTTP servers."""

from __future__ import annotations

from dataclasses import dataclass, field
from urllib.parse import urlsplit


@dataclass(frozen=True, slots=True)
class HTTPServerConfig:
    """Configuration for an MCP server using Streamable HTTP."""

    url: str
    headers: dict[str, str] = field(default_factory=dict)

    request_timeout_seconds: float = 30.0
    connect_timeout_seconds: float = 10.0
    read_timeout_seconds: float = 30.0

    verify_tls: bool = True
    terminate_on_close: bool = True
    max_sse_event_size: int = 1_048_576

    def __post_init__(self) -> None:
        parsed = urlsplit(self.url)

        if parsed.scheme not in {"http", "https"}:
            raise ValueError("MCP endpoint must use HTTP or HTTPS")

        if not parsed.hostname:
            raise ValueError("MCP endpoint must include a hostname")

        if parsed.username or parsed.password:
            raise ValueError("Credentials must not be embedded in the endpoint URL")

        if parsed.fragment:
            raise ValueError("MCP endpoint cannot contain a URL fragment")

        if self.verify_tls and parsed.scheme != "https":
            raise ValueError("TLS verification requires an HTTPS endpoint")

        for key, value in self.headers.items():
            if not key or not key.strip():
                raise ValueError("HTTP header names cannot be empty")

            if any(char in key or char in value for char in ("\r", "\n")):
                raise ValueError("HTTP headers cannot contain CR/LF characters")

        if self.request_timeout_seconds <= 0:
            raise ValueError("Request timeout must be positive")

        if self.connect_timeout_seconds <= 0:
            raise ValueError("Connect timeout must be positive")

        if self.read_timeout_seconds <= 0:
            raise ValueError("Read timeout must be positive")

        if self.max_sse_event_size <= 0:
            raise ValueError("Maximum SSE event size must be positive")
