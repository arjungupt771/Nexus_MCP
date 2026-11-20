from __future__ import annotations

import json

import pytest
from phase5_support import http_payload, stdio_payload

from core.exceptions import ConfigurationError
from core.models.server_config import ServerRegistration, resolve_references


def test_stdio_registration_normalizes_identity_and_labels():
    reg = ServerRegistration.from_payload(
        stdio_payload("GitHub", tags=["Code", "code", " Search "], capabilities=["Repo-Search"])
    )

    assert reg.server_id == "github"
    assert reg.tags == ["code", "search"]
    assert reg.capabilities == ["repo-search"]
    assert reg.to_descriptor().transport.value == "stdio"


def test_http_registration_accepts_reference_headers():
    reg = ServerRegistration.from_payload(http_payload())
    assert reg.config.transport.value == "streamable_http"
    assert reg.config.headers["Authorization"] == "Bearer ${REMOTE_TOKEN}"


@pytest.mark.parametrize(
    "config",
    [
        {"transport": "stdio", "command": ""},
        {"transport": "stdio", "command": "python", "read_timeout_seconds": 0},
        {"transport": "streamable_http", "url": "ftp://example.com/mcp"},
        {"transport": "streamable_http", "url": "http://example.com/mcp"},  # TLS verify on
        {"transport": "streamable_http", "url": "https://user:pw@example.com/mcp"},
        {"transport": "carrier_pigeon", "command": "x"},
    ],
)
def test_invalid_configurations_are_rejected(config):
    with pytest.raises(ConfigurationError) as exc_info:
        ServerRegistration.from_payload(stdio_payload(config=config))

    assert exc_info.value.details["issues"]


@pytest.mark.parametrize("server_id", ["", "Has Space", "bad/id", "ünï"])
def test_invalid_server_ids_are_rejected(server_id):
    with pytest.raises(ConfigurationError):
        ServerRegistration.from_payload(stdio_payload(server_id))


def test_unknown_fields_are_rejected():
    with pytest.raises(ConfigurationError):
        ServerRegistration.from_payload(stdio_payload(surprise=True))


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("env", "GITHUB_TOKEN", "literal-secret-value"),
        ("env", "API_KEY", "hunter2"),
        ("env", "HARMLESS", "ghp_abcdefghijklmnopqrstuvwxyz0123456789"),
        ("env", "OPENAI", "sk-abcdefghijklmnop"),
    ],
)
def test_literal_secrets_in_env_are_rejected_without_echoing_value(section, key, value):
    payload = stdio_payload(
        config={"transport": "stdio", "command": "python", section: {key: value}}
    )

    with pytest.raises(ConfigurationError) as exc_info:
        ServerRegistration.from_payload(payload)

    rendered = json.dumps(exc_info.value.to_dict()) + str(exc_info.value)
    assert value not in rendered


@pytest.mark.parametrize(
    "headers",
    [
        {"Authorization": "Bearer abc123"},
        {"X-Api-Key": "abc123"},
        {"X-Custom": "Bearer abc123"},
    ],
)
def test_literal_secrets_in_headers_are_rejected(headers):
    payload = http_payload()
    payload["config"]["headers"] = headers

    with pytest.raises(ConfigurationError):
        ServerRegistration.from_payload(payload)


def test_credentials_in_url_query_are_rejected():
    payload = http_payload()
    payload["config"]["url"] = "https://example.com/mcp?api_key=abc"

    with pytest.raises(ConfigurationError):
        ServerRegistration.from_payload(payload)


def test_non_sensitive_literals_are_allowed():
    reg = ServerRegistration.from_payload(
        stdio_payload(
            config={
                "transport": "stdio",
                "command": "python",
                "env": {"LOG_LEVEL": "debug", "GITHUB_TOKEN": "${GITHUB_TOKEN}"},
            }
        )
    )
    assert reg.config.env["LOG_LEVEL"] == "debug"


def test_public_view_masks_literals_but_keeps_references():
    reg = ServerRegistration.from_payload(
        stdio_payload(
            config={
                "transport": "stdio",
                "command": "python",
                "env": {"LOG_LEVEL": "debug", "GITHUB_TOKEN": "${GITHUB_TOKEN}"},
            }
        )
    )
    env = reg.public_view()["config"]["env"]
    assert env == {"LOG_LEVEL": "***", "GITHUB_TOKEN": "${GITHUB_TOKEN}"}


def test_references_resolve_only_at_runtime():
    reg = ServerRegistration.from_payload(http_payload())
    runtime = reg.config.to_runtime({"REMOTE_TOKEN": "tok-123"})
    assert runtime.headers["Authorization"] == "Bearer tok-123"
    # The persisted model itself still holds only the reference.
    assert "tok-123" not in reg.model_dump_json()


def test_missing_environment_variable_is_a_safe_configuration_error():
    with pytest.raises(ConfigurationError) as exc_info:
        resolve_references("Bearer ${NOT_SET}", {})

    assert exc_info.value.details == {"variable": "NOT_SET"}
    assert exc_info.value.retryable is False
