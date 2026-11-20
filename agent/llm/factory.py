"""Provider registry and environment-based configuration."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from typing import Any

import httpx

from agent.llm.errors import LLMConfigurationError
from agent.llm.gemini import GeminiProvider
from agent.llm.groq import GroqProvider
from agent.llm.models import ModelConfig, RetryConfig
from agent.llm.ollama import OllamaProvider
from agent.llm.provider import LLMProvider

ProviderBuilder = Callable[..., LLMProvider]

# Defaults only; override with NEXUS_LLM_MODEL or <PROVIDER>_MODEL.
DEFAULT_MODELS = {
    "gemini": "gemini-2.0-flash",
    "groq": "llama-3.3-70b-versatile",
    "ollama": "llama3.1",
}
_API_KEY_ENV = {"gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"), "groq": ("GROQ_API_KEY",)}


class ProviderRegistry:
    """Maps provider names to adapter classes.

    Register a new provider with ``registry.register("name", MyProvider)``;
    nothing in the agent core changes.
    """

    def __init__(self) -> None:
        self._builders: dict[str, type[Any]] = {}

    def register(self, name: str, provider_class: type[Any]) -> None:
        self._builders[name.strip().lower()] = provider_class

    def names(self) -> list[str]:
        return sorted(self._builders)

    def create(
        self,
        config: ModelConfig,
        *,
        env: Mapping[str, str] | None = None,
        client: httpx.AsyncClient | None = None,
        **kwargs: Any,
    ) -> LLMProvider:
        provider_class = self._builders.get(config.provider)

        if provider_class is None:
            raise LLMConfigurationError(
                f"Unknown LLM provider '{config.provider}'. Available: {', '.join(self.names())}."
            )

        environ = env if env is not None else os.environ
        api_key = next(
            (environ[v] for v in _API_KEY_ENV.get(config.provider, ()) if environ.get(v)), None
        )
        return provider_class(config, api_key=api_key, client=client, **kwargs)


def default_registry() -> ProviderRegistry:
    registry = ProviderRegistry()
    registry.register("gemini", GeminiProvider)
    registry.register("groq", GroqProvider)
    registry.register("ollama", OllamaProvider)
    return registry


def config_from_env(
    env: Mapping[str, str] | None = None,
    *,
    provider: str | None = None,
) -> ModelConfig:
    """Build a validated ``ModelConfig`` from environment variables.

    NEXUS_LLM_PROVIDER, NEXUS_LLM_MODEL, NEXUS_LLM_TEMPERATURE,
    NEXUS_LLM_MAX_OUTPUT_TOKENS, NEXUS_LLM_TIMEOUT_SECONDS,
    NEXUS_LLM_MAX_RETRIES, OLLAMA_BASE_URL (and <PROVIDER>_MODEL).
    """
    environ = env if env is not None else os.environ
    name = (provider or environ.get("NEXUS_LLM_PROVIDER") or "ollama").strip().lower()
    values: dict[str, Any] = {
        "provider": name,
        "model": environ.get("NEXUS_LLM_MODEL")
        or environ.get(f"{name.upper()}_MODEL")
        or DEFAULT_MODELS.get(name, ""),
    }
    mapping = {
        "temperature": ("NEXUS_LLM_TEMPERATURE", float),
        "max_output_tokens": ("NEXUS_LLM_MAX_OUTPUT_TOKENS", int),
        "request_timeout_seconds": ("NEXUS_LLM_TIMEOUT_SECONDS", float),
    }

    try:
        for field, (var, cast) in mapping.items():
            if environ.get(var):
                values[field] = cast(environ[var])

        if environ.get("NEXUS_LLM_MAX_RETRIES"):
            values["retry"] = RetryConfig(max_retries=int(environ["NEXUS_LLM_MAX_RETRIES"]))

        if name == "ollama" and environ.get("OLLAMA_BASE_URL"):
            values["base_url"] = environ["OLLAMA_BASE_URL"]

        return ModelConfig(**values)
    except (ValueError, TypeError) as exc:
        # Pydantic's ValidationError is a ValueError; show field names only.
        raise LLMConfigurationError(
            f"Invalid LLM configuration ({type(exc).__name__}); check NEXUS_LLM_* variables."
        ) from None
