"""Shared HTTP behaviour for provider adapters.

Adapters translate between Nexus models and one provider's wire format.
Everything else lives here: timeouts, error normalization, retries with
exponential backoff, streaming, structured-output repair and health checks.
"""

from __future__ import annotations

import asyncio
import json
import random
import re
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, ClassVar, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from agent.llm.errors import (
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMConnectionError,
    LLMError,
    LLMInvalidRequestError,
    LLMProviderUnavailableError,
    LLMRateLimitError,
    LLMResponseError,
    LLMStructuredOutputError,
    LLMTimeoutError,
)
from agent.llm.models import (
    LLMResponse,
    Message,
    ModelConfig,
    ModelInfo,
    Role,
    StreamChunk,
    ToolSpec,
)

T = TypeVar("T", bound=BaseModel)
SleepFn = Callable[[float], Awaitable[None]]

_MAX_RETRY_AFTER = 60.0


def parse_arguments(raw: Any) -> dict[str, Any]:
    """Tool arguments arrive as a JSON string or an object depending on provider."""
    if raw is None or raw == "":
        return {}

    if isinstance(raw, dict):
        return raw

    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise LLMResponseError("Model returned tool arguments that are not valid JSON.") from exc

    if not isinstance(parsed, dict):
        raise LLMResponseError("Model returned tool arguments that are not a JSON object.")

    return parsed


def extract_json_text(text: str) -> str:
    """Pull a JSON object out of model text (handles code fences / chatter)."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)

    if fenced:
        text = fenced.group(1).strip()

    start, end = text.find("{"), text.rfind("}")
    return text[start : end + 1] if start != -1 and end > start else text


class HTTPProvider(ABC):
    name: ClassVar[str]
    default_base_url: ClassVar[str]
    requires_api_key: ClassVar[bool] = True
    supports_tools: ClassVar[bool] = True

    def __init__(
        self,
        config: ModelConfig,
        *,
        api_key: str | None = None,
        client: httpx.AsyncClient | None = None,
        sleep: SleepFn | None = None,
        rng: Callable[[], float] | None = None,
    ) -> None:
        if self.requires_api_key and not api_key:
            raise LLMConfigurationError(
                f"Provider '{self.name}' requires an API key (set it in the environment).",
                provider=self.name,
            )

        self.config = config
        self._api_key = api_key
        self._base_url = (config.base_url or self.default_base_url).rstrip("/")
        self._client = client or httpx.AsyncClient()
        self._owns_client = client is None
        self._sleep: SleepFn = sleep or asyncio.sleep
        self._rng = rng or random.random

    # ---- adapter hooks -------------------------------------------------

    @abstractmethod
    def _endpoint(self, *, stream: bool) -> str: ...

    @abstractmethod
    def _headers(self) -> dict[str, str]: ...

    @abstractmethod
    def _build_payload(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None,
        json_schema: dict[str, Any] | None,
        stream: bool,
    ) -> dict[str, Any]: ...

    @abstractmethod
    def _parse_response(self, body: dict[str, Any]) -> LLMResponse: ...

    @abstractmethod
    def _parse_stream(self, lines: AsyncIterator[str]) -> AsyncIterator[StreamChunk]: ...

    @abstractmethod
    def _health_url(self) -> str: ...

    def _error_message(self, body: Any) -> str:
        if isinstance(body, dict):
            error = body.get("error")

            if isinstance(error, dict):
                return str(error.get("message", ""))

            if isinstance(error, str):
                return error

        return ""

    # ---- helpers -------------------------------------------------------

    def _timeout(self) -> httpx.Timeout:
        return httpx.Timeout(self.config.request_timeout_seconds)

    def _scrub(self, text: str) -> str:
        if self._api_key:
            text = text.replace(self._api_key, "***")

        return text[:300]

    def _http_error(self, response: httpx.Response) -> LLMError:
        status = response.status_code

        try:
            detail = self._scrub(self._error_message(response.json()))
        except ValueError:
            detail = ""

        suffix = f": {detail}" if detail else ""
        common = {"provider": self.name, "status_code": status}

        if status == 429:
            retry_after = None

            try:
                retry_after = float(response.headers.get("retry-after", ""))
            except ValueError:
                pass

            return LLMRateLimitError(
                f"{self.name} rate limit exceeded{suffix}", retry_after=retry_after, **common
            )

        if status in (401, 403):
            return LLMAuthenticationError(f"{self.name} rejected the credentials{suffix}", **common)

        if status == 408:
            return LLMTimeoutError(f"{self.name} request timed out{suffix}", **common)

        if status >= 500:
            return LLMProviderUnavailableError(
                f"{self.name} is unavailable ({status}){suffix}", **common
            )

        return LLMInvalidRequestError(
            f"{self.name} rejected the request ({status}){suffix}", **common
        )

    def _convert(self, exc: httpx.HTTPError) -> LLMError:
        if isinstance(exc, httpx.TimeoutException):
            return LLMTimeoutError(
                f"{self.name} request exceeded {self.config.request_timeout_seconds}s.",
                provider=self.name,
            )

        return LLMConnectionError(
            f"Could not reach {self.name} ({type(exc).__name__}).", provider=self.name
        )

    def _delay(self, attempt: int, exc: LLMError) -> float:
        retry = self.config.retry
        delay = min(
            retry.max_delay_seconds, retry.initial_delay_seconds * retry.multiplier**attempt
        )

        if exc.retry_after:
            delay = max(delay, min(exc.retry_after, _MAX_RETRY_AFTER))

        if retry.jitter:
            delay *= 0.5 + 0.5 * self._rng()

        return delay

    async def _with_retries[R](self, operation: Callable[[], Awaitable[R]]) -> R:
        retry = self.config.retry

        for attempt in range(retry.max_retries + 1):
            try:
                return await operation()
            except LLMError as exc:
                if not exc.retryable or attempt >= retry.max_retries:
                    raise

                await self._sleep(self._delay(attempt, exc))

        raise AssertionError("unreachable")  # pragma: no cover

    async def _post_once(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            response = await self._client.post(
                self._endpoint(stream=False),
                headers=self._headers(),
                json=payload,
                timeout=self._timeout(),
            )
        except httpx.HTTPError as exc:
            raise self._convert(exc) from exc

        if response.status_code >= 400:
            raise self._http_error(response)

        try:
            body = response.json()
        except ValueError as exc:
            raise LLMResponseError(
                f"{self.name} returned a non-JSON response.", provider=self.name
            ) from exc

        if not isinstance(body, dict):
            raise LLMResponseError(
                f"{self.name} returned an unexpected payload.", provider=self.name
            )

        return body

    # ---- public API ----------------------------------------------------

    async def _complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None,
        json_schema: dict[str, Any] | None,
    ) -> LLMResponse:
        payload = self._build_payload(messages, tools or None, json_schema, False)
        started = time.perf_counter()
        body = await self._with_retries(lambda: self._post_once(payload))
        response = self._parse_response(body)
        return response.model_copy(
            update={
                "provider": self.name,
                "model": response.model or self.config.model,
                "latency_ms": (time.perf_counter() - started) * 1000,
            }
        )

    async def generate(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
    ) -> LLMResponse:
        return await self._complete(messages, tools, None)

    async def stream(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        payload = self._build_payload(messages, tools or None, None, True)
        retry = self.config.retry

        for attempt in range(retry.max_retries + 1):
            emitted = False

            try:
                try:
                    async with self._client.stream(
                        "POST",
                        self._endpoint(stream=True),
                        headers=self._headers(),
                        json=payload,
                        timeout=self._timeout(),
                    ) as response:
                        if response.status_code >= 400:
                            await response.aread()
                            raise self._http_error(response)

                        async for chunk in self._parse_stream(response.aiter_lines()):
                            emitted = True
                            yield chunk

                    return
                except httpx.HTTPError as exc:
                    raise self._convert(exc) from exc
            except LLMError as exc:
                # Never retry once output has reached the caller.
                if emitted or not exc.retryable or attempt >= retry.max_retries:
                    raise

                await self._sleep(self._delay(attempt, exc))

    async def generate_structured(self, messages: list[Message], schema: type[T]) -> T:
        json_schema = schema.model_json_schema()
        conversation = list(messages)
        attempts = 1 + self.config.structured_output.max_repair_attempts
        last_error = "unknown"

        for _ in range(attempts):
            response = await self._complete(conversation, None, json_schema)

            try:
                return schema.model_validate_json(extract_json_text(response.content))
            except (ValidationError, ValueError) as exc:
                last_error = type(exc).__name__
                problem = (
                    "; ".join(
                        f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()[:5]
                    )
                    if isinstance(exc, ValidationError)
                    else "the reply was not valid JSON"
                )
                conversation = [
                    *conversation,
                    Message(role=Role.ASSISTANT, content=response.content),
                    Message(
                        role=Role.USER,
                        content=(
                            f"That reply was invalid ({problem}). Reply again with ONLY a JSON "
                            "object that matches the required schema."
                        ),
                    ),
                ]

        raise LLMStructuredOutputError(
            f"{self.name} could not produce valid structured output ({last_error}).",
            provider=self.name,
        )

    def model_info(self) -> ModelInfo:
        return ModelInfo(
            provider=self.name,
            model=self.config.model,
            supports_tools=self.supports_tools,
        )

    async def health_check(self) -> bool:
        try:
            response = await self._client.get(
                self._health_url(), headers=self._headers(), timeout=self._timeout()
            )
        except httpx.HTTPError:
            return False

        return response.status_code < 400

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model={self.config.model!r})"


def schema_instruction(json_schema: dict[str, Any]) -> str:
    """System-prompt text asking for JSON matching a schema (provider-neutral)."""
    return (
        "Respond with ONLY a single JSON object (no prose, no code fences) that conforms "
        f"to this JSON Schema:\n{json.dumps(json_schema, separators=(',', ':'))}"
    )
