from __future__ import annotations

import json

import httpx
import pytest
from pydantic import BaseModel
from support_llm import SECRET, BrokenStream, build, ok, text_body, tool_body

from agent.llm import (
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMConnectionError,
    LLMInvalidRequestError,
    LLMProvider,
    LLMProviderUnavailableError,
    LLMRateLimitError,
    LLMResponseError,
    LLMStructuredOutputError,
    LLMTimeoutError,
    Message,
    ModelConfig,
    ProviderRegistry,
    RetryConfig,
    Role,
    ToolCall,
    ToolSpec,
    config_from_env,
    default_registry,
)

PROVIDERS = ["gemini", "groq", "ollama"]
USER = [Message(role=Role.USER, content="hi")]
TOOL = ToolSpec(
    name="notes__create_note",
    description="Create a note",
    parameters={
        "type": "object",
        "properties": {"title": {"type": "string"}},
        "required": ["title"],
        "additionalProperties": False,
    },
)


@pytest.mark.parametrize("provider", PROVIDERS)
async def test_generate_normalizes_text_and_usage(provider):
    llm, _ = build(provider, lambda r: ok(text_body(provider)))

    response = await llm.generate(USER)

    assert response.content == "hello"
    assert response.provider == provider and response.model
    assert (response.usage.prompt_tokens, response.usage.completion_tokens) == (7, 3)
    assert response.usage.total_tokens == 10
    assert response.finish_reason.value == "stop" and response.latency_ms >= 0
    assert isinstance(llm, LLMProvider)


@pytest.mark.parametrize("provider", PROVIDERS)
async def test_tool_calls_are_extracted_with_parsed_arguments(provider):
    llm, _ = build(provider, lambda r: ok(tool_body(provider, args={"title": "Hi", "n": 2})))

    response = await llm.generate(USER, tools=[TOOL])

    (call,) = response.tool_calls
    assert call.name == "notes__create_note" and call.arguments == {"title": "Hi", "n": 2}
    assert call.id
    assert response.finish_reason.value == "tool_calls"


@pytest.mark.parametrize("provider", PROVIDERS)
async def test_request_carries_settings_tools_and_messages(provider):
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        seen["url"] = str(request.url)
        seen["headers"] = dict(request.headers)
        return ok(text_body(provider))

    llm, _ = build(provider, handler, temperature=0.7, max_output_tokens=321)
    await llm.generate([Message(role=Role.SYSTEM, content="be brief"), *USER], tools=[TOOL])
    body = json.dumps(seen["body"])

    assert "be brief" in body and "notes__create_note" in body
    assert "0.7" in body and "321" in body
    # Credentials travel in headers only, never in URL or body.
    assert SECRET not in seen["url"] and SECRET not in body
    if provider != "ollama":
        assert SECRET in "".join(seen["headers"].values())


async def test_gemini_tool_schema_is_sanitized():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return ok(text_body("gemini"))

    llm, _ = build("gemini", handler)
    await llm.generate(USER, tools=[TOOL])

    declaration = seen["body"]["tools"][0]["functionDeclarations"][0]
    assert "additionalProperties" not in declaration["parameters"]
    assert declaration["parameters"]["required"] == ["title"]


@pytest.mark.parametrize("provider", PROVIDERS)
async def test_tool_results_are_sent_back_in_provider_format(provider):
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return ok(text_body(provider))

    call = ToolCall(id="call_1", name="notes__create_note", arguments={"title": "x"})
    history = [
        *USER,
        Message(role=Role.ASSISTANT, tool_calls=[call]),
        Message(
            role=Role.TOOL, tool_call_id="call_1", name="notes__create_note", content='{"ok": true}'
        ),
    ]
    llm, _ = build(provider, handler)
    await llm.generate(history, tools=[TOOL])
    wire = json.dumps(seen["body"])

    assert "notes__create_note" in wire and "ok" in wire
    if provider == "groq":
        assert '"tool_call_id": "call_1"' in wire
    if provider == "gemini":
        assert "functionResponse" in wire and "functionCall" in wire


# ---------------------------------------------------------------- structured output


class Person(BaseModel):
    name: str
    age: int


@pytest.mark.parametrize("provider", PROVIDERS)
async def test_structured_output_parses_json(provider):
    llm, _ = build(provider, lambda r: ok(text_body(provider, '{"name": "Ada", "age": 36}')))

    person = await llm.generate_structured(USER, Person)

    assert person == Person(name="Ada", age=36)


async def test_structured_output_accepts_fenced_json():
    llm, _ = build(
        "groq", lambda r: ok(text_body("groq", 'Sure!\n```json\n{"name":"A","age":1}\n```'))
    )
    assert (await llm.generate_structured(USER, Person)).age == 1


async def test_structured_output_repairs_invalid_reply_once():
    replies = iter(['{"name": "Ada"}', '{"name": "Ada", "age": 36}'])
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return ok(text_body("groq", next(replies)))

    llm, _ = build("groq", handler)
    person = await llm.generate_structured(USER, Person)

    assert person.age == 36 and len(requests) == 2
    assert "invalid" in json.dumps(requests[1]["messages"][-1]).lower()


async def test_structured_output_gives_up_after_repair_budget():
    llm, _ = build("groq", lambda r: ok(text_body("groq", "not json at all")))

    with pytest.raises(LLMStructuredOutputError):
        await llm.generate_structured(USER, Person)


# ---------------------------------------------------------------- streaming


def sse(*events: dict | str) -> httpx.Response:
    lines = [e if isinstance(e, str) else json.dumps(e) for e in events]
    return httpx.Response(200, content="".join(f"data: {line}\n\n" for line in lines).encode())


async def collect(llm, **kwargs):
    return [chunk async for chunk in llm.stream(USER, **kwargs)]


async def test_groq_stream_yields_deltas_then_usage():
    response = sse(
        {"choices": [{"delta": {"content": "Hel"}}]},
        {"choices": [{"delta": {"content": "lo"}, "finish_reason": "stop"}]},
        {"choices": [], "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}},
        "[DONE]",
    )
    llm, _ = build("groq", lambda r: response)

    chunks = await collect(llm)

    assert "".join(c.delta for c in chunks) == "Hello"
    assert chunks[-1].usage.total_tokens == 6 and chunks[-1].finish_reason.value == "stop"


async def test_groq_stream_assembles_fragmented_tool_calls():
    response = sse(
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "c1",
                                "function": {"name": "t__x", "arguments": '{"a":'},
                            }
                        ]
                    }
                }
            ]
        },
        {
            "choices": [
                {
                    "delta": {"tool_calls": [{"index": 0, "function": {"arguments": " 1}"}}]},
                    "finish_reason": "tool_calls",
                }
            ]
        },
        "[DONE]",
    )
    llm, _ = build("groq", lambda r: response)

    final = (await collect(llm, tools=[TOOL]))[-1]

    assert [(c.id, c.name, c.arguments) for c in final.tool_calls] == [("c1", "t__x", {"a": 1})]
    assert final.finish_reason.value == "tool_calls"


async def test_gemini_stream():
    response = sse(
        {"candidates": [{"content": {"parts": [{"text": "He"}]}}]},
        {
            "candidates": [{"content": {"parts": [{"text": "y"}]}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 2, "candidatesTokenCount": 2},
        },
    )
    llm, _ = build("gemini", lambda r: response)

    chunks = await collect(llm)

    assert "".join(c.delta for c in chunks) == "Hey" and chunks[-1].usage.total_tokens == 4


async def test_ollama_ndjson_stream_with_tool_call():
    lines = [
        {"message": {"content": "ok"}, "done": False},
        {
            "message": {
                "content": "",
                "tool_calls": [{"function": {"name": "t__x", "arguments": {"q": 1}}}],
            },
            "done": False,
        },
        {"done": True, "done_reason": "stop", "prompt_eval_count": 5, "eval_count": 1},
    ]
    llm, _ = build(
        "ollama", lambda r: httpx.Response(200, content="\n".join(map(json.dumps, lines)).encode())
    )

    chunks = await collect(llm)

    assert chunks[0].delta == "ok"
    assert chunks[-1].tool_calls[0].arguments == {"q": 1} and chunks[-1].usage.total_tokens == 6


async def test_stream_retries_before_output_but_not_after():
    attempts = []

    def flaky(request):
        attempts.append(1)
        return (
            httpx.Response(503, json={"error": {"message": "busy"}})
            if len(attempts) == 1
            else sse({"choices": [{"delta": {"content": "ok"}}]}, "[DONE]")
        )

    llm, sleeps = build("groq", flaky)
    assert "ok" in "".join(c.delta for c in await collect(llm))
    assert len(attempts) == 2 and sleeps == [0.5]

    calls = []

    def dropping(request):
        calls.append(1)
        data = b'data: {"choices": [{"delta": {"content": "partial"}}]}\n\n'
        return httpx.Response(200, stream=BrokenStream(data))

    llm, _ = build("groq", dropping)
    received = []

    with pytest.raises(LLMConnectionError):
        async for chunk in llm.stream(USER):
            received.append(chunk.delta)

    assert received == ["partial"] and len(calls) == 1  # no replay after output


# ---------------------------------------------------------------- errors & retries


@pytest.mark.parametrize(
    ("status", "error", "retried"),
    [
        (400, LLMInvalidRequestError, False),
        (404, LLMInvalidRequestError, False),
        (401, LLMAuthenticationError, False),
        (403, LLMAuthenticationError, False),
        (429, LLMRateLimitError, True),
        (500, LLMProviderUnavailableError, True),
        (503, LLMProviderUnavailableError, True),
    ],
)
@pytest.mark.parametrize("provider", PROVIDERS)
async def test_http_errors_are_normalized_and_retried_only_when_recoverable(
    provider, status, error, retried
):
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(status, json={"error": {"message": "nope"}})

    llm, sleeps = build(provider, handler, retry=RetryConfig(max_retries=2, jitter=False))

    with pytest.raises(error) as exc_info:
        await llm.generate(USER)

    assert exc_info.value.status_code == status and exc_info.value.provider == provider
    assert exc_info.value.retryable is retried
    assert len(calls) == (3 if retried else 1) and len(sleeps) == (2 if retried else 0)


async def test_exponential_backoff_is_capped():
    llm, sleeps = build(
        "groq",
        lambda r: httpx.Response(503, json={}),
        retry=RetryConfig(
            max_retries=4, initial_delay_seconds=1, max_delay_seconds=3, multiplier=2, jitter=False
        ),
    )

    with pytest.raises(LLMProviderUnavailableError):
        await llm.generate(USER)

    assert sleeps == [1, 2, 3, 3]


async def test_retry_after_header_is_honored():
    responses = iter(
        [httpx.Response(429, headers={"retry-after": "7"}, json={}), ok(text_body("groq"))]
    )
    llm, sleeps = build(
        "groq", lambda r: next(responses), retry=RetryConfig(jitter=False, max_delay_seconds=8)
    )

    assert (await llm.generate(USER)).content == "hello"
    assert sleeps == [7.0]


async def test_recovers_after_transient_failures():
    responses = iter(
        [httpx.Response(503, json={}), httpx.Response(503, json={}), ok(text_body("groq"))]
    )
    llm, sleeps = build("groq", lambda r: next(responses))

    assert (await llm.generate(USER)).content == "hello" and sleeps == [0.5, 1.0]


async def test_zero_retries_means_one_attempt():
    calls = []
    llm, _ = build(
        "groq",
        lambda r: calls.append(1) or httpx.Response(503, json={}),
        retry=RetryConfig(max_retries=0),
    )

    with pytest.raises(LLMProviderUnavailableError):
        await llm.generate(USER)

    assert len(calls) == 1


async def test_timeouts_become_retryable_timeout_errors():
    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)

    llm, sleeps = build("ollama", handler, retry=RetryConfig(max_retries=1, jitter=False))

    with pytest.raises(LLMTimeoutError) as exc_info:
        await llm.generate(USER)

    assert exc_info.value.retryable and len(sleeps) == 1


async def test_connection_failures_become_connection_errors():
    def handler(request):
        raise httpx.ConnectError("refused", request=request)

    llm, _ = build("ollama", handler, retry=RetryConfig(max_retries=0))

    with pytest.raises(LLMConnectionError):
        await llm.generate(USER)


async def test_malformed_provider_payloads_are_non_retryable_response_errors():
    calls = []
    llm, _ = build("groq", lambda r: calls.append(1) or httpx.Response(200, content=b"<html>"))

    with pytest.raises(LLMResponseError) as exc_info:
        await llm.generate(USER)

    assert not exc_info.value.retryable and len(calls) == 1

    llm, _ = build("groq", lambda r: ok({"choices": []}))
    with pytest.raises(LLMResponseError):
        await llm.generate(USER)


async def test_bad_tool_argument_json_is_a_response_error():
    body = tool_body("groq")
    body["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = "{not json"
    llm, _ = build("groq", lambda r: ok(body))

    with pytest.raises(LLMResponseError):
        await llm.generate(USER)


async def test_api_keys_are_scrubbed_from_error_messages():
    llm, _ = build(
        "groq",
        lambda r: httpx.Response(401, json={"error": {"message": f"bad key {SECRET}"}}),
    )

    with pytest.raises(LLMAuthenticationError) as exc_info:
        await llm.generate(USER)

    assert SECRET not in str(exc_info.value) and "***" in str(exc_info.value)
    assert SECRET not in repr(llm)


# ---------------------------------------------------------------- config, registry, health


def test_missing_api_keys_are_configuration_errors_but_ollama_needs_none():
    registry = default_registry()

    for name in ("gemini", "groq"):
        with pytest.raises(LLMConfigurationError, match="API key"):
            registry.create(ModelConfig(provider=name, model="m"), env={})

    assert registry.create(ModelConfig(provider="ollama", model="m"), env={}).name == "ollama"
    assert registry.create(ModelConfig(provider="gemini", model="m"), env={"GOOGLE_API_KEY": "k"})


def test_unknown_provider_is_rejected():
    with pytest.raises(LLMConfigurationError, match="Unknown LLM provider"):
        default_registry().create(ModelConfig(provider="nope", model="m"), env={})


@pytest.mark.parametrize(
    "kwargs",
    [
        {"temperature": 5},
        {"max_output_tokens": 0},
        {"request_timeout_seconds": 0},
        {"base_url": "ftp://x"},
        {"base_url": "http://user:pw@host"},
        {"model": ""},
        {"surprise": 1},
    ],
)
def test_invalid_model_config_is_rejected(kwargs):
    with pytest.raises(ValueError):
        ModelConfig(**{"provider": "ollama", "model": "m", **kwargs})


def test_config_from_env():
    config = config_from_env(
        {
            "NEXUS_LLM_PROVIDER": "Groq",
            "GROQ_MODEL": "llama-x",
            "NEXUS_LLM_TEMPERATURE": "0.5",
            "NEXUS_LLM_MAX_OUTPUT_TOKENS": "256",
            "NEXUS_LLM_TIMEOUT_SECONDS": "15",
            "NEXUS_LLM_MAX_RETRIES": "5",
        }
    )
    assert (config.provider, config.model, config.temperature) == ("groq", "llama-x", 0.5)
    assert (config.max_output_tokens, config.request_timeout_seconds, config.retry.max_retries) == (
        256,
        15,
        5,
    )

    local = config_from_env({"OLLAMA_BASE_URL": "http://gpu-box:11434/"})
    assert local.provider == "ollama" and local.base_url == "http://gpu-box:11434"

    with pytest.raises(LLMConfigurationError):
        config_from_env({"NEXUS_LLM_TEMPERATURE": "hot"})

    with pytest.raises(LLMConfigurationError):
        config_from_env({"NEXUS_LLM_TEMPERATURE": "9"})


async def test_provider_switching_through_the_registry():
    registry = default_registry()
    env = {"GROQ_API_KEY": "k", "GEMINI_API_KEY": "k"}
    names = []

    for name in PROVIDERS:
        llm = registry.create(ModelConfig(provider=name, model="m"), env=env)
        names.append((llm.name, llm.model_info().provider))
        await llm.aclose()

    assert names == [(n, n) for n in PROVIDERS]


def test_new_providers_can_be_registered_without_touching_the_core():
    class EchoProvider:
        name = "echo"

        def __init__(self, config, *, api_key=None, client=None, **_):
            self.config = config

    registry = ProviderRegistry()
    registry.register("Echo", EchoProvider)

    assert registry.create(ModelConfig(provider="echo", model="m")).name == "echo"
    assert registry.names() == ["echo"]


@pytest.mark.parametrize("provider", PROVIDERS)
async def test_health_check(provider):
    llm, _ = build(provider, lambda r: httpx.Response(200, json={}))
    assert await llm.health_check() is True

    llm, _ = build(provider, lambda r: httpx.Response(500, json={}))
    assert await llm.health_check() is False

    def down(request):
        raise httpx.ConnectError("refused", request=request)

    llm, _ = build(provider, down)
    assert await llm.health_check() is False


def test_model_info():
    llm, _ = build("ollama", lambda r: ok({}))
    info = llm.model_info()
    assert (info.provider, info.model, info.supports_tools, info.supports_streaming) == (
        "ollama",
        "test-model",
        True,
        True,
    )
