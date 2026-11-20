# Phase 6 — Agent & LLM Provider Abstraction

MCP and LLM layers stay separate: `core/` has no LLM imports; `agent/` talks to MCP only through `NexusContainer.catalog` and `ToolExecutor`.

## Layout
| Path | Role |
|---|---|
| `agent/llm/models.py` | `Message`, `ToolSpec`, `ToolCall`, `LLMResponse`, `StreamChunk`, `TokenUsage`, `ModelInfo`, `ModelConfig` (validated) |
| `agent/llm/provider.py` | `LLMProvider` protocol: generate, stream, generate_structured, model_info, health_check |
| `agent/llm/errors.py` | Normalized errors with `retryable` flag |
| `agent/llm/http_base.py` | Shared timeouts, error mapping, exponential backoff, streaming, structured-output repair |
| `agent/llm/{gemini,groq,ollama}.py` | Wire-format adapters (Groq via `openai_compat.py`) |
| `agent/llm/factory.py` | `ProviderRegistry`, `config_from_env` |
| `agent/tools/bridge.py` | `MCPToolBridge`, `ToolPermissions` |
| `agent/core/` | `AgentDefinition`, `ExecutionSettings`, `Agent`, context trimming |
| `agent/graph/` | `AgentState`, `AgentNodes`, `build_agent_graph` |

## Usage
```python
config = config_from_env()                       # NEXUS_LLM_* env vars
definition = AgentDefinition(agent_id="assistant", name="Assistant", model=config,
    tool_permissions=ToolPermissions(allow=("notes.*", "github.search_*")))
agent = Agent.create(definition, container)      # provider built from env keys
result = await agent.run("Find MCP repos and save a note", history=prior_messages)
```
Switch provider: `agent.with_provider(other_provider)`. Add a provider: `registry.register("name", MyProvider)`.

## Graph
`user_input -> prepare_context -> invoke_llm -> detect_tool_calls`, then: tool calls -> `execute_tools -> process_results -> prepare_context` (loop); none -> `final_response`; budget exhausted -> `limit_reached` (final tool-free answer); provider error -> `handle_error`.

## Tool calling
Tools are named `<server_id>__<tool>` (provider-safe, collision-free, hashed if >64 chars). Calls are resolved against the **current** catalog, checked against permissions, schema-validated, then run via `ToolExecutor`. Unregistered/unauthorized/invalid calls never reach MCP; the error is returned to the model so it can recover. Multiple calls per response run in order (parallel/dependency scheduling is Phase 7).

## Resilience
Retryable: 429 (honors Retry-After), 5xx, timeouts, connection errors. Not retried: 400/401/403/404, malformed payloads. Streams are retried only before any output is emitted. API keys travel in headers only and are scrubbed from errors.

## Configuration (env)
`GEMINI_API_KEY` (or `GOOGLE_API_KEY`), `GROQ_API_KEY`, `OLLAMA_BASE_URL`, `NEXUS_LLM_PROVIDER|MODEL|TEMPERATURE|MAX_OUTPUT_TOKENS|TIMEOUT_SECONDS|MAX_RETRIES`, `<PROVIDER>_MODEL`. Default model names are placeholders; set the ones you use.
