# Phase 5 — Dynamic MCP Registry & Tool Discovery

## Architecture

| Component | Location | Role |
|---|---|---|
| `ServerRegistration`, `ServerUpdate`, `ServerRecord` | `core/models/server_config.py` | Validated, persistable server config (stdio / Streamable HTTP) |
| `ServerConfigStore` | `core/interfaces/config_store.py` | Persistence contract (SQLite now, PostgreSQL later) |
| `SQLiteServerConfigStore`, `InMemoryServerConfigStore` | `core/registry/` | Implementations |
| `ConfiguredServerConnector` | `core/services/configured_connector.py` | Builds transports from stored config; resolves `${ENV}` refs |
| `ToolSyncService` | `core/services/tool_sync.py` | Catalog diff, discovery tracking, capability capture, events |
| `ServerManagementService` | `core/services/server_manager.py` | Register/update/connect/disconnect/restart/enable/disable/unregister/status/restore |
| `InMemoryEventBus` | `core/events/` | Change notifications with sequence numbers, replay and streaming |
| `NexusContainer` | `core/container.py` | Wires everything; `container.servers`, `container.events`, `await container.initialize()` |

## Usage

```python
container = NexusContainer(config_store=SQLiteServerConfigStore("sqlite:///./nexus.db"))
await container.initialize()          # restores persisted servers, reconnects enabled ones
await container.servers.register({
    "server_id": "github", "name": "GitHub", "tags": ["code"],
    "config": {"transport": "stdio", "command": "python",
               "args": ["-m", "mcp_servers.github.main"],
               "env": {"GITHUB_TOKEN": "${GITHUB_TOKEN}"}},   # references only
})
await container.servers.connect("github")
```

## Secrets
Persisted config accepts only `${VAR}` references for sensitive keys. Literal secrets are rejected at validation; errors never echo values; status reports mask literals.

## Config updates
Metadata applies immediately. Connection-config changes on a connected server are persisted and flagged `config_pending_restart`; `update(..., restart=True)` applies immediately and rolls back on failure.

## Events
`server.registered|updated|unregistered|enabled|disabled|connecting|connected|disconnected|error`, `tools.updated` (added/removed/changed), `tools.discovery_failed`. Use `bus.stream()` or `bus.recent(since=seq)`.

## Compatibility notes
- `ServerDescriptor` additions all have defaults. `NexusContainer(connector)` works as before.
- `MCPSDKSession` now owns the SDK context in a dedicated task so connect/disconnect can run from different tasks (required for runtime management).
- Storage tables are created on `initialize()` with a `nexus_schema_meta` version row; no Alembic migration yet.
