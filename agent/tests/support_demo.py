"""Run the REAL GitHub, Calendar and Notes MCP servers in-process with fake backends."""

from __future__ import annotations

from typing import Any

from mcp import Client

from core.container import NexusContainer
from core.transports.sdk_session import MCPSDKSession
from mcp_servers.calendar.client import GoogleCalendarAPIError
from mcp_servers.calendar.server import create_calendar_server
from mcp_servers.github.client import GitHubAPIError
from mcp_servers.github.server import create_github_server
from mcp_servers.notes.server import create_notes_server

REPOS = {
    "total_count": 3,
    "items": [
        {"full_name": "acme/mcp-hub", "description": "MCP hub", "stargazers_count": 900},
        {"full_name": "acme/mcp-cli", "description": "MCP CLI", "stargazers_count": 500},
        {"full_name": "acme/mcp-kit", "description": "MCP toolkit", "stargazers_count": 120},
    ],
}
ISSUES = [
    {"number": 12, "title": "Crash on startup", "state": "open"},
    {"number": 15, "title": "Add retry option", "state": "open"},
]
EVENTS = {
    "items": [
        {"id": "e1", "summary": "Standup", "start": {"dateTime": "2026-11-17T09:00:00+00:00"}},
        {
            "id": "e2",
            "summary": "Design review",
            "start": {"dateTime": "2026-11-17T14:00:00+00:00"},
        },
    ]
}


class FakeGitHubClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.fail_with: GitHubAPIError | None = None

    def get(self, endpoint: str, *, params: dict | None = None) -> Any:
        self.calls.append((endpoint, params or {}))

        if self.fail_with:
            raise self.fail_with

        if endpoint == "/search/repositories":
            return REPOS

        if endpoint.endswith("/issues"):
            return ISSUES

        raise AssertionError(f"unexpected GitHub endpoint {endpoint}")


class FakeCalendarClient:
    def __init__(self) -> None:
        self.created: list[tuple[str, dict]] = []
        self.calls: list[str] = []
        self.fail_events: GoogleCalendarAPIError | None = None

    def list_events(self, calendar_id: str, **kwargs: Any) -> dict:
        self.calls.append("list_events")

        if self.fail_events:
            raise self.fail_events

        return EVENTS

    def query_freebusy(self, **kwargs: Any) -> dict:
        self.calls.append("query_freebusy")
        calendar = kwargs["calendar_ids"][0]
        return {
            "calendars": {
                calendar: {
                    "busy": [
                        {"start": "2026-11-17T09:00:00+00:00", "end": "2026-11-17T10:00:00+00:00"}
                    ]
                }
            }
        }

    def create_event(self, calendar_id: str, event: dict) -> dict:
        self.calls.append("create_event")
        self.created.append((calendar_id, event))
        return {"id": "evt-1", **event}


class InProcessTransport:
    """MCPTransport over an in-memory MCP server (real server code, no subprocess)."""

    def __init__(self, server: Any) -> None:
        self._session = MCPSDKSession(lambda: Client(server))

    @property
    def connected(self) -> bool:
        return self._session.connected

    @property
    def server_capabilities(self) -> Any:
        return self._session.server_capabilities

    async def connect(self) -> None:
        await self._session.connect()

    async def disconnect(self) -> None:
        await self._session.disconnect()

    async def list_tools(self, *, cursor: str | None = None) -> Any:
        return await self._session.list_tools(cursor=cursor)

    async def call_tool(self, name, arguments=None, *, read_timeout_seconds=None) -> Any:
        return await self._session.call_tool(
            name, arguments, read_timeout_seconds=read_timeout_seconds
        )


class InProcessConnector:
    def __init__(self, servers: dict[str, Any]) -> None:
        self._servers = servers

    async def connect(self, server) -> InProcessTransport:
        transport = InProcessTransport(self._servers[server.server_id])
        await transport.connect()
        return transport


class DemoWorld:
    def __init__(
        self, container: NexusContainer, github: FakeGitHubClient, calendar: FakeCalendarClient
    ):
        self.container = container
        self.github = github
        self.calendar = calendar


async def demo_world(
    tmp_path, *, servers: tuple[str, ...] = ("github", "calendar", "notes")
) -> DemoWorld:
    github, calendar = FakeGitHubClient(), FakeCalendarClient()
    factories = {
        "github": lambda: create_github_server(client=github),
        "calendar": lambda: create_calendar_server(client=calendar),
        "notes": lambda: create_notes_server(db_path=tmp_path / "notes.db"),
    }
    container = NexusContainer(InProcessConnector({s: factories[s]() for s in servers}))

    for server_id in servers:
        await container.servers.register(
            {
                "server_id": server_id,
                "name": f"{server_id.title()} MCP Server",
                "tags": [server_id],
                "config": {"transport": "stdio", "command": "python"},
            }
        )
        await container.servers.connect(server_id)

    return DemoWorld(container, github, calendar)
