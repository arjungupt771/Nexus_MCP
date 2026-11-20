"""End to end: Groq wire protocol (mocked HTTP) -> agent -> real notes MCP server."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx
from support_llm import SECRET

from agent.core import Agent, AgentDefinition
from agent.llm import ModelConfig, RetryConfig, default_registry
from core.container import NexusContainer

PROJECT_ROOT = Path(__file__).resolve().parents[2]


async def test_groq_agent_uses_a_real_mcp_server(tmp_path):
    requests: list[dict] = []
    replies = iter(
        [
            {
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "function": {
                                        "name": "notes__create_note",
                                        "arguments": json.dumps(
                                            {
                                                "title": "Agent note",
                                                "content": "written by the agent",
                                            }
                                        ),
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60},
            },
            {
                "choices": [
                    {"message": {"content": "I saved your note."}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 80, "completion_tokens": 5, "total_tokens": 85},
            },
        ]
    )

    def groq(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=next(replies))

    container = NexusContainer()
    await container.servers.register(
        {
            "server_id": "notes",
            "name": "Notes",
            "config": {
                "transport": "stdio",
                "command": sys.executable,
                "args": ["-m", "mcp_servers.notes.main"],
                "env": {
                    "PYTHONPATH": str(PROJECT_ROOT),
                    "NEXUS_NOTES_DB_PATH": str(tmp_path / "notes.db"),
                },
            },
        }
    )
    await container.servers.connect("notes")

    config = ModelConfig(provider="groq", model="llama-test", retry=RetryConfig(max_retries=0))
    provider = default_registry().create(
        config,
        env={"GROQ_API_KEY": SECRET},
        client=httpx.AsyncClient(transport=httpx.MockTransport(groq)),
    )
    agent = Agent.create(
        AgentDefinition(agent_id="notes-agent", name="Notes agent", model=config),
        container,
        provider=provider,
    )

    result = await agent.run("Save a note titled 'Agent note'.")

    assert result.response == "I saved your note." and result.stop_reason == "completed"
    outcome = result.tool_outcomes[0]
    assert outcome.success and (outcome.server_id, outcome.tool_name) == ("notes", "create_note")
    assert result.usage.total_tokens == 145

    # The model saw real MCP tool schemas and, afterwards, the real tool result.
    offered = {t["function"]["name"] for t in requests[0]["tools"]}
    assert {"notes__create_note", "notes__search_notes"} <= offered
    tool_message = requests[1]["messages"][-1]
    assert tool_message["role"] == "tool" and tool_message["tool_call_id"] == "call_1"
    assert json.loads(tool_message["content"])["success"] is True

    # And the note really exists in the MCP server's database.
    follow_up = await container.executor.execute(
        __import__("core.models", fromlist=["ToolCallRequest"]).ToolCallRequest(
            server_id="notes", tool_name="search_notes", arguments={"query": "Agent note"}
        )
    )
    assert follow_up.success and "Agent note" in json.dumps(follow_up.data)

    await agent.aclose()
    await container.close()
