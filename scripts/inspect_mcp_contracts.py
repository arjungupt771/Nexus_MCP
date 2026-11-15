
from __future__ import annotations

import asyncio
import json

from mcp import Client

from mcp_servers.notes.server import create_notes_server
from mcp_servers.github.server import create_github_server
from mcp_servers.calendar.server import create_calendar_server


async def inspect_server(name, factory):
    server = factory()

    async with Client(server, raise_exceptions=True) as client:
        result = await client.list_tools()

        return {
            "server": name,
            "tools": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.input_schema,
                }
                for tool in result.tools
            ],
        }


async def main():
    servers = [
        ("notes", create_notes_server),
        ("github", create_github_server),
        ("calendar", create_calendar_server),
    ]

    results = []

    for name, factory in servers:
        results.append(await inspect_server(name, factory))

    output = json.dumps(results, indent=2)

    with open(
        "mcp_contract_schemas.json",
        "w",
        encoding="utf-8",
    ) as file:
        file.write(output)

    print(output)
    print("\nSaved to mcp_contract_schemas.json")


if __name__ == "__main__":
    asyncio.run(main())
