import os
import sys

import pytest
from mcp import Client, StdioServerParameters


@pytest.mark.asyncio
async def test_github_server_over_stdio():
    env = os.environ.copy()

    parameters = StdioServerParameters(
        command=sys.executable,
        args=["-m", "mcp_servers.github.main"],
        env=env,
    )

    async with Client(parameters, raise_exceptions=True) as client:
        result = await client.list_tools()

    discovered = {tool.name for tool in result.tools}

    assert len(discovered) == 10
    assert "search_repositories" in discovered
    assert "create_issue" in discovered
    assert "search_code" in discovered
