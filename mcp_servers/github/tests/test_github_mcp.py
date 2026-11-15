from unittest.mock import Mock

import pytest
from mcp import Client

from mcp_servers.github.server import create_github_server

EXPECTED_TOOLS = {
    "search_repositories",
    "get_repository_details",
    "list_issues",
    "get_issue_details",
    "create_issue",
    "update_issue",
    "list_pull_requests",
    "get_pull_request_details",
    "create_pull_request_comment",
    "search_code",
}


@pytest.fixture
def github_client():
    return Mock()


@pytest.fixture
def github_server(github_client):
    return create_github_server(client=github_client)


@pytest.mark.asyncio
async def test_github_tool_discovery(github_server):
    async with Client(github_server, raise_exceptions=True) as client:
        result = await client.list_tools()

    discovered = {tool.name for tool in result.tools}

    assert discovered == EXPECTED_TOOLS


@pytest.mark.asyncio
async def test_search_repositories_over_mcp(
    github_server,
    github_client,
):
    github_client.get.return_value = {
        "total_count": 1,
        "items": [{"full_name": "octocat/hello-world"}],
    }

    async with Client(github_server, raise_exceptions=True) as client:
        result = await client.call_tool(
            "search_repositories",
            {"query": "machine learning"},
        )

    assert result.structured_content["success"] is True
    assert result.structured_content["data"]["total_count"] == 1


@pytest.mark.asyncio
async def test_get_repository_details_over_mcp(
    github_server,
    github_client,
):
    github_client.get.return_value = {
        "full_name": "octocat/hello-world",
    }

    async with Client(github_server, raise_exceptions=True) as client:
        result = await client.call_tool(
            "get_repository_details",
            {"owner": "octocat", "repo": "hello-world"},
        )

    assert result.structured_content["success"] is True
    assert result.structured_content["data"]["full_name"] == ("octocat/hello-world")


@pytest.mark.asyncio
async def test_list_issues_over_mcp(github_server, github_client):
    github_client.get.return_value = [{"number": 1}]

    async with Client(github_server, raise_exceptions=True) as client:
        result = await client.call_tool(
            "list_issues",
            {"owner": "octocat", "repo": "hello-world"},
        )

    assert result.structured_content["success"] is True
    assert result.structured_content["data"][0]["number"] == 1


@pytest.mark.asyncio
async def test_get_issue_details_over_mcp(
    github_server,
    github_client,
):
    github_client.get.return_value = {"number": 5, "title": "Bug"}

    async with Client(github_server, raise_exceptions=True) as client:
        result = await client.call_tool(
            "get_issue_details",
            {
                "owner": "octocat",
                "repo": "hello-world",
                "issue_number": 5,
            },
        )

    assert result.structured_content["success"] is True
    assert result.structured_content["data"]["number"] == 5


@pytest.mark.asyncio
async def test_create_issue_over_mcp(github_server, github_client):
    github_client.post.return_value = {"number": 10}

    async with Client(github_server, raise_exceptions=True) as client:
        result = await client.call_tool(
            "create_issue",
            {
                "owner": "octocat",
                "repo": "hello-world",
                "title": "Bug report",
                "body": "Something is broken",
            },
        )

    assert result.structured_content["success"] is True
    assert result.structured_content["data"]["number"] == 10


@pytest.mark.asyncio
async def test_update_issue_over_mcp(github_server, github_client):
    github_client.patch.return_value = {
        "number": 10,
        "state": "closed",
    }

    async with Client(github_server, raise_exceptions=True) as client:
        result = await client.call_tool(
            "update_issue",
            {
                "owner": "octocat",
                "repo": "hello-world",
                "issue_number": 10,
                "state": "closed",
            },
        )

    assert result.structured_content["success"] is True
    assert result.structured_content["data"]["state"] == "closed"


@pytest.mark.asyncio
async def test_list_pull_requests_over_mcp(
    github_server,
    github_client,
):
    github_client.get.return_value = [{"number": 3}]

    async with Client(github_server, raise_exceptions=True) as client:
        result = await client.call_tool(
            "list_pull_requests",
            {"owner": "octocat", "repo": "hello-world"},
        )

    assert result.structured_content["success"] is True
    assert result.structured_content["data"][0]["number"] == 3


@pytest.mark.asyncio
async def test_get_pull_request_details_over_mcp(
    github_server,
    github_client,
):
    github_client.get.return_value = {
        "number": 3,
        "title": "Feature",
    }

    async with Client(github_server, raise_exceptions=True) as client:
        result = await client.call_tool(
            "get_pull_request_details",
            {
                "owner": "octocat",
                "repo": "hello-world",
                "pull_number": 3,
            },
        )

    assert result.structured_content["success"] is True
    assert result.structured_content["data"]["number"] == 3


@pytest.mark.asyncio
async def test_create_pull_request_comment_over_mcp(
    github_server,
    github_client,
):
    github_client.post.return_value = {"id": 123}

    async with Client(github_server, raise_exceptions=True) as client:
        result = await client.call_tool(
            "create_pull_request_comment",
            {
                "owner": "octocat",
                "repo": "hello-world",
                "pull_number": 3,
                "body": "Looks good!",
            },
        )

    assert result.structured_content["success"] is True
    assert result.structured_content["data"]["id"] == 123


@pytest.mark.asyncio
async def test_search_code_over_mcp(github_server, github_client):
    github_client.get.return_value = {
        "total_count": 1,
        "items": [{"name": "auth.py"}],
    }

    async with Client(github_server, raise_exceptions=True) as client:
        result = await client.call_tool(
            "search_code",
            {"query": "authenticate"},
        )

    assert result.structured_content["success"] is True
    assert result.structured_content["data"]["total_count"] == 1


@pytest.mark.asyncio
async def test_validation_error_is_returned_safely(
    github_server,
):
    async with Client(github_server, raise_exceptions=True) as client:
        result = await client.call_tool(
            "get_repository_details",
            {"owner": "../invalid", "repo": "hello-world"},
        )

    response = result.structured_content

    assert response["success"] is False
    assert response["error"]["type"] == "validation_error"
