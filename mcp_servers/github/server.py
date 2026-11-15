from __future__ import annotations

from collections.abc import Callable
from typing import Any

from mcp.server import MCPServer

from mcp_servers.github.client import GitHubAPIError, GitHubClient
from mcp_servers.github.service import (
    GitHubService,
    GitHubValidationError,
)


def _execute(operation: Callable[[], Any]) -> dict[str, Any]:
    """Execute a service operation and return a safe MCP response."""

    try:
        return {
            "success": True,
            "data": operation(),
        }

    except GitHubAPIError as exc:
        return {
            "success": False,
            "error": exc.to_dict(),
        }

    except GitHubValidationError as exc:
        return {
            "success": False,
            "error": {
                "type": "validation_error",
                "message": str(exc),
            },
        }

    except Exception:
        return {
            "success": False,
            "error": {
                "type": "internal_error",
                "message": "An unexpected error occurred in the GitHub MCP server.",
            },
        }


def create_github_server(
    client: GitHubClient | None = None,
) -> MCPServer:
    """Create a GitHub MCP server with ten registered tools."""

    github_client = client or GitHubClient()
    service = GitHubService(github_client)

    server = MCPServer("Nexus GitHub MCP Server")

    @server.tool()
    def search_repositories(
        query: str,
        sort: str = "best-match",
        order: str = "desc",
        per_page: int = 10,
        page: int = 1,
    ) -> dict[str, Any]:
        """Search GitHub repositories by keywords, language, or topic."""
        return _execute(
            lambda: service.search_repositories(
                query,
                sort=sort,
                order=order,
                per_page=per_page,
                page=page,
            )
        )

    @server.tool()
    def get_repository_details(
        owner: str,
        repo: str,
    ) -> dict[str, Any]:
        """Retrieve repository metadata, statistics, and configuration."""
        return _execute(lambda: service.get_repository_details(owner, repo))

    @server.tool()
    def list_issues(
        owner: str,
        repo: str,
        state: str = "open",
        per_page: int = 20,
        page: int = 1,
    ) -> dict[str, Any]:
        """List repository issues, optionally filtered by state."""
        return _execute(
            lambda: service.list_issues(
                owner,
                repo,
                state=state,
                per_page=per_page,
                page=page,
            )
        )

    @server.tool()
    def get_issue_details(
        owner: str,
        repo: str,
        issue_number: int,
    ) -> dict[str, Any]:
        """Retrieve the details of a specific GitHub issue."""
        return _execute(
            lambda: service.get_issue_details(
                owner,
                repo,
                issue_number,
            )
        )

    @server.tool()
    def create_issue(
        owner: str,
        repo: str,
        title: str,
        body: str,
        labels: list[str] | None = None,
        assignees: list[str] | None = None,
    ) -> dict[str, Any]:
        """Create a GitHub issue in a repository."""
        return _execute(
            lambda: service.create_issue(
                owner,
                repo,
                title,
                body,
                labels=labels,
                assignees=assignees,
            )
        )

    @server.tool()
    def update_issue(
        owner: str,
        repo: str,
        issue_number: int,
        title: str | None = None,
        body: str | None = None,
        state: str | None = None,
    ) -> dict[str, Any]:
        """Update an existing GitHub issue."""
        return _execute(
            lambda: service.update_issue(
                owner,
                repo,
                issue_number,
                title=title,
                body=body,
                state=state,
            )
        )

    @server.tool()
    def list_pull_requests(
        owner: str,
        repo: str,
        state: str = "open",
        per_page: int = 20,
        page: int = 1,
    ) -> dict[str, Any]:
        """List pull requests in a GitHub repository."""
        return _execute(
            lambda: service.list_pull_requests(
                owner,
                repo,
                state=state,
                per_page=per_page,
                page=page,
            )
        )

    @server.tool()
    def get_pull_request_details(
        owner: str,
        repo: str,
        pull_number: int,
    ) -> dict[str, Any]:
        """Retrieve details of a specific pull request."""
        return _execute(
            lambda: service.get_pull_request_details(
                owner,
                repo,
                pull_number,
            )
        )

    @server.tool()
    def create_pull_request_comment(
        owner: str,
        repo: str,
        pull_number: int,
        body: str,
    ) -> dict[str, Any]:
        """Create a conversation comment on a pull request."""
        return _execute(
            lambda: service.create_pull_request_comment(
                owner,
                repo,
                pull_number,
                body,
            )
        )

    @server.tool()
    def search_code(
        query: str,
        owner: str | None = None,
        repo: str | None = None,
        per_page: int = 10,
        page: int = 1,
    ) -> dict[str, Any]:
        """Search GitHub code, optionally restricted to one repository."""
        return _execute(
            lambda: service.search_code(
                query,
                owner=owner,
                repo=repo,
                per_page=per_page,
                page=page,
            )
        )

    return server
