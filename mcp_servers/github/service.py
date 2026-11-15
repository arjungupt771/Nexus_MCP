from __future__ import annotations

import re
from typing import Any

from mcp_servers.github.client import GitHubClient

_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")


class GitHubValidationError(ValueError):
    """Raised when a GitHub tool receives invalid arguments."""


class GitHubService:
    def __init__(self, client: GitHubClient) -> None:
        self.client = client

    @staticmethod
    def _validate_identifier(value: str, field: str) -> str:
        value = value.strip()
        if not value or value in {".", ".."} or not _IDENTIFIER_PATTERN.fullmatch(value):
            raise GitHubValidationError(f"Invalid {field}.")
        return value

    @staticmethod
    def _validate_text(
        value: str,
        field: str,
        *,
        max_length: int = 5000,
    ) -> str:
        value = value.strip()

        if not value:
            raise GitHubValidationError(f"{field} cannot be empty.")

        if len(value) > max_length:
            raise GitHubValidationError(f"{field} cannot exceed {max_length} characters.")

        return value

    @staticmethod
    def _validate_pagination(per_page: int, page: int) -> None:
        if not 1 <= per_page <= 100:
            raise GitHubValidationError("per_page must be between 1 and 100.")

        if not 1 <= page <= 100:
            raise GitHubValidationError("page must be between 1 and 100.")

    @classmethod
    def _repository_path(cls, owner: str, repo: str) -> str:
        owner = cls._validate_identifier(owner, "owner")
        repo = cls._validate_identifier(repo, "repository name")
        return f"/repos/{owner}/{repo}"

    def search_repositories(
        self,
        query: str,
        *,
        sort: str = "best-match",
        order: str = "desc",
        per_page: int = 10,
        page: int = 1,
    ) -> dict[str, Any]:
        query = self._validate_text(query, "Search query", max_length=256)
        self._validate_pagination(per_page, page)

        if sort not in {"best-match", "stars", "forks", "help-wanted-issues", "updated"}:
            raise GitHubValidationError("Unsupported repository sort option.")

        if order not in {"asc", "desc"}:
            raise GitHubValidationError("order must be asc or desc.")

        return self.client.get(
            "/search/repositories",
            params={
                "q": query,
                "sort": sort,
                "order": order,
                "per_page": per_page,
                "page": page,
            },
        )

    def get_repository_details(
        self,
        owner: str,
        repo: str,
    ) -> dict[str, Any]:
        return self.client.get(self._repository_path(owner, repo))

    def list_issues(
        self,
        owner: str,
        repo: str,
        *,
        state: str = "open",
        per_page: int = 20,
        page: int = 1,
    ) -> list[dict[str, Any]]:
        self._validate_pagination(per_page, page)

        if state not in {"open", "closed", "all"}:
            raise GitHubValidationError("state must be open, closed, or all.")

        return self.client.get(
            f"{self._repository_path(owner, repo)}/issues",
            params={
                "state": state,
                "per_page": per_page,
                "page": page,
            },
        )

    def get_issue_details(
        self,
        owner: str,
        repo: str,
        issue_number: int,
    ) -> dict[str, Any]:
        self._validate_issue_number(issue_number)

        return self.client.get(f"{self._repository_path(owner, repo)}/issues/{issue_number}")

    def create_issue(
        self,
        owner: str,
        repo: str,
        title: str,
        body: str,
        *,
        labels: list[str] | None = None,
        assignees: list[str] | None = None,
    ) -> dict[str, Any]:
        title = self._validate_text(title, "Issue title", max_length=256)
        body = body.strip()

        if len(body) > 65536:
            raise GitHubValidationError("Issue body cannot exceed 65536 characters.")

        self._validate_string_list(labels, "labels")
        self._validate_string_list(assignees, "assignees")

        payload: dict[str, Any] = {
            "title": title,
            "body": body,
        }

        if labels:
            payload["labels"] = labels

        if assignees:
            payload["assignees"] = assignees

        return self.client.post(
            f"{self._repository_path(owner, repo)}/issues",
            json=payload,
        )

    def update_issue(
        self,
        owner: str,
        repo: str,
        issue_number: int,
        *,
        title: str | None = None,
        body: str | None = None,
        state: str | None = None,
    ) -> dict[str, Any]:
        self._validate_issue_number(issue_number)

        payload: dict[str, Any] = {}

        if title is not None:
            payload["title"] = self._validate_text(title, "Issue title", max_length=256)

        if body is not None:
            if len(body) > 65536:
                raise GitHubValidationError("Issue body cannot exceed 65536 characters.")
            payload["body"] = body

        if state is not None:
            if state not in {"open", "closed"}:
                raise GitHubValidationError("state must be open or closed.")
            payload["state"] = state

        if not payload:
            raise GitHubValidationError("Provide at least one field to update.")

        return self.client.patch(
            f"{self._repository_path(owner, repo)}/issues/{issue_number}",
            json=payload,
        )

    def list_pull_requests(
        self,
        owner: str,
        repo: str,
        *,
        state: str = "open",
        per_page: int = 20,
        page: int = 1,
    ) -> list[dict[str, Any]]:
        self._validate_pagination(per_page, page)

        if state not in {"open", "closed", "all"}:
            raise GitHubValidationError("state must be open, closed, or all.")

        return self.client.get(
            f"{self._repository_path(owner, repo)}/pulls",
            params={
                "state": state,
                "per_page": per_page,
                "page": page,
            },
        )

    def get_pull_request_details(
        self,
        owner: str,
        repo: str,
        pull_number: int,
    ) -> dict[str, Any]:
        self._validate_issue_number(pull_number)

        return self.client.get(f"{self._repository_path(owner, repo)}/pulls/{pull_number}")

    def create_pull_request_comment(
        self,
        owner: str,
        repo: str,
        pull_number: int,
        body: str,
    ) -> dict[str, Any]:
        self._validate_issue_number(pull_number)
        body = self._validate_text(body, "Comment", max_length=65536)

        # GitHub stores general PR conversation comments through
        # the Issues comments endpoint.
        return self.client.post(
            f"{self._repository_path(owner, repo)}/issues/{pull_number}/comments",
            json={"body": body},
        )

    def search_code(
        self,
        query: str,
        *,
        owner: str | None = None,
        repo: str | None = None,
        per_page: int = 10,
        page: int = 1,
    ) -> dict[str, Any]:
        query = self._validate_text(query, "Code search query", max_length=256)
        self._validate_pagination(per_page, page)

        if owner and not repo:
            raise GitHubValidationError("repo is required when owner is supplied.")

        if repo and not owner:
            raise GitHubValidationError("owner is required when repo is supplied.")

        if owner and repo:
            owner = self._validate_identifier(owner, "owner")
            repo = self._validate_identifier(repo, "repository name")
            query = f"{query} repo:{owner}/{repo}"

        return self.client.get(
            "/search/code",
            params={
                "q": query,
                "per_page": per_page,
                "page": page,
            },
        )

    @staticmethod
    def _validate_issue_number(number: int) -> None:
        if number < 1:
            raise GitHubValidationError("Issue or pull request number must be positive.")

    @classmethod
    def _validate_string_list(
        cls,
        values: list[str] | None,
        field: str,
    ) -> None:
        if values is None:
            return

        if len(values) > 20:
            raise GitHubValidationError(f"{field} cannot contain more than 20 values.")

        for value in values:
            cls._validate_text(value, field, max_length=100)
