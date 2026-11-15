from __future__ import annotations

import os
from typing import Any

import httpx

GITHUB_API_BASE_URL = "https://api.github.com"
GITHUB_API_VERSION = "2022-11-28"


class GitHubAPIError(Exception):
    """Normalized error returned by the GitHub API client."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        details: Any = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "github_api_error",
            "message": self.message,
            "status_code": self.status_code,
            "details": self.details,
        }


class GitHubClient:
    """Small, testable wrapper around the GitHub REST API."""

    def __init__(
        self,
        token: str | None = None,
        *,
        timeout: float = 15.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._token = token or os.getenv("GITHUB_TOKEN")

        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": GITHUB_API_VERSION,
            "User-Agent": "Nexus-MCP-GitHub-Server",
        }

        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"

        self._http = httpx.Client(
            base_url=GITHUB_API_BASE_URL,
            headers=headers,
            timeout=httpx.Timeout(timeout),
            transport=transport,
        )

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> GitHubClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def request(
        self,
        method: str,
        endpoint: str,
        *,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
    ) -> Any:
        """Send a request and normalize HTTP/network errors."""

        try:
            response = self._http.request(
                method,
                endpoint,
                params=params,
                json=json,
            )
        except httpx.TimeoutException as exc:
            raise GitHubAPIError("GitHub API request timed out.") from exc
        except httpx.RequestError as exc:
            raise GitHubAPIError("Could not connect to the GitHub API.") from exc

        if response.is_error:
            try:
                payload = response.json()
            except ValueError:
                payload = {}

            message = payload.get("message", "GitHub API request failed.")

            if response.status_code == 401:
                message = "GitHub authentication failed. Check GITHUB_TOKEN."
            elif response.status_code == 403:
                if response.headers.get("X-RateLimit-Remaining") == "0":
                    message = "GitHub API rate limit exceeded."
                else:
                    message = "GitHub access denied. Check token permissions and repository access."
            elif response.status_code == 404:
                message = "GitHub resource not found, or the token cannot access it."
            elif response.status_code == 422:
                message = payload.get(
                    "message",
                    "GitHub rejected the supplied validation parameters.",
                )

            raise GitHubAPIError(
                message,
                status_code=response.status_code,
                details=payload.get("errors"),
            )

        if response.status_code == 204 or not response.content:
            return None

        try:
            return response.json()
        except ValueError as exc:
            raise GitHubAPIError(
                "GitHub returned an invalid JSON response.",
                status_code=response.status_code,
            ) from exc

    def get(
        self,
        endpoint: str,
        *,
        params: dict[str, Any] | None = None,
    ) -> Any:
        return self.request("GET", endpoint, params=params)

    def post(
        self,
        endpoint: str,
        *,
        json: dict[str, Any],
    ) -> Any:
        return self.request("POST", endpoint, json=json)

    def patch(
        self,
        endpoint: str,
        *,
        json: dict[str, Any],
    ) -> Any:
        return self.request("PATCH", endpoint, json=json)
