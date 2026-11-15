from unittest.mock import Mock

import pytest

from mcp_servers.github.service import (
    GitHubService,
    GitHubValidationError,
)


@pytest.fixture
def client():
    return Mock()


@pytest.fixture
def service(client):
    return GitHubService(client)


def test_search_repositories(service, client):
    client.get.return_value = {"total_count": 1, "items": []}

    result = service.search_repositories(
        "machine learning",
        sort="stars",
        per_page=5,
    )

    assert result["total_count"] == 1
    client.get.assert_called_once_with(
        "/search/repositories",
        params={
            "q": "machine learning",
            "sort": "stars",
            "order": "desc",
            "per_page": 5,
            "page": 1,
        },
    )


def test_get_repository_details(service, client):
    client.get.return_value = {"full_name": "octocat/hello-world"}

    service.get_repository_details("octocat", "hello-world")

    client.get.assert_called_once_with("/repos/octocat/hello-world")


def test_list_issues(service, client):
    client.get.return_value = []

    service.list_issues("octocat", "hello-world", state="closed")

    client.get.assert_called_once_with(
        "/repos/octocat/hello-world/issues",
        params={
            "state": "closed",
            "per_page": 20,
            "page": 1,
        },
    )


def test_create_issue(service, client):
    client.post.return_value = {"number": 10}

    service.create_issue(
        "octocat",
        "hello-world",
        "Bug report",
        "Something is broken",
        labels=["bug"],
    )

    client.post.assert_called_once_with(
        "/repos/octocat/hello-world/issues",
        json={
            "title": "Bug report",
            "body": "Something is broken",
            "labels": ["bug"],
        },
    )


def test_update_issue(service, client):
    client.patch.return_value = {"state": "closed"}

    service.update_issue(
        "octocat",
        "hello-world",
        10,
        state="closed",
    )

    client.patch.assert_called_once_with(
        "/repos/octocat/hello-world/issues/10",
        json={"state": "closed"},
    )


def test_create_pull_request_comment(service, client):
    client.post.return_value = {"id": 123}

    service.create_pull_request_comment(
        "octocat",
        "hello-world",
        5,
        "Looks good!",
    )

    client.post.assert_called_once_with(
        "/repos/octocat/hello-world/issues/5/comments",
        json={"body": "Looks good!"},
    )


def test_search_code_with_repository_filter(service, client):
    client.get.return_value = {"total_count": 0, "items": []}

    service.search_code(
        "def authenticate",
        owner="octocat",
        repo="hello-world",
    )

    client.get.assert_called_once_with(
        "/search/code",
        params={
            "q": "def authenticate repo:octocat/hello-world",
            "per_page": 10,
            "page": 1,
        },
    )


@pytest.mark.parametrize(
    ("owner", "repo"),
    [
        ("", "hello-world"),
        ("../admin", "hello-world"),
        ("octocat", ".."),
        ("octocat", "bad/repo"),
    ],
)
def test_rejects_invalid_repository_identifiers(
    service,
    owner,
    repo,
):
    with pytest.raises(GitHubValidationError):
        service.get_repository_details(owner, repo)


def test_rejects_invalid_pagination(service):
    with pytest.raises(GitHubValidationError):
        service.list_issues(
            "octocat",
            "hello-world",
            per_page=101,
        )


def test_rejects_invalid_issue_number(service):
    with pytest.raises(GitHubValidationError):
        service.get_issue_details(
            "octocat",
            "hello-world",
            0,
        )


def test_rejects_empty_issue_title(service):
    with pytest.raises(GitHubValidationError):
        service.create_issue(
            "octocat",
            "hello-world",
            "",
            "Issue body",
        )


def test_rejects_empty_update(service):
    with pytest.raises(GitHubValidationError):
        service.update_issue(
            "octocat",
            "hello-world",
            10,
        )


def test_rejects_invalid_issue_state(service):
    with pytest.raises(GitHubValidationError):
        service.update_issue(
            "octocat",
            "hello-world",
            10,
            state="pending",
        )


def test_search_code_requires_owner_and_repo_together(service):
    with pytest.raises(GitHubValidationError):
        service.search_code("authentication", owner="octocat")
