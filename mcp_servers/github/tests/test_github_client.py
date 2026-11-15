import httpx
import pytest

from mcp_servers.github.client import GitHubAPIError, GitHubClient


def create_client(handler, token="test-token"):
    transport = httpx.MockTransport(handler)
    return GitHubClient(token=token, transport=transport)


def test_successful_get_request():
    def handler(request):
        assert request.method == "GET"
        assert request.url.path == "/repos/octocat/hello-world"
        assert request.headers["Authorization"] == "Bearer test-token"

        return httpx.Response(
            200,
            json={"name": "hello-world", "full_name": "octocat/hello-world"},
        )

    with create_client(handler) as client:
        result = client.get("/repos/octocat/hello-world")

    assert result["name"] == "hello-world"


def test_request_includes_github_headers():
    def handler(request):
        assert request.headers["Accept"] == "application/vnd.github+json"
        assert request.headers["X-GitHub-Api-Version"] == "2022-11-28"
        return httpx.Response(200, json={"ok": True})

    with create_client(handler) as client:
        result = client.get("/")

    assert result == {"ok": True}


def test_successful_post_request():
    def handler(request):
        assert request.method == "POST"
        assert request.read()
        return httpx.Response(201, json={"number": 12})

    with create_client(handler) as client:
        result = client.post(
            "/repos/octocat/hello-world/issues",
            json={"title": "Test issue"},
        )

    assert result["number"] == 12


def test_successful_patch_request():
    def handler(request):
        assert request.method == "PATCH"
        return httpx.Response(200, json={"state": "closed"})

    with create_client(handler) as client:
        result = client.patch(
            "/repos/octocat/hello-world/issues/1",
            json={"state": "closed"},
        )

    assert result["state"] == "closed"


def test_no_content_response_returns_none():
    def handler(request):
        return httpx.Response(204)

    with create_client(handler) as client:
        result = client.request("DELETE", "/repos/octocat/hello-world")

    assert result is None


@pytest.mark.parametrize(
    ("status_code", "payload", "expected_message"),
    [
        (401, {"message": "Bad credentials"}, "GitHub authentication failed"),
        (403, {"message": "Forbidden"}, "GitHub access denied"),
        (404, {"message": "Not Found"}, "GitHub resource not found"),
        (422, {"message": "Validation Failed"}, "Validation Failed"),
        (500, {"message": "Server Error"}, "Server Error"),
    ],
)
def test_http_errors_are_normalized(
    status_code,
    payload,
    expected_message,
):
    def handler(request):
        return httpx.Response(status_code, json=payload)

    with create_client(handler) as client:
        with pytest.raises(GitHubAPIError) as exc_info:
            client.get("/repos/octocat/hello-world")

    assert exc_info.value.status_code == status_code
    assert expected_message in exc_info.value.message


def test_rate_limit_error_is_identified():
    def handler(request):
        return httpx.Response(
            403,
            headers={"X-RateLimit-Remaining": "0"},
            json={"message": "API rate limit exceeded"},
        )

    with create_client(handler) as client:
        with pytest.raises(GitHubAPIError) as exc_info:
            client.get("/")

    assert "rate limit exceeded" in exc_info.value.message.lower()


def test_network_error_is_normalized():
    def handler(request):
        raise httpx.ConnectError("Connection refused", request=request)

    with create_client(handler) as client:
        with pytest.raises(GitHubAPIError) as exc_info:
            client.get("/")

    assert "Could not connect" in exc_info.value.message


def test_timeout_error_is_normalized():
    def handler(request):
        raise httpx.ReadTimeout("Request timed out", request=request)

    with create_client(handler) as client:
        with pytest.raises(GitHubAPIError) as exc_info:
            client.get("/")

    assert "timed out" in exc_info.value.message.lower()


def test_authentication_header_is_absent_without_token():
    def handler(request):
        assert "Authorization" not in request.headers
        return httpx.Response(200, json={"public": True})

    with create_client(handler, token=None) as client:
        result = client.get("/")

    assert result["public"] is True
