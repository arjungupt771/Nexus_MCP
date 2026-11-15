from fastapi.testclient import TestClient

from apps.backend.app.main import app

client = TestClient(app)


def test_root_endpoint() -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["name"] == "Nexus MCP"
    assert response.json()["health"] == "/api/v1/health"


def test_health_endpoint() -> None:
    response = client.get("/api/v1/health")

    assert response.status_code == 200

    payload = response.json()
    assert payload["status"] == "healthy"
    assert payload["application"] == "Nexus MCP"
    assert "timestamp" in payload
