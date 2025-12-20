"""Route-level tests for /api/diag endpoint."""

from unittest.mock import patch

from fastapi.testclient import TestClient


def test_diag_returns_provider_health():
    with patch("src.supervisor.app.get_provider_health") as mock_health:
        mock_health.return_value = {"openai": {"failures_recent": 0, "breaker_open": False, "cooldown_seconds": 0}}

        from src.supervisor.app import app

        client = TestClient(app)
        response = client.get("/api/diag")

        assert response.status_code == 200
        payload = response.json()
        assert payload["ok"] is True
        assert "provider_health" in payload
