"""Endpoint tests for ops routes."""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create test client."""
    from src.web_app import create_app

    app = create_app()
    return TestClient(app)


class TestOpsHealthEndpoint:
    def test_returns_200_or_503(self, client):
        """Endpoint should return 200 (found) or 503 (not found)."""
        resp = client.get("/ops/health")
        assert resp.status_code in (200, 503)

    def test_503_has_error_message(self, client, tmp_path, monkeypatch):
        """503 response should include error message."""
        monkeypatch.chdir(tmp_path)
        resp = client.get("/ops/health")
        if resp.status_code == 503:
            assert "error" in resp.json()


class TestOpsFidelityEndpoint:
    def test_returns_200_or_404(self, client):
        """Endpoint should return 200 (found) or 404 (not found)."""
        resp = client.get("/ops/fidelity/BTC")
        assert resp.status_code in (200, 404)

    def test_underlying_case_insensitive(self, client):
        """Underlying should be normalized to uppercase."""
        resp_lower = client.get("/ops/fidelity/btc")
        resp_upper = client.get("/ops/fidelity/BTC")
        assert resp_lower.status_code == resp_upper.status_code


class TestOpsGatesEndpoint:
    def test_returns_200(self, client):
        """Gates endpoint should always return 200."""
        resp = client.get("/ops/gates")
        assert resp.status_code == 200

    def test_has_can_trade(self, client):
        """Response must include can_trade boolean."""
        resp = client.get("/ops/gates")
        data = resp.json()
        assert "can_trade" in data
        assert isinstance(data["can_trade"], bool)

    def test_has_breakdown(self, client):
        """Response must include breakdown dict."""
        resp = client.get("/ops/gates")
        data = resp.json()
        assert "breakdown" in data
        assert isinstance(data["breakdown"], dict)

    def test_breakdown_has_truth_trust_trade(self, client):
        """Breakdown must have truth, trust, trade sections."""
        resp = client.get("/ops/gates")
        breakdown = resp.json()["breakdown"]
        assert "truth" in breakdown
        assert "trust" in breakdown
        assert "trade" in breakdown
