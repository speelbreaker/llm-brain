"""
Tests for the AI Steward API endpoints.
"""
from fastapi.testclient import TestClient
from src.web_app import app

client = TestClient(app)


def test_steward_run_endpoint():
    """Test that the steward run endpoint returns a valid report."""
    resp = client.post("/api/steward/run")
    assert resp.status_code == 200
    data = resp.json()
    assert "ok" in data
    assert "summary" in data
    assert "top_items" in data
    assert "builder_prompt" in data


def test_steward_report_endpoint_after_run():
    """Test that the report endpoint returns the last run's data."""
    client.post("/api/steward/run")
    resp = client.get("/api/steward/report")
    assert resp.status_code == 200
    data = resp.json()
    assert "summary" in data
    assert "top_items" in data
