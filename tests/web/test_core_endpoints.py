"""
Core endpoint tests - verify critical endpoints work correctly.

These tests ensure the main dashboard and health endpoints are functional.
"""
import pytest
from fastapi.testclient import TestClient

from src.web_app import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestRootEndpoint:
    """Tests for GET / (dashboard)."""

    def test_returns_200(self, client):
        """Root endpoint should return 200."""
        response = client.get("/")
        assert response.status_code == 200

    def test_returns_html(self, client):
        """Root endpoint should return HTML content."""
        response = client.get("/")
        assert "text/html" in response.headers.get("content-type", "")

    def test_contains_dashboard_title(self, client):
        """Response should contain dashboard title."""
        response = client.get("/")
        assert b"Options Trading Agent" in response.content or b"Options Agent" in response.content


class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_returns_200(self, client):
        """Health endpoint should return 200."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_returns_json(self, client):
        """Health endpoint should return JSON."""
        response = client.get("/health")
        assert "application/json" in response.headers.get("content-type", "")

    def test_contains_ok_field(self, client):
        """Health response should contain ok field."""
        response = client.get("/health")
        data = response.json()
        assert "ok" in data or "status" in data


class TestStatusEndpoint:
    """Tests for GET /status."""

    def test_returns_200(self, client):
        """Status endpoint should return 200."""
        response = client.get("/status")
        assert response.status_code == 200

    def test_returns_json(self, client):
        """Status endpoint should return JSON."""
        response = client.get("/status")
        assert "application/json" in response.headers.get("content-type", "")
