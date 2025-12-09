"""Tests for the LLM & Strategy Tuning API endpoints."""
import pytest
from fastapi.testclient import TestClient

from src.web_app import app
from src.config import settings


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


class TestLLMStatusEndpoint:
    """Tests for GET and POST /api/llm_status."""

    def test_get_llm_status(self, client):
        """GET /api/llm_status returns ok=True and expected fields."""
        response = client.get("/api/llm_status")
        assert response.status_code == 200
        data = response.json()
        assert data["ok"] is True
        assert "mode" in data
        assert "llm_enabled" in data
        assert "decision_mode" in data
        assert "explore_prob" in data

    def test_post_llm_status_update_llm_enabled(self, client):
        """POST /api/llm_status can toggle llm_enabled."""
        original_value = settings.llm_enabled
        try:
            response = client.post("/api/llm_status", json={"llm_enabled": True})
            assert response.status_code == 200
            data = response.json()
            assert data["ok"] is True
            assert data["llm_enabled"] is True
            assert settings.llm_enabled is True

            response = client.post("/api/llm_status", json={"llm_enabled": False})
            data = response.json()
            assert data["llm_enabled"] is False
            assert settings.llm_enabled is False
        finally:
            settings.llm_enabled = original_value

    def test_post_llm_status_update_explore_prob(self, client):
        """POST /api/llm_status can update explore_prob."""
        original_value = settings.explore_prob
        try:
            response = client.post("/api/llm_status", json={"explore_prob": 0.25})
            assert response.status_code == 200
            data = response.json()
            assert data["ok"] is True
            assert data["explore_prob"] == 0.25
            assert settings.explore_prob == 0.25
        finally:
            settings.explore_prob = original_value

    def test_post_llm_status_invalid_explore_prob(self, client):
        """POST /api/llm_status rejects invalid explore_prob values."""
        response = client.post("/api/llm_status", json={"explore_prob": 1.5})
        assert response.status_code == 400
        data = response.json()
        assert data["ok"] is False
        assert "explore_prob" in data["error"]

    def test_post_llm_status_invalid_decision_mode(self, client):
        """POST /api/llm_status rejects invalid decision_mode values."""
        response = client.post("/api/llm_status", json={"decision_mode": "invalid_mode"})
        assert response.status_code == 400
        data = response.json()
        assert data["ok"] is False
        assert "decision_mode" in data["error"]


class TestStrategyThresholdsEndpoint:
    """Tests for GET and POST /api/strategy_thresholds."""

    def test_get_strategy_thresholds(self, client):
        """GET /api/strategy_thresholds returns ok=True and effective values."""
        response = client.get("/api/strategy_thresholds")
        assert response.status_code == 200
        data = response.json()
        assert data["ok"] is True
        assert "mode" in data
        assert "is_research" in data
        assert "training_profile_mode" in data
        assert "prod" in data
        assert "research" in data
        assert "effective" in data
        assert "ivrv_min" in data["effective"]
        assert "delta_min" in data["effective"]
        assert "delta_max" in data["effective"]
        assert "dte_min" in data["effective"]
        assert "dte_max" in data["effective"]

    def test_post_strategy_thresholds_updates_effective(self, client):
        """POST /api/strategy_thresholds updates effective thresholds."""
        original_ivrv = settings.effective_ivrv_min
        try:
            response = client.post("/api/strategy_thresholds", json={"ivrv_min": 1.5})
            assert response.status_code == 200
            data = response.json()
            assert data["ok"] is True
            assert data["effective"]["ivrv_min"] == 1.5
        finally:
            if settings.is_research:
                settings.research_ivrv_min = original_ivrv
            else:
                settings.ivrv_min = original_ivrv

    def test_post_strategy_thresholds_training_profile_mode(self, client):
        """POST /api/strategy_thresholds can update training_profile_mode."""
        original_value = settings.training_profile_mode
        try:
            response = client.post("/api/strategy_thresholds", json={"training_profile_mode": "ladder"})
            assert response.status_code == 200
            data = response.json()
            assert data["ok"] is True
            assert data["training_profile_mode"] == "ladder"
            assert settings.training_profile_mode == "ladder"

            response = client.post("/api/strategy_thresholds", json={"training_profile_mode": "single"})
            data = response.json()
            assert data["training_profile_mode"] == "single"
        finally:
            settings.training_profile_mode = original_value

    def test_post_strategy_thresholds_invalid_training_profile_mode(self, client):
        """POST /api/strategy_thresholds rejects invalid training_profile_mode."""
        response = client.post("/api/strategy_thresholds", json={"training_profile_mode": "invalid"})
        assert response.status_code == 400
        data = response.json()
        assert data["ok"] is False
        assert "training_profile_mode" in data["error"]

    def test_post_strategy_thresholds_invalid_delta(self, client):
        """POST /api/strategy_thresholds rejects delta values outside 0-1."""
        response = client.post("/api/strategy_thresholds", json={"delta_min": 1.5})
        assert response.status_code == 400
        data = response.json()
        assert data["ok"] is False

    def test_post_strategy_thresholds_delta_min_greater_than_max(self, client):
        """POST /api/strategy_thresholds rejects delta_min > delta_max."""
        response = client.post("/api/strategy_thresholds", json={"delta_min": 0.5, "delta_max": 0.3})
        assert response.status_code == 400
        data = response.json()
        assert data["ok"] is False
        assert "delta_min" in data["error"]

    def test_post_strategy_thresholds_dte_min_greater_than_max(self, client):
        """POST /api/strategy_thresholds rejects dte_min > dte_max."""
        response = client.post("/api/strategy_thresholds", json={"dte_min": 30, "dte_max": 5})
        assert response.status_code == 400
        data = response.json()
        assert data["ok"] is False
        assert "dte_min" in data["error"]


class TestRiskLimitsEndpoint:
    """Tests for GET and POST /api/risk_limits."""

    def test_get_risk_limits(self, client):
        """GET /api/risk_limits returns ok=True and expected fields."""
        response = client.get("/api/risk_limits")
        assert response.status_code == 200
        data = response.json()
        assert data["ok"] is True
        assert "max_margin_used_pct" in data
        assert "max_net_delta_abs" in data
        assert "daily_drawdown_limit_pct" in data
        assert "kill_switch_enabled" in data

    def test_post_risk_limits_update_max_margin(self, client):
        """POST /api/risk_limits can update max_margin_used_pct."""
        original_value = settings.max_margin_used_pct
        try:
            response = client.post("/api/risk_limits", json={"max_margin_used_pct": 75.0})
            assert response.status_code == 200
            data = response.json()
            assert data["ok"] is True
            assert data["max_margin_used_pct"] == 75.0
            assert settings.max_margin_used_pct == 75.0
        finally:
            settings.max_margin_used_pct = original_value

    def test_post_risk_limits_update_max_net_delta(self, client):
        """POST /api/risk_limits can update max_net_delta_abs."""
        original_value = settings.max_net_delta_abs
        try:
            response = client.post("/api/risk_limits", json={"max_net_delta_abs": 2.5})
            assert response.status_code == 200
            data = response.json()
            assert data["ok"] is True
            assert data["max_net_delta_abs"] == 2.5
            assert settings.max_net_delta_abs == 2.5
        finally:
            settings.max_net_delta_abs = original_value

    def test_post_risk_limits_invalid_max_margin(self, client):
        """POST /api/risk_limits rejects max_margin_used_pct > 100."""
        response = client.post("/api/risk_limits", json={"max_margin_used_pct": 150.0})
        assert response.status_code == 400
        data = response.json()
        assert data["ok"] is False
        assert "max_margin_used_pct" in data["error"]

    def test_post_risk_limits_invalid_negative_delta(self, client):
        """POST /api/risk_limits rejects negative max_net_delta_abs."""
        response = client.post("/api/risk_limits", json={"max_net_delta_abs": -1.0})
        assert response.status_code == 400
        data = response.json()
        assert data["ok"] is False
