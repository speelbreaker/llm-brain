"""
Tests for the Bots API endpoints.
"""
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from src.web_app import app
from src.bots.types import StrategyCriterion, StrategyEvaluation


client = TestClient(app)


MOCK_SENSORS = {
    "BTC": {
        "vrp_30d": 8.5,
        "chop_factor_7d": 0.75,
        "iv_rank_6m": None,
        "term_structure_spread": None,
        "skew_25d": None,
        "adx_14d": 25.0,
        "rsi_14d": 55.0,
        "price_vs_ma200": 5.0,
    },
    "ETH": {
        "vrp_30d": 6.0,
        "chop_factor_7d": 0.80,
        "iv_rank_6m": None,
        "term_structure_spread": None,
        "skew_25d": None,
        "adx_14d": 22.0,
        "rsi_14d": 48.0,
        "price_vs_ma200": 3.0,
    },
}

MOCK_EVALUATIONS = [
    StrategyEvaluation(
        bot_name="GregBot",
        expert_id="greg_mandolini",
        underlying="BTC",
        strategy_key="STRATEGY_A_STRADDLE",
        label="ATM Straddle",
        status="pass",
        summary="All criteria met.",
        criteria=[
            StrategyCriterion(metric="vrp_30d", value=8.5, min=5.0, max=None, ok=True, note="ok"),
        ]
    ),
    StrategyEvaluation(
        bot_name="GregBot",
        expert_id="greg_mandolini",
        underlying="BTC",
        strategy_key="STRATEGY_B_STRANGLE",
        label="OTM Strangle",
        status="blocked",
        summary="Criteria not met.",
        criteria=[
            StrategyCriterion(metric="chop_factor_7d", value=0.75, min=0.8, max=None, ok=False, note="below min"),
        ]
    ),
]


class TestBotsMarketSensors:
    """Tests for GET /api/bots/market_sensors endpoint."""
    
    @patch("src.web_app.compute_all_sensors")
    def test_endpoint_returns_ok(self, mock_compute):
        """Should return ok: true with sensors dict."""
        mock_compute.return_value = MOCK_SENSORS
        response = client.get("/api/bots/market_sensors")
        assert response.status_code == 200
        data = response.json()
        assert data.get("ok") is True
        assert "sensors" in data
        assert isinstance(data["sensors"], dict)
    
    @patch("src.web_app.compute_all_sensors")
    def test_sensors_include_btc(self, mock_compute):
        """Should include BTC in sensors."""
        mock_compute.return_value = MOCK_SENSORS
        response = client.get("/api/bots/market_sensors")
        data = response.json()
        assert data.get("ok") is True
        assert "BTC" in data["sensors"]
    
    @patch("src.web_app.compute_all_sensors")
    def test_sensor_fields_present(self, mock_compute):
        """Each underlying should have expected sensor fields."""
        mock_compute.return_value = MOCK_SENSORS
        response = client.get("/api/bots/market_sensors")
        data = response.json()
        assert data.get("ok") is True
        for underlying, sensors in data["sensors"].items():
            expected_fields = [
                "vrp_30d", "chop_factor_7d", "iv_rank_6m",
                "term_structure_spread", "skew_25d", "adx_14d",
                "rsi_14d", "price_vs_ma200"
            ]
            for field in expected_fields:
                assert field in sensors, f"Missing {field} for {underlying}"
    
    @patch("src.web_app.compute_all_sensors")
    def test_sensor_values_returned(self, mock_compute):
        """Sensor values should be returned correctly."""
        mock_compute.return_value = MOCK_SENSORS
        response = client.get("/api/bots/market_sensors")
        data = response.json()
        assert data["sensors"]["BTC"]["vrp_30d"] == 8.5
        assert data["sensors"]["BTC"]["adx_14d"] == 25.0


class TestBotsStrategies:
    """Tests for GET /api/bots/strategies endpoint."""
    
    @patch("src.web_app.get_all_bot_evaluations")
    def test_endpoint_returns_ok(self, mock_evals):
        """Should return ok: true with strategies list."""
        mock_evals.return_value = MOCK_EVALUATIONS
        response = client.get("/api/bots/strategies")
        assert response.status_code == 200
        data = response.json()
        assert data.get("ok") is True
        assert "strategies" in data
        assert isinstance(data["strategies"], list)
    
    @patch("src.web_app.get_all_bot_evaluations")
    def test_strategy_shape(self, mock_evals):
        """Each strategy should have required fields."""
        mock_evals.return_value = MOCK_EVALUATIONS
        response = client.get("/api/bots/strategies")
        data = response.json()
        assert data.get("ok") is True
        for strat in data["strategies"]:
            assert "bot_name" in strat
            assert "expert_id" in strat
            assert "underlying" in strat
            assert "strategy_key" in strat
            assert "status" in strat
            assert strat["status"] in ["pass", "blocked", "no_data"]
            assert "criteria" in strat
            assert isinstance(strat["criteria"], list)
    
    @patch("src.web_app.get_all_bot_evaluations")
    def test_criteria_shape(self, mock_evals):
        """Each criterion should have required fields."""
        mock_evals.return_value = MOCK_EVALUATIONS
        response = client.get("/api/bots/strategies")
        data = response.json()
        assert data.get("ok") is True
        for strat in data["strategies"]:
            for criterion in strat.get("criteria", []):
                assert "metric" in criterion
                assert "ok" in criterion
                assert isinstance(criterion["ok"], bool)
    
    @patch("src.web_app.get_all_bot_evaluations")
    def test_gregbot_present(self, mock_evals):
        """GregBot evaluations should be present."""
        mock_evals.return_value = MOCK_EVALUATIONS
        response = client.get("/api/bots/strategies")
        data = response.json()
        assert data.get("ok") is True
        greg_strategies = [s for s in data["strategies"] if s.get("expert_id") == "greg_mandolini"]
        assert len(greg_strategies) > 0, "No GregBot strategies found"
    
    @patch("src.web_app.get_all_bot_evaluations")
    def test_pass_and_blocked_status(self, mock_evals):
        """Should have both pass and blocked strategies."""
        mock_evals.return_value = MOCK_EVALUATIONS
        response = client.get("/api/bots/strategies")
        data = response.json()
        statuses = [s["status"] for s in data["strategies"]]
        assert "pass" in statuses
        assert "blocked" in statuses


class TestBotsTypes:
    """Tests for Bots types module."""
    
    def test_strategy_criterion_model(self):
        """StrategyCriterion should be importable and usable."""
        criterion = StrategyCriterion(
            metric="vrp_30d",
            value=8.5,
            min=5.0,
            max=None,
            ok=True,
            note="ok"
        )
        assert criterion.metric == "vrp_30d"
        assert criterion.value == 8.5
        assert criterion.ok is True
    
    def test_strategy_evaluation_model(self):
        """StrategyEvaluation should be importable and usable."""
        evaluation = StrategyEvaluation(
            bot_name="GregBot",
            expert_id="greg_mandolini",
            underlying="BTC",
            strategy_key="atm_straddle",
            label="ATM Straddle",
            status="pass",
            summary="All criteria met.",
            criteria=[
                StrategyCriterion(metric="vrp_30d", value=8.5, ok=True)
            ]
        )
        assert evaluation.bot_name == "GregBot"
        assert evaluation.status == "pass"
        assert len(evaluation.criteria) == 1
    
    def test_strategy_criterion_defaults(self):
        """StrategyCriterion should have sensible defaults."""
        criterion = StrategyCriterion(metric="test", ok=False)
        assert criterion.value is None
        assert criterion.min is None
        assert criterion.max is None
        assert criterion.note is None


class TestBotsIntegration:
    """Integration tests that call actual endpoints (may depend on Deribit)."""
    
    def test_market_sensors_live(self):
        """Live endpoint should return valid structure."""
        response = client.get("/api/bots/market_sensors")
        assert response.status_code == 200
        data = response.json()
        assert data.get("ok") is True
        assert "sensors" in data
    
    def test_strategies_live(self):
        """Live endpoint should return valid structure."""
        response = client.get("/api/bots/strategies")
        assert response.status_code == 200
        data = response.json()
        assert data.get("ok") is True
        assert "strategies" in data
