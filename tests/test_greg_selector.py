"""
Tests for the Greg Mandolini VRP Harvester - Phase 1 Master Selector.
"""
import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock

from src.strategies.greg_selector import (
    GregSelectorSensors,
    GregSelectorDecision,
    build_sensors_from_state,
    evaluate_greg_selector,
    load_greg_spec,
)
from src.models import AgentState, VolState, MarketContext, PortfolioState


class TestGregSelectorModels:
    """Tests for Pydantic models."""
    
    def test_sensors_default_none(self):
        """GregSelectorSensors should have all None defaults."""
        sensors = GregSelectorSensors()
        assert sensors.vrp_30d is None
        assert sensors.chop_factor_7d is None
        assert sensors.iv_rank_6m is None
        assert sensors.adx_14d is None
    
    def test_sensors_with_values(self):
        """GregSelectorSensors should accept values."""
        sensors = GregSelectorSensors(
            vrp_30d=15.0,
            chop_factor_7d=0.5,
            adx_14d=25.0,
        )
        assert sensors.vrp_30d == 15.0
        assert sensors.chop_factor_7d == 0.5
        assert sensors.adx_14d == 25.0
    
    def test_decision_model(self):
        """GregSelectorDecision should be constructable."""
        sensors = GregSelectorSensors()
        decision = GregSelectorDecision(
            selected_strategy="STRATEGY_A_STRADDLE",
            reasoning="Test reasoning",
            sensors=sensors,
            rule_index=3,
            meta={"version": "4.1"},
        )
        assert decision.selected_strategy == "STRATEGY_A_STRADDLE"
        assert decision.reasoning == "Test reasoning"
        assert decision.rule_index == 3


class TestLoadGregSpec:
    """Tests for loading the JSON spec."""
    
    def test_load_spec_returns_dict(self):
        """load_greg_spec should return a dictionary."""
        spec = load_greg_spec()
        assert isinstance(spec, dict)
        assert "meta" in spec
        assert "decision_tree" in spec
        assert "strategy_definitions" in spec
    
    def test_spec_has_meta(self):
        """Spec should contain meta information."""
        spec = load_greg_spec()
        assert spec["meta"]["bot_name"] == "Magadini_VRP_Harvester"


class TestBuildSensorsFromState:
    """Tests for build_sensors_from_state."""
    
    def test_basic_state(self):
        """build_sensors_from_state should return a GregSelectorSensors object."""
        state = AgentState(
            timestamp=datetime.utcnow(),
            vol_state=VolState(),
            portfolio=PortfolioState(),
            spot={"BTC": 90000.0, "ETH": 3000.0},
        )
        sensors = build_sensors_from_state(state)
        assert isinstance(sensors, GregSelectorSensors)
    
    def test_vrp_calculation(self):
        """VRP should be IV - RV."""
        state = AgentState(
            timestamp=datetime.utcnow(),
            vol_state=VolState(btc_iv=60.0, btc_rv=45.0),
            portfolio=PortfolioState(),
            spot={"BTC": 90000.0},
        )
        sensors = build_sensors_from_state(state)
        assert sensors.vrp_30d == 15.0
    
    def test_chop_factor_calculation(self):
        """Chop factor should be RV_7d / IV_30d."""
        ctx = MarketContext(
            underlying="BTC",
            time=datetime.utcnow(),
            realized_vol_7d=30.0,
        )
        state = AgentState(
            timestamp=datetime.utcnow(),
            vol_state=VolState(btc_iv=60.0, btc_rv=45.0),
            portfolio=PortfolioState(),
            spot={"BTC": 90000.0},
            market_context=ctx,
        )
        sensors = build_sensors_from_state(state)
        assert sensors.chop_factor_7d == pytest.approx(0.5)
    
    def test_skew_from_vol_state(self):
        """Skew should come from vol_state.btc_skew."""
        state = AgentState(
            timestamp=datetime.utcnow(),
            vol_state=VolState(btc_skew=5.5),
            portfolio=PortfolioState(),
            spot={"BTC": 90000.0},
        )
        sensors = build_sensors_from_state(state)
        assert sensors.skew_25d == 5.5


class TestEvaluateGregSelector:
    """Tests for evaluate_greg_selector decision tree."""
    
    def test_no_trade_when_adx_too_high(self):
        """Should return NO_TRADE when ADX > 35."""
        sensors = GregSelectorSensors(adx_14d=40.0)
        decision = evaluate_greg_selector(sensors)
        assert decision.selected_strategy == "NO_TRADE"
        assert decision.rule_index == 0
    
    def test_no_trade_when_chop_too_high(self):
        """Should return NO_TRADE when chop_factor > 0.85."""
        sensors = GregSelectorSensors(chop_factor_7d=0.9)
        decision = evaluate_greg_selector(sensors)
        assert decision.selected_strategy == "NO_TRADE"
        assert decision.rule_index == 0
    
    def test_straddle_selection(self):
        """Should select STRATEGY_A_STRADDLE when conditions met."""
        sensors = GregSelectorSensors(
            vrp_30d=20.0,
            chop_factor_7d=0.4,
            adx_14d=15.0,
        )
        decision = evaluate_greg_selector(sensors)
        assert decision.selected_strategy == "STRATEGY_A_STRADDLE"
        assert decision.rule_index == 3
    
    def test_strangle_selection(self):
        """Should select STRATEGY_A_STRANGLE when conditions met."""
        sensors = GregSelectorSensors(
            vrp_30d=12.0,
            chop_factor_7d=0.7,
            adx_14d=25.0,
        )
        decision = evaluate_greg_selector(sensors)
        assert decision.selected_strategy == "STRATEGY_A_STRANGLE"
        assert decision.rule_index == 4
    
    def test_default_no_trade_when_no_match(self):
        """Should fall back to NO_TRADE when nothing matches."""
        sensors = GregSelectorSensors()
        decision = evaluate_greg_selector(sensors)
        assert decision.selected_strategy == "NO_TRADE"
    
    def test_decision_has_meta(self):
        """Decision should include meta from spec."""
        sensors = GregSelectorSensors()
        decision = evaluate_greg_selector(sensors)
        assert "bot_name" in decision.meta
    
    def test_decision_has_sensors(self):
        """Decision should include the sensors used."""
        sensors = GregSelectorSensors(vrp_30d=10.0)
        decision = evaluate_greg_selector(sensors)
        assert decision.sensors.vrp_30d == 10.0


class TestGregSelectorEndpoint:
    """Tests for the /api/strategies/greg/selector endpoint."""
    
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from src.web_app import app
        return TestClient(app)
    
    def test_endpoint_returns_ok(self, client):
        """GET /api/strategies/greg/selector should return ok."""
        with patch("src.web_app.status_store") as mock_store:
            mock_store.get.return_value = {
                "state": {
                    "timestamp": datetime.utcnow().isoformat(),
                    "vol_state": {"btc_iv": 50.0, "btc_rv": 40.0},
                    "portfolio": {},
                    "spot": {"BTC": 90000.0},
                }
            }
            response = client.get("/api/strategies/greg/selector")
        
        assert response.status_code == 200
        data = response.json()
        assert data["ok"] is True
        assert "selected_strategy" in data
        assert "reasoning" in data
        assert "sensors" in data
    
    def test_endpoint_has_timestamp(self, client):
        """Endpoint should include timestamp."""
        with patch("src.web_app.status_store") as mock_store:
            mock_store.get.return_value = {
                "state": {
                    "timestamp": datetime.utcnow().isoformat(),
                    "vol_state": {},
                    "portfolio": {},
                    "spot": {"BTC": 90000.0},
                }
            }
            response = client.get("/api/strategies/greg/selector")
        
        data = response.json()
        assert "timestamp" in data
