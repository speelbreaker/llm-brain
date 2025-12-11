"""
Tests for the Greg Mandolini VRP Harvester - ENTRY_ENGINE v8.0 Selector.
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
    get_calibration_spec,
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
            step_name="STRADDLE_CHECK",
            meta={"version": "8.0"},
        )
        assert decision.selected_strategy == "STRATEGY_A_STRADDLE"
        assert decision.reasoning == "Test reasoning"
        assert decision.rule_index == 3
        assert decision.step_name == "STRADDLE_CHECK"


class TestLoadGregSpec:
    """Tests for loading the JSON spec."""
    
    def test_load_spec_returns_dict(self):
        """load_greg_spec should return a dictionary."""
        spec = load_greg_spec()
        assert isinstance(spec, dict)
        assert "meta" in spec
        assert "decision_waterfall" in spec
        assert "strategy_definitions" in spec
    
    def test_spec_has_meta(self):
        """Spec should contain meta information."""
        spec = load_greg_spec()
        assert spec["meta"]["module"] == "ENTRY_ENGINE"
        assert "8.0" in spec["meta"]["version"]
    
    def test_spec_has_calibration(self):
        """Spec should have calibration block in global_entry_filters."""
        spec = load_greg_spec()
        assert "global_entry_filters" in spec
        assert "calibration" in spec["global_entry_filters"]
        cal = spec["global_entry_filters"]["calibration"]
        assert "skew_neutral_threshold" in cal
        assert "min_vrp_directional" in cal
        assert "rsi_thresholds" in cal


class TestGetCalibrationSpec:
    """Tests for get_calibration_spec helper."""
    
    def test_returns_calibration_dict(self):
        """Should return calibration dict."""
        cal = get_calibration_spec()
        assert isinstance(cal, dict)
        assert "skew_neutral_threshold" in cal
    
    def test_has_rsi_thresholds(self):
        """Should include RSI thresholds."""
        cal = get_calibration_spec()
        assert "rsi_thresholds" in cal
        assert cal["rsi_thresholds"]["lower"] == 30
        assert cal["rsi_thresholds"]["upper"] == 70


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
    """Tests for evaluate_greg_selector decision waterfall."""
    
    def test_no_trade_when_adx_too_high(self):
        """Should return NO_TRADE when ADX > 35 (safety filter)."""
        sensors = GregSelectorSensors(adx_14d=40.0)
        decision = evaluate_greg_selector(sensors)
        assert decision.selected_strategy == "NO_TRADE"
        assert decision.rule_index == 0
        assert decision.step_name == "SAFETY_FILTER"
    
    def test_no_trade_when_chop_too_high(self):
        """Should return NO_TRADE when chop_factor > 0.85 (safety filter)."""
        sensors = GregSelectorSensors(chop_factor_7d=0.9)
        decision = evaluate_greg_selector(sensors)
        assert decision.selected_strategy == "NO_TRADE"
        assert decision.rule_index == 0
        assert decision.step_name == "SAFETY_FILTER"
    
    def test_straddle_selection(self):
        """Should select STRATEGY_A_STRADDLE when conditions met."""
        sensors = GregSelectorSensors(
            vrp_30d=20.0,
            chop_factor_7d=0.4,
            adx_14d=15.0,
            skew_25d=0.0,
        )
        decision = evaluate_greg_selector(sensors)
        assert decision.selected_strategy == "STRATEGY_A_STRADDLE"
        assert decision.rule_index == 3
        assert decision.step_name == "STRADDLE_CHECK"
    
    def test_strangle_selection(self):
        """Should select STRATEGY_A_STRANGLE when conditions met."""
        sensors = GregSelectorSensors(
            vrp_30d=12.0,
            chop_factor_7d=0.7,
            adx_14d=25.0,
            skew_25d=0.0,
        )
        decision = evaluate_greg_selector(sensors)
        assert decision.selected_strategy == "STRATEGY_A_STRANGLE"
        assert decision.rule_index == 4
        assert decision.step_name == "STRANGLE_CHECK"
    
    def test_default_no_trade_when_no_match(self):
        """Should fall back to NO_TRADE when nothing matches."""
        sensors = GregSelectorSensors()
        decision = evaluate_greg_selector(sensors)
        assert decision.selected_strategy == "NO_TRADE"
    
    def test_decision_has_meta(self):
        """Decision should include meta from spec."""
        sensors = GregSelectorSensors()
        decision = evaluate_greg_selector(sensors)
        assert "module" in decision.meta
        assert decision.meta["module"] == "ENTRY_ENGINE"
    
    def test_decision_has_sensors(self):
        """Decision should include the sensors used."""
        sensors = GregSelectorSensors(vrp_30d=10.0)
        decision = evaluate_greg_selector(sensors)
        assert decision.sensors.vrp_30d == 10.0


class TestDecisionWaterfallBranches:
    """Tests for v8.0 decision waterfall branches."""
    
    def test_bull_put_spread_branch(self):
        """Bull Put Spread branch should trigger with skew > threshold, RSI < lower, VRP > directional."""
        sensors = GregSelectorSensors(
            vrp_30d=5.0,
            skew_25d=6.0,
            rsi_14d=25.0,
            adx_14d=20.0,
            chop_factor_7d=0.5,
        )
        decision = evaluate_greg_selector(sensors)
        assert decision.selected_strategy == "STRATEGY_F_BULL_PUT_SPREAD"
        assert decision.step_name == "DIRECTIONAL_SPREAD_CHECK"
    
    def test_bear_call_spread_branch(self):
        """Bear Call Spread branch should trigger with skew < -threshold, RSI > upper, VRP > directional."""
        sensors = GregSelectorSensors(
            vrp_30d=5.0,
            skew_25d=-6.0,
            rsi_14d=75.0,
            adx_14d=20.0,
            chop_factor_7d=0.5,
        )
        decision = evaluate_greg_selector(sensors)
        assert decision.selected_strategy == "STRATEGY_F_BEAR_CALL_SPREAD"
        assert decision.step_name == "DIRECTIONAL_SPREAD_CHECK"
    
    def test_directional_vrp_floor_enforced(self):
        """Directional spreads should require VRP > min_vrp_directional (2.0)."""
        sensors = GregSelectorSensors(
            vrp_30d=1.5,
            skew_25d=6.0,
            rsi_14d=25.0,
            adx_14d=20.0,
            chop_factor_7d=0.5,
        )
        decision = evaluate_greg_selector(sensors)
        assert decision.selected_strategy != "STRATEGY_F_BULL_PUT_SPREAD"


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
