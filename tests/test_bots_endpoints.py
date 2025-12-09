"""
Tests for the Bots API endpoints.
"""
import pytest
from fastapi.testclient import TestClient

from src.web_app import app
from src.bots.types import StrategyCriterion, StrategyEvaluation


client = TestClient(app)


class TestBotsMarketSensors:
    """Tests for GET /api/bots/market_sensors endpoint."""
    
    def test_endpoint_returns_ok(self):
        """Should return ok: true with sensors dict."""
        response = client.get("/api/bots/market_sensors")
        assert response.status_code == 200
        data = response.json()
        assert data.get("ok") is True
        assert "sensors" in data
        assert isinstance(data["sensors"], dict)
    
    def test_sensors_include_btc(self):
        """Should include BTC in sensors."""
        response = client.get("/api/bots/market_sensors")
        data = response.json()
        assert data.get("ok") is True
        assert "BTC" in data["sensors"]
    
    def test_sensor_fields_present(self):
        """Each underlying should have expected sensor fields."""
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


class TestBotsStrategies:
    """Tests for GET /api/bots/strategies endpoint."""
    
    def test_endpoint_returns_ok(self):
        """Should return ok: true with strategies list."""
        response = client.get("/api/bots/strategies")
        assert response.status_code == 200
        data = response.json()
        assert data.get("ok") is True
        assert "strategies" in data
        assert isinstance(data["strategies"], list)
    
    def test_strategy_shape(self):
        """Each strategy should have required fields."""
        response = client.get("/api/bots/strategies")
        data = response.json()
        assert data.get("ok") is True
        for strat in data["strategies"]:
            assert "bot_name" in strat
            assert "expert_id" in strat
            assert "underlying" in strat
            assert "strategy_key" in strat
            assert "label" in strat
            assert "status" in strat
            assert strat["status"] in ["pass", "blocked", "no_data"]
            assert "criteria" in strat
            assert isinstance(strat["criteria"], list)
    
    def test_criteria_shape(self):
        """Each criterion should have required fields."""
        response = client.get("/api/bots/strategies")
        data = response.json()
        assert data.get("ok") is True
        for strat in data["strategies"]:
            for criterion in strat.get("criteria", []):
                assert "metric" in criterion
                assert "ok" in criterion
                assert isinstance(criterion["ok"], bool)
    
    def test_gregbot_present(self):
        """GregBot evaluations should be present."""
        response = client.get("/api/bots/strategies")
        data = response.json()
        assert data.get("ok") is True
        greg_strategies = [s for s in data["strategies"] if s.get("expert_id") == "greg_mandolini"]
        assert len(greg_strategies) > 0, "No GregBot strategies found"
    
    def test_no_jade_lizard_or_ratio_spread(self):
        """Jade Lizard and Ratio Spread should not be present."""
        response = client.get("/api/bots/strategies")
        data = response.json()
        assert data.get("ok") is True
        for strat in data["strategies"]:
            assert "jade_lizard" not in strat["strategy_key"].lower()
            assert "ratio_spread" not in strat["strategy_key"].lower()
            assert "jade lizard" not in strat["label"].lower()
            assert "ratio spread" not in strat["label"].lower()
    
    def test_spread_labels_match_spec(self):
        """Spread strategies should have correct labels per JSON spec."""
        response = client.get("/api/bots/strategies")
        data = response.json()
        assert data.get("ok") is True
        
        labels = [s["label"] for s in data["strategies"]]
        keys = [s["strategy_key"] for s in data["strategies"]]
        
        if "STRATEGY_F_BULL_PUT_SPREAD" in keys:
            idx = keys.index("STRATEGY_F_BULL_PUT_SPREAD")
            assert "bull put spread" in labels[idx].lower()
        
        if "STRATEGY_F_BEAR_CALL_SPREAD" in keys:
            idx = keys.index("STRATEGY_F_BEAR_CALL_SPREAD")
            assert "bear call spread" in labels[idx].lower()


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


class TestGregSensors:
    """Tests for Greg sensor computation."""
    
    def test_compute_greg_sensors(self):
        """compute_greg_sensors should return all required keys."""
        from src.bots.gregbot import compute_greg_sensors
        
        sensors = compute_greg_sensors("BTC")
        
        required_keys = [
            "vrp_30d", "chop_factor_7d", "iv_rank_6m",
            "term_structure_spread", "skew_25d", "adx_14d",
            "rsi_14d", "price_vs_ma200"
        ]
        
        for key in required_keys:
            assert key in sensors, f"Missing sensor key: {key}"
    
    def test_gregbot_evaluations_structure(self):
        """get_gregbot_evaluations_for_underlying should return correct structure."""
        from src.bots.gregbot import get_gregbot_evaluations_for_underlying
        
        result = get_gregbot_evaluations_for_underlying("BTC")
        
        assert "underlying" in result
        assert result["underlying"] == "BTC"
        assert "sensors" in result
        assert "strategies" in result
        assert isinstance(result["strategies"], list)
        assert len(result["strategies"]) > 0
        
        for strat in result["strategies"]:
            assert isinstance(strat, StrategyEvaluation)
            assert strat.bot_name == "GregBot"
            assert strat.expert_id == "greg_mandolini"


class TestETHSensors:
    """Tests for ETH sensor computation - Task 1 validation."""
    
    def test_eth_sensors_present_in_market_sensors(self):
        """GET /api/bots/market_sensors should include ETH."""
        response = client.get("/api/bots/market_sensors")
        data = response.json()
        assert data.get("ok") is True
        assert "ETH" in data["sensors"], "ETH should be in sensors response"
    
    def test_eth_has_all_sensor_fields(self):
        """ETH should have all 8 sensor fields present (values may be None)."""
        response = client.get("/api/bots/market_sensors")
        data = response.json()
        assert data.get("ok") is True
        
        eth_sensors = data["sensors"].get("ETH", {})
        required_keys = [
            "vrp_30d", "chop_factor_7d", "iv_rank_6m",
            "term_structure_spread", "skew_25d", "adx_14d",
            "rsi_14d", "price_vs_ma200"
        ]
        for key in required_keys:
            assert key in eth_sensors, f"ETH missing sensor key: {key}"
    
    def test_eth_ohlc_sensors_computed(self):
        """ETH OHLC-based sensors (ADX, RSI, MA200) should be non-null."""
        response = client.get("/api/bots/market_sensors")
        data = response.json()
        assert data.get("ok") is True
        
        eth_sensors = data["sensors"].get("ETH", {})
        assert eth_sensors.get("adx_14d") is not None, "ETH adx_14d should be computed"
        assert eth_sensors.get("rsi_14d") is not None, "ETH rsi_14d should be computed"
        assert eth_sensors.get("price_vs_ma200") is not None, "ETH price_vs_ma200 should be computed"
    
    def test_eth_strategies_in_bots_strategies(self):
        """GET /api/bots/strategies should include ETH strategies."""
        response = client.get("/api/bots/strategies")
        data = response.json()
        assert data.get("ok") is True
        
        eth_strategies = [s for s in data["strategies"] if s.get("underlying") == "ETH"]
        assert len(eth_strategies) > 0, "Should have at least one ETH strategy"
    
    def test_eth_compute_greg_sensors(self):
        """compute_greg_sensors('ETH') should return all keys."""
        from src.bots.gregbot import compute_greg_sensors
        
        sensors = compute_greg_sensors("ETH")
        
        required_keys = [
            "vrp_30d", "chop_factor_7d", "iv_rank_6m",
            "term_structure_spread", "skew_25d", "adx_14d",
            "rsi_14d", "price_vs_ma200"
        ]
        for key in required_keys:
            assert key in sensors, f"ETH missing sensor key: {key}"


class TestSkew25d:
    """Tests for skew_25d computation - Task 2 validation."""
    
    def test_skew_25d_field_exists_for_btc(self):
        """BTC should have skew_25d field in sensors."""
        response = client.get("/api/bots/market_sensors")
        data = response.json()
        assert data.get("ok") is True
        assert "skew_25d" in data["sensors"].get("BTC", {}), "BTC should have skew_25d field"
    
    def test_skew_25d_field_exists_for_eth(self):
        """ETH should have skew_25d field in sensors."""
        response = client.get("/api/bots/market_sensors")
        data = response.json()
        assert data.get("ok") is True
        assert "skew_25d" in data["sensors"].get("ETH", {}), "ETH should have skew_25d field"
    
    def test_strategies_using_skew_have_criterion(self):
        """Strategies using skew (C_SHORT_PUT, F_BULL_PUT_SPREAD, F_BEAR_CALL_SPREAD) should have skew_25d criterion."""
        response = client.get("/api/bots/strategies")
        data = response.json()
        assert data.get("ok") is True
        
        skew_strategies = ["STRATEGY_C_SHORT_PUT", "STRATEGY_F_BULL_PUT_SPREAD", "STRATEGY_F_BEAR_CALL_SPREAD"]
        
        for strat in data["strategies"]:
            if strat["strategy_key"] in skew_strategies:
                criteria_metrics = [c["metric"] for c in strat.get("criteria", [])]
                assert "skew_25d" in criteria_metrics, f"{strat['strategy_key']} should have skew_25d criterion"
    
    def test_compute_skew_25d_returns_numeric_or_none(self):
        """compute_skew_25d should return float or None."""
        from src.bots.sensors import compute_skew_25d, get_ohlc_data
        
        df = get_ohlc_data("BTC")
        if not df.empty:
            spot = float(df["close"].iloc[-1])
            skew = compute_skew_25d("BTC", spot)
            assert skew is None or isinstance(skew, float), "skew_25d should be float or None"
    
    def test_sensor_bundle_includes_skew(self):
        """SensorBundle.to_dict() should include skew_25d."""
        from src.bots.sensors import compute_sensors_for_underlying
        
        bundle = compute_sensors_for_underlying("BTC")
        sensor_dict = bundle.to_dict()
        
        assert "skew_25d" in sensor_dict, "SensorBundle should include skew_25d"
