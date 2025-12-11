"""
Tests for Greg Position Management Engine.

Tests:
- Rule evaluation for each strategy type
- Correct threshold handling
- API endpoint responses
"""
import pytest
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

from src.greg_position_manager import (
    GregManagementSuggestion,
    GregPositionRules,
    evaluate_greg_positions,
    get_greg_position_rules,
    clear_rules_cache,
    _evaluate_straddle_strangle,
    _evaluate_calendar_spread,
    _evaluate_short_put,
    _evaluate_iron_fly,
    _evaluate_credit_spread,
    greg_management_store,
)
from src.models import AgentState


@pytest.fixture
def mock_rules() -> GregPositionRules:
    """Create mock rules for testing."""
    return GregPositionRules(
        meta={"module": "POSITION_ENGINE", "version": "1.0-test"},
        calibration={
            "straddle_delta_threshold": 0.15,
            "strangle_delta_threshold": 0.20,
            "straddle_profit_take_pct": 0.50,
            "strangle_profit_take_pct": 0.50,
            "straddle_stop_loss_multiple": 2.0,
            "strangle_stop_loss_multiple": 2.0,
            "straddle_close_at_dte": 21,
            "strangle_close_at_dte": 21,
            "calendar_price_move_stop_pct": 0.05,
            "short_put_profit_take_pct": 0.70,
            "short_put_stop_delta_abs": 0.80,
            "iron_fly_profit_take_pct": 0.30,
            "spread_profit_take_pct": 0.60,
        },
        strategies={},
    )


@pytest.fixture
def mock_state() -> AgentState:
    """Create mock AgentState for testing."""
    return AgentState(
        spot={"BTC": 100000.0, "ETH": 3500.0},
    )


class TestGregPositionRulesLoading:
    """Tests for rules loading."""
    
    def test_load_rules_returns_greg_position_rules(self):
        """Should return a GregPositionRules object."""
        rules = get_greg_position_rules()
        assert isinstance(rules, GregPositionRules)
        assert "POSITION_ENGINE" in rules.meta.get("module", "")
    
    def test_rules_have_calibration_values(self):
        """Should have calibration values."""
        rules = get_greg_position_rules()
        assert rules.calibration.get("straddle_delta_threshold") == 0.15
        assert rules.calibration.get("short_put_profit_take_pct") == 0.70
    
    def test_rules_have_strategies(self):
        """Should have strategy definitions."""
        rules = get_greg_position_rules()
        assert "STRATEGY_A_STRADDLE" in rules.strategies
        assert "STRATEGY_C_SHORT_PUT" in rules.strategies


class TestStraddleStrangleEvaluation:
    """Tests for straddle/strangle position evaluation."""
    
    def test_straddle_hedge_when_delta_exceeds_threshold(self, mock_rules):
        """Should suggest HEDGE when net_delta > threshold."""
        result = _evaluate_straddle_strangle(
            position_id="test:BTC-STRADDLE",
            underlying="BTC",
            strategy_code="STRATEGY_A_STRADDLE",
            net_delta=0.22,
            dte=30,
            profit_pct=0.10,
            loss_pct=0.0,
            rules=mock_rules,
        )
        
        assert result is not None
        assert result.action == "HEDGE"
        assert result.metrics["net_delta"] == 0.22
        assert result.metrics["target_delta_abs"] == 0.15
        assert "0.22" in result.reason
        assert "0.15" in result.reason
    
    def test_straddle_take_profit_when_profit_exceeds_threshold(self, mock_rules):
        """Should suggest TAKE_PROFIT when profit_pct >= threshold."""
        result = _evaluate_straddle_strangle(
            position_id="test:BTC-STRADDLE",
            underlying="BTC",
            strategy_code="STRATEGY_A_STRADDLE",
            net_delta=0.05,
            dte=30,
            profit_pct=0.55,
            loss_pct=0.0,
            rules=mock_rules,
        )
        
        assert result is not None
        assert result.action == "TAKE_PROFIT"
        assert "55" in result.summary or "0.55" in result.reason
    
    def test_straddle_close_when_loss_exceeds_multiple(self, mock_rules):
        """Should suggest CLOSE when loss exceeds stop-loss multiple."""
        result = _evaluate_straddle_strangle(
            position_id="test:BTC-STRADDLE",
            underlying="BTC",
            strategy_code="STRATEGY_A_STRADDLE",
            net_delta=0.05,
            dte=30,
            profit_pct=0.0,
            loss_pct=2.5,
            rules=mock_rules,
        )
        
        assert result is not None
        assert result.action == "CLOSE"
        assert "2.5" in result.summary or "stop-loss" in result.reason.lower()
    
    def test_straddle_roll_when_dte_below_threshold(self, mock_rules):
        """Should suggest ROLL when DTE <= threshold."""
        result = _evaluate_straddle_strangle(
            position_id="test:BTC-STRADDLE",
            underlying="BTC",
            strategy_code="STRATEGY_A_STRADDLE",
            net_delta=0.05,
            dte=19,
            profit_pct=0.10,
            loss_pct=0.0,
            rules=mock_rules,
        )
        
        assert result is not None
        assert result.action == "ROLL"
        assert "19" in result.summary or "21" in result.reason
    
    def test_straddle_hold_when_within_thresholds(self, mock_rules):
        """Should suggest HOLD when all metrics are within thresholds."""
        result = _evaluate_straddle_strangle(
            position_id="test:BTC-STRADDLE",
            underlying="BTC",
            strategy_code="STRATEGY_A_STRADDLE",
            net_delta=0.05,
            dte=35,
            profit_pct=0.20,
            loss_pct=0.0,
            rules=mock_rules,
        )
        
        assert result is not None
        assert result.action == "HOLD"
    
    def test_strangle_uses_different_delta_threshold(self, mock_rules):
        """Strangle should use 0.20 delta threshold instead of 0.15."""
        result = _evaluate_straddle_strangle(
            position_id="test:BTC-STRANGLE",
            underlying="BTC",
            strategy_code="STRATEGY_A_STRANGLE",
            net_delta=0.18,
            dte=30,
            profit_pct=0.10,
            loss_pct=0.0,
            rules=mock_rules,
        )
        
        assert result is not None
        assert result.action == "HOLD"
        assert result.metrics["target_delta_abs"] == 0.20


class TestShortPutEvaluation:
    """Tests for short put position evaluation."""
    
    def test_short_put_assign_when_delta_high(self, mock_rules):
        """Should suggest ASSIGN when delta_abs >= 0.80."""
        result = _evaluate_short_put(
            position_id="test:BTC-PUT",
            underlying="BTC",
            strategy_code="STRATEGY_C_SHORT_PUT",
            delta_abs=0.85,
            profit_pct=0.40,
            funding_rate=0.0002,
            rules=mock_rules,
        )
        
        assert result is not None
        assert result.action == "ASSIGN"
        assert "0.85" in result.reason or "0.80" in result.reason
        assert "perp" in result.reason.lower()
    
    def test_short_put_assign_negative_funding_prefers_spot(self, mock_rules):
        """Should suggest spot assignment when funding < 0."""
        result = _evaluate_short_put(
            position_id="test:BTC-PUT",
            underlying="BTC",
            strategy_code="STRATEGY_C_SHORT_PUT",
            delta_abs=0.85,
            profit_pct=0.40,
            funding_rate=-0.0002,
            rules=mock_rules,
        )
        
        assert result is not None
        assert result.action == "ASSIGN"
        assert "spot" in result.reason.lower()
    
    def test_short_put_take_profit(self, mock_rules):
        """Should suggest TAKE_PROFIT when profit_pct >= 0.70."""
        result = _evaluate_short_put(
            position_id="test:BTC-PUT",
            underlying="BTC",
            strategy_code="STRATEGY_C_SHORT_PUT",
            delta_abs=0.50,
            profit_pct=0.75,
            funding_rate=None,
            rules=mock_rules,
        )
        
        assert result is not None
        assert result.action == "TAKE_PROFIT"


class TestIronFlyEvaluation:
    """Tests for iron butterfly position evaluation."""
    
    def test_iron_fly_close_when_wing_touched(self, mock_rules):
        """Should suggest CLOSE when price touches wing."""
        result = _evaluate_iron_fly(
            position_id="test:BTC-IRON-FLY",
            underlying="BTC",
            strategy_code="STRATEGY_D_IRON_BUTTERFLY",
            spot_price=95000,
            center_strike=100000,
            wing_spread=5000,
            profit_pct=0.10,
            rules=mock_rules,
        )
        
        assert result is not None
        assert result.action == "CLOSE"
        assert "wing" in result.reason.lower()
    
    def test_iron_fly_take_profit(self, mock_rules):
        """Should suggest TAKE_PROFIT at 30%."""
        result = _evaluate_iron_fly(
            position_id="test:BTC-IRON-FLY",
            underlying="BTC",
            strategy_code="STRATEGY_D_IRON_BUTTERFLY",
            spot_price=100000,
            center_strike=100000,
            wing_spread=5000,
            profit_pct=0.35,
            rules=mock_rules,
        )
        
        assert result is not None
        assert result.action == "TAKE_PROFIT"


class TestCreditSpreadEvaluation:
    """Tests for credit spread position evaluation."""
    
    def test_bull_put_close_when_short_strike_touched(self, mock_rules):
        """Should suggest CLOSE when price drops to short strike."""
        result = _evaluate_credit_spread(
            position_id="test:BTC-BULL-PUT",
            underlying="BTC",
            strategy_code="STRATEGY_F_BULL_PUT_SPREAD",
            spot_price=95000,
            short_strike=95000,
            is_bull_put=True,
            profit_pct=0.10,
            rules=mock_rules,
        )
        
        assert result is not None
        assert result.action == "CLOSE"
    
    def test_bear_call_close_when_short_strike_touched(self, mock_rules):
        """Should suggest CLOSE when price rises to short strike."""
        result = _evaluate_credit_spread(
            position_id="test:BTC-BEAR-CALL",
            underlying="BTC",
            strategy_code="STRATEGY_F_BEAR_CALL_SPREAD",
            spot_price=105000,
            short_strike=105000,
            is_bull_put=False,
            profit_pct=0.10,
            rules=mock_rules,
        )
        
        assert result is not None
        assert result.action == "CLOSE"
    
    def test_credit_spread_take_profit(self, mock_rules):
        """Should suggest TAKE_PROFIT at 60%."""
        result = _evaluate_credit_spread(
            position_id="test:BTC-BULL-PUT",
            underlying="BTC",
            strategy_code="STRATEGY_F_BULL_PUT_SPREAD",
            spot_price=102000,
            short_strike=95000,
            is_bull_put=True,
            profit_pct=0.65,
            rules=mock_rules,
        )
        
        assert result is not None
        assert result.action == "TAKE_PROFIT"


class TestCalendarSpreadEvaluation:
    """Tests for calendar spread position evaluation."""
    
    def test_calendar_close_on_tent_breach(self, mock_rules):
        """Should suggest CLOSE when price moves > 5% from strike."""
        result = _evaluate_calendar_spread(
            position_id="test:BTC-CALENDAR",
            underlying="BTC",
            strategy_code="STRATEGY_B_CALENDAR",
            spot_price=95000,
            strike=100000,
            front_dte=7,
            profit_pct=0.10,
            rules=mock_rules,
        )
        
        assert result is not None
        assert result.action == "CLOSE"
        assert "tent" in result.reason.lower() or "5" in result.summary
    
    def test_calendar_close_near_expiry(self, mock_rules):
        """Should suggest CLOSE when front leg near expiry."""
        result = _evaluate_calendar_spread(
            position_id="test:BTC-CALENDAR",
            underlying="BTC",
            strategy_code="STRATEGY_B_CALENDAR",
            spot_price=100000,
            strike=100000,
            front_dte=1,
            profit_pct=0.10,
            rules=mock_rules,
        )
        
        assert result is not None
        assert result.action == "CLOSE"
        assert "1 day" in result.reason or "1" in result.summary


class TestEvaluateGregPositions:
    """Tests for the top-level evaluation function."""
    
    def test_evaluate_with_mock_positions(self, mock_state):
        """Should evaluate mock positions correctly."""
        mock_positions = [
            {
                "strategy_code": "STRATEGY_A_STRADDLE",
                "underlying": "BTC",
                "position_id": "test:BTC-STRADDLE",
                "net_delta": 0.22,
                "dte": 28,
                "profit_pct": 0.18,
                "loss_pct": 0.0,
            },
            {
                "strategy_code": "STRATEGY_C_SHORT_PUT",
                "underlying": "BTC",
                "position_id": "test:BTC-PUT",
                "delta": -0.85,
                "profit_pct": 0.40,
                "funding_rate": 0.0002,
            },
        ]
        
        suggestions = evaluate_greg_positions(mock_state, mock_positions=mock_positions)
        
        assert len(suggestions) == 2
        
        straddle = next((s for s in suggestions if "STRADDLE" in s.strategy_code), None)
        assert straddle is not None
        assert straddle.action == "HEDGE"
        assert straddle.metrics["net_delta"] == 0.22
        
        short_put = next((s for s in suggestions if "SHORT_PUT" in s.strategy_code), None)
        assert short_put is not None
        assert short_put.action == "ASSIGN"
    
    def test_evaluate_returns_empty_list_without_positions(self, mock_state):
        """Should return empty list when no positions to evaluate."""
        suggestions = evaluate_greg_positions(mock_state)
        assert suggestions == []


class TestGregManagementStore:
    """Tests for the management store."""
    
    def test_store_update_and_get(self):
        """Should store and retrieve suggestions."""
        suggestions = [
            GregManagementSuggestion(
                bot_name="GregBot",
                strategy_code="STRATEGY_A_STRADDLE",
                underlying="BTC",
                position_id="test:BTC-STRADDLE",
                summary="HEDGE: Net delta 0.22 > 0.15 threshold",
                action="HEDGE",
                reason="Test reason",
                metrics={"net_delta": 0.22},
            )
        ]
        
        greg_management_store.update(suggestions)
        result = greg_management_store.get()
        
        assert result["count"] == 1
        assert result["updated_at"] is not None
        assert len(result["suggestions"]) == 1
        assert result["suggestions"][0]["action"] == "HEDGE"


class TestAPIEndpoints:
    """Tests for API endpoints using TestClient."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi.testclient import TestClient
        from src.web_app import app
        return TestClient(app)
    
    def test_get_greg_management_endpoint(self, client):
        """Should return management suggestions."""
        response = client.get("/api/bots/greg/management")
        assert response.status_code == 200
        data = response.json()
        assert data["ok"] is True
        assert "suggestions" in data
        assert "count" in data
    
    def test_mock_greg_management_endpoint(self, client):
        """Should create and return mock suggestions."""
        response = client.post("/api/bots/greg/management/mock")
        assert response.status_code == 200
        data = response.json()
        assert data["ok"] is True
        assert data["mock"] is True
        assert data["count"] >= 1
        
        for suggestion in data["suggestions"]:
            assert "bot_name" in suggestion
            assert "strategy_code" in suggestion
            assert "action" in suggestion
            assert "reason" in suggestion
