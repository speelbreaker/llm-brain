"""One-tick integration tests for agent loop gate enforcement."""
from unittest.mock import patch, MagicMock
from src.ops.trade_permission import compute_trade_permission, PermissionCode
from src.config import TradingMode
from src.models import ActionType


class MockSettings:
    """Mock settings for integration tests."""
    def __init__(
        self,
        trade_mode: TradingMode = TradingMode.NORMAL,
        kill_switch_enabled: bool = False,
    ):
        self.trade_mode = trade_mode
        self.kill_switch_enabled = kill_switch_enabled
        self.is_training_on_testnet = False
        self.max_margin_used_pct = 80.0
        self.max_net_delta_abs = 5.0


class TestOneTickCloseOnlyMode:
    """Test gate enforcement in CLOSE_ONLY mode."""

    def test_open_blocked_in_close_only(self):
        """OPEN_COVERED_CALL blocked when trade_mode=CLOSE_ONLY."""
        cfg = MockSettings(trade_mode=TradingMode.CLOSE_ONLY)
        perm = compute_trade_permission(cfg, can_trade_from_health=True)
        
        assert perm.is_action_allowed(ActionType.OPEN_COVERED_CALL) is False
        assert perm.code == PermissionCode.BLOCKED_TRADE_MODE_CLOSE_ONLY

    def test_close_allowed_in_close_only(self):
        """CLOSE_COVERED_CALL allowed when trade_mode=CLOSE_ONLY."""
        cfg = MockSettings(trade_mode=TradingMode.CLOSE_ONLY)
        perm = compute_trade_permission(cfg, can_trade_from_health=True)
        
        assert perm.is_action_allowed(ActionType.CLOSE_COVERED_CALL) is True

    def test_roll_blocked_in_close_only(self):
        """ROLL_COVERED_CALL blocked when trade_mode=CLOSE_ONLY."""
        cfg = MockSettings(trade_mode=TradingMode.CLOSE_ONLY)
        perm = compute_trade_permission(cfg, can_trade_from_health=True)
        
        assert perm.is_action_allowed(ActionType.ROLL_COVERED_CALL) is False


class TestOneTickCanTradeFalse:
    """Test gate enforcement when can_trade=False."""

    def test_open_blocked_when_can_trade_false(self):
        """OPEN blocked when health/gates set can_trade=False."""
        cfg = MockSettings(trade_mode=TradingMode.NORMAL)
        perm = compute_trade_permission(cfg, can_trade_from_health=False)
        
        assert perm.is_action_allowed(ActionType.OPEN_COVERED_CALL) is False
        assert perm.code == PermissionCode.BLOCKED_CAN_TRADE_FALSE

    def test_close_allowed_for_risk_reduction(self):
        """CLOSE allowed when can_trade=False to reduce risk."""
        cfg = MockSettings(trade_mode=TradingMode.NORMAL)
        perm = compute_trade_permission(cfg, can_trade_from_health=False)
        
        assert perm.is_action_allowed(ActionType.CLOSE_COVERED_CALL) is True


class TestOneTickNormalMode:
    """Test gate enforcement in NORMAL mode with can_trade=True."""

    def test_all_actions_allowed(self):
        """All actions allowed in normal healthy state."""
        cfg = MockSettings(trade_mode=TradingMode.NORMAL)
        perm = compute_trade_permission(cfg, can_trade_from_health=True)
        
        assert perm.is_action_allowed(ActionType.OPEN_COVERED_CALL) is True
        assert perm.is_action_allowed(ActionType.CLOSE_COVERED_CALL) is True
        assert perm.is_action_allowed(ActionType.ROLL_COVERED_CALL) is True
        assert perm.code == PermissionCode.ALLOWED


class TestOneTickKillSwitch:
    """Test gate enforcement with kill switch enabled."""

    def test_kill_switch_blocks_all_trading(self):
        """Kill switch blocks all trading actions."""
        cfg = MockSettings(kill_switch_enabled=True)
        perm = compute_trade_permission(cfg, can_trade_from_health=True)
        
        assert perm.is_action_allowed(ActionType.OPEN_COVERED_CALL) is False
        assert perm.is_action_allowed(ActionType.CLOSE_COVERED_CALL) is False
        assert perm.is_action_allowed(ActionType.ROLL_COVERED_CALL) is False
        assert perm.is_action_allowed(ActionType.DO_NOTHING) is True
        assert perm.code == PermissionCode.BLOCKED_KILL_SWITCH


class TestRiskEngineIntegration:
    """Test that risk_engine enforces trade permission."""

    def test_risk_engine_blocks_open_in_close_only(self):
        """Verify risk_engine.check_action_allowed respects CLOSE_ONLY."""
        from src.risk_engine import check_action_allowed
        
        cfg = MockSettings(trade_mode=TradingMode.CLOSE_ONLY)
        
        mock_portfolio = MagicMock()
        mock_portfolio.equity_usd = 100000.0
        mock_portfolio.margin_used_pct = 20.0
        mock_portfolio.net_delta = 0.5
        mock_portfolio.option_positions = []
        
        mock_state = MagicMock()
        mock_state.portfolio = mock_portfolio
        mock_state.spot = {"BTC": 100000.0}
        mock_state.candidate_options = []
        
        proposed_action = {
            "action": ActionType.OPEN_COVERED_CALL.value,
            "params": {"symbol": "BTC-27DEC25-100000-C", "size": 0.1},
            "reasoning": "Test",
        }
        
        with patch("src.healthcheck.get_cached_health_status", return_value=None):
            allowed, reasons = check_action_allowed(mock_state, proposed_action, cfg)
        
        assert allowed is False
        assert any("CLOSE_ONLY" in r for r in reasons)
