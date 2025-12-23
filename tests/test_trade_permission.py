"""Unit tests for trade permission - single source of truth for trade authorization."""
import pytest
from src.ops.trade_permission import (
    compute_trade_permission,
    TradePermission,
    PermissionCode,
)
from src.config import TradingMode
from src.models import ActionType


class MockSettings:
    """Mock settings for testing trade permission."""
    def __init__(
        self,
        kill_switch_enabled: bool = False,
        trade_mode: TradingMode = TradingMode.NORMAL,
    ):
        self.kill_switch_enabled = kill_switch_enabled
        self.trade_mode = trade_mode


class TestComputeTradePermission:
    """Tests for compute_trade_permission function."""

    def test_normal_mode_allows_all_actions(self):
        """Normal mode with can_trade=True allows all actions."""
        cfg = MockSettings(trade_mode=TradingMode.NORMAL)
        perm = compute_trade_permission(cfg, can_trade_from_health=True)
        
        assert perm.allow_open is True
        assert perm.allow_roll is True
        assert perm.allow_close is True
        assert perm.allow_do_nothing is True
        assert perm.code == PermissionCode.ALLOWED

    def test_kill_switch_blocks_all_except_do_nothing(self):
        """Kill switch blocks OPEN, ROLL, CLOSE but allows DO_NOTHING."""
        cfg = MockSettings(kill_switch_enabled=True)
        perm = compute_trade_permission(cfg, can_trade_from_health=True)
        
        assert perm.allow_open is False
        assert perm.allow_roll is False
        assert perm.allow_close is False
        assert perm.allow_do_nothing is True
        assert perm.is_action_allowed(ActionType.DO_NOTHING) is True
        assert perm.code == PermissionCode.BLOCKED_KILL_SWITCH

    def test_close_only_mode_blocks_open_allows_close(self):
        """CLOSE_ONLY mode blocks OPEN and ROLL, allows CLOSE."""
        cfg = MockSettings(trade_mode=TradingMode.CLOSE_ONLY)
        perm = compute_trade_permission(cfg, can_trade_from_health=True)
        
        assert perm.allow_open is False
        assert perm.allow_roll is False
        assert perm.allow_close is True
        assert perm.code == PermissionCode.BLOCKED_TRADE_MODE_CLOSE_ONLY

    def test_can_trade_false_blocks_open_allows_close(self):
        """can_trade=False blocks OPEN/ROLL, allows CLOSE for risk reduction."""
        cfg = MockSettings(trade_mode=TradingMode.NORMAL)
        perm = compute_trade_permission(cfg, can_trade_from_health=False)
        
        assert perm.allow_open is False
        assert perm.allow_roll is False
        assert perm.allow_close is True  # Risk reduction allowed
        assert perm.code == PermissionCode.BLOCKED_CAN_TRADE_FALSE

    def test_can_trade_none_defaults_to_allowed(self):
        """can_trade=None (health unavailable) defaults to allowed."""
        cfg = MockSettings(trade_mode=TradingMode.NORMAL)
        perm = compute_trade_permission(cfg, can_trade_from_health=None)
        
        assert perm.allow_open is True
        assert perm.code == PermissionCode.ALLOWED

    def test_priority_kill_switch_over_trade_mode(self):
        """Kill switch takes priority over trade_mode setting."""
        cfg = MockSettings(
            kill_switch_enabled=True,
            trade_mode=TradingMode.NORMAL,
        )
        perm = compute_trade_permission(cfg)
        
        assert perm.code == PermissionCode.BLOCKED_KILL_SWITCH
        assert perm.effective_trade_mode == TradingMode.HALT


class TestIsActionAllowed:
    """Tests for TradePermission.is_action_allowed method."""

    def test_do_nothing_always_allowed_even_with_kill_switch(self):
        """DO_NOTHING is always allowed regardless of mode."""
        cfg = MockSettings(kill_switch_enabled=True)
        perm = compute_trade_permission(cfg)
        
        assert perm.is_action_allowed(ActionType.DO_NOTHING) is True

    def test_action_type_mapping(self):
        """Verify all action types are correctly mapped."""
        cfg = MockSettings(trade_mode=TradingMode.CLOSE_ONLY)
        perm = compute_trade_permission(cfg)
        
        assert perm.is_action_allowed(ActionType.OPEN_COVERED_CALL) is False
        assert perm.is_action_allowed(ActionType.ROLL_COVERED_CALL) is False
        assert perm.is_action_allowed(ActionType.CLOSE_COVERED_CALL) is True
        assert perm.is_action_allowed(ActionType.DO_NOTHING) is True
