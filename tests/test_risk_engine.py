"""
Tests for risk_engine module, specifically the kill switch and daily drawdown guard.
"""
from __future__ import annotations

from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Any, List
import pytest

from src.risk_engine import (
    check_action_allowed,
    _check_daily_drawdown_limit,
    _daily_drawdown_state,
)
from src.models import ActionType
from src.config import Settings


@dataclass
class MockPosition:
    symbol: str = "BTC-20DEC24-100000-C"
    side: Any = None
    size: float = 0.1


@dataclass
class MockPortfolio:
    equity_usd: float = 10000.0
    margin_used_pct: float = 20.0
    net_delta: float = 0.0
    option_positions: List[MockPosition] = field(default_factory=list)
    spot_positions: dict = field(default_factory=lambda: {"BTC": 1.0, "ETH": 10.0})


@dataclass
class MockVolState:
    btc_iv: float = 0.5
    btc_ivrv: float = 1.2
    eth_iv: float = 0.5
    eth_ivrv: float = 1.2


@dataclass
class MockCandidate:
    symbol: str = "BTC-20DEC24-100000-C"


@dataclass
class MockAgentState:
    portfolio: MockPortfolio = field(default_factory=MockPortfolio)
    candidate_options: list = field(default_factory=list)
    vol_state: MockVolState = field(default_factory=MockVolState)
    spot: dict = field(default_factory=lambda: {"BTC": 100000.0, "ETH": 3500.0})


def _reset_drawdown_state():
    """Reset the module-level drawdown state for clean tests."""
    _daily_drawdown_state["date"] = None
    _daily_drawdown_state["max_equity_usd"] = 0.0


class TestKillSwitch:
    """Tests for the global kill switch."""

    def test_kill_switch_blocks_open_covered_call(self):
        """Kill switch should block OPEN_COVERED_CALL action."""
        _reset_drawdown_state()
        cfg = Settings(kill_switch_enabled=True)
        state = MockAgentState()
        action = {"action": "OPEN_COVERED_CALL", "params": {"symbol": "BTC-20DEC24-100000-C", "size": 0.1}}
        
        allowed, reasons = check_action_allowed(state, action, config=cfg)
        
        assert allowed is False
        assert any("kill-switch" in r.lower() for r in reasons)

    def test_kill_switch_allows_do_nothing(self):
        """Kill switch should allow DO_NOTHING action."""
        _reset_drawdown_state()
        cfg = Settings(kill_switch_enabled=True)
        state = MockAgentState()
        action = {"action": "DO_NOTHING", "params": {}}
        
        allowed, reasons = check_action_allowed(state, action, config=cfg)
        
        assert allowed is True

    def test_kill_switch_blocks_close_covered_call(self):
        """Kill switch should block all trading actions including CLOSE_COVERED_CALL."""
        _reset_drawdown_state()
        cfg = Settings(kill_switch_enabled=True)
        state = MockAgentState()
        action = {"action": "CLOSE_COVERED_CALL", "params": {"symbol": "BTC-20DEC24-100000-C"}}
        
        allowed, reasons = check_action_allowed(state, action, config=cfg)
        
        assert allowed is False
        assert any("kill-switch" in r.lower() for r in reasons)

    def test_kill_switch_blocks_roll_covered_call(self):
        """Kill switch should block ROLL_COVERED_CALL action."""
        _reset_drawdown_state()
        cfg = Settings(kill_switch_enabled=True)
        state = MockAgentState()
        action = {"action": "ROLL_COVERED_CALL", "params": {"from_symbol": "A", "to_symbol": "B"}}
        
        allowed, reasons = check_action_allowed(state, action, config=cfg)
        
        assert allowed is False
        assert any("kill-switch" in r.lower() for r in reasons)

    def test_kill_switch_disabled_allows_trading(self):
        """When kill switch is disabled, trading should be allowed (subject to other checks)."""
        _reset_drawdown_state()
        cfg = Settings(kill_switch_enabled=False)
        state = MockAgentState()
        state.portfolio.equity_usd = 10000.0
        state.portfolio.margin_used_pct = 20.0
        state.portfolio.spot_positions = {"BTC": 1.0}
        action = {"action": "OPEN_COVERED_CALL", "params": {"symbol": "BTC-20DEC24-100000-C", "size": 0.1}}
        
        allowed, reasons = check_action_allowed(state, action, config=cfg)
        
        assert allowed is True

    def test_kill_switch_overrides_training_mode(self):
        """Kill switch should still work even in training mode on testnet."""
        _reset_drawdown_state()
        cfg = Settings(
            kill_switch_enabled=True,
            mode="research",
            training_mode=True,
            deribit_env="testnet",
        )
        state = MockAgentState()
        action = {"action": "OPEN_COVERED_CALL", "params": {"symbol": "BTC-20DEC24-100000-C", "size": 0.1}}
        
        allowed, reasons = check_action_allowed(state, action, config=cfg)
        
        assert allowed is False
        assert any("kill-switch" in r.lower() for r in reasons)


class TestDailyDrawdownGuard:
    """Tests for the daily drawdown guard."""

    def test_drawdown_guard_disabled_when_zero(self):
        """Drawdown guard should not block when limit is 0 (disabled)."""
        _reset_drawdown_state()
        cfg = Settings(daily_drawdown_limit_pct=0.0)
        state = MockAgentState()
        state.portfolio.equity_usd = 10000.0
        state.portfolio.spot_positions = {"BTC": 1.0}
        action = {"action": "OPEN_COVERED_CALL", "params": {"symbol": "BTC-20DEC24-100000-C", "size": 0.1}}
        
        allowed, reasons = check_action_allowed(state, action, config=cfg)
        
        assert allowed is True
        assert not any("drawdown" in r.lower() for r in reasons)

    def test_drawdown_guard_sets_baseline_on_first_call(self):
        """First call should set the baseline equity and allow trading."""
        _reset_drawdown_state()
        cfg = Settings(daily_drawdown_limit_pct=10.0)
        portfolio = MockPortfolio(equity_usd=10000.0)
        reasons = []
        
        result = _check_daily_drawdown_limit(portfolio, cfg, reasons)
        
        assert result is True
        assert _daily_drawdown_state["max_equity_usd"] == 10000.0

    def test_drawdown_guard_allows_when_equity_stable(self):
        """Trading should be allowed when equity is stable."""
        _reset_drawdown_state()
        cfg = Settings(daily_drawdown_limit_pct=10.0)
        state = MockAgentState()
        state.portfolio.equity_usd = 10000.0
        state.portfolio.spot_positions = {"BTC": 1.0}
        action = {"action": "OPEN_COVERED_CALL", "params": {"symbol": "BTC-20DEC24-100000-C", "size": 0.1}}
        
        check_action_allowed(state, action, config=cfg)
        
        allowed, reasons = check_action_allowed(state, action, config=cfg)
        
        assert allowed is True

    def test_drawdown_guard_blocks_after_limit_exceeded(self):
        """Trading should be blocked after drawdown limit is exceeded."""
        _reset_drawdown_state()
        cfg = Settings(daily_drawdown_limit_pct=10.0)
        portfolio = MockPortfolio(equity_usd=10000.0)
        reasons = []
        
        _check_daily_drawdown_limit(portfolio, cfg, reasons)
        
        portfolio.equity_usd = 8500.0
        reasons = []
        result = _check_daily_drawdown_limit(portfolio, cfg, reasons)
        
        assert result is False
        assert any("drawdown" in r.lower() for r in reasons)

    def test_drawdown_guard_allows_close_after_limit_exceeded(self):
        """CLOSE_COVERED_CALL should be allowed even after drawdown limit exceeded."""
        _reset_drawdown_state()
        cfg = Settings(daily_drawdown_limit_pct=10.0)
        state = MockAgentState()
        state.portfolio.equity_usd = 10000.0
        state.portfolio.spot_positions = {"BTC": 1.0}
        
        from src.models import Side
        state.portfolio.option_positions = [MockPosition(symbol="BTC-20DEC24-100000-C", side=Side.SELL, size=0.1)]
        
        open_action = {"action": "OPEN_COVERED_CALL", "params": {"symbol": "BTC-20DEC24-100000-C", "size": 0.1}}
        check_action_allowed(state, open_action, config=cfg)
        
        state.portfolio.equity_usd = 8500.0
        
        close_action = {"action": "CLOSE_COVERED_CALL", "params": {"symbol": "BTC-20DEC24-100000-C"}}
        allowed, reasons = check_action_allowed(state, close_action, config=cfg)
        
        assert allowed is True
        assert not any("drawdown" in r.lower() for r in reasons)

    def test_drawdown_guard_allows_do_nothing_after_limit_exceeded(self):
        """DO_NOTHING should always be allowed, even after drawdown limit exceeded."""
        _reset_drawdown_state()
        cfg = Settings(daily_drawdown_limit_pct=10.0)
        state = MockAgentState()
        state.portfolio.equity_usd = 10000.0
        state.portfolio.spot_positions = {"BTC": 1.0}
        
        open_action = {"action": "OPEN_COVERED_CALL", "params": {"symbol": "BTC-20DEC24-100000-C", "size": 0.1}}
        check_action_allowed(state, open_action, config=cfg)
        
        state.portfolio.equity_usd = 8000.0
        
        action = {"action": "DO_NOTHING", "params": {}}
        allowed, reasons = check_action_allowed(state, action, config=cfg)
        
        assert allowed is True

    def test_drawdown_guard_blocks_roll_after_limit_exceeded(self):
        """ROLL_COVERED_CALL should be blocked after drawdown limit exceeded (tested via helper)."""
        _reset_drawdown_state()
        cfg = Settings(daily_drawdown_limit_pct=10.0)
        portfolio = MockPortfolio(equity_usd=10000.0)
        reasons = []
        
        _check_daily_drawdown_limit(portfolio, cfg, reasons)
        
        portfolio.equity_usd = 8500.0
        reasons = []
        result = _check_daily_drawdown_limit(portfolio, cfg, reasons)
        
        assert result is False
        assert any("drawdown" in r.lower() for r in reasons)

    def test_drawdown_guard_updates_peak_equity(self):
        """Peak equity should be updated when equity increases."""
        _reset_drawdown_state()
        cfg = Settings(daily_drawdown_limit_pct=10.0)
        portfolio = MockPortfolio(equity_usd=10000.0)
        reasons = []
        
        _check_daily_drawdown_limit(portfolio, cfg, reasons)
        assert _daily_drawdown_state["max_equity_usd"] == 10000.0
        
        portfolio.equity_usd = 11000.0
        _check_daily_drawdown_limit(portfolio, cfg, reasons)
        assert _daily_drawdown_state["max_equity_usd"] == 11000.0
        
        portfolio.equity_usd = 10000.0
        result = _check_daily_drawdown_limit(portfolio, cfg, reasons)
        assert result is True
        assert _daily_drawdown_state["max_equity_usd"] == 11000.0

    def test_drawdown_guard_skipped_in_training_on_testnet(self):
        """Drawdown guard should be skipped in training mode on testnet."""
        _reset_drawdown_state()
        cfg = Settings(
            daily_drawdown_limit_pct=10.0,
            mode="research",
            training_mode=True,
            deribit_env="testnet",
        )
        state = MockAgentState()
        state.portfolio.equity_usd = 10000.0
        state.portfolio.spot_positions = {"BTC": 1.0}
        
        open_action = {"action": "OPEN_COVERED_CALL", "params": {"symbol": "BTC-20DEC24-100000-C", "size": 0.1}}
        check_action_allowed(state, open_action, config=cfg)
        
        state.portfolio.equity_usd = 5000.0
        
        allowed, reasons = check_action_allowed(state, open_action, config=cfg)
        
        assert allowed is True
        assert any("training" in r.lower() for r in reasons)


class TestDrawdownHelperFunction:
    """Tests for the _check_daily_drawdown_limit helper function."""

    def test_disabled_when_limit_zero(self):
        """Guard should return True (allow) when limit is 0."""
        _reset_drawdown_state()
        portfolio = MockPortfolio(equity_usd=10000.0)
        cfg = Settings(daily_drawdown_limit_pct=0.0)
        reasons = []
        
        result = _check_daily_drawdown_limit(portfolio, cfg, reasons)
        
        assert result is True
        assert len(reasons) == 0

    def test_allows_first_call(self):
        """First call should set baseline and return True."""
        _reset_drawdown_state()
        portfolio = MockPortfolio(equity_usd=10000.0)
        cfg = Settings(daily_drawdown_limit_pct=10.0)
        reasons = []
        
        result = _check_daily_drawdown_limit(portfolio, cfg, reasons)
        
        assert result is True
        assert _daily_drawdown_state["max_equity_usd"] == 10000.0

    def test_blocks_when_limit_exceeded(self):
        """Should block and append reason when limit exceeded."""
        _reset_drawdown_state()
        portfolio = MockPortfolio(equity_usd=10000.0)
        cfg = Settings(daily_drawdown_limit_pct=10.0)
        reasons = []
        
        _check_daily_drawdown_limit(portfolio, cfg, reasons)
        
        portfolio.equity_usd = 8500.0
        reasons = []
        result = _check_daily_drawdown_limit(portfolio, cfg, reasons)
        
        assert result is False
        assert len(reasons) == 1
        assert "15.00%" in reasons[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
