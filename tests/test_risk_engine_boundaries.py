"""
Risk engine boundary tests.
Freeze the behavior of risk limit checks to prevent accidental weakening.

Convention: Allow when metric <= limit, block when metric > limit.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List
import pytest

from src.risk_engine import check_action_allowed, _check_liquidity_limits, _daily_drawdown_state
from src.models import ActionType, Side
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
class MockCandidate:
    symbol: str = "BTC-20DEC24-100000-C"
    spread_pct: float = None
    open_interest: int = None


@dataclass
class MockAgentState:
    portfolio: MockPortfolio = field(default_factory=MockPortfolio)
    candidate_options: list = field(default_factory=list)
    spot: dict = field(default_factory=lambda: {"BTC": 100000.0, "ETH": 3500.0})


def _reset_drawdown_state():
    """Reset the module-level drawdown state for clean tests."""
    _daily_drawdown_state["date"] = None
    _daily_drawdown_state["max_equity_usd"] = 0.0


class TestMaxMarginUsedBoundary:
    """Tests for max_margin_used_pct boundary behavior.
    
    Note: For OPEN_COVERED_CALL actions, there's an additional 90% threshold check.
    The tests here verify the core margin limit behavior.
    """

    def test_margin_below_limit_allows_action(self):
        """Trading allowed when margin < limit."""
        _reset_drawdown_state()
        cfg = Settings(max_margin_used_pct=80.0, kill_switch_enabled=False, mode="production")
        state = MockAgentState()
        state.portfolio.margin_used_pct = 50.0
        state.portfolio.equity_usd = 10000.0
        state.portfolio.spot_positions = {"BTC": 1.0}
        action = {"action": "OPEN_COVERED_CALL", "params": {"symbol": "BTC-20DEC24-100000-C", "size": 0.1}}

        allowed, reasons = check_action_allowed(state, action, config=cfg)

        assert allowed is True
        assert not any("margin" in r.lower() for r in reasons)

    def test_margin_at_exact_limit_blocks_action(self):
        """Trading blocked when margin == limit (>= check)."""
        _reset_drawdown_state()
        cfg = Settings(max_margin_used_pct=80.0, kill_switch_enabled=False, mode="production")
        state = MockAgentState()
        state.portfolio.margin_used_pct = 80.0
        state.portfolio.equity_usd = 10000.0
        state.portfolio.spot_positions = {"BTC": 1.0}
        action = {"action": "OPEN_COVERED_CALL", "params": {"symbol": "BTC-20DEC24-100000-C", "size": 0.1}}

        allowed, reasons = check_action_allowed(state, action, config=cfg)

        assert allowed is False
        assert any("margin" in r.lower() for r in reasons)

    def test_margin_above_limit_blocks_action(self):
        """Trading blocked when margin > limit."""
        _reset_drawdown_state()
        cfg = Settings(max_margin_used_pct=80.0, kill_switch_enabled=False, mode="production")
        state = MockAgentState()
        state.portfolio.margin_used_pct = 85.0
        state.portfolio.equity_usd = 10000.0
        state.portfolio.spot_positions = {"BTC": 1.0}
        action = {"action": "OPEN_COVERED_CALL", "params": {"symbol": "BTC-20DEC24-100000-C", "size": 0.1}}

        allowed, reasons = check_action_allowed(state, action, config=cfg)

        assert allowed is False
        assert any("margin" in r.lower() for r in reasons)

    def test_margin_reason_includes_values(self):
        """Blocking reason includes actual vs limit values."""
        _reset_drawdown_state()
        cfg = Settings(max_margin_used_pct=80.0, kill_switch_enabled=False, mode="production")
        state = MockAgentState()
        state.portfolio.margin_used_pct = 85.0
        state.portfolio.equity_usd = 10000.0
        state.portfolio.spot_positions = {"BTC": 1.0}
        action = {"action": "OPEN_COVERED_CALL", "params": {"symbol": "BTC-20DEC24-100000-C", "size": 0.1}}

        _, reasons = check_action_allowed(state, action, config=cfg)

        margin_reasons = [r for r in reasons if "margin" in r.lower()]
        assert len(margin_reasons) >= 1
        assert "85" in margin_reasons[0]
        assert "80" in margin_reasons[0]


class TestMaxNetDeltaBoundary:
    """Tests for max_net_delta_abs boundary behavior."""

    def test_delta_below_limit_allows_action(self):
        """Trading allowed when |delta| < limit."""
        _reset_drawdown_state()
        cfg = Settings(max_net_delta_abs=5.0, kill_switch_enabled=False, mode="production")
        state = MockAgentState()
        state.portfolio.net_delta = 4.0
        state.portfolio.equity_usd = 10000.0
        state.portfolio.spot_positions = {"BTC": 1.0}
        action = {"action": "OPEN_COVERED_CALL", "params": {"symbol": "BTC-20DEC24-100000-C", "size": 0.1}}

        allowed, reasons = check_action_allowed(state, action, config=cfg)

        assert allowed is True
        assert not any("delta" in r.lower() for r in reasons)

    def test_delta_at_exact_limit_allows_action(self):
        """Trading allowed when |delta| == limit (> check, not >=)."""
        _reset_drawdown_state()
        cfg = Settings(max_net_delta_abs=5.0, kill_switch_enabled=False, mode="production")
        state = MockAgentState()
        state.portfolio.net_delta = 5.0
        state.portfolio.equity_usd = 10000.0
        state.portfolio.spot_positions = {"BTC": 1.0}
        action = {"action": "OPEN_COVERED_CALL", "params": {"symbol": "BTC-20DEC24-100000-C", "size": 0.1}}

        allowed, reasons = check_action_allowed(state, action, config=cfg)

        assert allowed is True
        assert not any("net delta" in r.lower() for r in reasons)

    def test_delta_above_limit_blocks_action(self):
        """Trading blocked when |delta| > limit."""
        _reset_drawdown_state()
        cfg = Settings(max_net_delta_abs=5.0, kill_switch_enabled=False, mode="production")
        state = MockAgentState()
        state.portfolio.net_delta = 6.0
        state.portfolio.equity_usd = 10000.0
        state.portfolio.spot_positions = {"BTC": 1.0}
        action = {"action": "OPEN_COVERED_CALL", "params": {"symbol": "BTC-20DEC24-100000-C", "size": 0.1}}

        allowed, reasons = check_action_allowed(state, action, config=cfg)

        assert allowed is False
        assert any("delta" in r.lower() for r in reasons)

    def test_negative_delta_uses_absolute_value(self):
        """Negative delta should use absolute value for comparison."""
        _reset_drawdown_state()
        cfg = Settings(max_net_delta_abs=5.0, kill_switch_enabled=False, mode="production")
        state = MockAgentState()
        state.portfolio.net_delta = -6.0
        state.portfolio.equity_usd = 10000.0
        state.portfolio.spot_positions = {"BTC": 1.0}
        action = {"action": "OPEN_COVERED_CALL", "params": {"symbol": "BTC-20DEC24-100000-C", "size": 0.1}}

        allowed, reasons = check_action_allowed(state, action, config=cfg)

        assert allowed is False
        assert any("delta" in r.lower() for r in reasons)


class TestMaxExpiryExposureBoundary:
    """Tests for max_expiry_exposure boundary behavior.
    
    Note: effective_max_expiry_exposure differs between research (1.0) and production (0.3).
    These tests use production mode to test the production limit.
    """

    def test_expiry_exposure_below_limit_allows_action(self):
        """Trading allowed when expiry exposure < limit."""
        _reset_drawdown_state()
        cfg = Settings(max_expiry_exposure=0.3, kill_switch_enabled=False, mode="production")
        state = MockAgentState()
        state.portfolio.equity_usd = 10000.0
        state.portfolio.spot_positions = {"BTC": 1.0}
        state.portfolio.option_positions = []
        action = {"action": "OPEN_COVERED_CALL", "params": {"symbol": "BTC-20DEC24-100000-C", "size": 0.1}}

        allowed, reasons = check_action_allowed(state, action, config=cfg)

        assert allowed is True

    def test_expiry_exposure_at_exact_limit_blocks_action(self):
        """Trading blocked when expiry exposure == limit (>= check)."""
        _reset_drawdown_state()
        cfg = Settings(max_expiry_exposure=0.3, kill_switch_enabled=False, mode="production")
        state = MockAgentState()
        state.portfolio.equity_usd = 10000.0
        state.portfolio.spot_positions = {"BTC": 1.0}
        state.portfolio.option_positions = [
            MockPosition(symbol="BTC-20DEC24-95000-C", side=Side.SELL, size=0.2)
        ]
        action = {"action": "OPEN_COVERED_CALL", "params": {"symbol": "BTC-20DEC24-100000-C", "size": 0.1}}

        allowed, reasons = check_action_allowed(state, action, config=cfg)

        assert allowed is False
        assert any("expiry" in r.lower() for r in reasons)

    def test_expiry_exposure_above_limit_blocks_action(self):
        """Trading blocked when expiry exposure would exceed limit."""
        _reset_drawdown_state()
        cfg = Settings(max_expiry_exposure=0.3, kill_switch_enabled=False, mode="production")
        state = MockAgentState()
        state.portfolio.equity_usd = 10000.0
        state.portfolio.spot_positions = {"BTC": 1.0}
        state.portfolio.option_positions = [
            MockPosition(symbol="BTC-20DEC24-95000-C", side=Side.SELL, size=0.25)
        ]
        action = {"action": "OPEN_COVERED_CALL", "params": {"symbol": "BTC-20DEC24-100000-C", "size": 0.1}}

        allowed, reasons = check_action_allowed(state, action, config=cfg)

        assert allowed is False
        assert any("expiry" in r.lower() for r in reasons)

    def test_expiry_exposure_reason_includes_values(self):
        """Blocking reason includes projected vs limit values."""
        _reset_drawdown_state()
        cfg = Settings(max_expiry_exposure=0.3, kill_switch_enabled=False, mode="production")
        state = MockAgentState()
        state.portfolio.equity_usd = 10000.0
        state.portfolio.spot_positions = {"BTC": 1.0}
        state.portfolio.option_positions = [
            MockPosition(symbol="BTC-20DEC24-95000-C", side=Side.SELL, size=0.25)
        ]
        action = {"action": "OPEN_COVERED_CALL", "params": {"symbol": "BTC-20DEC24-100000-C", "size": 0.1}}

        _, reasons = check_action_allowed(state, action, config=cfg)

        expiry_reasons = [r for r in reasons if "expiry" in r.lower()]
        assert len(expiry_reasons) >= 1
        assert "0.35" in expiry_reasons[0]
        assert "0.3" in expiry_reasons[0]


class TestLiquidityGuardsBoundary:
    """Tests for liquidity guard helper function."""

    def test_spread_below_limit_allows(self):
        """Candidate allowed when spread_pct < limit."""
        cfg = Settings(liquidity_max_spread_pct=5.0, liquidity_min_open_interest=50)
        candidate = MockCandidate(spread_pct=4.0, open_interest=100)

        reasons = _check_liquidity_limits(cfg, candidate)

        assert len(reasons) == 0

    def test_spread_at_exact_limit_allows(self):
        """Candidate allowed when spread_pct == limit (> check)."""
        cfg = Settings(liquidity_max_spread_pct=5.0, liquidity_min_open_interest=50)
        candidate = MockCandidate(spread_pct=5.0, open_interest=100)

        reasons = _check_liquidity_limits(cfg, candidate)

        assert len(reasons) == 0

    def test_spread_above_limit_blocks(self):
        """Candidate blocked when spread_pct > limit."""
        cfg = Settings(liquidity_max_spread_pct=5.0, liquidity_min_open_interest=50)
        candidate = MockCandidate(spread_pct=6.0, open_interest=100)

        reasons = _check_liquidity_limits(cfg, candidate)

        assert len(reasons) == 1
        assert "spread" in reasons[0].lower()
        assert "6.00" in reasons[0]
        assert "5.00" in reasons[0]

    def test_oi_above_limit_allows(self):
        """Candidate allowed when open_interest >= limit."""
        cfg = Settings(liquidity_max_spread_pct=5.0, liquidity_min_open_interest=50)
        candidate = MockCandidate(spread_pct=3.0, open_interest=50)

        reasons = _check_liquidity_limits(cfg, candidate)

        assert len(reasons) == 0

    def test_oi_below_limit_blocks(self):
        """Candidate blocked when open_interest < limit."""
        cfg = Settings(liquidity_max_spread_pct=5.0, liquidity_min_open_interest=50)
        candidate = MockCandidate(spread_pct=3.0, open_interest=40)

        reasons = _check_liquidity_limits(cfg, candidate)

        assert len(reasons) == 1
        assert "open interest" in reasons[0].lower()
        assert "40" in reasons[0]
        assert "50" in reasons[0]

    def test_both_limits_violated_returns_two_reasons(self):
        """Both violations should be reported."""
        cfg = Settings(liquidity_max_spread_pct=5.0, liquidity_min_open_interest=50)
        candidate = MockCandidate(spread_pct=10.0, open_interest=20)

        reasons = _check_liquidity_limits(cfg, candidate)

        assert len(reasons) == 2
        assert any("spread" in r.lower() for r in reasons)
        assert any("open interest" in r.lower() for r in reasons)

    def test_none_values_are_skipped(self):
        """None values should not trigger checks."""
        cfg = Settings(liquidity_max_spread_pct=5.0, liquidity_min_open_interest=50)
        candidate = MockCandidate(spread_pct=None, open_interest=None)

        reasons = _check_liquidity_limits(cfg, candidate)

        assert len(reasons) == 0


class TestLiquidityIntegration:
    """Integration tests for liquidity guards in check_action_allowed."""

    def test_open_action_blocked_for_illiquid_candidate(self):
        """OPEN_COVERED_CALL should be blocked for illiquid candidate."""
        _reset_drawdown_state()
        cfg = Settings(
            liquidity_max_spread_pct=5.0,
            liquidity_min_open_interest=50,
            kill_switch_enabled=False,
            mode="production",
        )
        candidate = MockCandidate(
            symbol="BTC-20DEC24-100000-C",
            spread_pct=10.0,
            open_interest=20,
        )
        state = MockAgentState()
        state.portfolio.equity_usd = 10000.0
        state.portfolio.spot_positions = {"BTC": 1.0}
        state.candidate_options = [candidate]
        action = {"action": "OPEN_COVERED_CALL", "params": {"symbol": "BTC-20DEC24-100000-C", "size": 0.1}}

        allowed, reasons = check_action_allowed(state, action, config=cfg)

        assert allowed is False
        assert any("liquidity" in r.lower() for r in reasons)

    def test_open_action_allowed_for_liquid_candidate(self):
        """OPEN_COVERED_CALL should be allowed for liquid candidate."""
        _reset_drawdown_state()
        cfg = Settings(
            liquidity_max_spread_pct=5.0,
            liquidity_min_open_interest=50,
            kill_switch_enabled=False,
            mode="production",
        )
        candidate = MockCandidate(
            symbol="BTC-20DEC24-100000-C",
            spread_pct=3.0,
            open_interest=100,
        )
        state = MockAgentState()
        state.portfolio.equity_usd = 10000.0
        state.portfolio.spot_positions = {"BTC": 1.0}
        state.candidate_options = [candidate]
        action = {"action": "OPEN_COVERED_CALL", "params": {"symbol": "BTC-20DEC24-100000-C", "size": 0.1}}

        allowed, reasons = check_action_allowed(state, action, config=cfg)

        assert allowed is True
        assert not any("liquidity" in r.lower() for r in reasons)

    def test_close_action_not_blocked_for_illiquidity(self):
        """CLOSE_COVERED_CALL should NOT be blocked for illiquidity."""
        _reset_drawdown_state()
        cfg = Settings(
            liquidity_max_spread_pct=5.0,
            liquidity_min_open_interest=50,
            kill_switch_enabled=False,
            mode="production",
        )
        state = MockAgentState()
        state.portfolio.equity_usd = 10000.0
        state.portfolio.option_positions = [
            MockPosition(symbol="BTC-20DEC24-100000-C", side=Side.SELL, size=0.1)
        ]
        action = {"action": "CLOSE_COVERED_CALL", "params": {"symbol": "BTC-20DEC24-100000-C"}}

        allowed, reasons = check_action_allowed(state, action, config=cfg)

        assert allowed is True
        assert not any("liquidity" in r.lower() for r in reasons)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
