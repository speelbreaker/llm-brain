"""
Tests for the rolling drawdown module.
"""
from __future__ import annotations

import time
import pytest
from unittest.mock import patch

from src.ops.rolling_drawdown import (
    record_equity,
    compute_rolling_drawdown,
    check_drawdown_breach,
    get_rolling_drawdown_status,
    reset_rolling_drawdown,
    reset_close_only_trigger,
    DrawdownState,
)


@pytest.fixture(autouse=True)
def reset_state():
    """Reset rolling drawdown state before each test."""
    reset_rolling_drawdown(clear_close_only_trigger=True)
    yield
    reset_rolling_drawdown(clear_close_only_trigger=True)


class TestRollingDrawdown:
    """Tests for rolling drawdown monitoring."""
    
    def test_no_snapshots_returns_zero(self):
        """With no snapshots, drawdown should be 0."""
        dd_pct, peak, trough, current = compute_rolling_drawdown(window_days=7)
        
        assert dd_pct == 0.0
        assert peak == 0.0
    
    def test_single_snapshot_no_drawdown(self):
        """Single snapshot means no drawdown."""
        record_equity(100000.0)
        
        dd_pct, peak, trough, current = compute_rolling_drawdown(window_days=7)
        
        assert dd_pct == 0.0
        assert peak == 100000.0
        assert current == 100000.0
    
    def test_drawdown_calculation(self):
        """Drawdown should be calculated correctly."""
        record_equity(100000.0)  # Peak
        record_equity(95000.0)   # Down 5%
        
        dd_pct, peak, trough, current = compute_rolling_drawdown(window_days=7)
        
        assert dd_pct == 5.0
        assert peak == 100000.0
        assert current == 95000.0
    
    def test_drawdown_tracks_peak_to_current(self):
        """Drawdown should track from peak to current, not peak to trough."""
        record_equity(100000.0)  # Peak
        record_equity(90000.0)   # Trough
        record_equity(95000.0)   # Recovery (current)
        
        dd_pct, peak, trough, current = compute_rolling_drawdown(window_days=7)
        
        # Drawdown is peak-to-current (5%), not peak-to-trough (10%)
        assert dd_pct == 5.0
        assert peak == 100000.0
        assert trough == 90000.0
        assert current == 95000.0
    
    def test_disabled_when_window_zero(self):
        """With window_days=0, should be disabled."""
        record_equity(100000.0)
        record_equity(90000.0)
        
        status = check_drawdown_breach(
            window_days=0,
            limit_pct=10.0,
            auto_close_only=True,
        )
        
        assert status.state == DrawdownState.DISABLED
        assert status.is_breached is False
    
    def test_breach_detection(self):
        """Should detect when limit is breached."""
        record_equity(100000.0)
        record_equity(88000.0)  # 12% drawdown
        
        status = check_drawdown_breach(
            window_days=7,
            limit_pct=10.0,
            auto_close_only=False,
        )
        
        assert status.state == DrawdownState.BREACHED
        assert status.is_breached is True
        assert status.current_dd_pct == 12.0
    
    def test_warning_at_70_percent_of_limit(self):
        """Should show warning when approaching limit."""
        record_equity(100000.0)
        record_equity(93000.0)  # 7% drawdown, which is 70% of 10% limit
        
        status = check_drawdown_breach(
            window_days=7,
            limit_pct=10.0,
            auto_close_only=False,
        )
        
        assert status.state == DrawdownState.WARNING
    
    def test_auto_close_only_trigger(self):
        """Auto close-only should be triggered on breach."""
        record_equity(100000.0)
        record_equity(85000.0)  # 15% drawdown
        
        with patch('src.ops.rolling_drawdown._trigger_close_only_mode') as mock_trigger:
            status = check_drawdown_breach(
                window_days=7,
                limit_pct=10.0,
                auto_close_only=True,
            )
            
            assert status.is_breached is True
            assert status.triggered_close_only is True
            mock_trigger.assert_called_once()
    
    def test_auto_close_only_only_triggers_once(self):
        """Auto close-only should only trigger once."""
        record_equity(100000.0)
        record_equity(85000.0)  # 15% drawdown
        
        with patch('src.ops.rolling_drawdown._trigger_close_only_mode') as mock_trigger:
            # First check triggers
            check_drawdown_breach(
                window_days=7,
                limit_pct=10.0,
                auto_close_only=True,
            )
            # Second check should not trigger again
            check_drawdown_breach(
                window_days=7,
                limit_pct=10.0,
                auto_close_only=True,
            )
            
            # Should only be called once
            mock_trigger.assert_called_once()
    
    def test_reset_clears_trigger(self):
        """Reset should clear the close-only trigger."""
        record_equity(100000.0)
        record_equity(85000.0)
        
        with patch('src.ops.rolling_drawdown._trigger_close_only_mode'):
            check_drawdown_breach(
                window_days=7,
                limit_pct=10.0,
                auto_close_only=True,
            )
        
        # Now reset
        reset_rolling_drawdown(clear_close_only_trigger=True)
        
        # Trigger flag should be cleared
        status = get_rolling_drawdown_status(window_days=7, limit_pct=10.0)
        assert status.triggered_close_only is False
    
    def test_to_dict_serialization(self):
        """Status should serialize to dict correctly."""
        record_equity(100000.0)
        record_equity(95000.0)
        
        status = get_rolling_drawdown_status(window_days=7, limit_pct=10.0)
        d = status.to_dict()
        
        assert isinstance(d, dict)
        assert "state" in d
        assert "current_dd_pct" in d
        assert "peak_equity" in d
        assert d["current_dd_pct"] == 5.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

