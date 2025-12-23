"""
Tests for the rate limiter module.
"""
from __future__ import annotations

import time
import pytest

from src.ops.rate_limiter import (
    record_order,
    check_rate_limit,
    get_rate_limit_status,
    reset_rate_limiter,
    RateLimitState,
)


@pytest.fixture(autouse=True)
def reset_state():
    """Reset rate limiter state before each test."""
    reset_rate_limiter()
    yield
    reset_rate_limiter()


class TestRateLimiter:
    """Tests for rate limiting."""
    
    def test_initial_state_allows_orders(self):
        """With no orders placed, should allow more."""
        allowed, status = check_rate_limit(max_orders=10, window_seconds=60)
        
        assert allowed is True
        assert status.state == RateLimitState.OK
        assert status.orders_in_window == 0
        assert status.can_place_order is True
    
    def test_disabled_when_max_is_zero(self):
        """When max_orders is 0, rate limiting should be disabled."""
        allowed, status = check_rate_limit(max_orders=0, window_seconds=60)
        
        assert allowed is True
        assert status.state == RateLimitState.DISABLED
        assert status.can_place_order is True
    
    def test_orders_count_increments(self):
        """Recording orders should increment the count."""
        record_order()
        record_order()
        record_order()
        
        _, status = check_rate_limit(max_orders=10, window_seconds=60)
        assert status.orders_in_window == 3
    
    def test_blocked_when_limit_reached(self):
        """Should block when limit is reached."""
        for _ in range(5):
            record_order()
        
        allowed, status = check_rate_limit(max_orders=5, window_seconds=60)
        
        assert allowed is False
        assert status.state == RateLimitState.BLOCKED
        assert status.can_place_order is False
        assert status.orders_in_window == 5
    
    def test_warning_at_80_percent(self):
        """Should show warning when approaching limit."""
        for _ in range(8):
            record_order()
        
        allowed, status = check_rate_limit(max_orders=10, window_seconds=60)
        
        assert allowed is True
        assert status.state == RateLimitState.WARNING
        assert status.usage_pct == 80.0
    
    def test_orders_expire_after_window(self):
        """Orders should expire after the window."""
        for _ in range(5):
            record_order()
        
        # All orders are in window
        allowed1, status1 = check_rate_limit(max_orders=5, window_seconds=60)
        assert allowed1 is False
        assert status1.orders_in_window == 5
        
        # With a 0.1s window, wait and check again
        time.sleep(0.15)
        allowed2, status2 = check_rate_limit(max_orders=5, window_seconds=0.1)
        assert allowed2 is True
        assert status2.orders_in_window == 0
    
    def test_usage_percentage_calculation(self):
        """Usage percentage should be calculated correctly."""
        record_order()
        record_order()
        
        _, status = check_rate_limit(max_orders=10, window_seconds=60)
        assert status.usage_pct == 20.0
    
    def test_to_dict_serialization(self):
        """RateLimitStatus should serialize to dict correctly."""
        record_order()
        _, status = check_rate_limit(max_orders=10, window_seconds=60)
        
        d = status.to_dict()
        assert isinstance(d, dict)
        assert "state" in d
        assert "orders_in_window" in d
        assert "limit" in d
        assert "can_place_order" in d
        assert d["orders_in_window"] == 1
    
    def test_get_rate_limit_status_alias(self):
        """get_rate_limit_status should work correctly."""
        record_order()
        status = get_rate_limit_status(max_orders=10, window_seconds=60)
        
        assert status.orders_in_window == 1
        assert status.state == RateLimitState.OK


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

