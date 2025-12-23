"""
Rate Limiter Module.

Tracks order placement frequency and prevents order spam.
The execution layer should call check_rate_limit() before placing orders
and record_order() after successfully placing an order.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


class RateLimitState(str, Enum):
    """Rate limiter health states."""
    OK = "ok"
    WARNING = "warning"  # Approaching limit (>80%)
    BLOCKED = "blocked"
    DISABLED = "disabled"


@dataclass
class RateLimitStatus:
    """Status of the rate limiter."""
    state: RateLimitState
    orders_in_window: int
    limit: int
    window_seconds: int
    can_place_order: bool
    usage_pct: float
    message: str
    
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "state": self.state.value,
            "orders_in_window": self.orders_in_window,
            "limit": self.limit,
            "window_seconds": self.window_seconds,
            "can_place_order": self.can_place_order,
            "usage_pct": round(self.usage_pct, 1),
            "message": self.message,
        }


# Module-level state: deque of order timestamps
_order_timestamps: deque[float] = deque()
_window_seconds: int = 60


def record_order() -> None:
    """
    Record that an order was placed.
    Call this after successfully placing an order.
    """
    _order_timestamps.append(time.time())


def _cleanup_old_timestamps(window_seconds: int) -> None:
    """Remove timestamps outside the current window."""
    cutoff = time.time() - window_seconds
    while _order_timestamps and _order_timestamps[0] < cutoff:
        _order_timestamps.popleft()


def check_rate_limit(max_orders: int, window_seconds: int = 60) -> tuple[bool, RateLimitStatus]:
    """
    Check if placing another order is allowed under the rate limit.
    
    Args:
        max_orders: Maximum orders allowed in the window. 0 means disabled.
        window_seconds: Time window in seconds.
    
    Returns:
        Tuple of (allowed: bool, status: RateLimitStatus)
    """
    if max_orders <= 0:
        return True, RateLimitStatus(
            state=RateLimitState.DISABLED,
            orders_in_window=0,
            limit=0,
            window_seconds=window_seconds,
            can_place_order=True,
            usage_pct=0.0,
            message="Rate limiting disabled",
        )
    
    _cleanup_old_timestamps(window_seconds)
    
    orders_in_window = len(_order_timestamps)
    usage_pct = (orders_in_window / max_orders) * 100 if max_orders > 0 else 0
    can_place = orders_in_window < max_orders
    
    if not can_place:
        state = RateLimitState.BLOCKED
        message = f"Rate limit exceeded: {orders_in_window}/{max_orders} orders in {window_seconds}s"
    elif usage_pct >= 80:
        state = RateLimitState.WARNING
        message = f"Approaching rate limit: {orders_in_window}/{max_orders} orders ({usage_pct:.0f}%)"
    else:
        state = RateLimitState.OK
        message = f"Rate limit OK: {orders_in_window}/{max_orders} orders"
    
    status = RateLimitStatus(
        state=state,
        orders_in_window=orders_in_window,
        limit=max_orders,
        window_seconds=window_seconds,
        can_place_order=can_place,
        usage_pct=usage_pct,
        message=message,
    )
    
    return can_place, status


def get_rate_limit_status(max_orders: int, window_seconds: int = 60) -> RateLimitStatus:
    """
    Get current rate limit status without checking if an order can be placed.
    """
    _, status = check_rate_limit(max_orders, window_seconds)
    return status


def reset_rate_limiter() -> None:
    """
    Reset rate limiter state. Useful for testing.
    """
    _order_timestamps.clear()

