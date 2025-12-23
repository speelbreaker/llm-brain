"""
Rolling Drawdown Monitor Module.

Tracks N-day rolling drawdown and can trigger auto close-only mode.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Optional, List, Tuple


class DrawdownState(str, Enum):
    """Rolling drawdown health states."""
    OK = "ok"
    WARNING = "warning"  # Approaching limit (>70% of threshold)
    BREACHED = "breached"
    DISABLED = "disabled"


@dataclass
class RollingDrawdownStatus:
    """Status of rolling drawdown."""
    state: DrawdownState
    current_dd_pct: float
    limit_pct: float
    peak_equity: float
    current_equity: float
    trough_equity: float
    window_days: int
    is_breached: bool
    triggered_close_only: bool
    message: str
    
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "state": self.state.value,
            "current_dd_pct": round(self.current_dd_pct, 2),
            "limit_pct": self.limit_pct,
            "peak_equity": round(self.peak_equity, 2),
            "current_equity": round(self.current_equity, 2),
            "trough_equity": round(self.trough_equity, 2),
            "window_days": self.window_days,
            "is_breached": self.is_breached,
            "triggered_close_only": self.triggered_close_only,
            "message": self.message,
        }


# Module-level state: list of (timestamp, equity) tuples
_equity_snapshots: List[Tuple[float, float]] = []
_triggered_close_only: bool = False


def record_equity(equity_usd: float) -> None:
    """
    Record an equity snapshot.
    Call this each loop iteration with the current portfolio equity.
    """
    _equity_snapshots.append((time.time(), equity_usd))


def _cleanup_old_snapshots(window_days: int) -> None:
    """Remove snapshots outside the rolling window."""
    if window_days <= 0:
        return
    
    cutoff = time.time() - (window_days * 24 * 60 * 60)
    while _equity_snapshots and _equity_snapshots[0][0] < cutoff:
        _equity_snapshots.pop(0)


def compute_rolling_drawdown(window_days: int) -> Tuple[float, float, float, float]:
    """
    Compute rolling drawdown from equity snapshots.
    
    Returns:
        Tuple of (current_dd_pct, peak, trough, current)
        - current_dd_pct: Drawdown from peak to current as percentage
        - peak: Maximum equity in window
        - trough: Minimum equity in window (for reference)
        - current: Most recent equity
    """
    if not _equity_snapshots:
        return 0.0, 0.0, 0.0, 0.0
    
    _cleanup_old_snapshots(window_days)
    
    if not _equity_snapshots:
        return 0.0, 0.0, 0.0, 0.0
    
    equities = [eq for _, eq in _equity_snapshots]
    peak = max(equities)
    trough = min(equities)
    current = equities[-1]
    
    if peak <= 0:
        return 0.0, peak, trough, current
    
    # Drawdown is peak-to-current (not peak-to-trough)
    dd_pct = ((peak - current) / peak) * 100.0
    
    return dd_pct, peak, trough, current


def check_drawdown_breach(
    window_days: int,
    limit_pct: float,
    auto_close_only: bool,
) -> RollingDrawdownStatus:
    """
    Check if rolling drawdown has breached the limit.
    
    Args:
        window_days: Rolling window in days. 0 means disabled.
        limit_pct: Maximum allowed drawdown percentage.
        auto_close_only: If True and breached, trigger close-only mode.
    
    Returns:
        RollingDrawdownStatus with current state.
    """
    global _triggered_close_only
    
    if window_days <= 0:
        return RollingDrawdownStatus(
            state=DrawdownState.DISABLED,
            current_dd_pct=0.0,
            limit_pct=limit_pct,
            peak_equity=0.0,
            current_equity=0.0,
            trough_equity=0.0,
            window_days=window_days,
            is_breached=False,
            triggered_close_only=False,
            message="Rolling drawdown monitoring disabled",
        )
    
    dd_pct, peak, trough, current = compute_rolling_drawdown(window_days)
    
    is_breached = dd_pct >= limit_pct
    
    # Auto-trigger close-only if breached and enabled
    if is_breached and auto_close_only and not _triggered_close_only:
        _triggered_close_only = True
        _trigger_close_only_mode()
    
    if is_breached:
        state = DrawdownState.BREACHED
        message = f"BREACHED: {dd_pct:.1f}% drawdown exceeds {limit_pct:.1f}% limit"
    elif dd_pct >= (limit_pct * 0.7):
        state = DrawdownState.WARNING
        message = f"Warning: {dd_pct:.1f}% drawdown approaching {limit_pct:.1f}% limit"
    else:
        state = DrawdownState.OK
        message = f"OK: {dd_pct:.1f}% drawdown (limit: {limit_pct:.1f}%)"
    
    return RollingDrawdownStatus(
        state=state,
        current_dd_pct=dd_pct,
        limit_pct=limit_pct,
        peak_equity=peak,
        current_equity=current,
        trough_equity=trough,
        window_days=window_days,
        is_breached=is_breached,
        triggered_close_only=_triggered_close_only,
        message=message,
    )


def _trigger_close_only_mode() -> None:
    """
    Trigger close-only mode by setting the trade_mode.
    This is imported dynamically to avoid circular imports.
    """
    try:
        from src.config import settings, TradingMode
        if settings.trade_mode != TradingMode.CLOSE_ONLY and settings.trade_mode != TradingMode.HALT:
            settings.trade_mode = TradingMode.CLOSE_ONLY
            print(f"[ROLLING DRAWDOWN] AUTO-TRIGGERED close_only mode due to drawdown breach")
    except Exception as e:
        print(f"[ROLLING DRAWDOWN] Failed to trigger close_only mode: {e}")


def get_rolling_drawdown_status(
    window_days: int,
    limit_pct: float,
) -> RollingDrawdownStatus:
    """
    Get current rolling drawdown status without triggering auto-actions.
    """
    if window_days <= 0:
        return RollingDrawdownStatus(
            state=DrawdownState.DISABLED,
            current_dd_pct=0.0,
            limit_pct=limit_pct,
            peak_equity=0.0,
            current_equity=0.0,
            trough_equity=0.0,
            window_days=window_days,
            is_breached=False,
            triggered_close_only=_triggered_close_only,
            message="Rolling drawdown monitoring disabled",
        )
    
    dd_pct, peak, trough, current = compute_rolling_drawdown(window_days)
    is_breached = dd_pct >= limit_pct
    
    if is_breached:
        state = DrawdownState.BREACHED
        message = f"BREACHED: {dd_pct:.1f}% drawdown exceeds {limit_pct:.1f}% limit"
    elif dd_pct >= (limit_pct * 0.7):
        state = DrawdownState.WARNING
        message = f"Warning: {dd_pct:.1f}% drawdown approaching {limit_pct:.1f}% limit"
    else:
        state = DrawdownState.OK
        message = f"OK: {dd_pct:.1f}% drawdown (limit: {limit_pct:.1f}%)"
    
    return RollingDrawdownStatus(
        state=state,
        current_dd_pct=dd_pct,
        limit_pct=limit_pct,
        peak_equity=peak,
        current_equity=current,
        trough_equity=trough,
        window_days=window_days,
        is_breached=is_breached,
        triggered_close_only=_triggered_close_only,
        message=message,
    )


def reset_rolling_drawdown(clear_close_only_trigger: bool = True) -> None:
    """
    Reset rolling drawdown state.
    
    Args:
        clear_close_only_trigger: If True, also reset the close-only trigger flag.
    """
    global _equity_snapshots, _triggered_close_only
    _equity_snapshots.clear()
    if clear_close_only_trigger:
        _triggered_close_only = False


def reset_close_only_trigger() -> None:
    """
    Reset only the close-only trigger flag without clearing equity history.
    Use this when manually resuming normal trading after a drawdown event.
    """
    global _triggered_close_only
    _triggered_close_only = False

