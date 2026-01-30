"""
Trade Permission Module - Single source of truth for trade authorization.

This module provides the unified "final permission" decision for the live trading path.
It combines:
- Kill switch state
- Trade mode (normal/close_only/halt)
- Health/gates can_trade status

Defense in depth: This is enforced at both:
1. The live loop (pre-execution)
2. The risk/execution layer (so no one can bypass by calling execution directly)
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal, Optional

from src.config import Settings, TradingMode
from src.models import ActionType


class PermissionCode(str, Enum):
    """Codes indicating why an action was allowed or blocked."""
    ALLOWED = "ALLOWED"
    BLOCKED_KILL_SWITCH = "BLOCKED_KILL_SWITCH"
    BLOCKED_TRADE_MODE_HALT = "BLOCKED_TRADE_MODE_HALT"
    BLOCKED_TRADE_MODE_CLOSE_ONLY = "BLOCKED_TRADE_MODE_CLOSE_ONLY"
    BLOCKED_CAN_TRADE_FALSE = "BLOCKED_CAN_TRADE_FALSE"
    BLOCKED_HEALTH_UNAVAILABLE = "BLOCKED_HEALTH_UNAVAILABLE"


@dataclass
class TradePermission:
    """Result of computing trade permission."""
    effective_trade_mode: TradingMode
    allow_open: bool
    allow_roll: bool
    allow_close: bool
    allow_do_nothing: bool  # Always True
    can_trade: bool  # From health/gates
    code: PermissionCode
    reason: str
    
    def is_action_allowed(self, action_type: ActionType) -> bool:
        """Check if a specific action type is allowed."""
        if action_type == ActionType.DO_NOTHING:
            return self.allow_do_nothing
        elif action_type == ActionType.OPEN_COVERED_CALL:
            return self.allow_open
        elif action_type == ActionType.ROLL_COVERED_CALL:
            return self.allow_roll
        elif action_type == ActionType.CLOSE_COVERED_CALL:
            return self.allow_close
        return False


def compute_trade_permission(
    cfg: Settings,
    can_trade_from_health: Optional[bool] = None,
) -> TradePermission:
    """
    Compute the effective trade permission based on all control factors.
    
    This is THE single source of truth for whether trading is allowed.
    
    Args:
        cfg: Settings instance with kill_switch_enabled and trade_mode
        can_trade_from_health: Optional boolean from health/gates system.
            If None, defaults to True (health not blocking).
    
    Returns:
        TradePermission with effective permissions and reason codes.
    
    Priority order (highest to lowest):
    1. Kill switch (blocks everything except DO_NOTHING)
    2. Trade mode = halt (blocks everything except DO_NOTHING)
    3. Trade mode = close_only (blocks OPEN and ROLL)
    4. Health/gates can_trade=False (blocks OPEN and ROLL, allows CLOSE)
    5. Trade mode = normal (allows all, subject to other risk checks)
    """
    # Default: allow everything
    allow_open = True
    allow_roll = True
    allow_close = True
    allow_do_nothing = True  # Always true
    
    # Resolve can_trade from health (None means health isn't blocking)
    can_trade = can_trade_from_health if can_trade_from_health is not None else True
    
    # Start with normal mode
    effective_mode = cfg.trade_mode
    code = PermissionCode.ALLOWED
    reason = "Trading allowed"
    
    # Priority 1: Kill switch overrides everything
    if cfg.kill_switch_enabled:
        allow_open = False
        allow_roll = False
        allow_close = False
        effective_mode = TradingMode.HALT
        code = PermissionCode.BLOCKED_KILL_SWITCH
        reason = "Kill-switch enabled - all trading blocked"
        return TradePermission(
            effective_trade_mode=effective_mode,
            allow_open=allow_open,
            allow_roll=allow_roll,
            allow_close=allow_close,
            allow_do_nothing=allow_do_nothing,
            can_trade=can_trade,
            code=code,
            reason=reason,
        )
    
    # Priority 2: Trade mode = halt
    if cfg.trade_mode == TradingMode.HALT:
        allow_open = False
        allow_roll = False
        allow_close = False
        code = PermissionCode.BLOCKED_TRADE_MODE_HALT
        reason = "Trade mode is HALT - all trading blocked"
        return TradePermission(
            effective_trade_mode=effective_mode,
            allow_open=allow_open,
            allow_roll=allow_roll,
            allow_close=allow_close,
            allow_do_nothing=allow_do_nothing,
            can_trade=can_trade,
            code=code,
            reason=reason,
        )
    
    # Priority 3: Trade mode = close_only
    if cfg.trade_mode == TradingMode.CLOSE_ONLY:
        allow_open = False
        allow_roll = False
        # allow_close remains True
        code = PermissionCode.BLOCKED_TRADE_MODE_CLOSE_ONLY
        reason = "Trade mode is CLOSE_ONLY - only CLOSE actions allowed"
        return TradePermission(
            effective_trade_mode=effective_mode,
            allow_open=allow_open,
            allow_roll=allow_roll,
            allow_close=allow_close,
            allow_do_nothing=allow_do_nothing,
            can_trade=can_trade,
            code=code,
            reason=reason,
        )
    
    # Priority 4: Health/gates can_trade=False
    # This blocks OPEN and ROLL but allows CLOSE (to reduce risk)
    if not can_trade:
        allow_open = False
        allow_roll = False
        # allow_close remains True (to reduce risk)
        code = PermissionCode.BLOCKED_CAN_TRADE_FALSE
        reason = "Health/gates can_trade=False - only CLOSE actions allowed"
        return TradePermission(
            effective_trade_mode=effective_mode,
            allow_open=allow_open,
            allow_roll=allow_roll,
            allow_close=allow_close,
            allow_do_nothing=allow_do_nothing,
            can_trade=can_trade,
            code=code,
            reason=reason,
        )
    
    # Priority 5: Normal mode - everything allowed
    return TradePermission(
        effective_trade_mode=effective_mode,
        allow_open=allow_open,
        allow_roll=allow_roll,
        allow_close=allow_close,
        allow_do_nothing=allow_do_nothing,
        can_trade=can_trade,
        code=code,
        reason=reason,
    )


def get_current_trade_permission(cfg: Settings) -> TradePermission:
    """
    Get current trade permission using cached health status.
    
    This is a convenience wrapper that fetches can_trade from the
    cached health status if available.
    """
    from src.healthcheck import get_cached_health_status
    
    cached_health = get_cached_health_status()
    can_trade = None
    
    if cached_health is not None:
        can_trade = cached_health.can_trade
    
    return compute_trade_permission(cfg, can_trade_from_health=can_trade)


def check_action_permission(
    action_type: ActionType,
    cfg: Settings,
    can_trade_from_health: Optional[bool] = None,
) -> tuple[bool, PermissionCode, str]:
    """
    Check if a specific action is permitted.
    
    Convenience function for checking a single action.
    
    Returns:
        Tuple of (allowed: bool, code: PermissionCode, reason: str)
    """
    permission = compute_trade_permission(cfg, can_trade_from_health)
    allowed = permission.is_action_allowed(action_type)
    return allowed, permission.code, permission.reason

