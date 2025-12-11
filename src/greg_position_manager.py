"""
Greg Position Management Engine - Advisory module for open Greg positions.

This module evaluates open positions that were entered via Greg strategies
and produces management suggestions (hedge, profit-take, stop-loss, roll, assign).

IMPORTANT: This is advice-only. No actual orders are sent.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Literal, Optional

from src.models import AgentState, OptionPosition, PortfolioState


GREG_POSITION_RULES_PATH = Path("docs/greg_mandolini/GREG_POSITION_RULES_V1.json")

ManagementAction = Literal["HEDGE", "TAKE_PROFIT", "ROLL", "ASSIGN", "CLOSE", "HOLD"]


@dataclass
class GregManagementSuggestion:
    """A management suggestion for an open Greg position."""
    bot_name: str
    strategy_code: str
    underlying: str
    position_id: str
    summary: str
    action: ManagementAction
    reason: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GregPositionRules:
    """Parsed position management rules from JSON spec."""
    meta: Dict[str, Any]
    calibration: Dict[str, float]
    strategies: Dict[str, Dict[str, Any]]
    
    def get_calibration(self, key: str, default: float = 0.0) -> float:
        return self.calibration.get(key, default)


@lru_cache(maxsize=1)
def load_greg_position_rules() -> GregPositionRules:
    """Load the Greg position management rules from JSON."""
    try:
        with open(GREG_POSITION_RULES_PATH, "r") as f:
            data = json.load(f)
        return GregPositionRules(
            meta=data.get("meta", {}),
            calibration=data.get("calibration", {}),
            strategies=data.get("strategies", {}),
        )
    except Exception as e:
        print(f"[GregPositionManager] Failed to load rules: {e}")
        return GregPositionRules(meta={}, calibration={}, strategies={})


def clear_rules_cache() -> None:
    """Clear the cached rules so they reload on next call."""
    load_greg_position_rules.cache_clear()


def get_greg_position_rules() -> GregPositionRules:
    """Get the Greg position management rules."""
    return load_greg_position_rules()


def _evaluate_straddle_strangle(
    position_id: str,
    underlying: str,
    strategy_code: str,
    net_delta: float,
    dte: int,
    profit_pct: float,
    loss_pct: float,
    rules: GregPositionRules,
) -> Optional[GregManagementSuggestion]:
    """
    Evaluate straddle or strangle position for management actions.
    
    Rules:
    - Hedge if net_delta exceeds threshold
    - Take profit if profit_pct >= 50% of credit
    - Stop loss if loss >= 2x initial credit
    - Close if DTE <= threshold
    """
    is_straddle = "STRADDLE" in strategy_code.upper()
    delta_threshold = rules.get_calibration(
        "straddle_delta_threshold" if is_straddle else "strangle_delta_threshold",
        0.15 if is_straddle else 0.20
    )
    profit_take_pct = rules.get_calibration(
        "straddle_profit_take_pct" if is_straddle else "strangle_profit_take_pct",
        0.50
    )
    stop_loss_multiple = rules.get_calibration(
        "straddle_stop_loss_multiple" if is_straddle else "strangle_stop_loss_multiple",
        2.0
    )
    close_at_dte = int(rules.get_calibration(
        "straddle_close_at_dte" if is_straddle else "strangle_close_at_dte",
        21
    ))
    
    strategy_spec = rules.strategies.get(strategy_code, {})
    display_name = strategy_spec.get("display_name", strategy_code)
    
    metrics = {
        "net_delta": net_delta,
        "target_delta_abs": delta_threshold,
        "dte": dte,
        "profit_pct": profit_pct,
        "loss_pct": loss_pct,
    }
    
    if loss_pct >= stop_loss_multiple:
        return GregManagementSuggestion(
            bot_name="GregBot",
            strategy_code=strategy_code,
            underlying=underlying,
            position_id=position_id,
            summary=f"CLOSE: Loss {loss_pct:.1f}x >= {stop_loss_multiple}x stop-loss threshold",
            action="CLOSE",
            reason=f"Loss has exceeded {stop_loss_multiple}x the initial credit ({loss_pct:.2f}x actual). Close to limit further downside.",
            metrics=metrics,
        )
    
    if profit_pct >= profit_take_pct:
        return GregManagementSuggestion(
            bot_name="GregBot",
            strategy_code=strategy_code,
            underlying=underlying,
            position_id=position_id,
            summary=f"TAKE_PROFIT: {profit_pct*100:.0f}% >= {profit_take_pct*100:.0f}% target",
            action="TAKE_PROFIT",
            reason=f"Profit of {profit_pct*100:.1f}% exceeds take-profit threshold of {profit_take_pct*100:.0f}%. Close to lock in gains.",
            metrics=metrics,
        )
    
    net_delta_abs = abs(net_delta)
    if net_delta_abs > delta_threshold:
        hedge_direction = "short" if net_delta > 0 else "long"
        return GregManagementSuggestion(
            bot_name="GregBot",
            strategy_code=strategy_code,
            underlying=underlying,
            position_id=position_id,
            summary=f"HEDGE: Net delta {net_delta:.2f} > {delta_threshold} threshold",
            action="HEDGE",
            reason=f"Net delta {net_delta:.2f} exceeds threshold {delta_threshold}. Recommend {hedge_direction} {net_delta_abs:.2f} {underlying} perp to neutralize.",
            metrics=metrics,
        )
    
    if dte <= close_at_dte:
        return GregManagementSuggestion(
            bot_name="GregBot",
            strategy_code=strategy_code,
            underlying=underlying,
            position_id=position_id,
            summary=f"ROLL: DTE {dte} <= {close_at_dte} days",
            action="ROLL",
            reason=f"Position has {dte} days to expiry, below the {close_at_dte}-day threshold. Roll out and recenter if VRP thesis is still valid, or close.",
            metrics=metrics,
        )
    
    return GregManagementSuggestion(
        bot_name="GregBot",
        strategy_code=strategy_code,
        underlying=underlying,
        position_id=position_id,
        summary=f"HOLD: Within thresholds (Δ={net_delta:.2f}, DTE={dte}, PnL={profit_pct*100:.1f}%)",
        action="HOLD",
        reason="Position is within all management thresholds. Continue to monitor.",
        metrics=metrics,
    )


def _evaluate_calendar_spread(
    position_id: str,
    underlying: str,
    strategy_code: str,
    spot_price: float,
    strike: float,
    front_dte: int,
    profit_pct: float,
    rules: GregPositionRules,
) -> Optional[GregManagementSuggestion]:
    """
    Evaluate calendar spread position for management actions.
    
    Rules:
    - Stop loss if price moves more than 5% from strike (tent breach)
    - Close front leg within 24 hours of expiry
    """
    price_move_stop = rules.get_calibration("calendar_price_move_stop_pct", 0.05)
    close_front_hours = int(rules.get_calibration("calendar_close_front_leg_hours", 24))
    
    price_move_pct = abs(spot_price - strike) / strike if strike > 0 else 0
    
    metrics = {
        "spot_price": spot_price,
        "strike": strike,
        "price_move_pct": price_move_pct,
        "front_dte": front_dte,
        "profit_pct": profit_pct,
    }
    
    if price_move_pct >= price_move_stop:
        return GregManagementSuggestion(
            bot_name="GregBot",
            strategy_code=strategy_code,
            underlying=underlying,
            position_id=position_id,
            summary=f"CLOSE: Price move {price_move_pct*100:.1f}% >= {price_move_stop*100:.0f}% tent breach",
            action="CLOSE",
            reason=f"Underlying has moved {price_move_pct*100:.1f}% away from strike, exceeding the {price_move_stop*100:.0f}% tent breach threshold. Close both legs.",
            metrics=metrics,
        )
    
    if front_dte <= 1:
        return GregManagementSuggestion(
            bot_name="GregBot",
            strategy_code=strategy_code,
            underlying=underlying,
            position_id=position_id,
            summary=f"CLOSE: Front leg DTE {front_dte} <= 1 day",
            action="CLOSE",
            reason=f"Front leg expires in {front_dte} day(s). Close both legs to avoid assignment risk.",
            metrics=metrics,
        )
    
    if profit_pct >= 0.30:
        return GregManagementSuggestion(
            bot_name="GregBot",
            strategy_code=strategy_code,
            underlying=underlying,
            position_id=position_id,
            summary=f"TAKE_PROFIT: {profit_pct*100:.0f}% profit (IV crush captured)",
            action="TAKE_PROFIT",
            reason=f"Calendar spread has captured {profit_pct*100:.1f}% profit. Front IV likely crushed vs back. Consider closing.",
            metrics=metrics,
        )
    
    return GregManagementSuggestion(
        bot_name="GregBot",
        strategy_code=strategy_code,
        underlying=underlying,
        position_id=position_id,
        summary=f"HOLD: Within tent (move={price_move_pct*100:.1f}%, DTE={front_dte})",
        action="HOLD",
        reason="Calendar spread is within the tent and has time remaining. Continue to monitor.",
        metrics=metrics,
    )


def _evaluate_short_put(
    position_id: str,
    underlying: str,
    strategy_code: str,
    delta_abs: float,
    profit_pct: float,
    funding_rate: Optional[float],
    rules: GregPositionRules,
) -> Optional[GregManagementSuggestion]:
    """
    Evaluate short put accumulation position for management actions.
    
    Rules:
    - Take profit at 70% of credit
    - Assignment consideration when delta >= 0.80
    - Funding-based assignment preference (perp vs spot)
    """
    profit_take_pct = rules.get_calibration("short_put_profit_take_pct", 0.70)
    stop_delta = rules.get_calibration("short_put_stop_delta_abs", 0.80)
    
    metrics = {
        "delta_abs": delta_abs,
        "profit_pct": profit_pct,
        "funding_rate": funding_rate,
    }
    
    if profit_pct >= profit_take_pct:
        return GregManagementSuggestion(
            bot_name="GregBot",
            strategy_code=strategy_code,
            underlying=underlying,
            position_id=position_id,
            summary=f"TAKE_PROFIT: {profit_pct*100:.0f}% >= {profit_take_pct*100:.0f}% target",
            action="TAKE_PROFIT",
            reason=f"Short put has captured {profit_pct*100:.1f}% of credit, exceeding {profit_take_pct*100:.0f}% target. Close to lock in gains.",
            metrics=metrics,
        )
    
    if delta_abs >= stop_delta:
        funding_str = ""
        if funding_rate is not None:
            if funding_rate >= 0:
                funding_str = f" Funding rate {funding_rate*100:.3f}% >= 0: prefer synthetic assignment via perp (collect funding)."
            else:
                funding_str = f" Funding rate {funding_rate*100:.3f}% < 0: prefer spot assignment (avoid paying funding)."
        
        return GregManagementSuggestion(
            bot_name="GregBot",
            strategy_code=strategy_code,
            underlying=underlying,
            position_id=position_id,
            summary=f"ASSIGN: Delta {delta_abs:.2f} >= {stop_delta} threshold",
            action="ASSIGN",
            reason=f"Short put delta ({delta_abs:.2f}) exceeds {stop_delta} threshold, indicating high ITM probability.{funding_str}",
            metrics=metrics,
        )
    
    return GregManagementSuggestion(
        bot_name="GregBot",
        strategy_code=strategy_code,
        underlying=underlying,
        position_id=position_id,
        summary=f"HOLD: Delta {delta_abs:.2f}, PnL {profit_pct*100:.1f}%",
        action="HOLD",
        reason="Short put is within management thresholds. Continue to monitor.",
        metrics=metrics,
    )


def _evaluate_iron_fly(
    position_id: str,
    underlying: str,
    strategy_code: str,
    spot_price: float,
    center_strike: float,
    wing_spread: float,
    profit_pct: float,
    rules: GregPositionRules,
) -> Optional[GregManagementSuggestion]:
    """
    Evaluate iron butterfly position for management actions.
    
    Rules:
    - Take profit at 30% of max profit
    - Close if price touches wing strikes
    """
    profit_take_pct = rules.get_calibration("iron_fly_profit_take_pct", 0.30)
    
    lower_wing = center_strike - wing_spread
    upper_wing = center_strike + wing_spread
    
    metrics = {
        "spot_price": spot_price,
        "center_strike": center_strike,
        "lower_wing": lower_wing,
        "upper_wing": upper_wing,
        "profit_pct": profit_pct,
    }
    
    if spot_price <= lower_wing or spot_price >= upper_wing:
        wing_touched = "lower" if spot_price <= lower_wing else "upper"
        return GregManagementSuggestion(
            bot_name="GregBot",
            strategy_code=strategy_code,
            underlying=underlying,
            position_id=position_id,
            summary=f"CLOSE: Price ${spot_price:.0f} touched {wing_touched} wing ${lower_wing if wing_touched == 'lower' else upper_wing:.0f}",
            action="CLOSE",
            reason=f"Underlying price ${spot_price:.0f} has touched the {wing_touched} wing strike. Close to limit further loss.",
            metrics=metrics,
        )
    
    if profit_pct >= profit_take_pct:
        return GregManagementSuggestion(
            bot_name="GregBot",
            strategy_code=strategy_code,
            underlying=underlying,
            position_id=position_id,
            summary=f"TAKE_PROFIT: {profit_pct*100:.0f}% >= {profit_take_pct*100:.0f}% of max profit",
            action="TAKE_PROFIT",
            reason=f"Iron butterfly has captured {profit_pct*100:.1f}% of maximum profit, exceeding {profit_take_pct*100:.0f}% target.",
            metrics=metrics,
        )
    
    return GregManagementSuggestion(
        bot_name="GregBot",
        strategy_code=strategy_code,
        underlying=underlying,
        position_id=position_id,
        summary=f"HOLD: Price ${spot_price:.0f} within wings [${lower_wing:.0f}, ${upper_wing:.0f}]",
        action="HOLD",
        reason="Iron butterfly price is within the wings. Continue to monitor.",
        metrics=metrics,
    )


def _evaluate_credit_spread(
    position_id: str,
    underlying: str,
    strategy_code: str,
    spot_price: float,
    short_strike: float,
    is_bull_put: bool,
    profit_pct: float,
    rules: GregPositionRules,
) -> Optional[GregManagementSuggestion]:
    """
    Evaluate credit spread (bull put or bear call) for management actions.
    
    Rules:
    - Take profit at 60% of max credit
    - Close if price touches short strike
    """
    profit_take_pct = rules.get_calibration("spread_profit_take_pct", 0.60)
    
    spread_type = "Bull Put" if is_bull_put else "Bear Call"
    
    metrics = {
        "spot_price": spot_price,
        "short_strike": short_strike,
        "is_bull_put": is_bull_put,
        "profit_pct": profit_pct,
    }
    
    touched = (is_bull_put and spot_price <= short_strike) or (not is_bull_put and spot_price >= short_strike)
    
    if touched:
        direction = "dropped to" if is_bull_put else "risen to"
        return GregManagementSuggestion(
            bot_name="GregBot",
            strategy_code=strategy_code,
            underlying=underlying,
            position_id=position_id,
            summary=f"CLOSE: Price ${spot_price:.0f} has {direction} short strike ${short_strike:.0f}",
            action="CLOSE",
            reason=f"{spread_type} spread: Price has {direction} the short strike. Close to limit further loss.",
            metrics=metrics,
        )
    
    if profit_pct >= profit_take_pct:
        return GregManagementSuggestion(
            bot_name="GregBot",
            strategy_code=strategy_code,
            underlying=underlying,
            position_id=position_id,
            summary=f"TAKE_PROFIT: {profit_pct*100:.0f}% >= {profit_take_pct*100:.0f}% of max credit",
            action="TAKE_PROFIT",
            reason=f"{spread_type} spread has captured {profit_pct*100:.1f}% of maximum credit, exceeding {profit_take_pct*100:.0f}% target.",
            metrics=metrics,
        )
    
    return GregManagementSuggestion(
        bot_name="GregBot",
        strategy_code=strategy_code,
        underlying=underlying,
        position_id=position_id,
        summary=f"HOLD: {spread_type} spread within thresholds",
        action="HOLD",
        reason=f"{spread_type} spread is within management thresholds. Continue to monitor.",
        metrics=metrics,
    )


def evaluate_greg_positions(
    state: AgentState,
    mock_positions: Optional[List[Dict[str, Any]]] = None,
) -> List[GregManagementSuggestion]:
    """
    Evaluate all open Greg positions and return management suggestions.
    
    Args:
        state: Current AgentState with portfolio, spots, etc.
        mock_positions: Optional list of mock positions for testing.
                       Each should have: strategy_code, underlying, position_id, 
                       and strategy-specific metrics.
    
    Returns:
        List of GregManagementSuggestion for each evaluated position.
    """
    rules = get_greg_position_rules()
    suggestions: List[GregManagementSuggestion] = []
    
    if mock_positions:
        for pos in mock_positions:
            suggestion = _evaluate_mock_position(pos, state, rules)
            if suggestion:
                suggestions.append(suggestion)
        return suggestions
    
    return suggestions


def _evaluate_mock_position(
    pos: Dict[str, Any],
    state: AgentState,
    rules: GregPositionRules,
) -> Optional[GregManagementSuggestion]:
    """Evaluate a mock/synthetic position for testing."""
    strategy_code = pos.get("strategy_code", "")
    underlying = pos.get("underlying", "BTC")
    position_id = pos.get("position_id", "mock")
    spot = state.spot.get(underlying, 0.0)
    
    if "STRADDLE" in strategy_code or "STRANGLE" in strategy_code:
        return _evaluate_straddle_strangle(
            position_id=position_id,
            underlying=underlying,
            strategy_code=strategy_code,
            net_delta=pos.get("net_delta", 0.0),
            dte=pos.get("dte", 30),
            profit_pct=pos.get("profit_pct", 0.0),
            loss_pct=pos.get("loss_pct", 0.0),
            rules=rules,
        )
    
    if "CALENDAR" in strategy_code:
        return _evaluate_calendar_spread(
            position_id=position_id,
            underlying=underlying,
            strategy_code=strategy_code,
            spot_price=spot,
            strike=pos.get("strike", spot),
            front_dte=pos.get("front_dte", 7),
            profit_pct=pos.get("profit_pct", 0.0),
            rules=rules,
        )
    
    if "SHORT_PUT" in strategy_code:
        return _evaluate_short_put(
            position_id=position_id,
            underlying=underlying,
            strategy_code=strategy_code,
            delta_abs=abs(pos.get("delta", 0.0)),
            profit_pct=pos.get("profit_pct", 0.0),
            funding_rate=pos.get("funding_rate"),
            rules=rules,
        )
    
    if "IRON" in strategy_code and ("FLY" in strategy_code or "BUTTERFLY" in strategy_code):
        return _evaluate_iron_fly(
            position_id=position_id,
            underlying=underlying,
            strategy_code=strategy_code,
            spot_price=spot,
            center_strike=pos.get("center_strike", spot),
            wing_spread=pos.get("wing_spread", 5000),
            profit_pct=pos.get("profit_pct", 0.0),
            rules=rules,
        )
    
    if "BULL_PUT" in strategy_code:
        return _evaluate_credit_spread(
            position_id=position_id,
            underlying=underlying,
            strategy_code=strategy_code,
            spot_price=spot,
            short_strike=pos.get("short_strike", spot * 0.95),
            is_bull_put=True,
            profit_pct=pos.get("profit_pct", 0.0),
            rules=rules,
        )
    
    if "BEAR_CALL" in strategy_code:
        return _evaluate_credit_spread(
            position_id=position_id,
            underlying=underlying,
            strategy_code=strategy_code,
            spot_price=spot,
            short_strike=pos.get("short_strike", spot * 1.05),
            is_bull_put=False,
            profit_pct=pos.get("profit_pct", 0.0),
            rules=rules,
        )
    
    return None


class GregManagementStore:
    """Thread-safe store for Greg management suggestions."""
    
    def __init__(self):
        self._lock = Lock()
        self._suggestions: List[GregManagementSuggestion] = []
        self._updated_at: Optional[datetime] = None
    
    def update(self, suggestions: List[GregManagementSuggestion]) -> None:
        with self._lock:
            self._suggestions = suggestions
            self._updated_at = datetime.now(timezone.utc)
    
    def get(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "suggestions": [s.to_dict() for s in self._suggestions],
                "count": len(self._suggestions),
                "updated_at": self._updated_at.isoformat() if self._updated_at else None,
            }


greg_management_store = GregManagementStore()
