"""
Risk engine module.
Validates proposed actions against risk limits before execution.
"""
from __future__ import annotations

from typing import Any

from src.config import Settings, settings
from src.models import ActionType, AgentState, OptionType, Side


def check_action_allowed(
    agent_state: AgentState,
    proposed_action: dict[str, Any],
    config: Settings | None = None,
) -> tuple[bool, list[str]]:
    """
    Check if a proposed action is allowed based on risk limits.
    
    Args:
        agent_state: Current agent state
        proposed_action: Proposed action dict with keys: action, params, reasoning
        config: Settings configuration (uses default if None)
    
    Returns:
        Tuple of (allowed: bool, reasons: list[str])
        If allowed is False, reasons contains the blocking reasons.
    """
    cfg = config or settings
    reasons: list[str] = []
    
    action_str = proposed_action.get("action", "DO_NOTHING")
    
    if isinstance(action_str, ActionType):
        action_type = action_str
    else:
        try:
            action_type = ActionType(action_str)
        except ValueError:
            reasons.append(f"Invalid action type: {action_str}")
            return False, reasons
    
    if action_type == ActionType.DO_NOTHING:
        return True, []
    
    portfolio = agent_state.portfolio
    params = proposed_action.get("params", {})
    
    margin_used_pct = portfolio.margin_used_pct
    if margin_used_pct >= cfg.max_margin_used_pct:
        reasons.append(
            f"Margin usage ({margin_used_pct:.1f}%) exceeds max ({cfg.max_margin_used_pct:.1f}%)"
        )
    
    if action_type in (ActionType.OPEN_COVERED_CALL, ActionType.ROLL_COVERED_CALL):
        margin_threshold = cfg.max_margin_used_pct * 0.9
        if margin_used_pct >= margin_threshold:
            reasons.append(
                f"Margin usage ({margin_used_pct:.1f}%) too high for new positions "
                f"(threshold: {margin_threshold:.1f}%)"
            )
    
    net_delta = abs(portfolio.net_delta)
    if net_delta > cfg.max_net_delta_abs:
        reasons.append(
            f"Net delta ({portfolio.net_delta:.2f}) exceeds max ({cfg.max_net_delta_abs:.2f})"
        )
    
    if action_type == ActionType.OPEN_COVERED_CALL:
        symbol = params.get("symbol", "")
        
        if symbol:
            try:
                expiry_str = symbol.split("-")[1] if "-" in symbol else ""
            except (IndexError, ValueError):
                expiry_str = ""
            
            expiry_exposure = 0.0
            for pos in portfolio.option_positions:
                pos_expiry = pos.symbol.split("-")[1] if "-" in pos.symbol else ""
                if pos_expiry == expiry_str and pos.side == Side.SELL:
                    expiry_exposure += pos.size
            
            size = params.get("size", cfg.default_order_size)
            projected_exposure = expiry_exposure + size
            
            if projected_exposure > cfg.max_expiry_exposure:
                reasons.append(
                    f"Per-expiry exposure ({projected_exposure:.2f}) would exceed max "
                    f"({cfg.max_expiry_exposure:.2f})"
                )
    
    if action_type == ActionType.CLOSE_COVERED_CALL:
        symbol = params.get("symbol", "")
        
        if symbol:
            position_exists = False
            for pos in portfolio.option_positions:
                if pos.symbol == symbol and pos.side == Side.SELL:
                    position_exists = True
                    break
            
            if not position_exists:
                reasons.append(f"No open short position found for symbol: {symbol}")
    
    if action_type == ActionType.ROLL_COVERED_CALL:
        from_symbol = params.get("from_symbol", "")
        to_symbol = params.get("to_symbol", "")
        
        if from_symbol:
            from_position_exists = False
            for pos in portfolio.option_positions:
                if pos.symbol == from_symbol and pos.side == Side.SELL:
                    from_position_exists = True
                    break
            
            if not from_position_exists:
                reasons.append(f"No open short position found to roll from: {from_symbol}")
        
        if not to_symbol:
            has_candidate = False
            for candidate in agent_state.candidate_options:
                if candidate.symbol != from_symbol:
                    has_candidate = True
                    break
            
            if not has_candidate:
                reasons.append("No valid candidate options available to roll into")
    
    allowed = len(reasons) == 0
    return allowed, reasons


def estimate_margin_impact(
    agent_state: AgentState,
    action_type: ActionType,
    params: dict[str, Any],
    config: Settings | None = None,
) -> float:
    """
    Estimate the margin impact of a proposed action.
    This is a simplified estimate - actual impact depends on Deribit's margin model.
    
    Args:
        agent_state: Current agent state
        action_type: Type of action
        params: Action parameters
        config: Settings configuration
    
    Returns:
        Estimated change in margin usage percentage (positive = more margin used)
    """
    cfg = config or settings
    
    if action_type == ActionType.DO_NOTHING:
        return 0.0
    
    size = params.get("size", cfg.default_order_size)
    symbol = params.get("symbol", params.get("to_symbol", ""))
    
    underlying = "BTC"
    if symbol:
        parts = symbol.split("-")
        if parts:
            underlying = parts[0]
    
    spot = agent_state.spot.get(underlying, 100000)
    
    notional = size * spot
    
    equity = agent_state.portfolio.equity_usd
    if equity <= 0:
        equity = 10000
    
    if action_type == ActionType.OPEN_COVERED_CALL:
        margin_pct_impact = (notional * 0.1 / equity) * 100
        return margin_pct_impact
    
    elif action_type == ActionType.CLOSE_COVERED_CALL:
        margin_pct_impact = -(notional * 0.1 / equity) * 100
        return margin_pct_impact
    
    elif action_type == ActionType.ROLL_COVERED_CALL:
        return 1.0
    
    return 0.0
