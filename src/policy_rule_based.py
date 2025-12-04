"""
Rule-based policy module.
Implements deterministic decision logic for covered call strategy.
"""
from __future__ import annotations

from typing import Any

from src.config import Settings, settings
from src.models import ActionType, AgentState, CandidateOption, OptionPosition, Side


def _get_open_covered_calls(
    agent_state: AgentState,
    underlying: str | None = None,
) -> list[OptionPosition]:
    """Get list of open short call positions (covered calls)."""
    covered_calls = []
    
    for pos in agent_state.portfolio.option_positions:
        if pos.side != Side.SELL:
            continue
        
        if pos.option_type.value != "call":
            continue
        
        if underlying and pos.underlying != underlying:
            continue
        
        covered_calls.append(pos)
    
    return covered_calls


def _select_best_candidate(
    candidates: list[CandidateOption],
    underlying: str | None = None,
    exclude_symbols: list[str] | None = None,
    config: Settings | None = None,
) -> CandidateOption | None:
    """
    Select the best candidate option for opening a covered call.
    
    Selection criteria (in order of priority):
    1. IVRV ratio >= configured minimum
    2. Highest premium
    3. DTE in preferred range (closer to middle of range)
    """
    cfg = config or settings
    exclude = set(exclude_symbols or [])
    
    filtered = []
    for c in candidates:
        if underlying and c.underlying != underlying:
            continue
        if c.symbol in exclude:
            continue
        if c.ivrv < cfg.ivrv_min:
            continue
        filtered.append(c)
    
    if not filtered:
        filtered = [
            c for c in candidates
            if (not underlying or c.underlying == underlying)
            and c.symbol not in exclude
        ]
    
    if not filtered:
        return None
    
    def score_candidate(c: CandidateOption) -> float:
        premium_score = c.premium_usd
        
        dte_mid = (cfg.dte_min + cfg.dte_max) / 2
        dte_range = cfg.dte_max - cfg.dte_min
        dte_deviation = abs(c.dte - dte_mid) / dte_range if dte_range > 0 else 0
        dte_score = 1.0 - dte_deviation
        
        ivrv_bonus = max(0, c.ivrv - cfg.ivrv_min) * 10
        
        return premium_score * (1 + dte_score * 0.1) + ivrv_bonus
    
    filtered.sort(key=score_candidate, reverse=True)
    return filtered[0]


def _should_roll_position(
    position: OptionPosition,
    agent_state: AgentState,
    config: Settings | None = None,
) -> tuple[bool, str]:
    """
    Determine if a position should be rolled.
    
    Roll conditions:
    1. DTE < 1 day (near expiry)
    2. Position is ITM (assignment risk)
    3. Position is ATM with low IV (not much premium left)
    """
    cfg = config or settings
    
    dte = position.expiry_dte or 0
    
    if dte < 1:
        return True, f"Near expiry (DTE={dte})"
    
    if position.moneyness == "ITM":
        if dte <= 2:
            return True, f"ITM with low DTE ({dte} days) - assignment risk"
    
    spot = agent_state.spot.get(position.underlying, 0)
    if spot > 0 and position.strike > 0:
        pct_from_strike = (position.strike - spot) / spot * 100
        if pct_from_strike < 2.0 and dte <= 1:
            return True, f"ATM (only {pct_from_strike:.1f}% OTM) with low DTE"
    
    return False, ""


def decide_action(
    agent_state: AgentState,
    config: Settings | None = None,
) -> dict[str, Any]:
    """
    Decide the next action based on current state using rule-based logic.
    
    Decision flow:
    1. Check for positions that need rolling
    2. If no open covered calls and good candidates exist, open new position
    3. Otherwise, do nothing
    
    Args:
        agent_state: Current agent state
        config: Settings configuration
    
    Returns:
        Dict with keys: action, params, reasoning
    """
    cfg = config or settings
    
    for underlying in cfg.underlyings:
        covered_calls = _get_open_covered_calls(agent_state, underlying)
        
        for cc in covered_calls:
            should_roll, roll_reason = _should_roll_position(cc, agent_state, cfg)
            
            if should_roll:
                new_candidate = _select_best_candidate(
                    agent_state.candidate_options,
                    underlying=underlying,
                    exclude_symbols=[cc.symbol],
                    config=cfg,
                )
                
                if new_candidate:
                    return {
                        "action": ActionType.ROLL_COVERED_CALL.value,
                        "params": {
                            "underlying": underlying,
                            "from_symbol": cc.symbol,
                            "to_symbol": new_candidate.symbol,
                            "size": cc.size,
                        },
                        "reasoning": f"Rolling {cc.symbol}: {roll_reason}. "
                                   f"New position: {new_candidate.symbol} "
                                   f"(DTE={new_candidate.dte}, premium=${new_candidate.premium_usd:.2f})",
                    }
                else:
                    return {
                        "action": ActionType.CLOSE_COVERED_CALL.value,
                        "params": {
                            "underlying": underlying,
                            "symbol": cc.symbol,
                            "size": cc.size,
                        },
                        "reasoning": f"Closing {cc.symbol}: {roll_reason}. "
                                   f"No suitable candidates available for rolling.",
                    }
    
    for underlying in cfg.underlyings:
        covered_calls = _get_open_covered_calls(agent_state, underlying)
        
        if not covered_calls:
            candidates = [
                c for c in agent_state.candidate_options
                if c.underlying == underlying
            ]
            
            if candidates:
                best = _select_best_candidate(candidates, underlying=underlying, config=cfg)
                
                if best:
                    return {
                        "action": ActionType.OPEN_COVERED_CALL.value,
                        "params": {
                            "underlying": underlying,
                            "symbol": best.symbol,
                            "size": cfg.default_order_size,
                        },
                        "reasoning": f"Opening covered call: {best.symbol} "
                                   f"(DTE={best.dte}, delta={best.delta:.2f}, "
                                   f"premium=${best.premium_usd:.2f}, IVRV={best.ivrv:.2f})",
                    }
    
    existing_positions = []
    for underlying in cfg.underlyings:
        ccs = _get_open_covered_calls(agent_state, underlying)
        existing_positions.extend([cc.symbol for cc in ccs])
    
    if existing_positions:
        reasoning = f"Existing positions: {', '.join(existing_positions)}. No action needed."
    elif not agent_state.candidate_options:
        reasoning = "No candidate options available that meet criteria."
    else:
        reasoning = "No suitable opportunities identified."
    
    return {
        "action": ActionType.DO_NOTHING.value,
        "params": {},
        "reasoning": reasoning,
    }
