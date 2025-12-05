"""
Rule-based policy module.
Implements deterministic decision logic for covered call strategy.
Supports research mode with exploration and production mode with strict filtering.
"""
from __future__ import annotations

import math
import random
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


def score_candidate(candidate: CandidateOption, cfg: Settings) -> float:
    """
    Score a candidate covered call. Higher is better.
    
    Rewards:
    - Higher premium (log-scaled to avoid huge skew)
    - Closeness to target delta and DTE (based on effective ranges)
    - IVRV above the minimum threshold
    
    Args:
        candidate: The candidate option to score
        cfg: Settings configuration
    
    Returns:
        Score value (higher is better)
    """
    target_delta = (cfg.effective_delta_min + cfg.effective_delta_max) / 2.0
    target_dte = (cfg.effective_dte_min + cfg.effective_dte_max) / 2.0

    delta_penalty = (candidate.delta - target_delta) ** 2
    dte_penalty = ((candidate.dte - target_dte) / max(target_dte, 1)) ** 2

    premium_score = math.log1p(max(candidate.premium_usd, 0.0))

    score = premium_score - 5.0 * delta_penalty - 2.0 * dte_penalty

    ivrv_excess = max(candidate.ivrv - cfg.effective_ivrv_min, 0.0)
    score += 1.0 * ivrv_excess

    return score


def choose_candidate_with_exploration(
    candidates: list[CandidateOption],
    cfg: Settings,
) -> tuple[CandidateOption | None, bool]:
    """
    Choose a candidate, with optional exploration in research mode.
    
    Args:
        candidates: List of candidate options
        cfg: Settings configuration
    
    Returns:
        Tuple of (chosen candidate or None, whether this was an exploration choice)
    """
    if not candidates:
        return None, False

    scored = [(score_candidate(c, cfg), c) for c in candidates]
    scored.sort(key=lambda x: x[0], reverse=True)

    best_score, best_candidate = scored[0]

    if not cfg.is_research or cfg.explore_prob <= 0.0:
        return best_candidate, False

    if random.random() < cfg.explore_prob:
        k = max(1, cfg.explore_top_k)
        top_k = scored[:k]
        _, chosen = random.choice(top_k)
        is_exploration = chosen.symbol != best_candidate.symbol
        return chosen, is_exploration

    return best_candidate, False


def _select_best_candidate(
    candidates: list[CandidateOption],
    underlying: str | None = None,
    exclude_symbols: list[str] | None = None,
    config: Settings | None = None,
) -> CandidateOption | None:
    """
    Select the best candidate option for opening a covered call.
    Uses scoring and exploration logic based on mode.
    
    Args:
        candidates: List of candidate options
        underlying: Filter to specific underlying (optional)
        exclude_symbols: Symbols to exclude (optional)
        config: Settings configuration
    
    Returns:
        Best candidate or None if no suitable candidates
    """
    cfg = config or settings
    exclude = set(exclude_symbols or [])
    
    filtered = []
    for c in candidates:
        if underlying and c.underlying != underlying:
            continue
        if c.symbol in exclude:
            continue
        if c.ivrv < cfg.effective_ivrv_min:
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
    
    chosen, _ = choose_candidate_with_exploration(filtered, cfg)
    return chosen


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
    Uses research vs production mode and exploration settings.
    
    Decision flow:
    1. Check for positions that need rolling
    2. If no open covered calls and good candidates exist, open new position
    3. Otherwise, do nothing
    
    Args:
        agent_state: Current agent state
        config: Settings configuration
    
    Returns:
        Dict with keys: action, params, reasoning, mode, policy_version
    """
    cfg = config or settings
    
    for underlying in cfg.underlyings:
        covered_calls = _get_open_covered_calls(agent_state, underlying)
        
        for cc in covered_calls:
            should_roll, roll_reason = _should_roll_position(cc, agent_state, cfg)
            
            if should_roll:
                candidates = [
                    c for c in agent_state.candidate_options
                    if c.underlying == underlying and c.symbol != cc.symbol
                ]
                
                if candidates:
                    new_candidate, was_exploration = choose_candidate_with_exploration(candidates, cfg)
                    
                    if new_candidate:
                        explore_tag = "Exploratory " if was_exploration else ""
                        return {
                            "action": ActionType.ROLL_COVERED_CALL.value,
                            "params": {
                                "underlying": underlying,
                                "from_symbol": cc.symbol,
                                "to_symbol": new_candidate.symbol,
                                "size": cc.size,
                            },
                            "reasoning": f"{explore_tag}Rolling {cc.symbol}: {roll_reason}. "
                                       f"New position: {new_candidate.symbol} "
                                       f"(DTE={new_candidate.dte}, delta={new_candidate.delta:.2f}, "
                                       f"premium=${new_candidate.premium_usd:.2f}, IVRV={new_candidate.ivrv:.2f}). "
                                       f"Mode={cfg.mode}, policy={cfg.policy_version}.",
                            "mode": cfg.mode,
                            "policy_version": cfg.policy_version,
                            "decision_source": "rule_based",
                        }
                
                return {
                    "action": ActionType.CLOSE_COVERED_CALL.value,
                    "params": {
                        "underlying": underlying,
                        "symbol": cc.symbol,
                        "size": cc.size,
                    },
                    "reasoning": f"Closing {cc.symbol}: {roll_reason}. "
                               f"No suitable candidates available for rolling. "
                               f"Mode={cfg.mode}, policy={cfg.policy_version}.",
                    "mode": cfg.mode,
                    "policy_version": cfg.policy_version,
                    "decision_source": "rule_based",
                }
    
    for underlying in cfg.underlyings:
        covered_calls = _get_open_covered_calls(agent_state, underlying)
        existing_symbols = {cc.symbol for cc in covered_calls}
        
        # In training mode on testnet: allow multiple calls per underlying
        # In normal mode: only open if no existing calls
        can_open_new = (
            cfg.is_training_on_testnet
            or not covered_calls
        )
        
        # In training mode, check against max_calls_per_underlying_training
        if cfg.is_training_on_testnet and covered_calls:
            if len(covered_calls) >= cfg.max_calls_per_underlying_training:
                can_open_new = False
        
        if can_open_new:
            # Exclude already-open symbols to avoid duplicates
            candidates = [
                c for c in agent_state.candidate_options
                if c.underlying == underlying and c.symbol not in existing_symbols
            ]
            
            if candidates:
                chosen, was_exploration = choose_candidate_with_exploration(candidates, cfg)
                
                if chosen:
                    explore_tag = "Exploratory " if was_exploration else ""
                    training_tag = "[TRAINING] " if cfg.is_training_on_testnet else ""
                    existing_count = len(covered_calls)
                    return {
                        "action": ActionType.OPEN_COVERED_CALL.value,
                        "params": {
                            "underlying": underlying,
                            "symbol": chosen.symbol,
                            "size": cfg.default_order_size,
                        },
                        "reasoning": f"{training_tag}{explore_tag}OPEN_COVERED_CALL on {chosen.symbol}: "
                                   f"DTE={chosen.dte}, delta={chosen.delta:.2f}, "
                                   f"premium=${chosen.premium_usd:.2f}, IVRV={chosen.ivrv:.2f}. "
                                   f"Existing calls for {underlying}: {existing_count}. "
                                   f"Mode={cfg.mode}, policy={cfg.policy_version}.",
                        "mode": cfg.mode,
                        "policy_version": cfg.policy_version,
                        "decision_source": "rule_based",
                    }
    
    existing_positions = []
    for underlying in cfg.underlyings:
        ccs = _get_open_covered_calls(agent_state, underlying)
        existing_positions.extend([cc.symbol for cc in ccs])
    
    if existing_positions:
        if cfg.is_training_on_testnet:
            # In training mode, reaching max positions is the only valid "done" state
            reasoning = (
                f"Training mode: max positions reached for all underlyings. "
                f"Existing: {', '.join(existing_positions)}."
            )
        else:
            reasoning = f"Existing positions: {', '.join(existing_positions)}. No action needed."
    elif not agent_state.candidate_options:
        reasoning = "No candidate options available that meet criteria."
    else:
        reasoning = "No suitable opportunities identified."
    
    return {
        "action": ActionType.DO_NOTHING.value,
        "params": {},
        "reasoning": f"{reasoning} Mode={cfg.mode}, policy={cfg.policy_version}.",
        "mode": cfg.mode,
        "policy_version": cfg.policy_version,
        "decision_source": "rule_based",
    }
