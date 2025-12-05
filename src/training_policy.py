"""
Training policy module for multi-strategy experimentation.
Builds multiple simulated actions for different strategy profiles.
"""
from __future__ import annotations

from typing import Any

from src.config import Settings, settings
from src.models import ActionType, AgentState, CandidateOption
from src.training_profiles import (
    TRAINING_PROFILES,
    pick_candidate_for_profile,
)


def build_training_actions(
    agent_state: AgentState,
    config: Settings | None = None,
) -> list[dict[str, Any]]:
    """
    Build multiple training actions for different strategy profiles.
    
    For each underlying, picks up to one candidate per profile
    (conservative / moderate / aggressive) and creates simulated
    OPEN_COVERED_CALL actions.
    
    Args:
        agent_state: Current agent state with candidates
        config: Settings configuration
    
    Returns:
        List of action dicts, each tagged with 'strategy' field
    """
    cfg = config or settings
    
    if not cfg.is_training_enabled:
        return []
    
    actions: list[dict[str, Any]] = []
    
    by_underlying: dict[str, list[CandidateOption]] = {u: [] for u in cfg.underlyings}
    for c in agent_state.candidate_options:
        if c.underlying in by_underlying:
            by_underlying[c.underlying].append(c)
    
    for underlying in cfg.underlyings:
        cands = by_underlying.get(underlying, [])
        if not cands:
            continue
        
        # Count existing open positions for this underlying
        existing_positions = [
            p for p in agent_state.portfolio.option_positions
            if p.underlying == underlying and p.side.value == "sell"
        ]
        opened_for_this_underlying = len(existing_positions)
        
        # Initialize used_symbols with already-open symbols to avoid duplicates
        used_symbols: set[str] = {p.symbol for p in existing_positions}
        
        # Skip if already at max positions for this underlying
        if opened_for_this_underlying >= cfg.max_calls_per_underlying_training:
            continue
        
        for profile_name in cfg.training_strategies:
            profile = TRAINING_PROFILES.get(profile_name)
            if not profile:
                continue
            
            available_cands = [c for c in cands if c.symbol not in used_symbols]
            if not available_cands:
                continue
            
            candidate_result = pick_candidate_for_profile(available_cands, profile)
            if candidate_result is None:
                continue
            
            if opened_for_this_underlying >= cfg.max_calls_per_underlying_training:
                break
            
            if isinstance(candidate_result, CandidateOption):
                sym = candidate_result.symbol
                delta = candidate_result.delta
                dte = candidate_result.dte
                premium = candidate_result.premium_usd
                ivrv = candidate_result.ivrv
            else:
                sym = str(candidate_result.get("symbol", ""))
                delta = float(candidate_result.get("delta", 0.0))
                dte = int(candidate_result.get("dte", 0))
                premium = float(candidate_result.get("premium_usd", 0.0))
                ivrv = float(candidate_result.get("ivrv", 1.0))
            
            used_symbols.add(sym)
            
            actions.append({
                "strategy": profile_name,
                "underlying": underlying,
                "action": ActionType.OPEN_COVERED_CALL.value,
                "params": {
                    "symbol": sym,
                    "size": cfg.default_order_size,
                },
                "reasoning": (
                    f"[training:{profile_name}] "
                    f"Selected {sym} with delta={delta:.3f}, "
                    f"dte={dte}, premium_usd={premium:.2f}, "
                    f"ivrv={ivrv:.2f}. "
                    f"Profile: {profile.get('description', '')}"
                ),
                "mode": cfg.mode,
                "policy_version": f"{cfg.policy_version}_training",
                "decision_source": "training_mode",
            })
            
            opened_for_this_underlying += 1
    
    return actions


def build_production_action(
    agent_state: AgentState,
    config: Settings | None = None,
) -> dict[str, Any]:
    """
    Build a single production action using the standard rule-based policy.
    This is a wrapper that imports the standard decide_action.
    
    Args:
        agent_state: Current agent state
        config: Settings configuration
    
    Returns:
        Single action dict
    """
    from src.policy_rule_based import decide_action
    return decide_action(agent_state, config)


def build_actions(
    agent_state: AgentState,
    config: Settings | None = None,
) -> list[dict[str, Any]]:
    """
    Build actions based on mode and training settings.
    
    In live/paper mode: returns single production action
    In research + training mode: returns multiple training actions
    
    Args:
        agent_state: Current agent state
        config: Settings configuration
    
    Returns:
        List of action dicts (length 0-1 for live, 0-N for training)
    """
    cfg = config or settings
    
    if cfg.is_training_enabled:
        training_actions = build_training_actions(agent_state, cfg)
        if training_actions:
            return training_actions
    
    production_action = build_production_action(agent_state, cfg)
    return [production_action]


def summarize_training_actions(actions: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Create a summary of training actions for logging/display.
    
    Args:
        actions: List of training action dicts
    
    Returns:
        Summary dict with counts by strategy and underlying
    """
    summary = {
        "total_actions": len(actions),
        "by_strategy": {},
        "by_underlying": {},
        "actions": [],
    }
    
    for a in actions:
        strategy = a.get("strategy", "unknown")
        underlying = a.get("underlying", "unknown")
        
        summary["by_strategy"][strategy] = summary["by_strategy"].get(strategy, 0) + 1
        summary["by_underlying"][underlying] = summary["by_underlying"].get(underlying, 0) + 1
        
        summary["actions"].append({
            "strategy": strategy,
            "underlying": underlying,
            "action": a.get("action"),
            "symbol": a.get("params", {}).get("symbol"),
            "size": a.get("params", {}).get("size"),
        })
    
    return summary
