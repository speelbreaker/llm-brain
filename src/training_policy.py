"""
Training policy module for multi-strategy experimentation.
Builds multiple simulated actions for different strategy profiles.

Supports two modes:
- "single": One action per profile per underlying (traditional)
- "ladder": Multiple actions per profile, building a delta ladder
"""
from __future__ import annotations

from typing import Any

from src.config import Settings, settings
from src.models import ActionType, AgentState, CandidateOption
from src.training_profiles import (
    TRAINING_PROFILES,
    pick_candidate_for_profile,
    pick_candidates_for_all_profiles,
)


def build_training_actions(
    agent_state: AgentState,
    config: Settings | None = None,
) -> list[dict[str, Any]]:
    """
    Build multiple training actions for different strategy profiles.
    
    Behavior depends on training_profile_mode:
    - "single": One action per profile per underlying (traditional)
    - "ladder": Multiple actions per profile, building a delta ladder
    
    In ladder mode:
    - Uses pick_candidates_for_all_profiles to get multiple candidates per profile
    - Respects per-expiry limits (training_max_calls_per_expiry)
    - Tags each action with strategy name and diagnostics for training data
    
    Args:
        agent_state: Current agent state with candidates
        config: Settings configuration
    
    Returns:
        List of action dicts, each tagged with 'strategy' and 'diagnostics' fields
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
        
        existing_positions = [
            p for p in agent_state.portfolio.option_positions
            if p.underlying == underlying and p.side.value == "sell"
        ]
        opened_for_this_underlying = len(existing_positions)
        
        used_symbols: set[str] = {p.symbol for p in existing_positions}
        
        if opened_for_this_underlying >= cfg.max_calls_per_underlying_training:
            continue
        
        remaining_slots = cfg.max_calls_per_underlying_training - opened_for_this_underlying
        
        # Filter out already-open symbols
        available_cands = [c for c in cands if c.symbol not in used_symbols]
        if not available_cands:
            continue
        
        calls_for_underlying = 0
        
        if cfg.training_profile_mode == "ladder":
            # Ladder mode: use profile-based multi-candidate selection
            all_profile_candidates = pick_candidates_for_all_profiles(
                candidates=available_cands,
                profile_names=cfg.training_strategies,
                max_calls_per_expiry=cfg.training_max_calls_per_expiry,
            )
            
            for profile_name in cfg.training_strategies:
                profile_candidates = all_profile_candidates.get(profile_name, [])
                profile = TRAINING_PROFILES.get(profile_name, {})
                
                for cand in profile_candidates:
                    if calls_for_underlying >= remaining_slots:
                        break
                    
                    if isinstance(cand, CandidateOption):
                        sym = cand.symbol
                        delta = cand.delta
                        dte = cand.dte
                        premium = cand.premium_usd
                        ivrv = cand.ivrv
                    else:
                        sym = str(cand.get("symbol", ""))
                        delta = float(cand.get("delta", 0.0))
                        dte = int(cand.get("dte", 0))
                        premium = float(cand.get("premium_usd", 0.0))
                        ivrv = float(cand.get("ivrv", 1.0))
                    
                    if sym in used_symbols:
                        continue
                    
                    actions.append({
                        "action": ActionType.OPEN_COVERED_CALL.value,
                        "mode": "training",
                        "strategy": profile_name,
                        "underlying": underlying,
                        "params": {
                            "symbol": sym,
                            "size": cfg.default_order_size,
                        },
                        "diagnostics": {
                            "delta": delta,
                            "dte": dte,
                            "premium_usd": premium,
                            "ivrv": ivrv,
                        },
                        "reasoning": (
                            f"Training mode [{profile_name}] ladder leg on {underlying}: "
                            f"{sym} Δ={delta:.3f}, DTE={dte}, premium=${premium:.2f}"
                        ),
                        "policy_version": f"{cfg.policy_version}_training",
                        "decision_source": "training_mode",
                    })
                    
                    used_symbols.add(sym)
                    calls_for_underlying += 1
                
                if calls_for_underlying >= remaining_slots:
                    break
        else:
            # Single mode: traditional one-per-profile selection
            matched_any_profile = False
            
            for profile_name in cfg.training_strategies:
                profile = TRAINING_PROFILES.get(profile_name)
                if not profile:
                    continue
                
                if calls_for_underlying >= remaining_slots:
                    break
                
                available_for_profile = [c for c in available_cands if c.symbol not in used_symbols]
                if not available_for_profile:
                    continue
                
                candidate_result = pick_candidate_for_profile(available_for_profile, profile)
                if candidate_result is None:
                    continue
                
                matched_any_profile = True
                
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
                    "action": ActionType.OPEN_COVERED_CALL.value,
                    "mode": "training",
                    "strategy": profile_name,
                    "underlying": underlying,
                    "params": {
                        "symbol": sym,
                        "size": cfg.default_order_size,
                    },
                    "diagnostics": {
                        "delta": delta,
                        "dte": dte,
                        "premium_usd": premium,
                        "ivrv": ivrv,
                    },
                    "reasoning": (
                        f"[training:{profile_name}] "
                        f"Selected {sym} with delta={delta:.3f}, "
                        f"dte={dte}, premium_usd={premium:.2f}, "
                        f"ivrv={ivrv:.2f}. "
                        f"Profile: {profile.get('description', '')}"
                    ),
                    "policy_version": f"{cfg.policy_version}_training",
                    "decision_source": "training_mode",
                })
                
                calls_for_underlying += 1
            
            # Fallback if no profile matched
            if not matched_any_profile and calls_for_underlying < remaining_slots:
                available_for_fallback = [c for c in available_cands if c.symbol not in used_symbols]
                if available_for_fallback:
                    best_cand = max(available_for_fallback, key=lambda x: x.premium_usd)
                    
                    actions.append({
                        "action": ActionType.OPEN_COVERED_CALL.value,
                        "mode": "training",
                        "strategy": "fallback",
                        "underlying": underlying,
                        "params": {
                            "symbol": best_cand.symbol,
                            "size": cfg.default_order_size,
                        },
                        "diagnostics": {
                            "delta": best_cand.delta,
                            "dte": best_cand.dte,
                            "premium_usd": best_cand.premium_usd,
                            "ivrv": best_cand.ivrv,
                        },
                        "reasoning": (
                            f"[training:fallback] "
                            f"No profile match, selecting best premium {best_cand.symbol} with delta={best_cand.delta:.3f}, "
                            f"dte={best_cand.dte}, premium_usd={best_cand.premium_usd:.2f}, "
                            f"ivrv={best_cand.ivrv:.2f}."
                        ),
                        "policy_version": f"{cfg.policy_version}_training",
                        "decision_source": "training_mode",
                    })
    
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
