"""
Training profiles for multi-strategy experimentation.
Defines conservative, moderate, and aggressive covered call strategies.
"""
from __future__ import annotations

from typing import Any, Optional

from src.models import CandidateOption
from src.scoring.candidates import score_option_candidate, ScoringProfile


TRAINING_PROFILES: dict[str, dict[str, Any]] = {
    "conservative": {
        "target_delta": 0.20,
        "delta_min": 0.10,
        "delta_max": 0.25,
        "min_dte": 1,
        "max_dte": 21,
        "max_legs": 2,
        "enabled": True,
        "description": "Low delta - lower assignment risk, steadier premium",
    },
    "moderate": {
        "target_delta": 0.30,
        "delta_min": 0.25,
        "delta_max": 0.35,
        "min_dte": 1,
        "max_dte": 21,
        "max_legs": 2,
        "enabled": True,
        "description": "Mid-range delta - balanced risk/reward",
    },
    "aggressive": {
        "target_delta": 0.40,
        "delta_min": 0.35,
        "delta_max": 0.50,
        "min_dte": 1,
        "max_dte": 21,
        "max_legs": 2,
        "enabled": True,
        "description": "Higher delta - higher premium but more assignment risk",
    },
}


def get_profile(name: str) -> dict[str, Any] | None:
    """Get a training profile by name."""
    return TRAINING_PROFILES.get(name)


def list_profiles() -> list[str]:
    """List all available profile names."""
    return list(TRAINING_PROFILES.keys())


def score_candidate_for_profile(
    candidate: CandidateOption | dict[str, Any],
    profile: dict[str, Any],
) -> float:
    """
    Score a candidate for a specific training profile.
    
    Uses the centralized scoring function (src/scoring/candidates.py) to ensure
    consistent scoring across live agent, backtests, and training.
    
    Args:
        candidate: CandidateOption or dict with delta, dte, premium_usd, ivrv
        profile: Profile dict with target_delta, delta_min/max, min/max_dte
    
    Returns:
        Score value (higher is better)
    """
    if isinstance(candidate, dict):
        features = {
            "delta": candidate.get("delta", 0.0),
            "dte": candidate.get("dte", 0),
            "premium_usd": candidate.get("premium_usd", 0.0),
            "ivrv": candidate.get("ivrv", 1.0),
        }
    else:
        features = {
            "delta": candidate.delta,
            "dte": candidate.dte,
            "premium_usd": candidate.premium_usd,
            "ivrv": candidate.ivrv,
        }

    profile_name = _get_profile_name_from_target_delta(profile.get("target_delta", 0.25))
    
    return score_option_candidate(
        features,
        profile=profile_name,
        config_overrides={"target_delta": profile.get("target_delta", 0.25)},
    )


def _get_profile_name_from_target_delta(target_delta: float) -> ScoringProfile:
    """Map target delta to scoring profile name."""
    if target_delta <= 0.22:
        return "conservative"
    elif target_delta <= 0.32:
        return "moderate"
    else:
        return "aggressive"


def pick_candidate_for_profile(
    candidates: list[CandidateOption] | list[dict[str, Any]],
    profile: dict[str, Any],
) -> Optional[CandidateOption | dict[str, Any]]:
    """
    Pick the best candidate for a given training profile.
    
    Filters candidates by delta and DTE ranges, then scores remaining.
    
    Args:
        candidates: List of CandidateOption or dicts
        profile: Profile dict with target_delta, delta_min/max, min/max_dte
    
    Returns:
        Best matching candidate or None if none match criteria
    """
    best = None
    best_score = float("-inf")

    delta_min = profile["delta_min"]
    delta_max = profile["delta_max"]
    min_dte = profile["min_dte"]
    max_dte = profile["max_dte"]

    for c in candidates:
        if isinstance(c, dict):
            delta = c.get("delta", 0.0)
            dte = c.get("dte", 0)
        else:
            delta = c.delta
            dte = c.dte

        if not (delta_min <= delta <= delta_max):
            continue
        if not (min_dte <= dte <= max_dte):
            continue

        score = score_candidate_for_profile(c, profile)

        if score > best_score:
            best_score = score
            best = c

    return best


def pick_candidates_for_all_profiles(
    candidates: list[CandidateOption] | list[dict[str, Any]],
    profile_names: list[str],
    max_calls_per_expiry: int = 3,
) -> dict[str, list[CandidateOption | dict[str, Any]]]:
    """
    Pick multiple candidates for each profile (for training ladders).
    
    Each profile can return up to `max_legs` candidates, subject to:
    - Per-expiry limits (max_calls_per_expiry) - keyed by expiry date, not DTE
    - Profile delta/DTE filtering
    - Sorted by premium (highest first)
    
    Args:
        candidates: List of candidate options
        profile_names: List of profile names to select for
        max_calls_per_expiry: Maximum calls per expiry date across all profiles
    
    Returns:
        Dict mapping profile_name -> list of candidates (may be empty)
    """
    result: dict[str, list[CandidateOption | dict[str, Any]]] = {
        name: [] for name in profile_names
    }
    
    per_expiry_counts: dict[str, int] = {}
    used_symbols: set[str] = set()
    
    for name in profile_names:
        profile = TRAINING_PROFILES.get(name)
        if not profile:
            continue
        
        if not profile.get("enabled", True):
            continue
        
        delta_min = profile["delta_min"]
        delta_max = profile["delta_max"]
        min_dte = profile["min_dte"]
        max_dte = profile["max_dte"]
        max_legs = profile.get("max_legs", 2)
        
        profile_candidates = []
        for c in candidates:
            if isinstance(c, dict):
                delta = c.get("delta", 0.0)
                dte = c.get("dte", 0)
                symbol = c.get("symbol", "")
            else:
                delta = c.delta
                dte = c.dte
                symbol = c.symbol
            
            if symbol in used_symbols:
                continue
            
            if not (delta_min <= delta <= delta_max):
                continue
            if not (min_dte <= dte <= max_dte):
                continue
            
            profile_candidates.append(c)
        
        profile_candidates.sort(
            key=lambda x: x.get("premium_usd", 0) if isinstance(x, dict) else x.premium_usd,
            reverse=True
        )
        
        for c in profile_candidates:
            if len(result[name]) >= max_legs:
                break
            
            if isinstance(c, dict):
                expiry = c.get("expiry")
                symbol = c.get("symbol", "")
            else:
                expiry = c.expiry
                symbol = c.symbol
            
            expiry_key = expiry.date().isoformat() if hasattr(expiry, 'date') else str(expiry)
            
            if per_expiry_counts.get(expiry_key, 0) >= max_calls_per_expiry:
                continue
            
            result[name].append(c)
            used_symbols.add(symbol)
            per_expiry_counts[expiry_key] = per_expiry_counts.get(expiry_key, 0) + 1
    
    return result


def pick_single_candidate_for_all_profiles(
    candidates: list[CandidateOption] | list[dict[str, Any]],
    profile_names: list[str],
) -> dict[str, Optional[CandidateOption | dict[str, Any]]]:
    """
    Pick single best candidate for each profile (legacy behavior).
    
    Args:
        candidates: List of candidate options
        profile_names: List of profile names to select for
    
    Returns:
        Dict mapping profile_name -> best candidate (or None)
    """
    result = {}
    for name in profile_names:
        profile = TRAINING_PROFILES.get(name)
        if profile:
            result[name] = pick_candidate_for_profile(candidates, profile)
        else:
            result[name] = None
    return result
