"""
Training profiles for multi-strategy experimentation.
Defines conservative, moderate, and aggressive covered call strategies.
"""
from __future__ import annotations

from typing import Any, Optional

from src.models import CandidateOption


TRAINING_PROFILES: dict[str, dict[str, Any]] = {
    "conservative": {
        "target_delta": 0.18,
        "delta_min": 0.10,
        "delta_max": 0.25,
        "min_dte": 10,
        "max_dte": 30,
        "description": "Low delta, longer DTE - lower assignment risk, steadier premium",
    },
    "moderate": {
        "target_delta": 0.25,
        "delta_min": 0.18,
        "delta_max": 0.35,
        "min_dte": 7,
        "max_dte": 21,
        "description": "Balanced delta and DTE - moderate risk/reward",
    },
    "aggressive": {
        "target_delta": 0.32,
        "delta_min": 0.25,
        "delta_max": 0.45,
        "min_dte": 3,
        "max_dte": 21,
        "description": "Higher delta, shorter DTE - higher premium but more assignment risk",
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
    
    Scoring logic:
    - Penalizes distance from target delta
    - Rewards higher premium (scaled)
    - Rewards IVRV > 1.0
    
    Args:
        candidate: CandidateOption or dict with delta, dte, premium_usd, ivrv
        profile: Profile dict with target_delta, delta_min/max, min/max_dte
    
    Returns:
        Score value (higher is better)
    """
    if isinstance(candidate, dict):
        delta = candidate.get("delta", 0.0)
        dte = candidate.get("dte", 0)
        premium = candidate.get("premium_usd", 0.0)
        ivrv = candidate.get("ivrv", 1.0)
    else:
        delta = candidate.delta
        dte = candidate.dte
        premium = candidate.premium_usd
        ivrv = candidate.ivrv

    target_delta = profile["target_delta"]
    delta_distance = abs(delta - target_delta)
    
    score = (
        premium / 100.0
        - delta_distance * 10
        + (ivrv - 1.0) * 5
    )
    
    return score


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
) -> dict[str, Optional[CandidateOption | dict[str, Any]]]:
    """
    Pick best candidates for multiple profiles at once.
    
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
