"""Centralized candidate option scoring.

This module provides a single scoring function used by:
- Live agent (policy_rule_based.py)
- Backtesting engine (covered_call_simulator.py)
- Training profiles (training_profiles.py)

By centralizing scoring logic, we ensure consistent candidate ranking
across all modes of operation.
"""
from __future__ import annotations

import math
from typing import Any, Literal, Mapping, TypedDict

ScoringProfile = Literal[
    "backtest",      # Full backtest scoring with market context
    "live",          # Live trading scoring
    "conservative",  # Training: low delta preference
    "moderate",      # Training: mid delta preference
    "aggressive",    # Training: high delta preference
]


class ScoringConfig(TypedDict, total=False):
    """Configuration for scoring function."""
    target_delta: float
    target_dte: float
    ivrv_min: float


SCORING_PROFILES: dict[str, ScoringConfig] = {
    "backtest": {
        "target_delta": 0.25,
        "target_dte": 7.0,
        "ivrv_min": 1.0,
    },
    "live": {
        "target_delta": 0.25,
        "target_dte": 7.0,
        "ivrv_min": 1.0,
    },
    "conservative": {
        "target_delta": 0.20,
        "target_dte": 7.0,
        "ivrv_min": 1.0,
    },
    "moderate": {
        "target_delta": 0.30,
        "target_dte": 7.0,
        "ivrv_min": 1.0,
    },
    "aggressive": {
        "target_delta": 0.40,
        "target_dte": 7.0,
        "ivrv_min": 1.0,
    },
}


def score_option_candidate(
    features: Mapping[str, Any],
    profile: ScoringProfile = "backtest",
    config_overrides: ScoringConfig | None = None,
) -> float:
    """
    Compute a numeric score for a single option candidate.
    Higher score = more attractive candidate.

    This is the single source of truth for candidate scoring across
    the live agent, backtests, and training data generation.

    Args:
        features: Dict-like object with candidate features:
            - delta: absolute delta (0.0 to 1.0)
            - dte: days to expiry
            - ivrv: IV/RV ratio (implied vol / realized vol)
            - premium_pct: premium as percentage of spot (for backtest)
            - premium_usd: premium in USD (for live/training)
            - otm_pct: out-of-the-money percentage (optional)
            - regime: market regime (-1=bear, 0=sideways, 1=bull) (optional)
            - return_7d_pct: 7-day return percentage (optional)
            - return_30d_pct: 30-day return percentage (optional)
            - realized_vol_7d: 7-day realized volatility (optional)
            - pct_from_200d_ma: distance from 200-day MA (optional)

        profile: Scoring profile to use:
            - "backtest": Full scoring with market context adjustments
            - "live": Live trading scoring
            - "conservative"/"moderate"/"aggressive": Training profiles

        config_overrides: Optional overrides for target_delta, target_dte, etc.

    Returns:
        Score value (higher is better). For "backtest" profile, returns
        value in [0, 10] range. For other profiles, range may vary.
    """
    base_config = SCORING_PROFILES.get(profile, SCORING_PROFILES["backtest"])
    config = {**base_config}
    if config_overrides:
        config.update(config_overrides)

    target_delta = config.get("target_delta", 0.25)
    target_dte = config.get("target_dte", 7.0)
    ivrv_min = config.get("ivrv_min", 1.0)

    delta = float(features.get("delta", 0.0))
    dte = float(features.get("dte", 7.0))
    ivrv = float(features.get("ivrv", 1.0))

    if profile == "backtest":
        return _score_backtest(features, target_delta, target_dte)
    elif profile == "live":
        return _score_live(features, target_delta, target_dte, ivrv_min)
    else:
        return _score_training_profile(features, target_delta)


def _score_backtest(
    features: Mapping[str, Any],
    target_delta: float,
    target_dte: float,
) -> float:
    """
    Full backtest scoring with market context adjustments.
    Output clamped to [0, 10].

    Components:
    1. IVRV bonus (0-3 points)
    2. Delta proximity to target (0-2 points)
    3. DTE proximity to target (0-1.5 points)
    4. Premium richness (0-2 points)
    5. Regime/volatility adjustments (-2 to 0 points)
    """
    delta = float(features.get("delta", 0.0))
    dte = float(features.get("dte", 7.0))
    ivrv = float(features.get("ivrv", 1.0))
    otm_pct = float(features.get("otm_pct", 5.0))
    premium_pct = float(features.get("premium_pct", 0.0))
    regime = int(features.get("regime", 0))
    ret_7d = float(features.get("return_7d_pct", 0.0))
    ret_30d = float(features.get("return_30d_pct", 0.0))
    rv7 = float(features.get("realized_vol_7d", 0.0))
    pct_from_200 = float(features.get("pct_from_200d_ma", 0.0))

    score = 0.0

    ivrv_clamped = max(1.0, min(ivrv, 1.5))
    score += (ivrv_clamped - 1.0) / 0.5 * 3.0

    delta_diff = abs(delta - target_delta)
    delta_score = max(0.0, 1.0 - delta_diff / 0.10) * 2.0
    score += delta_score

    dte_diff = abs(dte - target_dte)
    dte_score = max(0.0, 1.0 - dte_diff / 2.0) * 1.5
    score += dte_score

    prem_clamped = max(0.0, min(premium_pct, 1.5))
    premium_score = (prem_clamped / 1.5) * 2.0 if premium_pct > 0 else 0.0
    score += premium_score

    if regime == 1:
        if otm_pct < 5.0:
            score -= 1.0
        if ret_30d > 25.0:
            score -= 0.5
    elif regime == -1:
        if ret_7d < -10.0:
            score -= 0.5

    if rv7 > 0 and rv7 < 0.3:
        score -= 0.5

    if pct_from_200 > 20.0:
        score -= 0.5
    elif pct_from_200 < -20.0:
        score -= 0.5

    return max(0.0, min(score, 10.0))


def _score_live(
    features: Mapping[str, Any],
    target_delta: float,
    target_dte: float,
    ivrv_min: float,
) -> float:
    """
    Live trading scoring.
    Uses log-scaled premium and quadratic penalties for delta/DTE distance.
    """
    delta = float(features.get("delta", 0.0))
    dte = float(features.get("dte", 7.0))
    ivrv = float(features.get("ivrv", 1.0))
    premium_usd = float(features.get("premium_usd", 0.0))

    delta_penalty = (delta - target_delta) ** 2
    dte_penalty = ((dte - target_dte) / max(target_dte, 1)) ** 2

    premium_score = math.log1p(max(premium_usd, 0.0))

    score = premium_score - 5.0 * delta_penalty - 2.0 * dte_penalty

    ivrv_excess = max(ivrv - ivrv_min, 0.0)
    score += 1.0 * ivrv_excess

    return score


def _score_training_profile(
    features: Mapping[str, Any],
    target_delta: float,
) -> float:
    """
    Training profile scoring.
    Simple formula: premium/100 - delta_distance*10 + (ivrv-1)*5
    """
    delta = float(features.get("delta", 0.0))
    ivrv = float(features.get("ivrv", 1.0))
    premium_usd = float(features.get("premium_usd", 0.0))

    delta_distance = abs(delta - target_delta)

    score = (
        premium_usd / 100.0
        - delta_distance * 10
        + (ivrv - 1.0) * 5
    )

    return score
