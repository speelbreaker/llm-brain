"""
Backtest configuration schema with presets and rule toggles.
Supports ULTRA_SAFE, BALANCED, AGGRESSIVE, and CUSTOM presets.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional, Tuple

from pydantic import BaseModel, Field


class BacktestPreset(str, Enum):
    ULTRA_SAFE = "ULTRA_SAFE"
    BALANCED = "BALANCED"
    AGGRESSIVE = "AGGRESSIVE"
    CUSTOM = "CUSTOM"


class BacktestMode(str, Enum):
    TRAINING = "training"
    LIVE = "live"


class BacktestRuleToggles(BaseModel):
    enforce_per_expiry_exposure: Optional[bool] = Field(
        None, description="Cap exposure per expiry (e.g. notional or BTC amount)."
    )
    enforce_margin_cap: Optional[bool] = Field(
        None, description="Respect max_margin_used_pct limit."
    )
    enforce_net_delta_cap: Optional[bool] = Field(
        None, description="Respect max_net_delta_abs limit."
    )
    restrict_single_primary_call_per_expiry: Optional[bool] = Field(
        None, description="Only one main short call per underlying+expiry."
    )
    require_ivrv_filter: Optional[bool] = Field(
        None, description="Require IV/RV >= min_ivrv to sell premium."
    )
    use_synthetic_iv_and_skew: Optional[bool] = Field(
        None, description="Use synthetic IV/skew engine for scoring."
    )
    allow_multi_profile_laddering: Optional[bool] = Field(
        None, description="Allow conservative/moderate/aggressive ladders together."
    )
    respect_min_premium_filter: Optional[bool] = Field(
        None, description="Skip trades below min_premium_usd."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "enforce_per_expiry_exposure": True,
                "enforce_margin_cap": True,
                "enforce_net_delta_cap": True,
                "restrict_single_primary_call_per_expiry": True,
                "require_ivrv_filter": True,
                "use_synthetic_iv_and_skew": True,
                "allow_multi_profile_laddering": True,
                "respect_min_premium_filter": True,
            }
        }


class BacktestThresholds(BaseModel):
    max_margin_used_pct: Optional[float] = Field(
        None, ge=0, le=100, description="Max margin usage in percent of equity."
    )
    max_net_delta_abs: Optional[float] = Field(
        None, ge=0, description="Absolute portfolio delta limit (in BTC-equivalent)."
    )
    per_expiry_exposure_cap: Optional[float] = Field(
        None,
        ge=0,
        description="Cap for total short exposure per expiry (interpretation is up to engine).",
    )
    min_ivrv: Optional[float] = Field(
        None, ge=0, description="Minimum IV/RV to consider selling premium."
    )
    delta_range: Optional[Tuple[float, float]] = Field(
        None, description="Allowed delta range for short calls (min, max)."
    )
    dte_range: Optional[Tuple[int, int]] = Field(
        None, description="Allowed DTE range in days (min, max)."
    )
    min_premium_usd: Optional[float] = Field(
        None, ge=0, description="Minimum option premium in USD to accept."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "max_margin_used_pct": 60.0,
                "max_net_delta_abs": 3.5,
                "per_expiry_exposure_cap": 0.25,
                "min_ivrv": 1.15,
                "delta_range": [0.15, 0.35],
                "dte_range": [3, 21],
                "min_premium_usd": 200.0,
            }
        }


class BacktestConfig(BaseModel):
    """
    Incoming payload from UI for a backtest.
    If preset != CUSTOM, server will fill missing fields from preset defaults
    and then apply user overrides.
    """

    preset: BacktestPreset = Field(
        BacktestPreset.BALANCED,
        description="High-level preset for risk rules. ULTRA_SAFE / BALANCED / AGGRESSIVE / CUSTOM.",
    )
    mode: BacktestMode = Field(
        BacktestMode.TRAINING,
        description="Interpretation of rules: training (looser) vs live (stricter).",
    )

    rule_toggles: BacktestRuleToggles = Field(
        default_factory=lambda: BacktestRuleToggles(),
        description="Optional overrides for which rule families are active.",
    )
    thresholds: BacktestThresholds = Field(
        default_factory=lambda: BacktestThresholds(),
        description="Optional overrides for numeric thresholds.",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "preset": "BALANCED",
                "mode": "training",
                "rule_toggles": {
                    "enforce_per_expiry_exposure": True,
                    "enforce_margin_cap": True,
                    "enforce_net_delta_cap": True,
                    "restrict_single_primary_call_per_expiry": True,
                    "require_ivrv_filter": True,
                    "use_synthetic_iv_and_skew": True,
                    "allow_multi_profile_laddering": True,
                    "respect_min_premium_filter": True
                },
                "thresholds": {
                    "max_margin_used_pct": 60.0,
                    "max_net_delta_abs": 3.5,
                    "per_expiry_exposure_cap": 0.25,
                    "min_ivrv": 1.15,
                    "delta_range": [0.15, 0.35],
                    "dte_range": [3, 21],
                    "min_premium_usd": 200.0
                }
            }
        }


class ResolvedBacktestConfig(BaseModel):
    """
    Fully-resolved config that the backtest engine actually uses
    (after applying preset defaults + overrides).
    """

    preset: BacktestPreset
    mode: BacktestMode
    rule_toggles: BacktestRuleToggles
    thresholds: BacktestThresholds

    class Config:
        json_schema_extra = {
            "example": {
                "preset": "BALANCED",
                "mode": "training",
                "rule_toggles": {
                    "enforce_per_expiry_exposure": True,
                    "enforce_margin_cap": True,
                    "enforce_net_delta_cap": True,
                    "restrict_single_primary_call_per_expiry": True,
                    "require_ivrv_filter": True,
                    "use_synthetic_iv_and_skew": True,
                    "allow_multi_profile_laddering": True,
                    "respect_min_premium_filter": True
                },
                "thresholds": {
                    "max_margin_used_pct": 60.0,
                    "max_net_delta_abs": 3.5,
                    "per_expiry_exposure_cap": 0.25,
                    "min_ivrv": 1.15,
                    "delta_range": [0.15, 0.35],
                    "dte_range": [3, 21],
                    "min_premium_usd": 200.0
                }
            }
        }
