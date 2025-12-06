"""
Preset defaults and config resolution helper for backtests.
Provides ULTRA_SAFE, BALANCED, AGGRESSIVE preset configurations.
"""
from __future__ import annotations

from typing import Optional

from src.backtest.config_schema import (
    BacktestPreset,
    BacktestMode,
    BacktestRuleToggles,
    BacktestThresholds,
    BacktestConfig,
    ResolvedBacktestConfig,
)


PRESET_RULES = {
    BacktestPreset.ULTRA_SAFE: BacktestRuleToggles(
        enforce_per_expiry_exposure=True,
        enforce_margin_cap=True,
        enforce_net_delta_cap=True,
        restrict_single_primary_call_per_expiry=True,
        require_ivrv_filter=True,
        use_synthetic_iv_and_skew=True,
        allow_multi_profile_laddering=False,
        respect_min_premium_filter=True,
    ),
    BacktestPreset.BALANCED: BacktestRuleToggles(
        enforce_per_expiry_exposure=True,
        enforce_margin_cap=True,
        enforce_net_delta_cap=True,
        restrict_single_primary_call_per_expiry=True,
        require_ivrv_filter=True,
        use_synthetic_iv_and_skew=True,
        allow_multi_profile_laddering=True,
        respect_min_premium_filter=True,
    ),
    BacktestPreset.AGGRESSIVE: BacktestRuleToggles(
        enforce_per_expiry_exposure=False,
        enforce_margin_cap=True,
        enforce_net_delta_cap=True,
        restrict_single_primary_call_per_expiry=False,
        require_ivrv_filter=False,
        use_synthetic_iv_and_skew=True,
        allow_multi_profile_laddering=True,
        respect_min_premium_filter=False,
    ),
}

PRESET_THRESHOLDS = {
    BacktestPreset.ULTRA_SAFE: BacktestThresholds(
        max_margin_used_pct=40.0,
        max_net_delta_abs=2.5,
        per_expiry_exposure_cap=0.15,
        min_ivrv=1.3,
        delta_range=(0.15, 0.30),
        dte_range=(7, 21),
        min_premium_usd=300.0,
    ),
    BacktestPreset.BALANCED: BacktestThresholds(
        max_margin_used_pct=60.0,
        max_net_delta_abs=3.5,
        per_expiry_exposure_cap=0.25,
        min_ivrv=1.15,
        delta_range=(0.15, 0.35),
        dte_range=(3, 21),
        min_premium_usd=200.0,
    ),
    BacktestPreset.AGGRESSIVE: BacktestThresholds(
        max_margin_used_pct=80.0,
        max_net_delta_abs=5.0,
        per_expiry_exposure_cap=0.6,
        min_ivrv=1.0,
        delta_range=(0.10, 0.40),
        dte_range=(1, 21),
        min_premium_usd=100.0,
    ),
}


def resolve_backtest_config(config: BacktestConfig) -> ResolvedBacktestConfig:
    """
    Apply preset defaults and then overlay any user-provided overrides.
    If preset == CUSTOM, we just treat everything the user sent as-is and
    trust that your engine has sensible fallbacks.
    """

    if config.preset == BacktestPreset.CUSTOM:
        return ResolvedBacktestConfig(
            preset=config.preset,
            mode=config.mode,
            rule_toggles=config.rule_toggles,
            thresholds=config.thresholds,
        )

    base_rules = PRESET_RULES[config.preset].model_dump()
    base_thresholds = PRESET_THRESHOLDS[config.preset].model_dump()

    overrides_rules = config.rule_toggles.model_dump()
    overrides_thresholds = config.thresholds.model_dump()

    for key, value in overrides_rules.items():
        if value is not None:
            base_rules[key] = value

    for key, value in overrides_thresholds.items():
        if value is not None:
            base_thresholds[key] = value

    return ResolvedBacktestConfig(
        preset=config.preset,
        mode=config.mode,
        rule_toggles=BacktestRuleToggles(**base_rules),
        thresholds=BacktestThresholds(**base_thresholds),
    )


def get_preset_config(preset: BacktestPreset, mode: Optional[BacktestMode] = None) -> ResolvedBacktestConfig:
    """Get a fully resolved config for a given preset with no overrides."""
    default_mode = mode if mode is not None else BacktestMode.TRAINING
    
    if preset == BacktestPreset.CUSTOM:
        return ResolvedBacktestConfig(
            preset=preset,
            mode=default_mode,
            rule_toggles=PRESET_RULES[BacktestPreset.BALANCED].model_copy(),
            thresholds=PRESET_THRESHOLDS[BacktestPreset.BALANCED].model_copy(),
        )
    
    return ResolvedBacktestConfig(
        preset=preset,
        mode=default_mode,
        rule_toggles=PRESET_RULES[preset],
        thresholds=PRESET_THRESHOLDS[preset],
    )
