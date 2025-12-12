"""
Override system for TEST environment.

Allows experimentation with risk and entry rule settings
without affecting production LIVE configuration.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from src.config import EnvironmentMode

OVERRIDES_FILE = Path("data/bot_overrides_test.json")


class GlobalRiskOverrides(BaseModel):
    """Overrides for global risk engine caps."""
    max_margin_pct: Optional[float] = Field(
        default=None,
        description="Override max margin usage percentage"
    )
    max_net_delta: Optional[float] = Field(
        default=None,
        description="Override max absolute net delta"
    )
    daily_drawdown_limit_pct: Optional[float] = Field(
        default=None,
        description="Override daily drawdown limit percentage"
    )
    liquidity_max_spread_pct: Optional[float] = Field(
        default=None,
        description="Override max bid/ask spread percentage"
    )
    liquidity_min_open_interest: Optional[int] = Field(
        default=None,
        description="Override min open interest requirement"
    )


class BotRiskOverrides(BaseModel):
    """Overrides for per-bot risk configuration."""
    max_equity_share: Optional[float] = Field(
        default=None,
        description="Max percentage of equity this bot can use"
    )
    max_positions_per_underlying: Optional[int] = Field(
        default=None,
        description="Max positions per underlying for this bot"
    )
    max_notional_usd_per_position: Optional[float] = Field(
        default=None,
        description="Max notional USD per position"
    )
    max_notional_usd_per_underlying: Optional[float] = Field(
        default=None,
        description="Max notional USD per underlying"
    )


class EntryRuleOverrides(BaseModel):
    """Overrides for bot entry rule thresholds."""
    thresholds: Dict[str, float] = Field(
        default_factory=dict,
        description="Map of threshold key to override value"
    )


class BotOverrideConfig(BaseModel):
    """Top-level container for all TEST environment overrides."""
    use_global_risk_overrides: bool = Field(
        default=False,
        description="Enable global risk overrides"
    )
    use_bot_risk_overrides: bool = Field(
        default=False,
        description="Enable per-bot risk overrides"
    )
    use_entry_rule_overrides: bool = Field(
        default=False,
        description="Enable entry rule threshold overrides"
    )
    global_risk: GlobalRiskOverrides = Field(
        default_factory=GlobalRiskOverrides
    )
    bots: Dict[str, BotRiskOverrides] = Field(
        default_factory=dict,
        description="Per-bot risk overrides keyed by bot_id"
    )
    entry_rules: Dict[str, EntryRuleOverrides] = Field(
        default_factory=dict,
        description="Per-bot entry rule overrides keyed by bot_id"
    )


def load_overrides(env_mode: EnvironmentMode) -> BotOverrideConfig:
    """
    Load override configuration for the given environment mode.
    
    For LIVE mode, always returns defaults with all use_* flags False.
    For TEST mode, loads from the overrides JSON file if it exists.
    """
    if env_mode == EnvironmentMode.LIVE:
        return BotOverrideConfig()
    
    if not OVERRIDES_FILE.exists():
        return BotOverrideConfig()
    
    try:
        with open(OVERRIDES_FILE, "r") as f:
            data = json.load(f)
        return BotOverrideConfig.model_validate(data)
    except Exception:
        return BotOverrideConfig()


def save_overrides(env_mode: EnvironmentMode, cfg: BotOverrideConfig) -> bool:
    """
    Save override configuration for the given environment mode.
    
    Only allowed in TEST mode. Returns True on success.
    """
    if env_mode == EnvironmentMode.LIVE:
        return False
    
    OVERRIDES_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(OVERRIDES_FILE, "w") as f:
            json.dump(cfg.model_dump(), f, indent=2)
        return True
    except Exception:
        return False


def get_effective_global_risk_value(
    key: str,
    default_value: Any,
    env_mode: EnvironmentMode,
    overrides: Optional[BotOverrideConfig] = None,
) -> Any:
    """
    Get the effective value for a global risk setting.
    
    In TEST mode with overrides enabled, returns the override value if set.
    Otherwise returns the default value.
    """
    if env_mode != EnvironmentMode.TEST:
        return default_value
    
    if overrides is None:
        overrides = load_overrides(env_mode)
    
    if not overrides.use_global_risk_overrides:
        return default_value
    
    override_value = getattr(overrides.global_risk, key, None)
    if override_value is not None:
        return override_value
    
    return default_value


def get_effective_bot_risk_value(
    bot_id: str,
    key: str,
    default_value: Any,
    env_mode: EnvironmentMode,
    overrides: Optional[BotOverrideConfig] = None,
) -> Any:
    """
    Get the effective value for a bot-specific risk setting.
    
    In TEST mode with overrides enabled, returns the override value if set.
    Otherwise returns the default value.
    """
    if env_mode != EnvironmentMode.TEST:
        return default_value
    
    if overrides is None:
        overrides = load_overrides(env_mode)
    
    if not overrides.use_bot_risk_overrides:
        return default_value
    
    bot_overrides = overrides.bots.get(bot_id)
    if bot_overrides is None:
        return default_value
    
    override_value = getattr(bot_overrides, key, None)
    if override_value is not None:
        return override_value
    
    return default_value


def get_effective_entry_threshold(
    bot_id: str,
    threshold_key: str,
    default_value: float,
    env_mode: EnvironmentMode,
    overrides: Optional[BotOverrideConfig] = None,
) -> float:
    """
    Get the effective value for an entry rule threshold.
    
    In TEST mode with overrides enabled, returns the override value if set.
    Otherwise returns the default value.
    """
    if env_mode != EnvironmentMode.TEST:
        return default_value
    
    if overrides is None:
        overrides = load_overrides(env_mode)
    
    if not overrides.use_entry_rule_overrides:
        return default_value
    
    entry_overrides = overrides.entry_rules.get(bot_id)
    if entry_overrides is None:
        return default_value
    
    return entry_overrides.thresholds.get(threshold_key, default_value)


def get_greg_calibration_defaults() -> Dict[str, Dict[str, Any]]:
    """
    Get the default calibration thresholds for GregBot from the JSON spec.
    Returns a dict mapping threshold keys to their metadata.
    """
    greg_spec_file = Path("docs/greg_mandolini/GREG_SELECTOR_RULES_FINAL.json")
    
    if not greg_spec_file.exists():
        return {}
    
    try:
        with open(greg_spec_file, "r") as f:
            spec = json.load(f)
        
        calibration = spec.get("global_entry_filters", {}).get("calibration", {})
        
        thresholds = {}
        
        threshold_labels = {
            "skew_neutral_threshold": ("Skew Neutral Threshold", "vol pts"),
            "min_vrp_floor": ("Min VRP Floor", "vol pts"),
            "min_vrp_directional": ("Min VRP Directional", "vol pts"),
            "safety_adx_high": ("Safety ADX High", ""),
            "safety_chop_high": ("Safety Chop High", ""),
            "straddle_vrp_min": ("Straddle VRP Min", "vol pts"),
            "straddle_adx_max": ("Straddle ADX Max", ""),
            "straddle_chop_max": ("Straddle Chop Max", ""),
            "strangle_vrp_min": ("Strangle VRP Min", "vol pts"),
            "strangle_adx_max": ("Strangle ADX Max", ""),
            "strangle_chop_max": ("Strangle Chop Max", ""),
            "calendar_term_spread_min": ("Calendar Term Spread Min", "vol pts"),
            "calendar_front_rv_iv_ratio_max": ("Calendar Front RV/IV Max", ""),
            "calendar_vrp_7d_min": ("Calendar VRP 7d Min", "vol pts"),
            "iron_fly_iv_rank_min": ("Iron Fly IV Rank Min", ""),
            "iron_fly_vrp_min": ("Iron Fly VRP Min", "vol pts"),
        }
        
        for key, value in calibration.items():
            if isinstance(value, (int, float)):
                label, unit = threshold_labels.get(key, (key.replace("_", " ").title(), ""))
                thresholds[key] = {
                    "key": key,
                    "label": label,
                    "default_value": float(value),
                    "unit": unit,
                }
        
        return thresholds
    except Exception:
        return {}


def get_entry_rules_for_ui(
    bot_id: str,
    env_mode: EnvironmentMode,
) -> Dict[str, Any]:
    """
    Get entry rules for UI display, including defaults, current values, and override state.
    """
    defaults = get_greg_calibration_defaults()
    overrides = load_overrides(env_mode)
    
    use_overrides = (
        env_mode == EnvironmentMode.TEST and 
        overrides.use_entry_rule_overrides
    )
    
    entry_overrides = overrides.entry_rules.get(bot_id, EntryRuleOverrides())
    
    rules = []
    for key, meta in defaults.items():
        current_value = entry_overrides.thresholds.get(key, meta["default_value"])
        rules.append({
            "key": key,
            "label": meta["label"],
            "default_value": meta["default_value"],
            "current_value": current_value if use_overrides else meta["default_value"],
            "unit": meta["unit"],
        })
    
    return {
        "env": env_mode.value,
        "bot_id": bot_id,
        "rules": rules,
        "use_overrides": use_overrides,
    }


def get_global_risk_for_ui(env_mode: EnvironmentMode) -> Dict[str, Any]:
    """
    Get global risk settings for UI display.
    """
    from src.config import settings
    
    overrides = load_overrides(env_mode)
    use_overrides = (
        env_mode == EnvironmentMode.TEST and
        overrides.use_global_risk_overrides
    )
    
    fields = [
        {
            "key": "max_margin_pct",
            "label": "Max Margin Usage",
            "default_value": settings.max_margin_used_pct,
            "current_value": overrides.global_risk.max_margin_pct if use_overrides and overrides.global_risk.max_margin_pct is not None else settings.max_margin_used_pct,
            "unit": "%",
        },
        {
            "key": "max_net_delta",
            "label": "Max Net Delta",
            "default_value": settings.max_net_delta_abs,
            "current_value": overrides.global_risk.max_net_delta if use_overrides and overrides.global_risk.max_net_delta is not None else settings.max_net_delta_abs,
            "unit": "",
        },
        {
            "key": "daily_drawdown_limit_pct",
            "label": "Daily Drawdown Limit",
            "default_value": settings.daily_drawdown_limit_pct,
            "current_value": overrides.global_risk.daily_drawdown_limit_pct if use_overrides and overrides.global_risk.daily_drawdown_limit_pct is not None else settings.daily_drawdown_limit_pct,
            "unit": "%",
        },
        {
            "key": "liquidity_max_spread_pct",
            "label": "Max Bid/Ask Spread",
            "default_value": settings.liquidity_max_spread_pct,
            "current_value": overrides.global_risk.liquidity_max_spread_pct if use_overrides and overrides.global_risk.liquidity_max_spread_pct is not None else settings.liquidity_max_spread_pct,
            "unit": "%",
        },
        {
            "key": "liquidity_min_open_interest",
            "label": "Min Open Interest",
            "default_value": settings.liquidity_min_open_interest,
            "current_value": overrides.global_risk.liquidity_min_open_interest if use_overrides and overrides.global_risk.liquidity_min_open_interest is not None else settings.liquidity_min_open_interest,
            "unit": "contracts",
        },
    ]
    
    return {
        "env": env_mode.value,
        "fields": fields,
        "use_overrides": use_overrides,
    }


def get_bot_risk_for_ui(bot_id: str, env_mode: EnvironmentMode) -> Dict[str, Any]:
    """
    Get per-bot risk settings for UI display.
    """
    from src.config import settings
    
    overrides = load_overrides(env_mode)
    use_overrides = (
        env_mode == EnvironmentMode.TEST and
        overrides.use_bot_risk_overrides
    )
    
    bot_overrides = overrides.bots.get(bot_id, BotRiskOverrides())
    
    fields = [
        {
            "key": "max_equity_share",
            "label": "Max Equity Share",
            "default_value": 25.0,
            "current_value": bot_overrides.max_equity_share if use_overrides and bot_overrides.max_equity_share is not None else 25.0,
            "unit": "%",
        },
        {
            "key": "max_notional_usd_per_position",
            "label": "Max Notional USD/Position",
            "default_value": settings.greg_live_max_notional_usd_per_position,
            "current_value": bot_overrides.max_notional_usd_per_position if use_overrides and bot_overrides.max_notional_usd_per_position is not None else settings.greg_live_max_notional_usd_per_position,
            "unit": "USD",
        },
        {
            "key": "max_notional_usd_per_underlying",
            "label": "Max Notional USD/Underlying",
            "default_value": settings.greg_live_max_notional_usd_per_underlying,
            "current_value": bot_overrides.max_notional_usd_per_underlying if use_overrides and bot_overrides.max_notional_usd_per_underlying is not None else settings.greg_live_max_notional_usd_per_underlying,
            "unit": "USD",
        },
        {
            "key": "max_positions_per_underlying",
            "label": "Max Positions/Underlying",
            "default_value": 3,
            "current_value": bot_overrides.max_positions_per_underlying if use_overrides and bot_overrides.max_positions_per_underlying is not None else 3,
            "unit": "",
        },
    ]
    
    return {
        "env": env_mode.value,
        "bot_id": bot_id,
        "fields": fields,
        "use_overrides": use_overrides,
    }
