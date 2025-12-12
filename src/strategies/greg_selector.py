"""
Greg Mandolini VRP Harvester - ENTRY_ENGINE v8.0 Selector.

This module implements the strategy selection logic based on market sensors.
Phase 1 is read-only: computes sensors, runs decision waterfall, returns recommendation.
No trades are placed.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from src.models import AgentState


class GregSelectorSensors(BaseModel):
    """Sensor values used by the Greg selector decision waterfall (v8.0)."""
    vrp_30d: Optional[float] = Field(
        default=None,
        description="Volatility Risk Premium (IV_30d - RV_30d)"
    )
    vrp_7d: Optional[float] = Field(
        default=None,
        description="7-day VRP (IV_7d - RV_7d)"
    )
    front_rv_iv_ratio: Optional[float] = Field(
        default=None,
        description="Front RV/IV ratio (RV_7d / IV_7d)"
    )
    chop_factor_7d: Optional[float] = Field(
        default=None,
        description="Chop factor (RV_7d / IV_30d), < 0.6 = calm"
    )
    iv_rank_6m: Optional[float] = Field(
        default=None,
        description="IV Rank over 6 months (0-1 scale)"
    )
    term_structure_spread: Optional[float] = Field(
        default=None,
        description="IV_7d - IV_30d term structure spread"
    )
    skew_25d: Optional[float] = Field(
        default=None,
        description="25-delta skew (Put IV - Call IV)"
    )
    adx_14d: Optional[float] = Field(
        default=None,
        description="Average Directional Index (14-day)"
    )
    rsi_14d: Optional[float] = Field(
        default=None,
        description="Relative Strength Index (14-day)"
    )
    price_vs_ma200: Optional[float] = Field(
        default=None,
        description="Current price minus 200-day MA"
    )
    predicted_funding_rate: Optional[float] = Field(
        default=None,
        description="Predicted perpetual funding rate"
    )


class GregSelectorDecision(BaseModel):
    """Result of running the Greg selector decision waterfall."""
    selected_strategy: str = Field(
        ...,
        description="Selected strategy name (e.g., STRATEGY_A_STRADDLE) or NO_TRADE"
    )
    reasoning: str = Field(
        ...,
        description="Explanation for why this strategy was selected"
    )
    sensors: GregSelectorSensors = Field(
        ...,
        description="Sensor values used in the decision"
    )
    rule_index: int = Field(
        ...,
        description="Index of the matched rule in the decision waterfall (0-based)"
    )
    step_name: str = Field(
        default="",
        description="Name of the matched step in the waterfall"
    )
    meta: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata from the Greg selector spec"
    )


@lru_cache(maxsize=1)
def load_greg_spec() -> Dict[str, Any]:
    """Load and cache the Greg selector JSON specification."""
    spec_path = Path(__file__).parent.parent.parent / "docs" / "greg_mandolini" / "GREG_SELECTOR_RULES_FINAL.json"
    with open(spec_path, "r") as f:
        return json.load(f)


def clear_greg_spec_cache() -> None:
    """Clear the cached spec so it reloads on next call."""
    load_greg_spec.cache_clear()


def get_calibration_spec() -> Dict[str, Any]:
    """
    Get the calibration configuration from the Greg spec.
    Supports both v8.0 (global_entry_filters.calibration) and v6.0 (global_constraints.calibration).
    Returns the calibration block, or an empty dict if neither exists.
    """
    spec = load_greg_spec()
    
    calib = spec.get("global_entry_filters", {}).get("calibration", {})
    if calib:
        return calib
    
    calib = spec.get("global_constraints", {}).get("calibration", {})
    return calib


def get_calibration_spec_with_overrides(env_mode: Optional[str] = None) -> Dict[str, Any]:
    """
    Get calibration spec with TEST environment overrides applied.
    
    In TEST mode with entry rule overrides enabled, override values
    from data/bot_overrides_test.json take precedence over spec defaults.
    In LIVE mode or with overrides disabled, returns the base spec.
    
    Args:
        env_mode: Environment mode string ("test" or "live"). 
                  If None, uses settings.env_mode.
    
    Returns:
        Calibration dict with overrides applied where set.
    """
    from src.config import settings, EnvironmentMode
    from src.bots.overrides import load_overrides, EntryRuleOverrides
    
    base_calib = get_calibration_spec()
    
    if env_mode is None:
        effective_mode = settings.env_mode
    else:
        try:
            effective_mode = EnvironmentMode(env_mode)
        except ValueError:
            effective_mode = EnvironmentMode.LIVE
    
    if effective_mode != EnvironmentMode.TEST:
        return base_calib
    
    overrides = load_overrides(effective_mode)
    if not overrides.use_entry_rule_overrides:
        return base_calib
    
    greg_overrides = overrides.entry_rules.get("gregbot", EntryRuleOverrides())
    if not greg_overrides.thresholds:
        return base_calib
    
    result = dict(base_calib)
    for key, value in greg_overrides.thresholds.items():
        if "." in key:
            parts = key.split(".", 1)
            parent_key, child_key = parts
            if parent_key in result and isinstance(result[parent_key], dict):
                result[parent_key] = dict(result[parent_key])
                result[parent_key][child_key] = value
        else:
            result[key] = value
    
    return result


def build_sensors_from_state(state: AgentState) -> GregSelectorSensors:
    """
    Map AgentState / market_context / vol_state into the Greg selector sensors.
    Uses whatever fields are available; leaves others as None.
    
    Currently available mappings:
    - vrp_30d: Computed as btc_iv - btc_rv from vol_state
    - chop_factor_7d: Computed as rv_7d / btc_iv
    - skew_25d: From vol_state.btc_skew
    - price_vs_ma200: Computed from market_context.pct_from_200d_ma
    
    Not yet available (require external data sources):
    - iv_rank_6m: Requires 6-month IV history for percentile calculation
    - term_structure_spread: Requires term structure curve (7d vs 30d IV)
    - adx_14d: Requires 14-day price history for ADX calculation
    - rsi_14d: Requires 14-day price history for RSI calculation
    """
    sensors = GregSelectorSensors()
    
    vol = state.vol_state
    ctx = state.market_context
    spot_btc = state.spot.get("BTC", 0)
    
    btc_iv = vol.btc_iv if vol.btc_iv > 0 else None
    btc_rv = vol.btc_rv if vol.btc_rv > 0 else None
    
    if btc_rv is None and btc_iv is not None and vol.btc_ivrv > 0:
        btc_rv = btc_iv / vol.btc_ivrv
    
    rv_30d = None
    if ctx and ctx.realized_vol_30d > 0:
        rv_30d = ctx.realized_vol_30d
    elif btc_rv is not None:
        rv_30d = btc_rv
    
    if btc_iv is not None and rv_30d is not None:
        sensors.vrp_30d = btc_iv - rv_30d
    
    rv_7d = None
    if ctx and ctx.realized_vol_7d > 0:
        rv_7d = ctx.realized_vol_7d
    elif btc_rv is not None:
        rv_7d = btc_rv
    
    if rv_7d is not None and btc_iv is not None and btc_iv > 0:
        sensors.chop_factor_7d = rv_7d / btc_iv
    
    if vol.btc_skew != 0:
        sensors.skew_25d = vol.btc_skew
    
    if ctx and spot_btc > 0:
        if ctx.pct_from_200d_ma != 0:
            sensors.price_vs_ma200 = ctx.pct_from_200d_ma * spot_btc / 100.0
        elif ctx.pct_from_200d_ma == 0:
            sensors.price_vs_ma200 = 0.0
    
    return sensors


def _safe_check(value: Optional[float], op: str, threshold: float) -> bool:
    """Safely compare a sensor value, returning False if value is None."""
    if value is None:
        return False
    if op == ">":
        return value > threshold
    elif op == ">=":
        return value >= threshold
    elif op == "<":
        return value < threshold
    elif op == "<=":
        return value <= threshold
    elif op == "==":
        return value == threshold
    return False


def _get_calibration_context(spec: Dict[str, Any], env_mode: Optional[str] = None) -> Dict[str, float]:
    """
    Extract all calibration variables from the spec for expression evaluation.
    
    Supports both v6.0 (global_constraints.calibration) and v8.0 (global_entry_filters.calibration).
    Returns a dict with all numeric calibration values that can be used
    as variables in decision waterfall condition expressions.
    
    Args:
        spec: The loaded Greg spec (ignored if env_mode is provided, uses override-aware version)
        env_mode: If provided, uses get_calibration_spec_with_overrides for TEST override support
    """
    if env_mode is not None:
        calibration = get_calibration_spec_with_overrides(env_mode)
    else:
        calibration = spec.get("global_entry_filters", {}).get("calibration", {})
        if not calibration:
            calibration = spec.get("global_constraints", {}).get("calibration", {})
    
    ctx: Dict[str, float] = {}
    
    def extract_values(d: Dict[str, Any], prefix: str = "") -> None:
        for key, value in d.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, (int, float)):
                ctx[full_key] = float(value)
                if not prefix:
                    ctx[key] = float(value)
            elif isinstance(value, dict):
                extract_values(value, full_key)
    
    extract_values(calibration)
    
    if "skew_neutral_threshold" not in ctx:
        ctx["skew_neutral_threshold"] = 4.0
    if "min_vrp_floor" not in ctx:
        ctx["min_vrp_floor"] = 0.0
    if "min_vrp_directional" not in ctx:
        ctx["min_vrp_directional"] = 2.0
    if "rsi_thresholds.lower" not in ctx:
        ctx["rsi_thresholds.lower"] = 30.0
    if "rsi_thresholds.upper" not in ctx:
        ctx["rsi_thresholds.upper"] = 70.0
    
    return ctx


def evaluate_greg_selector(
    sensors: GregSelectorSensors,
    env_mode: Optional[str] = None,
) -> GregSelectorDecision:
    """
    Walk the decision waterfall from the JSON spec in order; first match wins.
    Supports v8.0 decision_waterfall format with branches.
    None values are treated as "condition not satisfied".
    
    Args:
        sensors: Sensor values for evaluation
        env_mode: Environment mode ("test" or "live") for TEST override support
    """
    spec = load_greg_spec()
    meta = spec.get("meta", {})
    
    decision_waterfall = spec.get("decision_waterfall", [])
    if not decision_waterfall:
        decision_waterfall = spec.get("decision_tree", [])
    
    calibration_ctx = _get_calibration_context(spec, env_mode=env_mode)
    
    for idx, rule in enumerate(decision_waterfall):
        step_name = rule.get("name", f"step_{idx}")
        condition = rule.get("condition", "")
        outcome = rule.get("outcome") or rule.get("selected_strategy", "NO_TRADE")
        reasoning = rule.get("reasoning", "")
        branches = rule.get("branches", [])
        
        if condition == "default" or (not condition and not branches and outcome):
            return GregSelectorDecision(
                selected_strategy=outcome,
                reasoning=reasoning or "No valid setup found.",
                sensors=sensors,
                rule_index=idx,
                step_name=step_name,
                meta=meta,
            )
        
        if branches:
            for branch in branches:
                branch_condition = branch.get("condition", "")
                branch_outcome = branch.get("outcome", "NO_TRADE")
                branch_reasoning = branch.get("reasoning", "")
                
                if _matches_condition(branch_condition, sensors, calibration_ctx):
                    return GregSelectorDecision(
                        selected_strategy=branch_outcome,
                        reasoning=branch_reasoning,
                        sensors=sensors,
                        rule_index=idx,
                        step_name=step_name,
                        meta=meta,
                    )
        elif condition and _matches_condition(condition, sensors, calibration_ctx):
            return GregSelectorDecision(
                selected_strategy=outcome,
                reasoning=reasoning,
                sensors=sensors,
                rule_index=idx,
                step_name=step_name,
                meta=meta,
            )
    
    return GregSelectorDecision(
        selected_strategy="NO_TRADE",
        reasoning="No valid setup found (fallback).",
        sensors=sensors,
        rule_index=len(decision_waterfall),
        step_name="FALLBACK",
        meta=meta,
    )


def _matches_condition(
    condition: str,
    sensors: GregSelectorSensors,
    calibration_ctx: Optional[Dict[str, float]] = None,
) -> bool:
    """
    Parse and evaluate a condition string against sensor values.
    Supports AND/OR operators, basic comparisons, ABS(), and calibration variables.
    """
    calibration_ctx = calibration_ctx or {}
    
    if " OR " in condition:
        parts = condition.split(" OR ")
        return any(_evaluate_simple_condition(p.strip(), sensors, calibration_ctx) for p in parts)
    
    if " AND " in condition:
        parts = condition.split(" AND ")
        return all(_evaluate_simple_condition(p.strip(), sensors, calibration_ctx) for p in parts)
    
    return _evaluate_simple_condition(condition.strip(), sensors, calibration_ctx)


def _resolve_value(
    expr: str,
    sensors: GregSelectorSensors,
    calibration_ctx: Dict[str, float],
) -> Optional[float]:
    """
    Resolve an expression to a float value.
    Supports:
    - Sensor names: vrp_30d, skew_25d, etc.
    - Calibration variables: skew_neutral_threshold, min_vrp_floor, rsi_thresholds.lower
    - ABS(sensor_name): absolute value of a sensor
    - Numeric literals: 15.0, 0.8, etc.
    - Expressions: (skew_neutral_threshold * -1)
    """
    expr = expr.strip()
    
    if expr.startswith("ABS(") and expr.endswith(")"):
        inner = expr[4:-1].strip()
        inner_val = _resolve_value(inner, sensors, calibration_ctx)
        if inner_val is None:
            return None
        return abs(inner_val)
    
    if expr.startswith("(") and expr.endswith(")"):
        inner = expr[1:-1].strip()
        if " * " in inner:
            parts = inner.split(" * ")
            if len(parts) == 2:
                left = _resolve_value(parts[0].strip(), sensors, calibration_ctx)
                right = _resolve_value(parts[1].strip(), sensors, calibration_ctx)
                if left is not None and right is not None:
                    return left * right
        return _resolve_value(inner, sensors, calibration_ctx)
    
    if expr in calibration_ctx:
        return calibration_ctx[expr]
    
    sensor_val = getattr(sensors, expr, None)
    if sensor_val is not None:
        return sensor_val
    
    try:
        return float(expr)
    except ValueError:
        return None


def _evaluate_simple_condition(
    cond: str,
    sensors: GregSelectorSensors,
    calibration_ctx: Dict[str, float],
) -> bool:
    """
    Evaluate a simple condition like 'adx_14d > 35' or 'ABS(skew_25d) < skew_neutral_threshold'.
    Returns False if any sensor value is None.
    """
    operators = [">=", "<=", ">", "<", "=="]
    
    for op in operators:
        if op in cond:
            idx = cond.find(op)
            left_expr = cond[:idx].strip()
            right_expr = cond[idx + len(op):].strip()
            
            left_val = _resolve_value(left_expr, sensors, calibration_ctx)
            right_val = _resolve_value(right_expr, sensors, calibration_ctx)
            
            if left_val is None or right_val is None:
                return False
            
            return _safe_check(left_val, op, right_val)
    
    return False
