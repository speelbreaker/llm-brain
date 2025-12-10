"""
Greg Mandolini VRP Harvester - Phase 1 Master Selector.

This module implements the strategy selection logic based on market sensors.
Phase 1 is read-only: computes sensors, runs decision tree, returns recommendation.
No trades are placed.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from src.models import AgentState


class GregSelectorSensors(BaseModel):
    """Sensor values used by the Greg selector decision tree (v6.0)."""
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
    """Result of running the Greg selector decision tree."""
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
        description="Index of the matched rule in the decision tree (0-based)"
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


def _get_calibration_context(spec: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract all calibration variables from the spec for expression evaluation.
    
    Returns a dict with all numeric calibration values that can be used
    as variables in decision tree condition expressions.
    """
    calibration = spec.get("global_constraints", {}).get("calibration", {})
    ctx: Dict[str, float] = {}
    for key, value in calibration.items():
        if isinstance(value, (int, float)):
            ctx[key] = float(value)
    if "skew_neutral_threshold" not in ctx:
        ctx["skew_neutral_threshold"] = 4.0
    if "min_vrp_floor" not in ctx:
        ctx["min_vrp_floor"] = 0.0
    return ctx


def evaluate_greg_selector(sensors: GregSelectorSensors) -> GregSelectorDecision:
    """
    Walk the decision tree from the JSON spec in order; first match wins.
    Implements each condition explicitly in Python (no eval).
    None values are treated as "condition not satisfied".
    Supports v6.0 calibration variables and ABS() function.
    """
    spec = load_greg_spec()
    meta = spec.get("meta", {})
    decision_tree = spec.get("decision_tree", [])
    calibration_ctx = _get_calibration_context(spec)
    
    for idx, rule in enumerate(decision_tree):
        condition = rule.get("condition", "")
        selected_strategy = rule.get("selected_strategy", "NO_TRADE")
        reasoning = rule.get("reasoning", "")
        
        if condition == "default":
            return GregSelectorDecision(
                selected_strategy=selected_strategy,
                reasoning=reasoning,
                sensors=sensors,
                rule_index=idx,
                meta=meta,
            )
        
        if _matches_condition(condition, sensors, calibration_ctx):
            return GregSelectorDecision(
                selected_strategy=selected_strategy,
                reasoning=reasoning,
                sensors=sensors,
                rule_index=idx,
                meta=meta,
            )
    
    return GregSelectorDecision(
        selected_strategy="NO_TRADE",
        reasoning="No valid setup found (fallback).",
        sensors=sensors,
        rule_index=len(decision_tree),
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
    - Calibration variables: skew_neutral_threshold, min_vrp_floor
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
