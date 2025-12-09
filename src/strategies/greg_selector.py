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
    """Sensor values used by the Greg selector decision tree."""
    vrp_30d: Optional[float] = Field(
        default=None,
        description="Volatility Risk Premium (IV_30d - RV_30d)"
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
    rv_30d = ctx.realized_vol_30d if ctx and ctx.realized_vol_30d > 0 else btc_rv
    
    if btc_iv is not None and rv_30d is not None:
        sensors.vrp_30d = btc_iv - rv_30d
    
    rv_7d = None
    if ctx and ctx.realized_vol_7d > 0:
        rv_7d = ctx.realized_vol_7d
    
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


def evaluate_greg_selector(sensors: GregSelectorSensors) -> GregSelectorDecision:
    """
    Walk the decision tree from the JSON spec in order; first match wins.
    Implements each condition explicitly in Python (no eval).
    None values are treated as "condition not satisfied".
    """
    spec = load_greg_spec()
    meta = spec.get("meta", {})
    decision_tree = spec.get("decision_tree", [])
    
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
        
        if _matches_condition(condition, sensors):
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


def _matches_condition(condition: str, sensors: GregSelectorSensors) -> bool:
    """
    Parse and evaluate a condition string against sensor values.
    Supports AND/OR operators and basic comparisons.
    """
    if " OR " in condition:
        parts = condition.split(" OR ")
        return any(_evaluate_simple_condition(p.strip(), sensors) for p in parts)
    
    if " AND " in condition:
        parts = condition.split(" AND ")
        return all(_evaluate_simple_condition(p.strip(), sensors) for p in parts)
    
    return _evaluate_simple_condition(condition.strip(), sensors)


def _evaluate_simple_condition(cond: str, sensors: GregSelectorSensors) -> bool:
    """
    Evaluate a simple condition like 'adx_14d > 35' or 'vrp_30d >= 10.0'.
    Returns False if the sensor value is None.
    """
    operators = [">=", "<=", ">", "<", "=="]
    
    for op in operators:
        if op in cond:
            parts = cond.split(op)
            if len(parts) == 2:
                sensor_name = parts[0].strip()
                try:
                    threshold = float(parts[1].strip())
                except ValueError:
                    return False
                
                sensor_value = getattr(sensors, sensor_name, None)
                return _safe_check(sensor_value, op, threshold)
    
    return False
