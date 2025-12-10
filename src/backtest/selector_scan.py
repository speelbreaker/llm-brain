"""
Selector Frequency Scan module for backtesting.

Analyzes how often a selector's rules would allow trading in a synthetic universe.
Backtest-only - no orders, no Deribit calls, no DB writes.
"""
from __future__ import annotations

import math
import random
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Literal, Optional

import numpy as np
from pydantic import BaseModel, Field


class SelectorScanConfig(BaseModel):
    """Configuration for a selector frequency scan."""
    selector_id: Literal["greg"] = "greg"
    underlyings: List[Literal["BTC", "ETH"]] = ["BTC", "ETH"]
    num_paths: int = 1
    horizon_days: int = 365
    decision_interval_days: float = 1.0
    threshold_overrides: Dict[str, float] = Field(default_factory=dict)


class SelectorScanResult(BaseModel):
    """Result of a selector frequency scan."""
    summary: Dict[str, Dict[str, Dict[str, float]]] = Field(
        default_factory=dict,
        description="Per underlying, per strategy: {pass_count, total_steps, pass_pct}"
    )
    total_steps: Dict[str, int] = Field(
        default_factory=dict,
        description="Total decision points per underlying"
    )


class SelectorHeatmapConfig(BaseModel):
    """Configuration for a selector heatmap scan."""
    selector_id: str = "greg"
    underlying: str = "BTC"
    strategy_key: str = "STRATEGY_A_STRADDLE"
    metric_x: str = "vrp_30d_min"
    metric_y: str = "adx_14d_max"
    grid_x: List[float] = Field(default_factory=list)
    grid_y: List[float] = Field(default_factory=list)
    horizon_days: int = 365
    decision_interval_days: float = 1.0
    num_paths: int = 1
    base_threshold_overrides: Dict[str, float] = Field(default_factory=dict)


class SelectorHeatmapResult(BaseModel):
    """Result of a selector heatmap scan."""
    metric_x: str
    metric_y: str
    grid_x: List[float]
    grid_y: List[float]
    values: List[List[float]] = Field(
        default_factory=list,
        description="values[y_index][x_index] = pass_pct (0-1)"
    )


GREG_STRATEGIES = [
    "STRATEGY_A_STRADDLE",
    "STRATEGY_A_STRANGLE",
    "STRATEGY_B_CALENDAR",
    "STRATEGY_C_SHORT_PUT",
    "STRATEGY_D_IRON_BUTTERFLY",
    "STRATEGY_F_BULL_PUT_SPREAD",
    "STRATEGY_F_BEAR_CALL_SPREAD",
    "NO_TRADE",
]

GREG_STRATEGY_CRITERIA = {
    "STRATEGY_A_STRADDLE": [
        ("vrp_30d", "min", 15.0),
        ("chop_factor_7d", "max", 0.6),
        ("adx_14d", "max", 20.0),
    ],
    "STRATEGY_A_STRANGLE": [
        ("vrp_30d", "min", 10.0),
        ("chop_factor_7d", "max", 0.8),
        ("adx_14d", "max", 30.0),
    ],
    "STRATEGY_B_CALENDAR": [
        ("term_structure_spread", "min", 5.0),
        ("iv_rank_6m", "max", 0.80),
    ],
    "STRATEGY_C_SHORT_PUT": [
        ("skew_25d", "min", 5.0),
        ("price_vs_ma200", "min", 0.0),
        ("iv_rank_6m", "min", 0.50),
    ],
    "STRATEGY_D_IRON_BUTTERFLY": [
        ("iv_rank_6m", "min", 0.80),
        ("vrp_30d", "min", 10.0),
    ],
    "STRATEGY_F_BULL_PUT_SPREAD": [
        ("skew_25d", "min", 5.0),
        ("rsi_14d", "max", 30.0),
    ],
    "STRATEGY_F_BEAR_CALL_SPREAD": [
        ("skew_25d", "max", -5.0),
        ("rsi_14d", "min", 70.0),
    ],
    "NO_TRADE": [],
}


def _generate_synthetic_sensors(
    underlying: str,
    step_index: int,
    rng: np.random.Generator,
) -> Dict[str, Optional[float]]:
    """
    Generate synthetic sensor values for a given step.
    
    Uses realistic distributions based on typical crypto market behavior.
    """
    base_seed_offset = hash(underlying) % 1000
    
    base_iv = 55.0 if underlying == "BTC" else 65.0
    iv_cycle = 10.0 * math.sin(2 * math.pi * step_index / 60)
    iv_noise = rng.normal(0, 5)
    iv_30d = base_iv + iv_cycle + iv_noise
    iv_30d = max(20.0, min(150.0, iv_30d))
    
    base_rv = 45.0 if underlying == "BTC" else 55.0
    rv_cycle = 8.0 * math.sin(2 * math.pi * step_index / 45 + 0.5)
    rv_noise = rng.normal(0, 4)
    rv_30d = base_rv + rv_cycle + rv_noise
    rv_30d = max(15.0, min(120.0, rv_30d))
    
    vrp_30d = iv_30d - rv_30d
    
    rv_7d = rv_30d * (0.8 + rng.uniform(0, 0.4))
    chop_factor_7d = rv_7d / iv_30d if iv_30d > 0 else 0.5
    
    iv_low = 35.0 if underlying == "BTC" else 45.0
    iv_high = 100.0 if underlying == "BTC" else 120.0
    iv_rank_6m = (iv_30d - iv_low) / (iv_high - iv_low)
    iv_rank_6m = max(0.0, min(1.0, iv_rank_6m))
    
    term_spread_base = rng.normal(0, 3)
    term_structure_spread = term_spread_base + 2.0 * math.sin(2 * math.pi * step_index / 30)
    
    skew_base = rng.normal(2.0, 4.0)
    skew_25d = skew_base + 3.0 * math.sin(2 * math.pi * step_index / 90)
    
    adx_base = 25.0 + rng.normal(0, 8)
    adx_cycle = 10.0 * math.sin(2 * math.pi * step_index / 45)
    adx_14d = adx_base + adx_cycle
    adx_14d = max(10.0, min(60.0, adx_14d))
    
    rsi_base = 50.0 + rng.normal(0, 15)
    rsi_cycle = 20.0 * math.sin(2 * math.pi * step_index / 20)
    rsi_14d = rsi_base + rsi_cycle
    rsi_14d = max(10.0, min(90.0, rsi_14d))
    
    price_vs_ma200 = rng.normal(5.0, 15.0)
    
    return {
        "vrp_30d": vrp_30d,
        "chop_factor_7d": chop_factor_7d,
        "iv_rank_6m": iv_rank_6m,
        "term_structure_spread": term_structure_spread,
        "skew_25d": skew_25d,
        "adx_14d": adx_14d,
        "rsi_14d": rsi_14d,
        "price_vs_ma200": price_vs_ma200,
    }


def _check_strategy_passes(
    sensors: Dict[str, Optional[float]],
    strategy_key: str,
    threshold_overrides: Dict[str, float],
) -> bool:
    """
    Check if a strategy passes its criteria with optional threshold overrides.
    NO_TRADE is not a real strategy - it never "passes" for scanning purposes.
    """
    if strategy_key == "NO_TRADE":
        return False
    
    criteria = GREG_STRATEGY_CRITERIA.get(strategy_key, [])
    if not criteria:
        return False
    
    for metric, bound_type, default_threshold in criteria:
        value = sensors.get(metric)
        if value is None:
            return False
        
        override_key = f"{metric}_{bound_type}"
        threshold = threshold_overrides.get(override_key, default_threshold)
        
        if bound_type == "min":
            if value < threshold:
                return False
        elif bound_type == "max":
            if value > threshold:
                return False
    
    return True


def run_selector_scan(config: SelectorScanConfig) -> SelectorScanResult:
    """
    Run a selector frequency scan on the synthetic universe.
    
    For each synthetic path and each decision timestamp, build sensor values
    and run the selector (Greg for now), counting how often each strategy passes.
    """
    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    total_steps: Dict[str, int] = {}
    
    num_steps = int(config.horizon_days / config.decision_interval_days)
    
    for underlying in config.underlyings:
        strategy_counts: Dict[str, int] = {s: 0 for s in GREG_STRATEGIES}
        total = 0
        
        for path_idx in range(config.num_paths):
            seed = hash((underlying, path_idx, config.horizon_days)) % (2**31)
            rng = np.random.default_rng(seed)
            
            for step_idx in range(num_steps):
                sensors = _generate_synthetic_sensors(underlying, step_idx, rng)
                
                for strategy_key in GREG_STRATEGIES:
                    if _check_strategy_passes(sensors, strategy_key, config.threshold_overrides):
                        strategy_counts[strategy_key] += 1
                
                total += 1
        
        strategy_summary: Dict[str, Dict[str, float]] = {}
        for strategy_key, count in strategy_counts.items():
            pass_pct = count / total if total > 0 else 0.0
            strategy_summary[strategy_key] = {
                "pass_count": float(count),
                "total_steps": float(total),
                "pass_pct": pass_pct,
            }
        
        summary[underlying] = strategy_summary
        total_steps[underlying] = total
    
    return SelectorScanResult(summary=summary, total_steps=total_steps)


def run_selector_heatmap(config: SelectorHeatmapConfig) -> SelectorHeatmapResult:
    """
    For a given selector, underlying and strategy, sweep two threshold metrics
    over a grid and compute the pass% for that strategy at each grid point.
    """
    values: List[List[float]] = []
    
    for y_val in config.grid_y:
        row: List[float] = []
        for x_val in config.grid_x:
            overrides = dict(config.base_threshold_overrides or {})
            overrides[config.metric_x] = x_val
            overrides[config.metric_y] = y_val
            
            scan_cfg = SelectorScanConfig(
                selector_id=config.selector_id,
                underlyings=[config.underlying],
                num_paths=config.num_paths,
                horizon_days=config.horizon_days,
                decision_interval_days=config.decision_interval_days,
                threshold_overrides=overrides,
            )
            scan_result = run_selector_scan(scan_cfg)
            
            underlying_summary = scan_result.summary.get(config.underlying, {})
            strat_stats = underlying_summary.get(config.strategy_key, {})
            pass_pct = float(strat_stats.get("pass_pct", 0.0))
            
            row.append(pass_pct)
        values.append(row)
    
    return SelectorHeatmapResult(
        metric_x=config.metric_x,
        metric_y=config.metric_y,
        grid_x=config.grid_x,
        grid_y=config.grid_y,
        values=values,
    )
