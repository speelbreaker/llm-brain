"""
Core calibration logic for IV multiplier fitting.

This module provides the shared calibration logic that can be used by:
- auto_calibrate_iv.py CLI script (using harvester data)
- API endpoints for calibration

Uses Black-Scholes pricing to fit an IV multiplier that minimizes MAE
between synthetic prices and observed mark prices.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from src.calibration import black_scholes_call_price


@dataclass
class CalibrationSample:
    """A single option sample for calibration."""
    underlying: str
    strike: float
    dte_days: float
    underlying_price: float
    mark_price: float
    mark_iv: float


@dataclass
class CalibrationFitResult:
    """Result of IV multiplier fitting."""
    best_multiplier: float
    mae_pct: float
    num_samples: int
    multiplier_range: Tuple[float, float]
    search_points: int


def extract_calibration_samples(
    df: pd.DataFrame,
    underlying: str,
    dte_min: int,
    dte_max: int,
    moneyness_range: float = 0.10,
    max_samples: int = 5000,
) -> List[CalibrationSample]:
    """
    Extract calibration samples from a harvester DataFrame.
    
    Filters to:
    - Specified underlying
    - DTE in [dte_min, dte_max]
    - Call options only
    - Near-ATM (within moneyness_range)
    - Valid mark_iv and mark_price
    
    Args:
        df: DataFrame from build_live_deribit_exam_dataset
        underlying: Asset symbol (BTC, ETH)
        dte_min: Minimum days to expiry
        dte_max: Maximum days to expiry
        moneyness_range: Max |strike/underlying_price - 1.0| (default 0.10 = ±10%)
        max_samples: Maximum number of samples to return (randomly sampled if exceeded)
    
    Returns:
        List of CalibrationSample objects
    """
    underlying = underlying.upper()
    
    required_cols = ["underlying", "option_type", "strike", "underlying_price", "mark_price", "mark_iv"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    if "dte_days" not in df.columns:
        raise ValueError("DataFrame missing dte_days column. Use build_live_deribit_exam_dataset to compute it.")
    
    mask = (
        (df["underlying"].str.upper() == underlying) &
        (df["option_type"].str.upper().isin(["CALL", "C"])) &
        (df["dte_days"] >= dte_min) &
        (df["dte_days"] <= dte_max) &
        (df["mark_iv"].notna()) &
        (df["mark_iv"] > 0) &
        (df["mark_price"].notna()) &
        (df["mark_price"] > 0) &
        (df["underlying_price"].notna()) &
        (df["underlying_price"] > 0) &
        (df["strike"].notna()) &
        (df["strike"] > 0)
    )
    
    filtered = df[mask].copy()
    
    if len(filtered) == 0:
        return []
    
    filtered["moneyness"] = abs(filtered["strike"] / filtered["underlying_price"] - 1.0)
    filtered = filtered[filtered["moneyness"] <= moneyness_range]
    
    if len(filtered) == 0:
        return []
    
    if len(filtered) > max_samples:
        filtered = filtered.sample(n=max_samples, random_state=42)
    
    samples = []
    for _, row in filtered.iterrows():
        samples.append(CalibrationSample(
            underlying=underlying,
            strike=float(row["strike"]),
            dte_days=float(row["dte_days"]),
            underlying_price=float(row["underlying_price"]),
            mark_price=float(row["mark_price"]),
            mark_iv=float(row["mark_iv"]) / 100.0,
        ))
    
    return samples


def calibrate_iv_multiplier(
    samples: List[CalibrationSample],
    multiplier_min: float = 0.5,
    multiplier_max: float = 1.5,
    search_points: int = 100,
) -> CalibrationFitResult:
    """
    Find the IV multiplier that minimizes MAE between synthetic and mark prices.
    
    Uses a grid search over the specified multiplier range.
    
    Args:
        samples: List of CalibrationSample objects
        multiplier_min: Lower bound for multiplier search
        multiplier_max: Upper bound for multiplier search
        search_points: Number of points in the grid search
    
    Returns:
        CalibrationFitResult with the best multiplier and metrics
    """
    if not samples:
        return CalibrationFitResult(
            best_multiplier=1.0,
            mae_pct=0.0,
            num_samples=0,
            multiplier_range=(multiplier_min, multiplier_max),
            search_points=search_points,
        )
    
    multipliers = np.linspace(multiplier_min, multiplier_max, search_points)
    
    strikes = np.array([s.strike for s in samples])
    dtes = np.array([s.dte_days for s in samples])
    spots = np.array([s.underlying_price for s in samples])
    marks = np.array([s.mark_price for s in samples])
    ivs = np.array([s.mark_iv for s in samples])
    
    t_years = np.maximum(0.0001, dtes / 365.0)
    
    best_multiplier = 1.0
    best_mae = float("inf")
    
    for m in multipliers:
        sigma = ivs * m
        
        errors_pct = []
        for i in range(len(samples)):
            syn_price = black_scholes_call_price(
                spot=spots[i],
                strike=strikes[i],
                t_years=t_years[i],
                sigma=sigma[i],
                r=0.0,
            )
            
            if marks[i] > 0:
                err = abs(syn_price - marks[i]) / marks[i] * 100.0
            else:
                err = 0.0
            errors_pct.append(err)
        
        mae = np.mean(errors_pct)
        
        if mae < best_mae:
            best_mae = mae
            best_multiplier = m
    
    return CalibrationFitResult(
        best_multiplier=round(best_multiplier, 6),
        mae_pct=round(best_mae, 6),
        num_samples=len(samples),
        multiplier_range=(multiplier_min, multiplier_max),
        search_points=search_points,
    )
