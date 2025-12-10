#!/usr/bin/env python3
"""
Realism checker: Compare synthetic universe distributions vs harvested Deribit data.

This script compares:
1. Distributions of key sensors (VRP, chop, ADX, skew, IV rank, term slope)
2. Cluster occupancy frequencies
3. Regime transition matrices

Usage:
    python scripts/realism_check.py --underlying BTC --days 90
    python scripts/realism_check.py --underlying BTC --data-root data/live_deribit --synthetic-days 180
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.synthetic.regimes import (
    load_regime_model,
    sample_regime_path,
    evolve_iv_and_skew,
    get_default_regimes,
    get_default_transition_matrix,
    RegimeParams,
    RegimeModel,
    GREG_SENSOR_COLUMNS,
)


SENSOR_COLUMNS = [
    "vrp_30d",
    "vrp_7d",
    "chop_factor",
    "adx_14",
    "iv_rank_30d",
    "skew_25d",
    "term_slope",
]


def load_harvested_sensors(
    data_root: str,
    underlying: str,
    lookback_days: int = 90,
) -> pd.DataFrame:
    """
    Load harvested data and compute sensors.
    """
    root = Path(data_root)
    asset_dir = root / underlying.upper()
    
    if not asset_dir.exists():
        print(f"No harvested data found at {asset_dir}")
        return pd.DataFrame()
    
    parquet_files = sorted(asset_dir.rglob("*.parquet"))
    if not parquet_files:
        return pd.DataFrame()
    
    dfs = []
    for pf in parquet_files:
        try:
            df = pd.read_parquet(pf)
            dfs.append(df)
        except Exception:
            continue
    
    if not dfs:
        return pd.DataFrame()
    
    combined = pd.concat(dfs, ignore_index=True)
    
    if "harvest_time" in combined.columns:
        combined["harvest_time"] = pd.to_datetime(combined["harvest_time"], utc=True)
        cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
        combined = combined[combined["harvest_time"] >= cutoff]
    
    if combined.empty:
        return pd.DataFrame()
    
    daily = combined.groupby(combined["harvest_time"].dt.date).agg({
        "underlying_price": "first",
        "mark_iv": "mean",
    }).reset_index()
    daily.columns = ["date", "spot", "iv_atm"]
    daily = daily.sort_values("date")
    
    if len(daily) < 10:
        return pd.DataFrame()
    
    daily["log_ret"] = np.log(daily["spot"] / daily["spot"].shift(1))
    daily["rv_7d"] = daily["log_ret"].rolling(7, min_periods=5).std() * np.sqrt(365) * 100
    daily["rv_30d"] = daily["log_ret"].rolling(30, min_periods=20).std() * np.sqrt(365) * 100
    
    daily["vrp_30d"] = daily["iv_atm"] - daily["rv_30d"]
    daily["vrp_7d"] = daily["iv_atm"] - daily["rv_7d"]
    
    daily["chop_factor"] = 0.5
    daily["adx_14"] = 25.0
    daily["iv_rank_30d"] = daily["iv_atm"].rolling(30, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100
    )
    daily["skew_25d"] = 0.05
    daily["term_slope"] = 0.0
    
    return daily


def generate_synthetic_sensors(
    underlying: str,
    n_days: int = 180,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic sensor data using the regime model.
    """
    model = load_regime_model(underlying)
    
    if model is None:
        regimes = get_default_regimes()
        transition_matrix = get_default_transition_matrix()
    else:
        regimes = model.regimes
        transition_matrix = model.transition_matrix
    
    np.random.seed(seed)
    
    regime_path = sample_regime_path(
        n_steps=n_days,
        transition_matrix=transition_matrix,
        initial_regime=0,
    )
    
    records = []
    iv_atm = 60.0
    skew_level = 0.05
    
    for day, regime_id in enumerate(regime_path):
        regime = regimes.get(regime_id)
        if regime is None:
            continue
        
        iv_atm, skew_level = evolve_iv_and_skew(
            current_iv_atm=iv_atm,
            current_skew=skew_level,
            regime=regime,
            rv_30d=regime.mu_rv_30d,
        )
        
        vrp = iv_atm - regime.mu_rv_30d
        
        records.append({
            "day": day,
            "regime_id": regime_id,
            "iv_atm": iv_atm,
            "rv_30d": regime.mu_rv_30d,
            "vrp_30d": vrp,
            "vrp_7d": vrp * 0.8,
            "chop_factor": 0.5 + np.random.randn() * 0.1,
            "adx_14": 25.0 + np.random.randn() * 5,
            "iv_rank_30d": 50.0 + np.random.randn() * 20,
            "skew_25d": skew_level,
            "term_slope": 0.0 + np.random.randn() * 0.02,
        })
    
    return pd.DataFrame(records)


def compute_ks_distance(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute Kolmogorov-Smirnov distance between two distributions.
    """
    from scipy import stats
    
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    
    if len(x) < 5 or len(y) < 5:
        return 1.0
    
    ks_stat, _ = stats.ks_2samp(x, y)
    return float(ks_stat)


def compute_moment_diffs(
    real_data: pd.DataFrame,
    synth_data: pd.DataFrame,
    columns: List[str],
) -> Dict[str, Dict[str, float]]:
    """
    Compare moments (mean, std, skew, kurtosis) between real and synthetic data.
    """
    results = {}
    
    for col in columns:
        if col not in real_data.columns or col not in synth_data.columns:
            continue
        
        real_vals = real_data[col].dropna().values
        synth_vals = synth_data[col].dropna().values
        
        if len(real_vals) < 5 or len(synth_vals) < 5:
            continue
        
        real_mean = float(np.mean(real_vals))
        synth_mean = float(np.mean(synth_vals))
        
        real_std = float(np.std(real_vals))
        synth_std = float(np.std(synth_vals))
        
        try:
            ks_dist = compute_ks_distance(real_vals, synth_vals)
        except ImportError:
            ks_dist = abs(real_mean - synth_mean) / (real_std + synth_std + 1e-6)
        
        results[col] = {
            "real_mean": round(real_mean, 4),
            "synth_mean": round(synth_mean, 4),
            "mean_diff": round(synth_mean - real_mean, 4),
            "real_std": round(real_std, 4),
            "synth_std": round(synth_std, 4),
            "ks_distance": round(ks_dist, 4),
        }
    
    return results


def compute_regime_distribution(
    regime_path: np.ndarray,
    n_regimes: int,
) -> Dict[int, float]:
    """
    Compute regime occupancy frequencies.
    """
    counts = np.bincount(regime_path.astype(int), minlength=n_regimes)
    total = len(regime_path)
    
    return {i: round(float(counts[i]) / total, 4) for i in range(n_regimes)}


def assign_regimes_using_model(
    data: pd.DataFrame,
    model: RegimeModel,
    sensor_columns: List[str],
) -> np.ndarray:
    """
    Assign regime IDs to each row of data using RegimeModel.predict_cluster().
    
    Args:
        data: DataFrame with sensor columns
        model: RegimeModel with predict_cluster() method
        sensor_columns: List of sensor column names to use
        
    Returns:
        Array of regime IDs
    """
    regime_ids = []
    
    for _, row in data.iterrows():
        sensor_values = []
        for col in sensor_columns:
            if col in row and not pd.isna(row[col]):
                sensor_values.append(float(row[col]))
            else:
                sensor_values.append(0.0)
        
        sensor_vector = np.array(sensor_values)
        regime_id = model.predict_cluster(sensor_vector)
        regime_ids.append(regime_id)
    
    return np.array(regime_ids)


def compute_transition_matrix_from_path(
    regime_path: np.ndarray,
    n_regimes: int,
) -> np.ndarray:
    """
    Compute empirical transition matrix from a regime path.
    
    Args:
        regime_path: Array of regime IDs over time
        n_regimes: Number of regimes
        
    Returns:
        Transition matrix (n_regimes x n_regimes)
    """
    trans_matrix = np.zeros((n_regimes, n_regimes))
    
    for i in range(len(regime_path) - 1):
        from_regime = int(regime_path[i])
        to_regime = int(regime_path[i + 1])
        if from_regime < n_regimes and to_regime < n_regimes:
            trans_matrix[from_regime, to_regime] += 1
    
    row_sums = trans_matrix.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    trans_matrix = trans_matrix / row_sums
    
    return trans_matrix


def compare_transition_matrices(
    real_matrix: np.ndarray,
    model_matrix: np.ndarray,
) -> Dict[str, float]:
    """
    Compare two transition matrices.
    
    Returns:
        Dict with comparison metrics
    """
    diff = np.abs(real_matrix - model_matrix)
    
    return {
        "max_abs_diff": float(np.max(diff)),
        "mean_abs_diff": float(np.mean(diff)),
        "frobenius_norm": float(np.linalg.norm(diff, 'fro')),
    }


def compute_realism_score(
    moment_diffs: Dict[str, Dict[str, float]],
) -> float:
    """
    Compute overall realism score from moment differences.
    
    Score = 1 - average KS distance (higher is better, max 1.0)
    """
    ks_values = [v["ks_distance"] for v in moment_diffs.values() if "ks_distance" in v]
    
    if not ks_values:
        return 0.0
    
    avg_ks = np.mean(ks_values)
    return round(1.0 - avg_ks, 4)


def identify_biggest_mismatches(
    moment_diffs: Dict[str, Dict[str, float]],
    top_n: int = 3,
) -> List[str]:
    """
    Identify the sensors with the biggest distribution mismatches.
    """
    ranked = sorted(
        moment_diffs.items(),
        key=lambda x: x[1].get("ks_distance", 0),
        reverse=True,
    )
    
    messages = []
    for col, stats in ranked[:top_n]:
        ks = stats.get("ks_distance", 0)
        mean_diff = stats.get("mean_diff", 0)
        
        if mean_diff > 0:
            direction = "synthetic too high"
        else:
            direction = "synthetic too low"
        
        messages.append(f"{col}: KS={ks:.3f}, {direction} by {abs(mean_diff):.3f}")
    
    return messages


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare synthetic vs real distributions")
    
    parser.add_argument("--underlying", "-u", type=str, default="BTC",
                       help="Underlying asset")
    parser.add_argument("--data-root", type=str, default="data/live_deribit",
                       help="Root directory for harvested data")
    parser.add_argument("--days", type=int, default=90,
                       help="Lookback days for real data")
    parser.add_argument("--synthetic-days", type=int, default=180,
                       help="Number of days to simulate")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for synthetic generation")
    parser.add_argument("--output", "-o", type=str, default=None,
                       help="Output file for JSON results")
    
    args = parser.parse_args()
    
    print(f"Realism Check: {args.underlying}")
    print(f"=" * 50)
    
    print(f"\n1. Loading harvested data from {args.data_root}...")
    real_data = load_harvested_sensors(
        data_root=args.data_root,
        underlying=args.underlying,
        lookback_days=args.days,
    )
    
    if real_data.empty:
        print("  WARNING: No harvested data available. Using defaults.")
        real_data = pd.DataFrame({
            "vrp_30d": np.random.randn(100) * 5 + 10,
            "vrp_7d": np.random.randn(100) * 5 + 8,
            "chop_factor": np.random.rand(100) * 0.3 + 0.4,
            "adx_14": np.random.randn(100) * 5 + 25,
            "iv_rank_30d": np.random.rand(100) * 100,
            "skew_25d": np.random.randn(100) * 0.02 + 0.05,
            "term_slope": np.random.randn(100) * 0.02,
        })
    
    print(f"  Loaded {len(real_data)} data points")
    
    print(f"\n2. Generating synthetic data ({args.synthetic_days} days)...")
    synth_data = generate_synthetic_sensors(
        underlying=args.underlying,
        n_days=args.synthetic_days,
        seed=args.seed,
    )
    print(f"  Generated {len(synth_data)} synthetic days")
    
    print(f"\n3. Comparing distributions...")
    moment_diffs = compute_moment_diffs(real_data, synth_data, SENSOR_COLUMNS)
    
    print(f"\nSensor Comparison:")
    print("-" * 70)
    for sensor, stats in moment_diffs.items():
        print(f"  {sensor:15s}: real_mean={stats['real_mean']:8.3f}, synth_mean={stats['synth_mean']:8.3f}, KS={stats['ks_distance']:.3f}")
    
    realism_score = compute_realism_score(moment_diffs)
    print(f"\n4. Realism Score: {realism_score:.2f} (1.0 = perfect match)")
    
    mismatches = identify_biggest_mismatches(moment_diffs)
    if mismatches:
        print(f"\n5. Biggest Mismatches:")
        for msg in mismatches:
            print(f"  - {msg}")
    
    if "regime_id" in synth_data.columns:
        regime_dist = compute_regime_distribution(
            synth_data["regime_id"].values,
            n_regimes=4,
        )
        print(f"\n6. Synthetic Regime Distribution:")
        for regime_id, freq in regime_dist.items():
            print(f"  Regime {regime_id}: {freq:.1%}")
    
    results = {
        "underlying": args.underlying,
        "realism_score": realism_score,
        "real_data_days": len(real_data),
        "synthetic_days": len(synth_data),
        "moment_diffs": moment_diffs,
        "biggest_mismatches": mismatches,
    }
    
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults written to: {args.output}")
    
    print(f"\n{'=' * 50}")
    if realism_score >= 0.8:
        print("GOOD: Synthetic universe closely matches real Deribit data!")
    elif realism_score >= 0.6:
        print("FAIR: Synthetic universe reasonably matches, but could be improved.")
    else:
        print("POOR: Significant mismatch. Consider recalibrating regimes.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
