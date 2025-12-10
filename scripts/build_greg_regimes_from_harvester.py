#!/usr/bin/env python3
"""
Build Greg-sensor cluster regimes from harvester data.

This script:
1. Loads harvested data and/or Greg-sensor CSVs
2. Computes or loads daily Greg sensors per underlying
3. Runs KMeans clustering on sensor data
4. Fits RegimeParams from cluster statistics
5. Estimates Markov transition matrix between regimes
6. Saves results to data/greg_regimes.json

Usage:
    python scripts/build_greg_regimes_from_harvester.py --underlying BTC
    python scripts/build_greg_regimes_from_harvester.py --underlying ETH --n-clusters 4
    python scripts/build_greg_regimes_from_harvester.py --underlying BTC --lookback-days 180

Output:
    data/greg_regimes.json - RegimeParams and transition matrix per underlying
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from math import log, sqrt
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.synthetic.regimes import (
    cluster_greg_sensors_from_real_data,
    fit_regime_params_from_clusters,
    estimate_transition_matrix,
    RegimeModel,
    save_regime_model,
    GREG_REGIME_FILE,
)


def load_harvester_data(underlying: str, lookback_days: int = 90) -> pd.DataFrame:
    """
    Load harvester data from CSV files.
    
    Args:
        underlying: BTC or ETH
        lookback_days: Number of days of history to load
        
    Returns:
        DataFrame with date index and sensor-like columns
    """
    data_dir = Path("data/harvester")
    
    if not data_dir.exists():
        print(f"Harvester data directory not found: {data_dir}")
        return pd.DataFrame()
    
    pattern = f"{underlying.lower()}_*.parquet"
    parquet_files = list(data_dir.glob(pattern))
    
    if not parquet_files:
        pattern = f"{underlying.lower()}_*.csv"
        csv_files = list(data_dir.glob(pattern))
        if csv_files:
            dfs = []
            for f in csv_files:
                try:
                    df = pd.read_csv(f, parse_dates=["timestamp"])
                    dfs.append(df)
                except Exception as e:
                    print(f"Error loading {f}: {e}")
            if dfs:
                return pd.concat(dfs, ignore_index=True)
    else:
        dfs = []
        for f in parquet_files:
            try:
                df = pd.read_parquet(f)
                dfs.append(df)
            except Exception as e:
                print(f"Error loading {f}: {e}")
        if dfs:
            return pd.concat(dfs, ignore_index=True)
    
    return pd.DataFrame()


def compute_sensors_from_ohlc(
    spot_df: pd.DataFrame,
    iv_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Compute Greg-style sensors from OHLC and IV data.
    
    Args:
        spot_df: DataFrame with OHLC columns and date index
        iv_df: Optional DataFrame with IV data
        
    Returns:
        DataFrame with sensor columns
    """
    if spot_df.empty:
        return pd.DataFrame()
    
    df = spot_df.copy()
    
    if "close" not in df.columns:
        return pd.DataFrame()
    
    df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
    
    df["rv_7d"] = df["log_ret"].rolling(7, min_periods=5).std() * np.sqrt(365) * 100
    df["rv_30d"] = df["log_ret"].rolling(30, min_periods=20).std() * np.sqrt(365) * 100
    
    df["ma_200"] = df["close"].rolling(200, min_periods=50).mean()
    df["price_vs_ma200"] = ((df["close"] - df["ma_200"]) / (df["ma_200"] + 1e-9)) * 100
    
    df["rsi_14d"] = _compute_rsi_series(df["close"], 14)
    
    df["adx_14d"] = _compute_adx_series(df, 14)
    
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"] - df["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(7, min_periods=5).mean()
    range_size = df["high"].rolling(7, min_periods=5).max() - df["low"].rolling(7, min_periods=5).min()
    df["chop_factor_7d"] = (
        100 * np.log10(atr.rolling(7, min_periods=5).sum() / (range_size + 1e-9))
    ) / np.log10(7)
    df["chop_factor_7d"] = df["chop_factor_7d"].clip(0, 1)
    
    if iv_df is not None and "iv_atm" in iv_df.columns:
        df = df.join(iv_df[["iv_atm"]], how="left")
        df["iv_atm_30d"] = df["iv_atm"].rolling(30, min_periods=10).mean()
        df["vrp_30d"] = df["iv_atm_30d"] - df["rv_30d"]
        df["vrp_7d"] = df["iv_atm"].rolling(7, min_periods=5).mean() - df["rv_7d"]
        
        iv_6m = df["iv_atm"].rolling(180, min_periods=30).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100)
        df["iv_rank_6m"] = iv_6m
    else:
        df["iv_atm_30d"] = df["rv_30d"] * 1.1
        df["vrp_30d"] = df["iv_atm_30d"] - df["rv_30d"]
        df["vrp_7d"] = df["vrp_30d"]
        df["iv_rank_6m"] = 50.0
    
    df["term_structure_spread"] = 0.0
    df["skew_25d"] = 0.05
    
    for col in ["rv_30d", "adx_14d", "vrp_30d", "chop_factor_7d", "rsi_14d"]:
        if col in df.columns:
            df[col] = df[col].ffill().bfill()
    
    return df


def _compute_rsi_series(closes: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI as a Series."""
    delta = closes.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _compute_adx_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute ADX as a Series."""
    high = df["high"]
    low = df["low"]
    close = df["close"]
    
    plus_dm = high.diff()
    minus_dm = low.diff().abs() * -1
    
    plus_dm = plus_dm.where((plus_dm > minus_dm.abs()) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.abs().where((minus_dm.abs() > plus_dm) & (minus_dm.abs() > 0), 0.0)
    
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.rolling(window=period).mean()
    
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / (atr + 1e-9))
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / (atr + 1e-9))
    
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
    adx = dx.rolling(window=period).mean()
    
    return adx


def fetch_spot_ohlc_from_api(underlying: str, lookback_days: int = 250) -> pd.DataFrame:
    """
    Fetch OHLC data from Deribit API.
    
    Args:
        underlying: BTC or ETH
        lookback_days: Number of days of history
        
    Returns:
        DataFrame with OHLC columns
    """
    try:
        from src.deribit_client import DeribitClient
        
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=lookback_days)
        
        with DeribitClient() as client:
            index_name = f"{underlying.lower()}_usd"
            try:
                res = client.get_tradingview_chart_data(
                    instrument_name=index_name,
                    start=start,
                    end=end,
                    resolution="1D",
                )
            except Exception:
                perp_name = f"{underlying}-PERPETUAL"
                res = client.get_tradingview_chart_data(
                    instrument_name=perp_name,
                    start=start,
                    end=end,
                    resolution="1D",
                )
        
        if not res.get("ticks"):
            return pd.DataFrame()
        
        timestamps = [
            datetime.fromtimestamp(ts / 1000, tz=timezone.utc) for ts in res["ticks"]
        ]
        df = pd.DataFrame(
            {
                "open": res["open"],
                "high": res["high"],
                "low": res["low"],
                "close": res["close"],
                "volume": res.get("volume", [0] * len(res["open"])),
            },
            index=pd.DatetimeIndex(timestamps, name="timestamp"),
        )
        return df
    except Exception as e:
        print(f"Error fetching OHLC from API: {e}")
        return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(
        description="Build Greg-sensor cluster regimes from harvester data"
    )
    parser.add_argument(
        "--underlying",
        type=str,
        required=True,
        choices=["BTC", "ETH", "btc", "eth"],
        help="Underlying asset (BTC or ETH)",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=6,
        help="Number of regime clusters (default: 6)",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=180,
        help="Days of history to use (default: 180)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path (default: data/greg_regimes.json)",
    )
    
    args = parser.parse_args()
    underlying = args.underlying.upper()
    
    print(f"Building Greg regimes for {underlying}...")
    print(f"  Clusters: {args.n_clusters}")
    print(f"  Lookback: {args.lookback_days} days")
    
    print("\n1. Loading spot OHLC data...")
    spot_df = fetch_spot_ohlc_from_api(underlying, args.lookback_days)
    
    if spot_df.empty:
        print("  API fetch failed, trying harvester data...")
        harvester_df = load_harvester_data(underlying, args.lookback_days)
        if not harvester_df.empty and "close" in harvester_df.columns:
            spot_df = harvester_df
    
    if spot_df.empty:
        print("ERROR: No spot data available. Cannot build regimes.")
        print("\nTo fix this:")
        print("  1. Ensure internet connectivity for Deribit API access")
        print("  2. Or run the data harvester first: python -m scripts.data_harvester")
        print("  3. Or place OHLC CSV files in data/harvester/")
        sys.exit(1)
    
    min_required = args.n_clusters * 5
    if len(spot_df) < min_required:
        print(f"WARNING: Only {len(spot_df)} data points available, minimum {min_required} recommended.")
        if len(spot_df) < args.n_clusters * 2:
            print(f"ERROR: Not enough data for {args.n_clusters} clusters. Reduce --n-clusters or increase --lookback-days.")
            sys.exit(1)
    
    print(f"  Loaded {len(spot_df)} data points")
    
    print("\n2. Computing sensors from OHLC data...")
    sensors_df = compute_sensors_from_ohlc(spot_df)
    
    if sensors_df.empty:
        print("ERROR: Failed to compute sensors.")
        sys.exit(1)
    
    sensors_df = sensors_df.dropna(subset=["rv_30d", "adx_14d"])
    print(f"  Computed sensors for {len(sensors_df)} days")
    
    print("\n3. Clustering sensors into regimes...")
    try:
        df_with_regime, kmeans = cluster_greg_sensors_from_real_data(
            sensors_df,
            n_clusters=args.n_clusters,
        )
        print(f"  Clustered into {args.n_clusters} regimes")
        
        regime_counts = df_with_regime["regime_id"].value_counts().sort_index()
        print("  Regime distribution:")
        for regime_id, count in regime_counts.items():
            pct = count / len(df_with_regime) * 100
            print(f"    Regime {regime_id}: {count} days ({pct:.1f}%)")
    except Exception as e:
        print(f"ERROR: Clustering failed: {e}")
        sys.exit(1)
    
    print("\n4. Fitting regime parameters...")
    regimes = fit_regime_params_from_clusters(df_with_regime)
    
    for regime_id, params in regimes.items():
        print(f"  Regime {regime_id} ({params.name}):")
        print(f"    mu_rv_30d: {params.mu_rv_30d:.1f}%")
        print(f"    mu_vrp_30d: {params.mu_vrp_30d:.2%}")
        print(f"    iv_level_sigma: {params.iv_level_sigma:.1f}")
        print(f"    phi_iv: {params.phi_iv:.2f}")
    
    print("\n5. Estimating transition matrix...")
    regime_sequence = df_with_regime["regime_id"].values
    transition_matrix = estimate_transition_matrix(regime_sequence, args.n_clusters)
    
    print("  Transition probabilities (row = from, col = to):")
    for i in range(args.n_clusters):
        probs = " ".join([f"{p:.2f}" for p in transition_matrix[i]])
        print(f"    {i}: [{probs}]")
    
    print("\n6. Saving regime model...")
    model = RegimeModel(
        underlying=underlying,
        n_clusters=args.n_clusters,
        regimes=regimes,
        transition_matrix=transition_matrix,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    
    output_path = Path(args.output) if args.output else GREG_REGIME_FILE
    save_regime_model(model, output_path)
    print(f"  Saved to: {output_path}")
    
    print("\nDone! Regime model ready for synthetic universe generation.")


if __name__ == "__main__":
    main()
