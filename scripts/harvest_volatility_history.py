#!/usr/bin/env python3
"""
Volatility History Harvester

Downloads historical DVOL (implied volatility) and RV (realized volatility)
time series from the Deribit API for use in synthetic universe calibration
and historical replay backtesting.

Data sources:
- DVOL: Deribit Volatility Index via public/get_volatility_index_data
- RV: Official 30-day historical volatility via public/get_historical_volatility

Usage:
    python -m scripts.harvest_volatility_history
    python -m scripts.harvest_volatility_history --days 365
    python -m scripts.harvest_volatility_history --currency BTC
"""
from __future__ import annotations

import argparse
import datetime
import logging
import os
import sys
from pathlib import Path
from typing import Any, List, Optional, Tuple

import httpx
import pandas as pd

DATA_DIR = Path("data/volatility_history")
DERIBIT_BASE_URL = "https://www.deribit.com/api/v2"
SUPPORTED_CURRENCIES = ["BTC", "ETH"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def fetch_dvol_history(
    currency: str,
    start: datetime.datetime,
    end: datetime.datetime,
    resolution: str = "3600",
) -> pd.DataFrame:
    """
    Fetch DVOL (Deribit Volatility Index) historical data.
    
    Args:
        currency: Currency code (BTC or ETH)
        start: Start datetime
        end: End datetime
        resolution: Resolution in seconds ('60', '3600', '43200')
        
    Returns:
        DataFrame with columns: timestamp, dvol_open, dvol_high, dvol_low, dvol_close
    """
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    
    url = f"{DERIBIT_BASE_URL}/public/get_volatility_index_data"
    params = {
        "currency": currency.upper(),
        "start_timestamp": start_ms,
        "end_timestamp": end_ms,
        "resolution": resolution,
    }
    
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
        
        result = data.get("result", {})
        raw_data = result.get("data", [])
        
        if not raw_data:
            logger.warning(f"No DVOL data returned for {currency}")
            return pd.DataFrame()
        
        records = []
        for point in raw_data:
            if len(point) >= 5:
                records.append({
                    "timestamp": datetime.datetime.fromtimestamp(
                        point[0] / 1000, tz=datetime.timezone.utc
                    ),
                    "dvol_open": float(point[1]),
                    "dvol_high": float(point[2]),
                    "dvol_low": float(point[3]),
                    "dvol_close": float(point[4]),
                })
        
        df = pd.DataFrame(records)
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df
        
    except Exception as e:
        logger.error(f"Error fetching DVOL for {currency}: {e}")
        return pd.DataFrame()


def fetch_historical_volatility(currency: str) -> pd.DataFrame:
    """
    Fetch official Deribit historical volatility (30-day RV) time series.
    
    The API returns a list of [timestamp_ms, rv_value] pairs.
    
    Args:
        currency: Currency code (BTC or ETH)
        
    Returns:
        DataFrame with columns: timestamp, rv_30d
    """
    url = f"{DERIBIT_BASE_URL}/public/get_historical_volatility"
    params = {"currency": currency.upper()}
    
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
        
        result = data.get("result", [])
        
        if not result:
            logger.warning(f"No RV data returned for {currency}")
            return pd.DataFrame()
        
        records = []
        for point in result:
            if len(point) >= 2:
                records.append({
                    "timestamp": datetime.datetime.fromtimestamp(
                        point[0] / 1000, tz=datetime.timezone.utc
                    ),
                    "rv_30d": float(point[1]),
                })
        
        df = pd.DataFrame(records)
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df
        
    except Exception as e:
        logger.error(f"Error fetching RV for {currency}: {e}")
        return pd.DataFrame()


def fetch_dvol_in_chunks(
    currency: str,
    start: datetime.datetime,
    end: datetime.datetime,
    resolution: str = "3600",
    chunk_days: int = 30,
) -> pd.DataFrame:
    """
    Fetch DVOL history in chunks to handle API limits.
    
    Args:
        currency: Currency code
        start: Start datetime
        end: End datetime
        resolution: Resolution in seconds
        chunk_days: Number of days per chunk
        
    Returns:
        Combined DataFrame
    """
    all_dfs = []
    current_start = start
    
    while current_start < end:
        current_end = min(
            current_start + datetime.timedelta(days=chunk_days),
            end
        )
        
        logger.info(
            f"Fetching DVOL {currency} from {current_start.date()} to {current_end.date()}"
        )
        
        df_chunk = fetch_dvol_history(currency, current_start, current_end, resolution)
        
        if not df_chunk.empty:
            all_dfs.append(df_chunk)
        
        current_start = current_end
    
    if not all_dfs:
        return pd.DataFrame()
    
    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    return combined.reset_index(drop=True)


def merge_iv_rv_data(
    df_dvol: pd.DataFrame,
    df_rv: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge DVOL and RV data on timestamp, computing VRP.
    
    Args:
        df_dvol: DataFrame with DVOL data
        df_rv: DataFrame with RV data
        
    Returns:
        Merged DataFrame with iv_30d, rv_30d, vrp_30d columns
    """
    if df_dvol.empty:
        logger.warning("Empty DVOL data for merge")
        return pd.DataFrame()
    
    df_iv = df_dvol[["timestamp", "dvol_close"]].copy()
    df_iv = df_iv.rename(columns={"dvol_close": "iv_30d"})
    
    if df_rv.empty:
        logger.warning("Empty RV data, using only IV")
        df_iv["rv_30d"] = None
        df_iv["vrp_30d"] = None
        return df_iv
    
    df_merged = pd.merge_asof(
        df_iv.sort_values("timestamp"),
        df_rv.sort_values("timestamp"),
        on="timestamp",
        direction="nearest",
        tolerance=pd.Timedelta(hours=2),
    )
    
    df_merged["vrp_30d"] = df_merged["iv_30d"] - df_merged["rv_30d"]
    
    return df_merged


def save_volatility_history(
    currency: str,
    df: pd.DataFrame,
    output_dir: Path = DATA_DIR,
) -> Path:
    """
    Save volatility history to Parquet file.
    
    Args:
        currency: Currency code
        df: DataFrame with volatility data
        output_dir: Output directory
        
    Returns:
        Path to saved file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    now = datetime.datetime.now(datetime.timezone.utc)
    filename = f"{currency.lower()}_volatility_history.parquet"
    filepath = output_dir / filename
    
    df["currency"] = currency.upper()
    df["harvest_time"] = now
    
    df.to_parquet(filepath, engine="pyarrow", index=False)
    logger.info(f"Saved {len(df)} rows to {filepath}")
    
    return filepath


def load_volatility_history(
    currency: str,
    data_dir: Path = DATA_DIR,
) -> Optional[pd.DataFrame]:
    """
    Load volatility history from Parquet file.
    
    Args:
        currency: Currency code
        data_dir: Data directory
        
    Returns:
        DataFrame or None if not found
    """
    filepath = data_dir / f"{currency.lower()}_volatility_history.parquet"
    
    if not filepath.exists():
        return None
    
    return pd.read_parquet(filepath)


def harvest_volatility(
    currency: str,
    days: int = 365,
    resolution: str = "3600",
) -> Tuple[pd.DataFrame, Path]:
    """
    Main harvest function for a single currency.
    
    Args:
        currency: Currency code (BTC or ETH)
        days: Number of days of history to fetch
        resolution: DVOL resolution in seconds
        
    Returns:
        Tuple of (DataFrame, output path)
    """
    end = datetime.datetime.now(datetime.timezone.utc)
    start = end - datetime.timedelta(days=days)
    
    logger.info(f"Harvesting {currency} volatility data for past {days} days")
    logger.info(f"Date range: {start.date()} to {end.date()}")
    
    df_dvol = fetch_dvol_in_chunks(currency, start, end, resolution)
    logger.info(f"Fetched {len(df_dvol)} DVOL data points")
    
    df_rv = fetch_historical_volatility(currency)
    logger.info(f"Fetched {len(df_rv)} RV data points")
    
    df_merged = merge_iv_rv_data(df_dvol, df_rv)
    logger.info(f"Merged data has {len(df_merged)} rows")
    
    if not df_merged.empty:
        filepath = save_volatility_history(currency, df_merged)
        
        logger.info(f"\n{currency} Volatility Summary:")
        logger.info(f"  IV 30d: mean={df_merged['iv_30d'].mean():.2f}, "
                   f"std={df_merged['iv_30d'].std():.2f}")
        if df_merged["rv_30d"].notna().any():
            logger.info(f"  RV 30d: mean={df_merged['rv_30d'].mean():.2f}, "
                       f"std={df_merged['rv_30d'].std():.2f}")
            logger.info(f"  VRP 30d: mean={df_merged['vrp_30d'].mean():.2f}, "
                       f"std={df_merged['vrp_30d'].std():.2f}")
        
        return df_merged, filepath
    else:
        logger.warning(f"No data harvested for {currency}")
        return pd.DataFrame(), Path()


def run_full_harvest(
    currencies: Optional[List[str]] = None,
    days: int = 365,
) -> dict[str, pd.DataFrame]:
    """
    Run full harvest for all currencies.
    
    Args:
        currencies: List of currencies (defaults to BTC, ETH)
        days: Number of days of history
        
    Returns:
        Dict mapping currency to DataFrame
    """
    if currencies is None:
        currencies = SUPPORTED_CURRENCIES
    
    results = {}
    
    for currency in currencies:
        if currency.upper() not in SUPPORTED_CURRENCIES:
            logger.warning(f"Skipping unsupported currency: {currency}")
            continue
        
        df, _ = harvest_volatility(currency.upper(), days)
        results[currency.upper()] = df
    
    return results


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Harvest volatility history from Deribit"
    )
    parser.add_argument(
        "--currency", "-c",
        type=str,
        default=None,
        help="Currency to harvest (BTC, ETH, or 'all'). Default: all"
    )
    parser.add_argument(
        "--days", "-d",
        type=int,
        default=365,
        help="Number of days of history to fetch. Default: 365"
    )
    parser.add_argument(
        "--resolution", "-r",
        type=str,
        default="3600",
        choices=["60", "3600", "43200"],
        help="DVOL resolution in seconds. Default: 3600 (hourly)"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Volatility History Harvester")
    logger.info("=" * 60)
    
    if args.currency is None or args.currency.lower() == "all":
        currencies = SUPPORTED_CURRENCIES
    else:
        currencies = [args.currency.upper()]
    
    for currency in currencies:
        if currency not in SUPPORTED_CURRENCIES:
            logger.error(f"Unsupported currency: {currency}")
            continue
        
        harvest_volatility(currency, args.days, args.resolution)
    
    logger.info("=" * 60)
    logger.info("Harvest complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
