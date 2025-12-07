"""
Real Scraper Data Loader

Loads normalized Real Scraper data from data/real_scraper directory
for use in backtests. Data is expected to be in the format produced by
scripts/import_real_scraper_deribit.py.

File structure:
    data/real_scraper/<UNDERLYING>/<YYYY-MM-DD>/<UNDERLYING>_<YYYY-MM-DD>.parquet
"""

from datetime import datetime, date, timedelta, timezone
from pathlib import Path
from typing import List, Optional

import pandas as pd


DEFAULT_DATA_ROOT = Path("data/real_scraper")


def discover_real_scraper_files(
    underlying: str,
    start_date: date,
    end_date: date,
    data_root: Path = DEFAULT_DATA_ROOT,
) -> List[Path]:
    """
    Discover all Real Scraper Parquet files for the given underlying and date range.
    
    Args:
        underlying: Asset symbol (e.g., "BTC", "ETH")
        start_date: Start date (inclusive)
        end_date: End date (inclusive)
        data_root: Root directory for Real Scraper data
        
    Returns:
        List of Parquet file paths sorted by date.
    """
    base_dir = data_root / underlying.upper()
    
    if not base_dir.exists():
        return []
    
    files = []
    current_date = start_date
    
    while current_date <= end_date:
        date_str = current_date.strftime("%Y-%m-%d")
        expected_path = base_dir / date_str / f"{underlying.upper()}_{date_str}.parquet"
        
        if expected_path.exists():
            files.append(expected_path)
        
        current_date += timedelta(days=1)
    
    return files


def load_real_scraper_data(
    underlying: str,
    start_ts: datetime,
    end_ts: datetime,
    data_root: Path = DEFAULT_DATA_ROOT,
) -> pd.DataFrame:
    """
    Load Real Scraper data for the given underlying and time range.
    
    Args:
        underlying: Asset symbol (e.g., "BTC", "ETH")
        start_ts: Start timestamp (UTC)
        end_ts: End timestamp (UTC)
        data_root: Root directory for Real Scraper data
        
    Returns:
        DataFrame with canonical columns, or empty DataFrame if no data found.
    """
    start_date = start_ts.date() if hasattr(start_ts, "date") else start_ts
    end_date = end_ts.date() if hasattr(end_ts, "date") else end_ts
    
    files = discover_real_scraper_files(underlying, start_date, end_date, data_root)
    
    if not files:
        print(f"[RealScraperLoader] No data files found for {underlying} "
              f"from {start_date} to {end_date}")
        return pd.DataFrame()
    
    print(f"[RealScraperLoader] Loading {len(files)} file(s) for {underlying}")
    
    dfs = []
    for filepath in files:
        try:
            df = pd.read_parquet(filepath)
            dfs.append(df)
        except Exception as e:
            print(f"[RealScraperLoader] Warning: Failed to read {filepath}: {e}")
    
    if not dfs:
        return pd.DataFrame()
    
    combined = pd.concat(dfs, ignore_index=True)
    
    if "harvest_time" in combined.columns:
        combined["harvest_time"] = pd.to_datetime(combined["harvest_time"], utc=True)
        
        start_utc = start_ts if start_ts.tzinfo else start_ts.replace(tzinfo=timezone.utc)
        end_utc = end_ts if end_ts.tzinfo else end_ts.replace(tzinfo=timezone.utc)
        
        mask = (combined["harvest_time"] >= start_utc) & (combined["harvest_time"] <= end_utc)
        combined = combined[mask]
    
    sort_cols = []
    if "harvest_time" in combined.columns:
        sort_cols.append("harvest_time")
    if "instrument_name" in combined.columns:
        sort_cols.append("instrument_name")
    
    if sort_cols:
        combined = combined.sort_values(sort_cols).reset_index(drop=True)
    
    print(f"[RealScraperLoader] Loaded {len(combined)} rows, "
          f"{combined['instrument_name'].nunique() if 'instrument_name' in combined.columns else 0} instruments")
    
    return combined


def get_real_scraper_snapshot(
    df: pd.DataFrame,
    as_of: datetime,
    tolerance_minutes: int = 60,
) -> pd.DataFrame:
    """
    Get the option chain snapshot closest to the given timestamp.
    
    Args:
        df: DataFrame from load_real_scraper_data
        as_of: Target timestamp
        tolerance_minutes: Maximum time difference to consider valid
        
    Returns:
        DataFrame containing options from the closest snapshot.
    """
    if df.empty or "harvest_time" not in df.columns:
        return pd.DataFrame()
    
    unique_times = df["harvest_time"].unique()
    unique_times = pd.to_datetime(unique_times, utc=True)
    
    as_of_utc = as_of if as_of.tzinfo else as_of.replace(tzinfo=timezone.utc)
    
    time_diffs = abs(unique_times - as_of_utc)
    closest_idx = time_diffs.argmin()
    closest_time = unique_times[closest_idx]
    
    diff_minutes = time_diffs[closest_idx].total_seconds() / 60
    if diff_minutes > tolerance_minutes:
        return pd.DataFrame()
    
    return df[df["harvest_time"] == closest_time].copy()
