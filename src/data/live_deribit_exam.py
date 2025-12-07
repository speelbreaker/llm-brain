"""
Reusable function for building exam datasets from live Deribit captures.

This module provides build_live_deribit_exam_dataset() which can be called
programmatically from backtester or CLI scripts.
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, date, timezone
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "instrument_name",
    "underlying",
    "expiry_timestamp",
    "option_type",
    "strike",
    "underlying_price",
    "mark_price",
    "mark_iv",
    "greek_delta",
]


def parse_timestamp_from_filename(filepath: Path, underlying: str) -> Optional[datetime]:
    """
    Try to parse the snapshot timestamp from the filename.
    
    Expected pattern: <UNDERLYING>_<YYYY-MM-DD_HHMM>.parquet
    Example: BTC_2025-12-07_1400.parquet
    
    Returns:
        datetime with UTC timezone, or None if parsing fails.
    """
    pattern = rf"^{underlying}_(\d{{4}}-\d{{2}}-\d{{2}}_\d{{4}})\.parquet$"
    match = re.match(pattern, filepath.name)
    
    if match:
        try:
            dt_str = match.group(1)
            dt = datetime.strptime(dt_str, "%Y-%m-%d_%H%M")
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


def get_timestamp_from_parquet(filepath: Path) -> Optional[datetime]:
    """
    Fallback: read harvest_time column from Parquet and return the minimum value.
    
    Returns:
        datetime with UTC timezone, or None if unavailable.
    """
    try:
        df = pd.read_parquet(filepath, columns=["harvest_time"])
        if "harvest_time" in df.columns and len(df) > 0:
            min_ts = df["harvest_time"].min()
            if pd.notna(min_ts):
                if hasattr(min_ts, "tzinfo") and min_ts.tzinfo is None:
                    return min_ts.replace(tzinfo=timezone.utc)
                return min_ts.to_pydatetime() if hasattr(min_ts, "to_pydatetime") else min_ts
    except Exception:
        pass
    return None


def get_file_timestamp(filepath: Path, underlying: str) -> Optional[datetime]:
    """
    Get the snapshot timestamp for a file, trying filename first, then Parquet content.
    """
    ts = parse_timestamp_from_filename(filepath, underlying)
    if ts is not None:
        return ts
    
    return get_timestamp_from_parquet(filepath)


def discover_files(
    data_root: Path,
    underlying: str,
    start_date: date,
    end_date: date,
) -> List[Tuple[Path, datetime]]:
    """
    Discover all Parquet files for the given underlying within the date range.
    
    Returns:
        List of (filepath, timestamp) tuples for files within the date range.
    """
    base_dir = data_root / underlying
    
    if not base_dir.exists():
        return []
    
    all_parquets = list(base_dir.glob("**/*.parquet"))
    
    if not all_parquets:
        return []
    
    valid_files = []
    
    for fp in all_parquets:
        ts = get_file_timestamp(fp, underlying)
        
        if ts is None:
            continue
        
        ts_date = ts.date()
        if start_date <= ts_date <= end_date:
            valid_files.append((fp, ts))
    
    valid_files.sort(key=lambda x: x[1])
    
    return valid_files


def validate_schema(df: pd.DataFrame) -> List[str]:
    """
    Validate that required columns exist in the DataFrame.
    
    Returns:
        List of missing column names (empty if all present).
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    return missing


def compute_dte_days(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add dte_days column: days to expiry from harvest_time.
    
    Args:
        df: DataFrame with harvest_time and expiry_timestamp columns.
        
    Returns:
        DataFrame with dte_days column added (or unchanged if columns missing).
    """
    if "expiry_timestamp" not in df.columns or "harvest_time" not in df.columns:
        return df
    
    harvest_dt = pd.to_datetime(df["harvest_time"], utc=True)
    harvest_ts = harvest_dt.apply(lambda x: x.timestamp() if pd.notna(x) else np.nan)
    
    expiry_ts = pd.to_numeric(df["expiry_timestamp"], errors="coerce")
    df["dte_days"] = (expiry_ts - harvest_ts) / 86400.0
    
    return df


def load_and_stitch(files: List[Tuple[Path, datetime]]) -> pd.DataFrame:
    """
    Load all Parquet files and concatenate into a single DataFrame.
    """
    dfs = []
    
    for filepath, file_ts in files:
        try:
            df = pd.read_parquet(filepath)
            
            if "harvest_time" not in df.columns:
                df["harvest_time"] = file_ts
            
            dfs.append(df)
        except Exception:
            continue
    
    if not dfs:
        return pd.DataFrame()
    
    combined = pd.concat(dfs, ignore_index=True)
    
    if "harvest_time" in combined.columns:
        combined["harvest_time"] = pd.to_datetime(combined["harvest_time"], utc=True)
    
    combined = compute_dte_days(combined)
    
    sort_cols = []
    if "harvest_time" in combined.columns:
        sort_cols.append("harvest_time")
    if "instrument_name" in combined.columns:
        sort_cols.append("instrument_name")
    
    if sort_cols:
        combined = combined.sort_values(sort_cols).reset_index(drop=True)
    
    return combined


def compute_summary(
    underlying: str,
    num_files: int,
    df: pd.DataFrame,
    start_date: date,
    end_date: date,
) -> Dict[str, Any]:
    """
    Compute summary statistics for the exam dataset.
    """
    num_rows = len(df)
    columns = sorted(df.columns.tolist())
    
    num_snapshots = None
    min_harvest = None
    max_harvest = None
    
    if "harvest_time" in df.columns and num_rows > 0:
        num_snapshots = df["harvest_time"].nunique()
        min_harvest = df["harvest_time"].min()
        max_harvest = df["harvest_time"].max()
    
    num_instruments = None
    if "instrument_name" in df.columns and num_rows > 0:
        num_instruments = df["instrument_name"].nunique()
    
    unique_underlyings = None
    if "underlying" in df.columns and num_rows > 0:
        unique_underlyings = df["underlying"].dropna().unique().tolist()[:10]
    
    min_dte = None
    max_dte = None
    if "dte_days" in df.columns and num_rows > 0:
        min_dte = df["dte_days"].min()
        max_dte = df["dte_days"].max()
    
    missing = validate_schema(df)
    
    summary = {
        "underlying": underlying,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "num_files": num_files,
        "num_rows": num_rows,
        "num_snapshots": num_snapshots,
        "num_instruments": num_instruments,
        "unique_underlyings": unique_underlyings,
        "time_min": min_harvest.isoformat() if min_harvest else None,
        "time_max": max_harvest.isoformat() if max_harvest else None,
        "dte_min": float(min_dte) if min_dte is not None and not pd.isna(min_dte) else None,
        "dte_max": float(max_dte) if max_dte is not None and not pd.isna(max_dte) else None,
        "columns": columns,
        "missing_required_columns": missing,
    }
    
    return summary


def build_live_deribit_exam_dataset(
    underlying: str,
    start_date: date,
    end_date: date,
    base_dir: Path | str = "data/live_deribit",
    exams_dir: Path | str = "data/exams",
    write_files: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Build an exam dataset from live Deribit captures.
    
    This function finds all snapshot Parquet files for the specified underlying
    between start_date and end_date, loads and concatenates them into a single
    DataFrame, enforces the canonical schema, and optionally writes output files.
    
    Args:
        underlying: Asset symbol (e.g., "BTC", "ETH")
        start_date: Start date (inclusive)
        end_date: End date (inclusive)
        base_dir: Root directory for harvester data (default: "data/live_deribit")
        exams_dir: Output directory for exam dataset (default: "data/exams")
        write_files: If True, write Parquet and JSON summary files
    
    Returns:
        Tuple of (DataFrame, summary_dict)
        
    Raises:
        ValueError: If no files found or no data after stitching
    """
    underlying = underlying.upper()
    base_dir = Path(base_dir)
    exams_dir = Path(exams_dir)
    
    files = discover_files(base_dir, underlying, start_date, end_date)
    
    if not files:
        raise ValueError(
            f"No files found for {underlying} in date range "
            f"{start_date.isoformat()} to {end_date.isoformat()}"
        )
    
    df = load_and_stitch(files)
    
    if df.empty:
        raise ValueError("No data after stitching. DataFrame is empty.")
    
    summary = compute_summary(underlying, len(files), df, start_date, end_date)
    
    if write_files:
        os.makedirs(exams_dir, exist_ok=True)
        
        start_str = start_date.isoformat()
        end_str = end_date.isoformat()
        
        out_filename = f"{underlying}_{start_str}_{end_str}_live_deribit.parquet"
        out_path = exams_dir / out_filename
        
        summary_filename = f"{underlying}_{start_str}_{end_str}_live_deribit_summary.json"
        summary_path = exams_dir / summary_filename
        
        df.to_parquet(out_path, engine="pyarrow", index=False)
        
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
    
    return df, summary
