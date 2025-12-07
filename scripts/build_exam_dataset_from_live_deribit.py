#!/usr/bin/env python3
"""
Build Exam Dataset from Live Deribit Captures

Reads and stitches together Deribit options data captured by the data harvester
(stored in Parquet files under data/live_deribit/...) and produces a single
"exam dataset" file for a given underlying and date range.

Usage:
    python scripts/build_exam_dataset_from_live_deribit.py \
        --underlying BTC \
        --start 2025-02-01 \
        --end 2025-03-01 \
        [--data-root data/live_deribit] \
        [--out-dir data/exams]
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

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
    start_date: datetime,
    end_date: datetime,
) -> list[tuple[Path, datetime]]:
    """
    Discover all Parquet files for the given underlying within the date range.
    
    Returns:
        List of (filepath, timestamp) tuples for files within the date range.
    """
    base_dir = data_root / underlying
    
    if not base_dir.exists():
        print(f"ERROR: Directory does not exist: {base_dir}")
        sys.exit(1)
    
    all_parquets = list(base_dir.glob("**/*.parquet"))
    
    if not all_parquets:
        print(f"No Parquet files found under {base_dir}")
        return []
    
    valid_files = []
    parse_errors = 0
    
    for fp in all_parquets:
        ts = get_file_timestamp(fp, underlying)
        
        if ts is None:
            print(f"WARNING: Could not determine timestamp for {fp}")
            parse_errors += 1
            continue
        
        ts_date = ts.date()
        if start_date.date() <= ts_date <= end_date.date():
            valid_files.append((fp, ts))
    
    if parse_errors > 0:
        print(f"WARNING: {parse_errors} file(s) had parsing errors and were skipped.")
    
    valid_files.sort(key=lambda x: x[1])
    
    return valid_files


def validate_schema(df: pd.DataFrame) -> list[str]:
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


def load_and_stitch(files: list[tuple[Path, datetime]]) -> pd.DataFrame:
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
        except Exception as e:
            print(f"WARNING: Failed to read {filepath}: {e}")
    
    if not dfs:
        return pd.DataFrame()
    
    combined = pd.concat(dfs, ignore_index=True)
    
    if "harvest_time" in combined.columns:
        combined["harvest_time"] = pd.to_datetime(combined["harvest_time"], utc=True)
    
    missing_cols = validate_schema(combined)
    if missing_cols:
        print("\n" + "!" * 60)
        print("WARNING: MISSING REQUIRED COLUMNS")
        print("!" * 60)
        for col in missing_cols:
            print(f"  - {col}")
        print("!" * 60 + "\n")
    
    combined = compute_dte_days(combined)
    
    sort_cols = []
    if "harvest_time" in combined.columns:
        sort_cols.append("harvest_time")
    if "instrument_name" in combined.columns:
        sort_cols.append("instrument_name")
    
    if sort_cols:
        combined = combined.sort_values(sort_cols).reset_index(drop=True)
    
    return combined


def print_summary(
    underlying: str,
    num_files: int,
    df: pd.DataFrame,
) -> dict:
    """
    Print summary statistics and return a summary dict for JSON output.
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
    
    print("\n" + "=" * 60)
    print("EXAM DATASET SUMMARY")
    print("=" * 60)
    print(f"Underlying:           {underlying}")
    print(f"Files stitched:       {num_files}")
    print(f"Total rows:           {num_rows}")
    
    if num_snapshots is not None:
        print(f"Unique snapshots:     {num_snapshots}")
    
    if num_instruments is not None:
        print(f"Unique instruments:   {num_instruments}")
    
    if unique_underlyings is not None:
        print(f"Unique underlyings:   {unique_underlyings}")
    
    if min_harvest is not None:
        print(f"Min harvest_time:     {min_harvest}")
        print(f"Max harvest_time:     {max_harvest}")
    
    if min_dte is not None and max_dte is not None:
        print(f"Min dte_days:         {min_dte:.2f}")
        print(f"Max dte_days:         {max_dte:.2f}")
    
    print(f"\nColumns ({len(columns)}):")
    for col in columns:
        print(f"  - {col}")
    
    missing = validate_schema(df)
    if missing:
        print("\nWARNINGS:")
        for col in missing:
            print(f"  - Missing required column: {col}")
    
    print("=" * 60)
    
    summary = {
        "underlying": underlying,
        "num_files": num_files,
        "num_rows": num_rows,
        "num_snapshots": num_snapshots,
        "num_instruments": num_instruments,
        "unique_underlyings": unique_underlyings,
        "min_harvest_time": min_harvest.isoformat() if min_harvest else None,
        "max_harvest_time": max_harvest.isoformat() if max_harvest else None,
        "min_dte_days": float(min_dte) if min_dte is not None else None,
        "max_dte_days": float(max_dte) if max_dte is not None else None,
        "columns": columns,
        "missing_required_columns": missing,
    }
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Build exam dataset from live Deribit captures"
    )
    parser.add_argument(
        "--underlying",
        required=True,
        help="Underlying asset (e.g., BTC, ETH, SOL)",
    )
    parser.add_argument(
        "--start",
        required=True,
        help="Start date (YYYY-MM-DD, UTC, inclusive)",
    )
    parser.add_argument(
        "--end",
        required=True,
        help="End date (YYYY-MM-DD, UTC, inclusive)",
    )
    parser.add_argument(
        "--data-root",
        default="data/live_deribit",
        help="Root directory for harvester data (default: data/live_deribit)",
    )
    parser.add_argument(
        "--out-dir",
        default="data/exams",
        help="Output directory for exam dataset (default: data/exams)",
    )
    
    args = parser.parse_args()
    
    underlying = args.underlying.upper()
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    
    try:
        start_date = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end_date = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError as e:
        print(f"ERROR: Invalid date format: {e}")
        print("Use YYYY-MM-DD format.")
        sys.exit(1)
    
    if start_date > end_date:
        print("ERROR: Start date must be before or equal to end date.")
        sys.exit(1)
    
    print(f"Building exam dataset for {underlying}")
    print(f"Date range: {args.start} to {args.end}")
    print(f"Data root: {data_root}")
    print(f"Output dir: {out_dir}")
    
    files = discover_files(data_root, underlying, start_date, end_date)
    
    if not files:
        print(f"\nNo files found for {underlying} in date range {args.start} to {args.end}")
        sys.exit(1)
    
    print(f"\nFound {len(files)} file(s) in date range")
    
    df = load_and_stitch(files)
    
    if df.empty:
        print("\nERROR: No data after stitching. DataFrame is empty.")
        sys.exit(1)
    
    summary = print_summary(underlying, len(files), df)
    summary["start_date"] = args.start
    summary["end_date"] = args.end
    
    os.makedirs(out_dir, exist_ok=True)
    
    out_filename = f"{underlying}_{args.start}_{args.end}_live_deribit.parquet"
    out_path = out_dir / out_filename
    
    summary_filename = f"{underlying}_{args.start}_{args.end}_live_deribit_summary.json"
    summary_path = out_dir / summary_filename
    
    df.to_parquet(out_path, engine="pyarrow", index=False)
    print(f"\nWrote exam dataset to: {out_path}")
    
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote summary to: {summary_path}")


if __name__ == "__main__":
    main()
