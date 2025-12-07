#!/usr/bin/env python3
"""
Import Real Scraper Deribit Data

Converts Real Scraper options data (e.g., from Kaggle datasets) into our 
canonical snapshot format used by live_deribit and the backtest engine.

Usage:
    python scripts/import_real_scraper_deribit.py \
        --underlying BTC \
        --date 2023-05-26 \
        --input data/real_scraper/raw/DeriBit_BTC_26MAY23_allStrikes_aggregated.csv

Output:
    data/real_scraper/BTC/2023-05-26/BTC_2023-05-26.parquet
"""

import argparse
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


CANONICAL_COLUMNS = [
    "harvest_time",
    "instrument_name",
    "underlying",
    "expiry",
    "expiry_timestamp",
    "option_type",
    "strike",
    "underlying_price",
    "mark_price",
    "best_bid_price",
    "best_ask_price",
    "bid_iv",
    "ask_iv",
    "mark_iv",
    "open_interest",
    "volume",
    "greek_delta",
    "greek_gamma",
    "greek_theta",
    "greek_vega",
    "dte_days",
]


def parse_deribit_expiry(expiry_str: str) -> Tuple[datetime, str]:
    """
    Parse Deribit-style expiry string like '26MAY23' or '2023-05-26'.
    
    Returns:
        (expiry_datetime, formatted_expiry_str YYYY-MM-DD)
    """
    expiry_str = expiry_str.strip().upper()
    
    if re.match(r"^\d{4}-\d{2}-\d{2}$", expiry_str):
        dt = datetime.strptime(expiry_str, "%Y-%m-%d")
        return dt.replace(tzinfo=timezone.utc, hour=8), expiry_str
    
    match = re.match(r"^(\d{1,2})([A-Z]{3})(\d{2,4})$", expiry_str)
    if match:
        day = int(match.group(1))
        month_str = match.group(2)
        year_str = match.group(3)
        
        months = {
            "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4,
            "MAY": 5, "JUN": 6, "JUL": 7, "AUG": 8,
            "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12
        }
        month = months.get(month_str, 1)
        
        if len(year_str) == 2:
            year = 2000 + int(year_str)
        else:
            year = int(year_str)
        
        dt = datetime(year, month, day, 8, 0, 0, tzinfo=timezone.utc)
        formatted = dt.strftime("%Y-%m-%d")
        return dt, formatted
    
    raise ValueError(f"Cannot parse expiry: {expiry_str}")


def construct_instrument_name(
    underlying: str,
    expiry_dt: datetime,
    strike: float,
    option_type: str,
) -> str:
    """
    Construct Deribit-style instrument name like BTC-26MAY23-30000-C.
    """
    expiry_str = expiry_dt.strftime("%d%b%y").upper()
    strike_int = int(strike) if strike == int(strike) else strike
    opt_char = "C" if option_type.upper().startswith("C") else "P"
    return f"{underlying}-{expiry_str}-{strike_int}-{opt_char}"


def parse_instrument_name(instrument_name: str) -> Tuple[str, datetime, float, str]:
    """
    Parse instrument name like BTC-26MAY23-30000-C.
    
    Returns:
        (underlying, expiry_datetime, strike, option_type)
    """
    parts = instrument_name.split("-")
    if len(parts) < 4:
        raise ValueError(f"Invalid instrument name: {instrument_name}")
    
    underlying = parts[0]
    expiry_str = parts[1]
    strike = float(parts[2])
    option_type = parts[3]
    
    expiry_dt, _ = parse_deribit_expiry(expiry_str)
    
    return underlying, expiry_dt, strike, option_type


def normalize_column_name(col: str) -> str:
    """Normalize column names to lowercase with underscores."""
    col = col.strip().lower()
    col = re.sub(r"[^a-z0-9]+", "_", col)
    col = col.strip("_")
    return col


def read_input_file(input_path: Path) -> pd.DataFrame:
    """Read CSV or Parquet file."""
    ext = input_path.suffix.lower()
    
    if ext == ".parquet":
        return pd.read_parquet(input_path)
    elif ext == ".csv":
        return pd.read_csv(input_path)
    elif ext == ".gz":
        return pd.read_csv(input_path, compression="gzip")
    else:
        return pd.read_csv(input_path)


def map_real_scraper_columns(df: pd.DataFrame, underlying: str, target_date: str) -> pd.DataFrame:
    """
    Map Real Scraper columns to our canonical schema.
    
    This handles various Real Scraper CSV formats from Kaggle.
    """
    df = df.copy()
    df.columns = [normalize_column_name(c) for c in df.columns]
    
    print(f"Input columns: {list(df.columns)}")
    
    result = pd.DataFrame()
    
    timestamp_cols = ["timestamp", "time", "datetime", "date", "snap_time", "collection_time"]
    for col in timestamp_cols:
        if col in df.columns:
            try:
                result["harvest_time"] = pd.to_datetime(df[col], utc=True)
                break
            except Exception:
                continue
    
    if "harvest_time" not in result.columns:
        target_dt = datetime.strptime(target_date, "%Y-%m-%d").replace(
            hour=12, tzinfo=timezone.utc
        )
        result["harvest_time"] = target_dt
        print(f"WARNING: No timestamp column found, using {target_dt}")
    
    result["underlying"] = underlying
    
    if "instrument_name" in df.columns:
        result["instrument_name"] = df["instrument_name"]
        
        parsed_data = []
        for inst in df["instrument_name"]:
            try:
                _, exp_dt, strike, opt_type = parse_instrument_name(str(inst))
                parsed_data.append({
                    "expiry": exp_dt.strftime("%Y-%m-%d"),
                    "expiry_timestamp": exp_dt.timestamp(),
                    "strike": strike,
                    "option_type": opt_type,
                })
            except Exception:
                parsed_data.append({
                    "expiry": None,
                    "expiry_timestamp": None,
                    "strike": None,
                    "option_type": None,
                })
        
        parsed_df = pd.DataFrame(parsed_data)
        for col in ["expiry", "expiry_timestamp", "strike", "option_type"]:
            result[col] = parsed_df[col]
    else:
        if "strike" in df.columns:
            result["strike"] = pd.to_numeric(df["strike"], errors="coerce")
        elif "strike_price" in df.columns:
            result["strike"] = pd.to_numeric(df["strike_price"], errors="coerce")
        
        if "option_type" in df.columns:
            result["option_type"] = df["option_type"].str.upper().str[0]
        elif "type" in df.columns:
            result["option_type"] = df["type"].str.upper().str[0]
        elif "cp_flag" in df.columns:
            result["option_type"] = df["cp_flag"].str.upper().str[0]
        
        expiry_cols = ["expiry", "expiration", "expiry_date", "expiration_date", "maturity"]
        for col in expiry_cols:
            if col in df.columns:
                try:
                    exp_dt_series = pd.to_datetime(df[col], utc=True)
                    result["expiry"] = exp_dt_series.dt.strftime("%Y-%m-%d")
                    result["expiry_timestamp"] = exp_dt_series.apply(lambda x: x.timestamp() if pd.notna(x) else np.nan)
                    break
                except Exception:
                    continue
        
        if "instrument_name" not in result.columns and all(
            col in result.columns for col in ["strike", "option_type", "expiry"]
        ):
            def make_name(row):
                try:
                    exp_dt = datetime.strptime(str(row["expiry"]), "%Y-%m-%d").replace(tzinfo=timezone.utc)
                    return construct_instrument_name(
                        underlying,
                        exp_dt,
                        float(row["strike"]),
                        str(row["option_type"]),
                    )
                except Exception:
                    return None
            
            result["instrument_name"] = df.apply(make_name, axis=1)
    
    price_mappings = {
        "underlying_price": ["underlying_price", "spot_price", "spot", "index_price", "btc_price", "eth_price"],
        "mark_price": ["mark_price", "mark", "mid_price", "option_price", "price"],
        "best_bid_price": ["best_bid_price", "bid_price", "bid", "best_bid"],
        "best_ask_price": ["best_ask_price", "ask_price", "ask", "best_ask"],
    }
    
    for target_col, source_cols in price_mappings.items():
        for col in source_cols:
            if col in df.columns:
                result[target_col] = pd.to_numeric(df[col], errors="coerce")
                break
        if target_col not in result.columns:
            result[target_col] = np.nan
    
    iv_mappings = {
        "mark_iv": ["mark_iv", "iv", "implied_volatility", "impl_vol", "mid_iv", "volatility"],
        "bid_iv": ["bid_iv", "bid_implied_vol"],
        "ask_iv": ["ask_iv", "ask_implied_vol"],
    }
    
    for target_col, source_cols in iv_mappings.items():
        for col in source_cols:
            if col in df.columns:
                vals = pd.to_numeric(df[col], errors="coerce")
                if vals.median() > 1 and vals.median() < 200:
                    vals = vals / 100.0
                result[target_col] = vals
                break
        if target_col not in result.columns:
            result[target_col] = np.nan
    
    greek_mappings = {
        "greek_delta": ["delta", "greek_delta", "option_delta"],
        "greek_gamma": ["gamma", "greek_gamma", "option_gamma"],
        "greek_theta": ["theta", "greek_theta", "option_theta"],
        "greek_vega": ["vega", "greek_vega", "option_vega"],
    }
    
    for target_col, source_cols in greek_mappings.items():
        for col in source_cols:
            if col in df.columns:
                result[target_col] = pd.to_numeric(df[col], errors="coerce")
                break
        if target_col not in result.columns:
            result[target_col] = np.nan
    
    other_mappings = {
        "open_interest": ["open_interest", "oi", "openinterest"],
        "volume": ["volume", "trading_volume", "vol"],
    }
    
    for target_col, source_cols in other_mappings.items():
        for col in source_cols:
            if col in df.columns:
                result[target_col] = pd.to_numeric(df[col], errors="coerce")
                break
        if target_col not in result.columns:
            result[target_col] = np.nan
    
    if "harvest_time" in result.columns and "expiry_timestamp" in result.columns:
        harvest_ts = result["harvest_time"].apply(
            lambda x: x.timestamp() if pd.notna(x) else np.nan
        )
        result["dte_days"] = (result["expiry_timestamp"] - harvest_ts) / 86400.0
    else:
        result["dte_days"] = np.nan
    
    for col in CANONICAL_COLUMNS:
        if col not in result.columns:
            result[col] = np.nan
    
    result = result[CANONICAL_COLUMNS]
    
    result = result.dropna(subset=["instrument_name", "strike", "option_type"])
    
    return result


def print_summary(df: pd.DataFrame, underlying: str, target_date: str) -> None:
    """Print summary statistics."""
    print("\n" + "=" * 60)
    print("IMPORT SUMMARY")
    print("=" * 60)
    print(f"Underlying:           {underlying}")
    print(f"Target date:          {target_date}")
    print(f"Total rows:           {len(df)}")
    print(f"Unique instruments:   {df['instrument_name'].nunique()}")
    
    if "harvest_time" in df.columns:
        print(f"Min harvest_time:     {df['harvest_time'].min()}")
        print(f"Max harvest_time:     {df['harvest_time'].max()}")
    
    if "dte_days" in df.columns:
        valid_dte = df["dte_days"].dropna()
        if len(valid_dte) > 0:
            print(f"Min dte_days:         {valid_dte.min():.2f}")
            print(f"Max dte_days:         {valid_dte.max():.2f}")
    
    if "mark_price" in df.columns:
        valid_marks = df["mark_price"].dropna()
        if len(valid_marks) > 0:
            print(f"Mark price range:     {valid_marks.min():.6f} - {valid_marks.max():.6f}")
    
    if "underlying_price" in df.columns:
        valid_spots = df["underlying_price"].dropna()
        if len(valid_spots) > 0:
            print(f"Spot price range:     ${valid_spots.min():,.2f} - ${valid_spots.max():,.2f}")
    
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Import Real Scraper Deribit options data"
    )
    parser.add_argument(
        "--underlying",
        default="BTC",
        help="Underlying asset (default: BTC)",
    )
    parser.add_argument(
        "--date",
        default="2023-05-26",
        help="Target date YYYY-MM-DD (default: 2023-05-26)",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input file (CSV, CSV.GZ, or Parquet)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/real_scraper",
        help="Output directory (default: data/real_scraper)",
    )
    
    args = parser.parse_args()
    
    underlying = args.underlying.upper()
    target_date = args.date
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        sys.exit(1)
    
    try:
        datetime.strptime(target_date, "%Y-%m-%d")
    except ValueError:
        print(f"ERROR: Invalid date format: {target_date}")
        print("Use YYYY-MM-DD format.")
        sys.exit(1)
    
    print(f"Importing Real Scraper data")
    print(f"  Input:      {input_path}")
    print(f"  Underlying: {underlying}")
    print(f"  Date:       {target_date}")
    
    df_raw = read_input_file(input_path)
    print(f"  Raw rows:   {len(df_raw)}")
    
    df_normalized = map_real_scraper_columns(df_raw, underlying, target_date)
    
    print_summary(df_normalized, underlying, target_date)
    
    out_subdir = output_dir / underlying / target_date
    out_subdir.mkdir(parents=True, exist_ok=True)
    
    out_path = out_subdir / f"{underlying}_{target_date}.parquet"
    df_normalized.to_parquet(out_path, engine="pyarrow", index=False)
    
    print(f"\nWrote normalized data to: {out_path}")


if __name__ == "__main__":
    main()
