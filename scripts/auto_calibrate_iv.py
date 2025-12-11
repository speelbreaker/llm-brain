#!/usr/bin/env python3
"""
Auto-calibrate IV multiplier using harvester data.

This script:
1. Loads multi-day option snapshots from the harvester (parquet files)
2. Filters to near-ATM, near-dated calls
3. Fits an IV multiplier that minimizes MAE between synthetic BS prices and observed mark prices
4. Stores the result in the calibration_history database table

Usage:
    python scripts/auto_calibrate_iv.py --underlying BTC --dte-min 3 --dte-max 10 --lookback-days 14
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data.live_deribit_exam import build_live_deribit_exam_dataset
from src.calibration_core import extract_calibration_samples, calibrate_iv_multiplier
from src.db.models_calibration import CalibrationHistoryEntry, insert_calibration_history
from src.db import init_db


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Auto-calibrate IV multiplier using harvester data."
    )
    parser.add_argument(
        "--underlying",
        type=str,
        required=True,
        choices=["BTC", "ETH"],
        help="Underlying asset (BTC or ETH)",
    )
    parser.add_argument(
        "--dte-min",
        type=int,
        default=3,
        help="Minimum days to expiry (default: 3)",
    )
    parser.add_argument(
        "--dte-max",
        type=int,
        default=10,
        help="Maximum days to expiry (default: 10)",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=14,
        help="Number of days of harvester data to use (default: 14)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=5000,
        help="Maximum number of option snapshots for calibration (default: 5000)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/live_deribit",
        help="Harvester data directory (default: data/live_deribit)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't save to database, just print results",
    )
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("Auto IV Calibration")
    print(f"{'='*60}")
    print(f"  Underlying: {args.underlying}")
    print(f"  DTE range: {args.dte_min}–{args.dte_max} days")
    print(f"  Lookback: {args.lookback_days} days")
    print(f"  Max samples: {args.max_samples}")
    print(f"  Data dir: {args.data_dir}")
    print(f"{'='*60}\n")
    
    end_date = date.today()
    start_date = end_date - timedelta(days=args.lookback_days)
    
    print(f"Loading harvester data from {start_date} to {end_date}...")
    
    try:
        df, summary = build_live_deribit_exam_dataset(
            underlying=args.underlying,
            start_date=start_date,
            end_date=end_date,
            base_dir=args.data_dir,
            write_files=False,
        )
    except ValueError as e:
        print(f"\nError: {e}")
        print("Make sure the data harvester has collected data for the specified period.")
        sys.exit(1)
    
    print(f"  Loaded {len(df):,} rows from {summary.get('num_files', 0)} files")
    print(f"  Snapshots: {summary.get('num_snapshots', 0)}")
    print(f"  Time range: {summary.get('time_min', 'N/A')} to {summary.get('time_max', 'N/A')}")
    
    print(f"\nExtracting calibration samples...")
    samples = extract_calibration_samples(
        df=df,
        underlying=args.underlying,
        dte_min=args.dte_min,
        dte_max=args.dte_max,
        moneyness_range=0.10,
        max_samples=args.max_samples,
    )
    
    if not samples:
        print("\nNo valid samples found after filtering. Check:")
        print("  - DTE range may be too narrow")
        print("  - Moneyness filter may be too strict")
        print("  - Data may not have the required columns")
        sys.exit(1)
    
    print(f"  Found {len(samples):,} calibration samples")
    
    print(f"\nFitting IV multiplier...")
    result = calibrate_iv_multiplier(
        samples=samples,
        multiplier_min=0.5,
        multiplier_max=1.5,
        search_points=100,
    )
    
    print(f"\n{'='*60}")
    print("Auto IV calibration complete:")
    print(f"{'='*60}")
    print(f"  Underlying: {args.underlying}")
    print(f"  DTE range: {args.dte_min}–{args.dte_max} days")
    print(f"  Lookback: {args.lookback_days} days")
    print(f"  Samples: {result.num_samples:,}")
    print(f"  Recommended multiplier: {result.best_multiplier:.4f}")
    print(f"  MAE: {result.mae_pct:.2f}% of mark")
    print(f"{'='*60}")
    
    if args.dry_run:
        print("\n[DRY RUN] Skipping database save.")
    else:
        print("\nSaving to calibration_history...")
        init_db()
        
        entry = CalibrationHistoryEntry(
            underlying=args.underlying,
            dte_min=args.dte_min,
            dte_max=args.dte_max,
            lookback_days=args.lookback_days,
            multiplier=float(result.best_multiplier),
            mae_pct=float(result.mae_pct),
            num_samples=int(result.num_samples),
        )
        
        row_id = insert_calibration_history(entry)
        print(f"Row saved to calibration_history (id={row_id})")
    
    print("\nDone.")


if __name__ == "__main__":
    main()
