#!/usr/bin/env python3
"""
Build Exam Dataset from Live Deribit Captures

CLI wrapper for the reusable build_live_deribit_exam_dataset function.

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
import sys
from datetime import datetime, timezone
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.live_deribit_exam import build_live_deribit_exam_dataset


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
    
    try:
        start_date = datetime.strptime(args.start, "%Y-%m-%d").date()
        end_date = datetime.strptime(args.end, "%Y-%m-%d").date()
    except ValueError as e:
        print(f"ERROR: Invalid date format: {e}")
        print("Use YYYY-MM-DD format.")
        sys.exit(1)
    
    if start_date > end_date:
        print("ERROR: Start date must be before or equal to end date.")
        sys.exit(1)
    
    underlying = args.underlying.upper()
    
    print(f"Building exam dataset for {underlying}")
    print(f"Date range: {args.start} to {args.end}")
    print(f"Data root: {args.data_root}")
    print(f"Output dir: {args.out_dir}")
    
    try:
        df, summary = build_live_deribit_exam_dataset(
            underlying=underlying,
            start_date=start_date,
            end_date=end_date,
            base_dir=Path(args.data_root),
            exams_dir=Path(args.out_dir),
            write_files=True,
        )
        
        print("\n" + "=" * 60)
        print("EXAM DATASET SUMMARY")
        print("=" * 60)
        print(f"Underlying:           {summary.get('underlying')}")
        print(f"Files stitched:       {summary.get('num_files')}")
        print(f"Total rows:           {summary.get('num_rows')}")
        
        if summary.get("num_snapshots"):
            print(f"Unique snapshots:     {summary.get('num_snapshots')}")
        
        if summary.get("num_instruments"):
            print(f"Unique instruments:   {summary.get('num_instruments')}")
        
        if summary.get("time_min"):
            print(f"Min harvest_time:     {summary.get('time_min')}")
            print(f"Max harvest_time:     {summary.get('time_max')}")
        
        if summary.get("dte_min") is not None:
            print(f"Min dte_days:         {summary.get('dte_min'):.2f}")
            print(f"Max dte_days:         {summary.get('dte_max'):.2f}")
        
        print(f"\nColumns ({len(summary.get('columns', []))}):")
        for col in summary.get("columns", []):
            print(f"  - {col}")
        
        missing = summary.get("missing_required_columns", [])
        if missing:
            print("\nWARNINGS:")
            for col in missing:
                print(f"  - Missing required column: {col}")
        
        print("=" * 60)
        
        out_filename = f"{underlying}_{args.start}_{args.end}_live_deribit.parquet"
        summary_filename = f"{underlying}_{args.start}_{args.end}_live_deribit_summary.json"
        print(f"\nWrote exam dataset to: {args.out_dir}/{out_filename}")
        print(f"Wrote summary to: {args.out_dir}/{summary_filename}")
        
    except ValueError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
