#!/usr/bin/env python3
"""
Auto-calibrate IV multiplier using harvester data.

This script uses the extended calibration engine (run_historical_calibration_from_harvest)
with realism guardrails to ensure only valid calibrations are marked as usable.

Usage:
    python scripts/auto_calibrate_iv.py --underlying BTC --dte-min 3 --dte-max 10 --lookback-days 14
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.calibration_config import (
    CalibrationConfig,
    HarvestConfig,
    BandConfig,
    DEFAULT_TERM_BANDS,
)
from src.calibration_extended import run_historical_calibration_from_harvest
from src.db.models_calibration import (
    CalibrationHistoryEntry,
    insert_calibration_history,
    assess_calibration_realism,
)
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
    print("Auto IV Calibration (Extended Engine)")
    print(f"{'='*60}")
    print(f"  Underlying: {args.underlying}")
    print(f"  DTE range: {args.dte_min}–{args.dte_max} days")
    print(f"  Lookback: {args.lookback_days} days")
    print(f"  Max samples: {args.max_samples}")
    print(f"  Data dir: {args.data_dir}")
    print(f"{'='*60}\n")
    
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=args.lookback_days)
    
    print(f"Building calibration config...")
    print(f"  Time range: {start_time.isoformat()} to {end_time.isoformat()}")
    
    config = CalibrationConfig(
        underlying=args.underlying,
        min_dte=float(args.dte_min),
        max_dte=float(args.dte_max),
        source="harvested",
        harvest=HarvestConfig(
            data_root=args.data_dir,
            underlying=args.underlying,
            start_time=start_time,
            end_time=end_time,
            snapshot_step=1,
            max_snapshots=None,
        ),
        option_types=["C"],
        bands=[
            BandConfig(name="target", min_dte=args.dte_min, max_dte=args.dte_max),
        ],
        fit_skew=True,
        emit_recommended_vol_surface=True,
        max_samples=args.max_samples,
        return_rows=False,
    )
    
    print(f"\nRunning historical calibration from harvest...")
    
    try:
        result = run_historical_calibration_from_harvest(config, validate_quality=True)
    except Exception as e:
        print(f"\nError during calibration: {e}")
        print("Check harvester data availability and schema.")
        sys.exit(1)
    
    rec_mult = result.recommended_iv_multiplier or result.iv_multiplier
    global_metrics = result.global_metrics
    data_quality = result.data_quality
    
    mae_pct = global_metrics.mae_pct if global_metrics else result.mae_pct
    vega_weighted_mae_pct = global_metrics.vega_weighted_mae_pct if global_metrics else None
    bias_pct = global_metrics.bias_pct if global_metrics else result.bias_pct
    dq_status = data_quality.status if data_quality else "ok"
    
    status, reason = assess_calibration_realism(
        multiplier=rec_mult,
        mae_pct=mae_pct,
        vega_weighted_mae_pct=vega_weighted_mae_pct,
        data_quality_status=dq_status,
    )
    
    print(f"\n{'='*60}")
    print("Auto IV Calibration Complete")
    print(f"{'='*60}")
    print(f"  Underlying: {args.underlying}")
    print(f"  DTE range: {args.dte_min}–{args.dte_max} days")
    print(f"  Lookback: {args.lookback_days} days")
    print(f"  Samples: {result.count:,}")
    print(f"  Snapshots: {result.snapshot_count}")
    print(f"{'='*60}")
    print(f"\nMetrics:")
    print(f"  Recommended multiplier: {rec_mult:.4f}")
    if vega_weighted_mae_pct is not None:
        print(f"  Vega-weighted MAE: {vega_weighted_mae_pct:.2f}%")
    print(f"  MAE: {mae_pct:.2f}%")
    print(f"  Bias: {bias_pct:.2f}%")
    
    if result.bands:
        print(f"\nPer-band metrics:")
        for band in result.bands:
            band_mult = band.recommended_iv_multiplier or "-"
            band_vmae = f"{band.vega_weighted_mae_pct:.2f}%" if band.vega_weighted_mae_pct else "-"
            print(f"  {band.name}: mult={band_mult}, vMAE={band_vmae}, count={band.count}")
    
    if result.skew_misfit:
        print(f"\nSkew misfit:")
        print(f"  Max abs diff: {result.skew_misfit.max_abs_diff:.4f}")
        for anchor, diff in result.skew_misfit.anchor_diffs.items():
            print(f"  {anchor}: {diff:+.4f}")
    
    if data_quality:
        print(f"\nData quality:")
        print(f"  Status: {data_quality.status}")
        print(f"  Snapshots: {data_quality.num_snapshots}")
        print(f"  Schema failures: {data_quality.num_schema_failures}")
        if data_quality.issues:
            for issue in data_quality.issues:
                print(f"  - {issue}")
    
    print(f"\n{'='*60}")
    print(f"Realism Assessment:")
    print(f"{'='*60}")
    
    if status == "ok":
        print(f"  Status: OK (green)")
        print(f"  Reason: {reason}")
        print(f"\nResult looks reasonable. See Calibration UI → Calibration History (Auto-Calibrate).")
    elif status == "degraded":
        print(f"  Status: DEGRADED (orange)")
        print(f"  Reason: {reason}")
        print(f"\nResult usable but exercise caution.")
    else:
        print(f"\n{'!'*60}")
        print(f"  WARNING: Auto-calibration flagged as FAILED")
        print(f"{'!'*60}")
        print(f"  Status: FAILED (red)")
        print(f"  Reason: {reason}")
        print(f"\nThis run will be recorded for debugging but should NOT be")
        print(f"used to update the vol surface.")
        print(f"{'!'*60}")
    
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
            multiplier=float(rec_mult),
            mae_pct=float(mae_pct) if mae_pct is not None else 0.0,
            vega_weighted_mae_pct=float(vega_weighted_mae_pct) if vega_weighted_mae_pct is not None else None,
            bias_pct=float(bias_pct) if bias_pct is not None else None,
            num_samples=int(result.count),
            source="harvested",
            status=status,
            reason=reason,
        )
        
        row_id = insert_calibration_history(entry)
        print(f"Row saved to calibration_history (id={row_id}, status={status})")
    
    print("\nDone.")


if __name__ == "__main__":
    main()
