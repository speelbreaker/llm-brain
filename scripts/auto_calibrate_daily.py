#!/usr/bin/env python3
"""
Daily auto-calibration script for BTC and ETH.

This script:
1. Runs calibration for both BTC and ETH underlyings
2. Applies the update policy (min_delta, min_samples, min_vega, smoothing)
3. Records results to calibration_history table
4. Exits with non-zero if all runs failed or all are degraded

Usage:
    python -m scripts.auto_calibrate_daily
    python -m scripts.auto_calibrate_daily --dry-run
    python -m scripts.auto_calibrate_daily --underlyings BTC,ETH
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Literal, Optional, cast

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.calibration_config import (
    CalibrationConfig,
    HarvestConfig,
    BandConfig,
)
from src.calibration_extended import run_historical_calibration_from_harvest
from src.db.models_calibration import (
    CalibrationHistoryEntry,
    insert_calibration_history,
    assess_calibration_realism,
)
from src.calibration_update_policy import (
    CalibrationUpdatePolicy,
    get_policy,
    get_current_applied_multipliers,
    get_smoothed_multipliers,
    should_apply_update,
    load_recent_calibration_history,
    BandMultiplier,
)
from src.calibration_store import set_applied_multiplier
from src.db import init_db


@dataclass
class CalibrationRunResult:
    """Result of a single underlying calibration run."""
    underlying: str
    status: Literal["ok", "degraded", "failed"]
    reason: str
    multiplier: Optional[float] = None
    smoothed_multiplier: Optional[float] = None
    mae_pct: Optional[float] = None
    vega_weighted_mae_pct: Optional[float] = None
    num_samples: int = 0
    applied: bool = False
    applied_reason: str = ""


def run_calibration_for_underlying(
    underlying: str,
    dte_min: int,
    dte_max: int,
    lookback_days: int,
    max_samples: int,
    data_dir: str,
    policy: CalibrationUpdatePolicy,
    dry_run: bool = False,
) -> CalibrationRunResult:
    """
    Run calibration for a single underlying with policy enforcement.
    
    Returns CalibrationRunResult with status and metrics.
    """
    print(f"\n{'='*60}")
    print(f"Auto IV Calibration: {underlying}")
    print(f"{'='*60}")
    print(f"  DTE range: {dte_min}–{dte_max} days")
    print(f"  Lookback: {lookback_days} days")
    print(f"  Max samples: {max_samples}")
    print(f"  Data dir: {data_dir}")
    
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=lookback_days)
    
    config = CalibrationConfig(
        underlying=underlying,
        min_dte=float(dte_min),
        max_dte=float(dte_max),
        source="harvested",
        harvest=HarvestConfig(
            data_root=data_dir,
            underlying=underlying,
            start_time=start_time,
            end_time=end_time,
            snapshot_step=1,
            max_snapshots=None,
        ),
        option_types=["C"],
        bands=[
            BandConfig(name="target", min_dte=dte_min, max_dte=dte_max),
        ],
        fit_skew=True,
        emit_recommended_vol_surface=True,
        max_samples=max_samples,
        return_rows=False,
    )
    
    try:
        result = run_historical_calibration_from_harvest(config, validate_quality=True)
    except Exception as e:
        print(f"\n  ERROR: Calibration failed - {e}")
        return CalibrationRunResult(
            underlying=underlying,
            status="failed",
            reason=f"Calibration exception: {str(e)}",
        )
    
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
    
    print(f"\n  Metrics:")
    print(f"    Recommended multiplier: {rec_mult:.4f}")
    print(f"    Samples: {result.count:,}")
    if vega_weighted_mae_pct is not None:
        print(f"    Vega-weighted MAE: {vega_weighted_mae_pct:.2f}%")
    print(f"    MAE: {mae_pct:.2f}%")
    print(f"    Bias: {bias_pct:.2f}%")
    print(f"    Realism status: {status.upper()}")
    print(f"    Reason: {reason}")
    
    history = load_recent_calibration_history(underlying, limit=50)
    
    recommended_bands: Optional[List[BandMultiplier]] = None
    if result.bands:
        recommended_bands = [
            BandMultiplier(
                name=b.name,
                min_dte=b.min_dte,
                max_dte=b.max_dte,
                iv_multiplier=b.recommended_iv_multiplier or 1.0,
            )
            for b in result.bands
            if b.recommended_iv_multiplier is not None
        ]
    
    smoothed_global, smoothed_bands = get_smoothed_multipliers(
        history, rec_mult, recommended_bands, policy
    )
    
    print(f"\n  Smoothing:")
    print(f"    Raw recommended: {rec_mult:.4f}")
    print(f"    Smoothed (EWMA): {smoothed_global:.4f}")
    print(f"    History window: {policy.smoothing_window_days} days")
    print(f"    EWMA alpha: {policy.ewma_alpha}")
    
    current_applied = get_current_applied_multipliers(underlying)
    sample_size = result.count
    vega_sum = sample_size * 10.0
    
    update_decision = should_apply_update(
        current_applied, smoothed_global, smoothed_bands,
        policy, sample_size, vega_sum
    )
    
    applied = False
    applied_reason = ""
    
    if status == "failed":
        applied = False
        applied_reason = f"Realism check failed: {reason}"
        print(f"\n  Update: NOT APPLIED (realism check failed)")
    elif update_decision.should_apply:
        applied = True
        applied_reason = update_decision.reason
        print(f"\n  Update: WILL APPLY - {applied_reason}")
        
        if not dry_run:
            band_multipliers_dict = None
            if smoothed_bands:
                band_multipliers_dict = {b.name: b.iv_multiplier for b in smoothed_bands}
            
            set_applied_multiplier(
                underlying=underlying,
                global_multiplier=smoothed_global,
                band_multipliers=band_multipliers_dict,
                source="harvested",
                applied_reason=f"Daily auto-calibration: {applied_reason}",
            )
            print(f"    Applied multiplier {smoothed_global:.4f} to {underlying}")
    else:
        applied = False
        applied_reason = update_decision.reason
        print(f"\n  Update: NOT APPLIED - {applied_reason}")
    
    if not dry_run:
        entry = CalibrationHistoryEntry(
            underlying=underlying,
            dte_min=dte_min,
            dte_max=dte_max,
            lookback_days=lookback_days,
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
        print(f"    Saved to calibration_history (id={row_id})")
    else:
        print(f"    [DRY RUN] Would save to calibration_history")
    
    typed_status = cast(Literal["ok", "degraded", "failed"], status)
    return CalibrationRunResult(
        underlying=underlying,
        status=typed_status,
        reason=reason,
        multiplier=rec_mult,
        smoothed_multiplier=smoothed_global,
        mae_pct=mae_pct,
        vega_weighted_mae_pct=vega_weighted_mae_pct,
        num_samples=result.count,
        applied=applied,
        applied_reason=applied_reason,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Daily auto-calibration for BTC and ETH with update policy."
    )
    parser.add_argument(
        "--underlyings",
        type=str,
        default="BTC,ETH",
        help="Comma-separated list of underlyings (default: BTC,ETH)",
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
        help="Don't save to database or apply updates, just print results",
    )
    
    args = parser.parse_args()
    
    underlyings = [u.strip().upper() for u in args.underlyings.split(",")]
    for u in underlyings:
        if u not in ("BTC", "ETH"):
            print(f"ERROR: Invalid underlying '{u}'. Must be BTC or ETH.")
            sys.exit(1)
    
    print(f"\n{'#'*60}")
    print("Daily Auto-Calibration")
    print(f"{'#'*60}")
    print(f"  Underlyings: {', '.join(underlyings)}")
    print(f"  DTE range: {args.dte_min}–{args.dte_max} days")
    print(f"  Lookback: {args.lookback_days} days")
    print(f"  Max samples: {args.max_samples}")
    print(f"  Data dir: {args.data_dir}")
    print(f"  Dry run: {args.dry_run}")
    print(f"  Started: {datetime.now(timezone.utc).isoformat()}")
    print(f"{'#'*60}")
    
    if not args.dry_run:
        init_db()
    
    policy = get_policy()
    print(f"\nUpdate Policy:")
    print(f"  min_delta_global: {policy.min_delta_global}")
    print(f"  min_sample_size: {policy.min_sample_size}")
    print(f"  min_vega_sum: {policy.min_vega_sum}")
    print(f"  smoothing_window_days: {policy.smoothing_window_days}")
    print(f"  ewma_alpha: {policy.ewma_alpha}")
    
    results: List[CalibrationRunResult] = []
    
    for underlying in underlyings:
        result = run_calibration_for_underlying(
            underlying=underlying,
            dte_min=args.dte_min,
            dte_max=args.dte_max,
            lookback_days=args.lookback_days,
            max_samples=args.max_samples,
            data_dir=args.data_dir,
            policy=policy,
            dry_run=args.dry_run,
        )
        results.append(result)
    
    print(f"\n{'#'*60}")
    print("Summary")
    print(f"{'#'*60}")
    
    ok_count = sum(1 for r in results if r.status == "ok")
    degraded_count = sum(1 for r in results if r.status == "degraded")
    failed_count = sum(1 for r in results if r.status == "failed")
    applied_count = sum(1 for r in results if r.applied)
    
    for r in results:
        status_icon = {"ok": "OK", "degraded": "DEGRADED", "failed": "FAILED"}[r.status]
        applied_str = "APPLIED" if r.applied else "not applied"
        mult_str = f"{r.smoothed_multiplier:.4f}" if r.smoothed_multiplier else "N/A"
        print(f"  {r.underlying}: {status_icon}, mult={mult_str}, {applied_str}")
    
    print(f"\nTotals:")
    print(f"  OK: {ok_count}")
    print(f"  Degraded: {degraded_count}")
    print(f"  Failed: {failed_count}")
    print(f"  Applied: {applied_count}")
    print(f"\nFinished: {datetime.now(timezone.utc).isoformat()}")
    print(f"{'#'*60}\n")
    
    if failed_count == len(results):
        print("EXIT 2: All calibrations failed")
        sys.exit(2)
    elif degraded_count == len(results):
        print("EXIT 1: All calibrations degraded")
        sys.exit(1)
    elif ok_count == 0:
        print("EXIT 1: No successful calibrations")
        sys.exit(1)
    else:
        print("EXIT 0: At least one successful calibration")
        sys.exit(0)


if __name__ == "__main__":
    main()
