#!/usr/bin/env python3
"""
CLI script to run calibration and update synthetic vol_surface configuration.

Now with update policy support: smoothing, thresholds, and history tracking.

Usage:
    python scripts/update_vol_surface_from_calibration.py --underlying BTC --min-dte 3 --max-dte 30
    python scripts/update_vol_surface_from_calibration.py --config calibration_config.json
    python scripts/update_vol_surface_from_calibration.py --underlying BTC --source harvested --data-root data/live_deribit
    python scripts/update_vol_surface_from_calibration.py --underlying BTC --force  # Force apply regardless of thresholds

Outputs a YAML/JSON snippet for vol_surface ready to paste into synthetic config.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.calibration_config import CalibrationConfig, HarvestConfig, BandConfig
from src.calibration_extended import run_calibration_extended, build_vol_surface_from_calibration
from src.calibration_update_policy import (
    CalibrationUpdatePolicy,
    run_calibration_with_policy,
    get_policy,
    load_recent_calibration_history,
    get_current_applied_multipliers,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run calibration and generate vol_surface config"
    )
    
    parser.add_argument("--underlying", "-u", type=str, default="BTC",
                       help="Underlying asset (BTC or ETH)")
    parser.add_argument("--min-dte", type=float, default=3.0,
                       help="Minimum DTE for calibration")
    parser.add_argument("--max-dte", type=float, default=30.0,
                       help="Maximum DTE for calibration")
    parser.add_argument("--iv-multiplier", type=float, default=1.0,
                       help="Current IV multiplier")
    parser.add_argument("--rv-window", type=int, default=7,
                       help="RV window in days")
    parser.add_argument("--max-samples", type=int, default=80,
                       help="Max samples to process")
    
    parser.add_argument("--source", type=str, choices=["live", "harvested"], default="live",
                       help="Data source: live API or harvested Parquet")
    parser.add_argument("--data-root", type=str, default="data/live_deribit",
                       help="Root directory for harvested data")
    parser.add_argument("--start-time", type=str, default=None,
                       help="Start time for harvested data (ISO format)")
    parser.add_argument("--end-time", type=str, default=None,
                       help="End time for harvested data (ISO format)")
    
    parser.add_argument("--fit-skew", action="store_true",
                       help="Fit skew anchor ratios")
    parser.add_argument("--bands", type=str, default=None,
                       help="DTE bands as JSON: [{\"name\":\"weekly\",\"min_dte\":3,\"max_dte\":10}]")
    
    parser.add_argument("--config", type=str, default=None,
                       help="Path to calibration config JSON file")
    
    parser.add_argument("--output", "-o", type=str, default=None,
                       help="Output file path (default: stdout)")
    parser.add_argument("--format", type=str, choices=["json", "yaml"], default="json",
                       help="Output format")
    
    parser.add_argument("--force", "-f", action="store_true",
                       help="Force apply calibration regardless of policy thresholds")
    parser.add_argument("--no-policy", action="store_true",
                       help="Skip policy checks and just output recommended values (legacy mode)")
    
    parser.add_argument("--min-delta", type=float, default=None,
                       help="Override policy min_delta_global threshold")
    parser.add_argument("--min-samples", type=int, default=None,
                       help="Override policy min_sample_size threshold")
    parser.add_argument("--smoothing-days", type=int, default=None,
                       help="Override policy smoothing_window_days")
    
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    
    if args.no_policy:
        return run_legacy_mode(args)
    
    return run_with_policy(args)


def run_with_policy(args: argparse.Namespace) -> int:
    """Run calibration with update policy (new behavior)."""
    
    policy = get_policy()
    if args.min_delta is not None:
        policy.min_delta_global = args.min_delta
        policy.min_delta_band = args.min_delta
    if args.min_samples is not None:
        policy.min_sample_size = args.min_samples
    if args.smoothing_days is not None:
        policy.smoothing_window_days = args.smoothing_days
    
    source: Literal["live", "harvested"] = "live" if args.source == "live" else "harvested"
    
    print(f"Running calibration for {args.underlying}...", file=sys.stderr)
    print(f"  Source: {source}", file=sys.stderr)
    print(f"  DTE range: [{args.min_dte}, {args.max_dte}]", file=sys.stderr)
    print(f"  Policy: min_delta={policy.min_delta_global}, min_samples={policy.min_sample_size}, smoothing={policy.smoothing_window_days}d", file=sys.stderr)
    if args.force:
        print(f"  FORCE mode: will apply regardless of thresholds", file=sys.stderr)
    
    try:
        config_kwargs = {
            "min_dte": args.min_dte,
            "max_dte": args.max_dte,
            "iv_multiplier": args.iv_multiplier,
            "rv_window_days": args.rv_window,
            "max_samples": args.max_samples,
            "fit_skew": args.fit_skew,
        }
        
        if source == "harvested":
            config_kwargs["data_root"] = args.data_root
        
        if args.bands:
            bands_list = json.loads(args.bands)
            config_kwargs["bands"] = [BandConfig(**b) for b in bands_list]
        
        record, decision = run_calibration_with_policy(
            underlying=args.underlying,
            source=source,
            force=args.force,
            policy=policy,
            **config_kwargs,
        )
        
    except Exception as e:
        print(f"ERROR: Calibration failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1
    
    print(f"\nCalibration results:", file=sys.stderr)
    print(f"  Sample size: {record.sample_size}", file=sys.stderr)
    print(f"  Vega sum: {record.vega_sum:.1f}", file=sys.stderr)
    print(f"  Recommended IV multiplier: {record.recommended_iv_multiplier:.4f}", file=sys.stderr)
    print(f"  Smoothed IV multiplier: {record.smoothed_global_multiplier:.4f}", file=sys.stderr)
    
    if record.recommended_band_multipliers:
        print(f"  Band multipliers:", file=sys.stderr)
        for band in record.recommended_band_multipliers:
            smoothed = None
            if record.smoothed_band_multipliers:
                for sb in record.smoothed_band_multipliers:
                    if sb.name == band.name:
                        smoothed = sb.iv_multiplier
                        break
            smoothed_str = f" -> smoothed: {smoothed:.4f}" if smoothed else ""
            print(f"    {band.name}: {band.iv_multiplier:.4f}{smoothed_str}", file=sys.stderr)
    
    print(f"\nPolicy decision:", file=sys.stderr)
    if decision.should_apply:
        print(f"  ✅ APPLIED: {decision.reason}", file=sys.stderr)
    else:
        print(f"  ❌ NOT APPLIED: {decision.reason}", file=sys.stderr)
    
    print(f"\nRun saved to: data/calibration_runs/", file=sys.stderr)
    
    current = get_current_applied_multipliers()
    output_data = {
        "timestamp": record.timestamp.isoformat(),
        "underlying": record.underlying,
        "source": record.source,
        "recommended_iv_multiplier": record.recommended_iv_multiplier,
        "smoothed_iv_multiplier": record.smoothed_global_multiplier,
        "applied": record.applied,
        "applied_reason": record.applied_reason,
        "current_global_multiplier": current.global_multiplier,
    }
    
    if args.format == "yaml":
        try:
            import yaml
            output = yaml.dump(output_data, default_flow_style=False, sort_keys=False)
        except ImportError:
            output = json.dumps(output_data, indent=2)
    else:
        output = json.dumps(output_data, indent=2)
    
    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"\nResult written to: {args.output}", file=sys.stderr)
    else:
        print("\n# Calibration result:", file=sys.stderr)
        print(output)
    
    return 0


def run_legacy_mode(args: argparse.Namespace) -> int:
    """Run calibration without policy (legacy behavior for compatibility)."""
    
    if args.config:
        with open(args.config) as f:
            config_dict = json.load(f)
        config = CalibrationConfig(**config_dict)
    else:
        harvest = None
        if args.source == "harvested":
            start_time = None
            end_time = None
            if args.start_time:
                start_time = datetime.fromisoformat(args.start_time.replace("Z", "+00:00"))
            if args.end_time:
                end_time = datetime.fromisoformat(args.end_time.replace("Z", "+00:00"))
            
            harvest = HarvestConfig(
                data_root=args.data_root,
                underlying=args.underlying,
                start_time=start_time,
                end_time=end_time,
            )
        
        bands = None
        if args.bands:
            bands_list = json.loads(args.bands)
            bands = [BandConfig(**b) for b in bands_list]
        
        config = CalibrationConfig(
            underlying=args.underlying,
            min_dte=args.min_dte,
            max_dte=args.max_dte,
            iv_multiplier=args.iv_multiplier,
            rv_window_days=args.rv_window,
            max_samples=args.max_samples,
            source=args.source,
            harvest=harvest,
            bands=bands,
            fit_skew=args.fit_skew,
            emit_recommended_vol_surface=True,
            return_rows=False,
        )
    
    print(f"Running calibration for {config.underlying} (legacy mode)...", file=sys.stderr)
    print(f"  Source: {config.source}", file=sys.stderr)
    print(f"  DTE range: [{config.min_dte}, {config.max_dte}]", file=sys.stderr)
    
    try:
        result = run_calibration_extended(config)
    except Exception as e:
        print(f"ERROR: Calibration failed: {e}", file=sys.stderr)
        return 1
    
    print(f"\nCalibration results:", file=sys.stderr)
    print(f"  Count: {result.count}", file=sys.stderr)
    print(f"  MAE: {result.mae_pct:.2f}%", file=sys.stderr)
    print(f"  Bias: {result.bias_pct:.2f}%", file=sys.stderr)
    print(f"  RV (annualized): {result.rv_annualized:.2%}" if result.rv_annualized else "", file=sys.stderr)
    print(f"  Recommended IV multiplier: {result.recommended_iv_multiplier}" if result.recommended_iv_multiplier else "", file=sys.stderr)
    
    vol_surface = build_vol_surface_from_calibration(result)
    
    if args.format == "yaml":
        try:
            import yaml
            output = yaml.dump(vol_surface, default_flow_style=False, sort_keys=False)
        except ImportError:
            output = json.dumps(vol_surface, indent=2)
    else:
        output = json.dumps(vol_surface, indent=2)
    
    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"\nVol surface config written to: {args.output}", file=sys.stderr)
    else:
        print("\n# Recommended vol_surface configuration:", file=sys.stderr)
        print(output)
    
    if result.recommended_skew:
        print(f"\nSkew anchor ratios:", file=sys.stderr)
        for delta, ratio in result.recommended_skew.anchor_ratios.items():
            print(f"  Delta {delta}: {ratio:.4f}", file=sys.stderr)
    
    if result.vol_surface_diff:
        print(f"\nVol surface diff from current:", file=sys.stderr)
        print(f"  IV multiplier delta: {result.vol_surface_diff.iv_multiplier_delta:+.4f}", file=sys.stderr)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
