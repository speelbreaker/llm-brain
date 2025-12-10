#!/usr/bin/env python3
"""
CLI script to run calibration and update synthetic vol_surface configuration.

Usage:
    python scripts/update_vol_surface_from_calibration.py --underlying BTC --min-dte 3 --max-dte 30
    python scripts/update_vol_surface_from_calibration.py --config calibration_config.json
    python scripts/update_vol_surface_from_calibration.py --underlying BTC --source harvested --data-root data/live_deribit

Outputs a YAML/JSON snippet for vol_surface ready to paste into synthetic config.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.calibration_config import CalibrationConfig, HarvestConfig, BandConfig
from src.calibration_extended import run_calibration_extended, build_vol_surface_from_calibration


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
    
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    
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
    
    print(f"Running calibration for {config.underlying}...", file=sys.stderr)
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
