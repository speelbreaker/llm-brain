#!/usr/bin/env python3
"""
Greg Environment Heatmaps CLI Script.

Sweeps metric pairs and finds "sweet spot" regions where:
- The environment spends non-trivial time
- Greg's selector likes a given strategy

Example usage:
    python scripts/run_greg_environment_heatmaps.py \
      --underlying BTC \
      --metrics vrp_30d adx_14d iv_rank_6m skew_25d rsi_14d price_vs_ma200 \
      --strategies "STRATEGY_A_STRADDLE" "STRATEGY_A_STRANGLE" \
      --x-bins 20 \
      --y-bins 20 \
      --min-env-frac 0.005 \
      --min-pass-frac 0.4
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtest.environment_heatmap import (
    run_full_heatmap_analysis,
    save_sweetspots_json,
    save_sweetspots_markdown,
    AVAILABLE_METRICS,
    GREG_STRATEGIES,
    STRATEGY_LABELS,
)


def main():
    parser = argparse.ArgumentParser(
        description="Greg Environment Heatmaps - Find strategy sweet spots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic BTC analysis with default strategies
  python scripts/run_greg_environment_heatmaps.py --underlying BTC

  # Full analysis with specific metrics and strategies
  python scripts/run_greg_environment_heatmaps.py \\
    --underlying BTC ETH \\
    --metrics vrp_30d adx_14d iv_rank_6m skew_25d \\
    --strategies STRATEGY_A_STRADDLE STRATEGY_A_STRANGLE \\
    --min-env-frac 0.01 \\
    --min-pass-frac 0.3

Available Metrics:
  vrp_30d, chop_factor_7d, iv_rank_6m, term_structure_spread,
  skew_25d, adx_14d, rsi_14d, price_vs_ma200

Available Strategies:
  STRATEGY_A_STRADDLE, STRATEGY_A_STRANGLE, STRATEGY_B_CALENDAR,
  STRATEGY_C_SHORT_PUT, STRATEGY_D_IRON_BUTTERFLY, STRATEGY_F_BULL_PUT_SPREAD,
  STRATEGY_F_BEAR_CALL_SPREAD
        """
    )
    
    parser.add_argument(
        "--underlying",
        nargs="+",
        default=["BTC"],
        help="Underlying assets to analyze (default: BTC)",
    )
    
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=AVAILABLE_METRICS,
        help=f"Metrics to analyze (default: all)",
    )
    
    default_strategies = [
        "STRATEGY_A_STRADDLE",
        "STRATEGY_A_STRANGLE",
        "STRATEGY_B_CALENDAR",
        "STRATEGY_C_SHORT_PUT",
        "STRATEGY_D_IRON_BUTTERFLY",
        "STRATEGY_F_BULL_PUT_SPREAD",
        "STRATEGY_F_BEAR_CALL_SPREAD",
    ]
    
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=default_strategies,
        help="Greg strategy keys to analyze (default: all)",
    )
    
    parser.add_argument(
        "--x-bins",
        type=int,
        default=20,
        help="Number of bins on X axis (default: 20)",
    )
    
    parser.add_argument(
        "--y-bins",
        type=int,
        default=20,
        help="Number of bins on Y axis (default: 20)",
    )
    
    parser.add_argument(
        "--min-env-frac",
        type=float,
        default=0.005,
        help="Minimum environment occupancy fraction (default: 0.005 = 0.5%%)",
    )
    
    parser.add_argument(
        "--min-pass-frac",
        type=float,
        default=0.4,
        help="Minimum strategy pass fraction (default: 0.4 = 40%%)",
    )
    
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=365,
        help="Historical lookback period in days (default: 365)",
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("backtest/output"),
        help="Output directory for CSV/JSON/Markdown files",
    )
    
    args = parser.parse_args()
    
    for metric in args.metrics:
        if metric not in AVAILABLE_METRICS:
            print(f"Error: Unknown metric '{metric}'")
            print(f"Available metrics: {', '.join(AVAILABLE_METRICS)}")
            sys.exit(1)
    
    for strategy in args.strategies:
        valid_keys = [s for s in GREG_STRATEGIES if s != "NO_TRADE"]
        if strategy not in valid_keys and strategy not in STRATEGY_LABELS.values():
            print(f"Error: Unknown strategy '{strategy}'")
            print(f"Available strategies: {', '.join(valid_keys)}")
            sys.exit(1)
    
    print("=" * 60)
    print("Greg Environment Heatmaps Analysis")
    print("=" * 60)
    print(f"Underlyings: {', '.join(args.underlying)}")
    print(f"Metrics: {', '.join(args.metrics)}")
    print(f"Strategies: {', '.join(args.strategies)}")
    print(f"Grid size: {args.x_bins} x {args.y_bins}")
    print(f"Min env occupancy: {args.min_env_frac * 100:.1f}%")
    print(f"Min strategy pass rate: {args.min_pass_frac * 100:.0f}%")
    print(f"Lookback: {args.lookback_days} days")
    print(f"Output dir: {args.output_dir}")
    print("=" * 60)
    
    summaries = run_full_heatmap_analysis(
        underlyings=args.underlying,
        metrics=args.metrics,
        strategies=args.strategies,
        x_bins=args.x_bins,
        y_bins=args.y_bins,
        min_env_frac=args.min_env_frac,
        min_pass_frac=args.min_pass_frac,
        lookback_days=args.lookback_days,
        output_dir=args.output_dir,
    )
    
    json_path = args.output_dir / "greg_heatmap_sweetspots.json"
    save_sweetspots_json(summaries, json_path)
    print(f"\nSaved JSON: {json_path}")
    
    md_path = args.output_dir / "greg_heatmap_sweetspots.md"
    save_sweetspots_markdown(summaries, md_path, args.min_env_frac, args.min_pass_frac)
    print(f"Saved Markdown: {md_path}")
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    total_spots = sum(len(s.sweet_spots) for s in summaries)
    print(f"Total sweet spots found: {total_spots}")
    print(f"Strategy/metric pair combinations with spots: {len(summaries)}")
    
    if summaries:
        print("\nTop sweet spots by sweetness score:")
        all_spots = []
        for s in summaries:
            for sp in s.sweet_spots:
                all_spots.append((s, sp))
        
        all_spots.sort(key=lambda x: x[1].sweetness, reverse=True)
        
        for i, (summary, spot) in enumerate(all_spots[:10], 1):
            print(f"  {i}. {summary.underlying} / {summary.strategy}")
            print(f"     {summary.x_metric} ∈ [{spot.x_low:.1f}, {spot.x_high:.1f}]")
            print(f"     {summary.y_metric} ∈ [{spot.y_low:.1f}, {spot.y_high:.1f}]")
            print(f"     Occupancy: {spot.occupancy_frac * 100:.1f}%, Pass: {spot.strategy_pass_frac * 100:.0f}%, Score: {spot.sweetness:.4f}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
