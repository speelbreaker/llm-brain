#!/usr/bin/env python3
"""
Compare SYNTHETIC vs LIVE_DERIBIT backtests over the same period.

This script runs two backtests (one with synthetic pricing, one with captured
Deribit data) and prints a side-by-side comparison of metrics.

Usage:
    python scripts/compare_synthetic_vs_live.py \
        --underlying BTC \
        --start 2025-12-01 \
        --end 2025-12-07 \
        [--decision-interval-minutes 1440] \
        [--exit-style tp_and_roll]
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.db import get_db_session
from src.db.backtest_service import (
    create_backtest_run,
    complete_run,
    fail_run,
    get_run_with_details,
)
from src.backtest.config_schema import DataSourceType


def run_backtest_with_data_source(
    underlying: str,
    start_ts: datetime,
    end_ts: datetime,
    data_source: DataSourceType,
    decision_interval_minutes: int,
    exit_style: str,
) -> str:
    """
    Run a backtest and return the run_id.
    """
    from src.backtest.covered_call_simulator import CoveredCallSimulator
    from src.backtest.deribit_data_source import DeribitDataSource
    from src.backtest.types import CallSimulationConfig
    from src.backtest.manager import _compute_equity_curve, _compute_enhanced_metrics
    
    with get_db_session() as db:
        try:
            run = create_backtest_run(
                db=db,
                underlying=underlying,
                start_ts=start_ts,
                end_ts=end_ts,
                data_source=data_source.value,
                decision_interval_minutes=decision_interval_minutes,
                primary_exit_style=exit_style,
                config_json={
                    "underlying": underlying,
                    "data_source": data_source.value,
                    "decision_interval_minutes": decision_interval_minutes,
                    "exit_style": exit_style,
                },
            )
            
            print(f"  Created run: {run.run_id} (data_source={data_source.value})")
            
            decision_interval_hours = decision_interval_minutes / 60
            decision_interval_bars = max(1, int(decision_interval_hours))
            
            pricing_mode = "deribit_live" if data_source == DataSourceType.LIVE_DERIBIT else "synthetic_bs"
            
            config = CallSimulationConfig(
                underlying=underlying,
                start=start_ts,
                end=end_ts,
                timeframe="1h",
                decision_interval_bars=decision_interval_bars,
                initial_spot_position=1.0,
                contract_size=1.0,
                fee_rate=0.0003,
                target_dte=7,
                dte_tolerance=3,
                target_delta=0.25,
                delta_tolerance=0.10,
                min_dte=1,
                max_dte=21,
                delta_min=0.10,
                delta_max=0.40,
                option_margin_type="linear",
                option_settlement_ccy="USDC",
                tp_threshold_pct=80.0,
                min_score_to_trade=3.0,
                pricing_mode=pricing_mode,
            )
            
            if data_source == DataSourceType.LIVE_DERIBIT:
                from src.backtest.live_deribit_data_source import LiveDeribitDataSource
                
                data_src = LiveDeribitDataSource(
                    underlying=underlying,
                    start_date=start_ts.date(),
                    end_date=end_ts.date(),
                )
            else:
                data_src = DeribitDataSource()
            
            simulator = CoveredCallSimulator(data_source=data_src, config=config)
            
            def always_trade_policy(candidates, state):
                return True
            
            result = simulator.simulate_policy(policy=always_trade_policy, size=1.0)
            
            trades = result.trades if hasattr(result, 'trades') else []
            
            metrics = result.metrics if hasattr(result, 'metrics') else {}
            
            chains_list = []
            for trade in trades:
                chain = getattr(trade, "chain", None)
                if chain:
                    chains_list.append({
                        "open_time": chain.decision_time.isoformat(),
                        "instrument_name": getattr(chain, "instrument_name", None),
                        "num_legs": len(getattr(chain, "legs", [])),
                        "num_rolls": max(0, len(getattr(chain, "legs", [])) - 1),
                        "pnl": float(chain.total_pnl),
                        "pnl_vs_hodl": float(getattr(chain, "pnl_vs_hodl", 0)),
                        "max_drawdown_pct": float(chain.max_drawdown_pct),
                    })
            
            formatted_metrics = {
                "initial_equity": metrics.get("initial_equity", 0),
                "final_equity": metrics.get("final_equity", 0),
                "net_profit_usd": metrics.get("final_pnl", 0),
                "net_profit_pct": metrics.get("total_return_pct", 0),
                "max_drawdown_pct": metrics.get("max_drawdown_pct", 0),
                "num_trades": metrics.get("num_trades", 0),
                "win_rate": metrics.get("win_rate", 0) * 100 if metrics.get("win_rate") else 0,
                "sharpe_ratio": metrics.get("sharpe_ratio", 0),
                "sortino_ratio": metrics.get("sortino_ratio", 0),
                "profit_factor": metrics.get("profit_factor", 0),
                "final_pnl_vs_hodl": metrics.get("total_pnl_vs_hodl", 0),
            }
            
            metrics_by_style = {exit_style: formatted_metrics}
            chains_by_style = {exit_style: chains_list}
            
            complete_run(
                db=db,
                run=run,
                metrics_by_style=metrics_by_style,
                chains_by_style=chains_by_style,
                primary_exit_style=exit_style,
            )
            
            if hasattr(data_src, 'close'):
                data_src.close()
            
            print(f"  Completed run: {run.run_id}")
            return run.run_id
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            if 'run' in locals():
                fail_run(db, run, str(e))
            raise


def get_metrics_for_run(run_id: str, exit_style: str) -> Optional[Dict[str, Any]]:
    """Get metrics for a completed run."""
    with get_db_session() as db:
        result = get_run_with_details(db, run_id)
        if not result:
            return None
        
        metrics = result.get("metrics", {})
        return metrics.get(exit_style, {})


def print_comparison(
    underlying: str,
    start_date: str,
    end_date: str,
    exit_style: str,
    synth_run_id: str,
    live_run_id: str,
    synth_metrics: Dict[str, Any],
    live_metrics: Dict[str, Any],
) -> None:
    """Print side-by-side comparison of metrics."""
    print("\n" + "=" * 70)
    print("BACKTEST COMPARISON: SYNTHETIC vs LIVE_DERIBIT")
    print("=" * 70)
    print(f"Underlying:    {underlying}")
    print(f"Period:        {start_date} -> {end_date}")
    print(f"Exit style:    {exit_style}")
    print("-" * 70)
    
    print(f"\nSYNTHETIC (run_id: {synth_run_id})")
    print("-" * 35)
    _print_metrics(synth_metrics)
    
    print(f"\nLIVE_DERIBIT (run_id: {live_run_id})")
    print("-" * 35)
    _print_metrics(live_metrics)
    
    print("\n" + "=" * 70)
    print("SIDE-BY-SIDE")
    print("=" * 70)
    print(f"{'Metric':<25} {'SYNTHETIC':>15} {'LIVE_DERIBIT':>15} {'Diff':>12}")
    print("-" * 70)
    
    compare_keys = [
        ("net_profit_pct", "Net Profit %"),
        ("max_drawdown_pct", "Max Drawdown %"),
        ("sharpe_ratio", "Sharpe Ratio"),
        ("sortino_ratio", "Sortino Ratio"),
        ("num_trades", "Num Trades"),
        ("win_rate", "Win Rate %"),
        ("profit_factor", "Profit Factor"),
        ("final_pnl_vs_hodl", "PnL vs HODL"),
    ]
    
    for key, label in compare_keys:
        synth_val = synth_metrics.get(key, 0) or 0
        live_val = live_metrics.get(key, 0) or 0
        
        if isinstance(synth_val, (int, float)) and isinstance(live_val, (int, float)):
            diff = live_val - synth_val
            diff_str = f"{diff:+.2f}" if abs(diff) > 0.001 else "0.00"
        else:
            diff_str = "N/A"
        
        synth_str = f"{synth_val:.2f}" if isinstance(synth_val, float) else str(synth_val)
        live_str = f"{live_val:.2f}" if isinstance(live_val, float) else str(live_val)
        
        print(f"{label:<25} {synth_str:>15} {live_str:>15} {diff_str:>12}")
    
    print("=" * 70)


def _print_metrics(metrics: Dict[str, Any]) -> None:
    """Print metrics in a readable format."""
    print(f"  Net Profit %:      {metrics.get('net_profit_pct', 'N/A')}")
    print(f"  Max Drawdown %:    {metrics.get('max_drawdown_pct', 'N/A')}")
    print(f"  Sharpe Ratio:      {metrics.get('sharpe_ratio', 'N/A')}")
    print(f"  Sortino Ratio:     {metrics.get('sortino_ratio', 'N/A')}")
    print(f"  Num Trades:        {metrics.get('num_trades', 'N/A')}")
    print(f"  Win Rate %:        {metrics.get('win_rate', 'N/A')}")
    print(f"  Profit Factor:     {metrics.get('profit_factor', 'N/A')}")
    print(f"  PnL vs HODL:       {metrics.get('final_pnl_vs_hodl', 'N/A')}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare SYNTHETIC vs LIVE_DERIBIT backtests"
    )
    parser.add_argument(
        "--underlying",
        default="BTC",
        help="Underlying asset (default: BTC)",
    )
    parser.add_argument(
        "--start",
        required=True,
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        required=True,
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--decision-interval-minutes",
        type=int,
        default=1440,
        help="Decision interval in minutes (default: 1440 = daily)",
    )
    parser.add_argument(
        "--exit-style",
        default="tp_and_roll",
        choices=["hold_to_expiry", "tp_and_roll"],
        help="Exit style (default: tp_and_roll)",
    )
    
    args = parser.parse_args()
    
    try:
        start_date = datetime.strptime(args.start, "%Y-%m-%d")
        end_date = datetime.strptime(args.end, "%Y-%m-%d")
    except ValueError as e:
        print(f"ERROR: Invalid date format: {e}")
        print("Use YYYY-MM-DD format.")
        sys.exit(1)
    
    if start_date > end_date:
        print("ERROR: Start date must be before or equal to end date.")
        sys.exit(1)
    
    start_ts = start_date.replace(hour=0, minute=0, second=0, tzinfo=timezone.utc)
    end_ts = end_date.replace(hour=23, minute=59, second=59, tzinfo=timezone.utc)
    
    underlying = args.underlying.upper()
    exit_style = args.exit_style
    decision_interval = args.decision_interval_minutes
    
    print(f"\n{'='*70}")
    print("RUNNING COMPARISON BACKTESTS")
    print(f"{'='*70}")
    print(f"Underlying:      {underlying}")
    print(f"Period:          {args.start} to {args.end}")
    print(f"Decision Int:    {decision_interval} minutes")
    print(f"Exit Style:      {exit_style}")
    print(f"{'='*70}\n")
    
    print("Running SYNTHETIC backtest...")
    try:
        synth_run_id = run_backtest_with_data_source(
            underlying=underlying,
            start_ts=start_ts,
            end_ts=end_ts,
            data_source=DataSourceType.SYNTHETIC,
            decision_interval_minutes=decision_interval,
            exit_style=exit_style,
        )
    except Exception as e:
        print(f"ERROR: SYNTHETIC backtest failed: {e}")
        sys.exit(1)
    
    print("\nRunning LIVE_DERIBIT backtest...")
    try:
        live_run_id = run_backtest_with_data_source(
            underlying=underlying,
            start_ts=start_ts,
            end_ts=end_ts,
            data_source=DataSourceType.LIVE_DERIBIT,
            decision_interval_minutes=decision_interval,
            exit_style=exit_style,
        )
    except Exception as e:
        print(f"ERROR: LIVE_DERIBIT backtest failed: {e}")
        print("Make sure you have harvested data for this date range.")
        sys.exit(1)
    
    synth_metrics = get_metrics_for_run(synth_run_id, exit_style)
    live_metrics = get_metrics_for_run(live_run_id, exit_style)
    
    if not synth_metrics:
        print(f"ERROR: Could not retrieve SYNTHETIC metrics for {synth_run_id}")
        sys.exit(1)
    
    if not live_metrics:
        print(f"ERROR: Could not retrieve LIVE_DERIBIT metrics for {live_run_id}")
        sys.exit(1)
    
    print_comparison(
        underlying=underlying,
        start_date=args.start,
        end_date=args.end,
        exit_style=exit_style,
        synth_run_id=synth_run_id,
        live_run_id=live_run_id,
        synth_metrics=synth_metrics,
        live_metrics=live_metrics,
    )
    
    print("\nComparison complete!")
    print(f"SYNTHETIC run_id:     {synth_run_id}")
    print(f"LIVE_DERIBIT run_id:  {live_run_id}")
    print("\nYou can view these runs in the Backtesting Lab UI or via the API.")


if __name__ == "__main__":
    main()
