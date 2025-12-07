#!/usr/bin/env python3
"""
Diff report for comparing two backtest runs stored in PostgreSQL.

Usage:
    python scripts/diff_backtest_runs.py \
        --run-a 2025-12-07T23-17-31Z_BTC_414864df \
        --run-b 2025-12-07T23-17-31Z_BTC_f3f6156a \
        --exit-style tp_and_roll
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.db import get_db_session
from src.db.models_backtest import BacktestRun, BacktestMetric


METRICS_FIELDS = [
    ("net_profit_pct", "pct"),
    ("net_profit_usd", "usd"),
    ("final_pnl_vs_hodl", "usd"),
    ("max_drawdown_pct", "pct"),
    ("max_drawdown_usd", "usd"),
    ("num_trades", "int"),
    ("win_rate", "pct"),
    ("profit_factor", "float"),
    ("avg_trade_usd", "usd"),
    ("sharpe_ratio", "float"),
    ("sortino_ratio", "float"),
]


def fetch_run(db, run_id: str) -> Optional[BacktestRun]:
    """Fetch a backtest run by its run_id string."""
    return db.query(BacktestRun).filter(BacktestRun.run_id == run_id).first()


def fetch_metrics(db, run_numeric_id: int, exit_style: str) -> Optional[BacktestMetric]:
    """Fetch metrics for a run by numeric ID and exit style."""
    return db.query(BacktestMetric).filter(
        BacktestMetric.run_id == run_numeric_id,
        BacktestMetric.exit_style == exit_style,
    ).first()


def get_metric_value(metrics: BacktestMetric, field: str) -> float:
    """Extract a metric value, returning 0.0 if not found."""
    if hasattr(metrics, field):
        val = getattr(metrics, field)
        return float(val) if val is not None else 0.0
    if hasattr(metrics, 'metrics_json') and metrics.metrics_json:
        return float(metrics.metrics_json.get(field, 0.0))
    return 0.0


def format_value(val: float, fmt_type: str) -> str:
    """Format a value based on its type."""
    if fmt_type == "pct":
        return f"{val:.1f} %"
    elif fmt_type == "usd":
        return f"{val:,.2f}"
    elif fmt_type == "int":
        return f"{int(val)}"
    else:
        return f"{val:.2f}"


def format_diff(diff: float, fmt_type: str) -> str:
    """Format a diff value with +/- sign."""
    sign = "+" if diff > 0 else ""
    if fmt_type == "pct":
        return f"{sign}{diff:.1f} pp"
    elif fmt_type == "usd":
        return f"{sign}{diff:,.2f}"
    elif fmt_type == "int":
        return f"{sign}{int(diff)}"
    else:
        return f"{sign}{diff:.2f}"


def print_diff_report(
    run_a: BacktestRun,
    run_b: BacktestRun,
    metrics_a: BacktestMetric,
    metrics_b: BacktestMetric,
    exit_style: str,
):
    """Print the formatted diff report."""
    print()
    print("=" * 80)
    print("BACKTEST DIFF REPORT")
    print("=" * 80)
    print()
    print("Comparing runs:")
    print(f"  A: {run_a.run_id}  (data_source={run_a.data_source})")
    print(f"  B: {run_b.run_id}  (data_source={run_b.data_source})")
    print(f"  Underlying: {run_a.underlying}")
    print(f"  Exit style: {exit_style}")
    print(f"  Period A: {run_a.start_ts} -> {run_a.end_ts}")
    print(f"  Period B: {run_b.start_ts} -> {run_b.end_ts}")
    print(f"  Decision interval: {run_a.decision_interval_minutes} minutes")
    print()
    
    col_metric = 24
    col_val = 20
    
    header = f"{'Metric':<{col_metric}} {'A (' + run_a.data_source + ')':<{col_val}} {'B (' + run_b.data_source + ')':<{col_val}} {'Diff (B - A)':<{col_val}}"
    print(header)
    print("-" * len(header))
    
    for field, fmt_type in METRICS_FIELDS:
        val_a = get_metric_value(metrics_a, field)
        val_b = get_metric_value(metrics_b, field)
        diff = val_b - val_a
        
        str_a = format_value(val_a, fmt_type)
        str_b = format_value(val_b, fmt_type)
        str_diff = format_diff(diff, fmt_type)
        
        print(f"{field:<{col_metric}} {str_a:<{col_val}} {str_b:<{col_val}} {str_diff:<{col_val}}")
    
    print("=" * 80)
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Compare two backtest runs and show a diff report."
    )
    parser.add_argument(
        "--run-a",
        required=True,
        help="Run ID of the first backtest (A)",
    )
    parser.add_argument(
        "--run-b",
        required=True,
        help="Run ID of the second backtest (B)",
    )
    parser.add_argument(
        "--exit-style",
        default=None,
        help="Exit style to compare (default: use primary_exit_style from runs)",
    )
    
    args = parser.parse_args()
    
    with get_db_session() as db:
        run_a = fetch_run(db, args.run_a)
        if not run_a:
            print(f"ERROR: Run A not found: {args.run_a}", file=sys.stderr)
            sys.exit(1)
        
        run_b = fetch_run(db, args.run_b)
        if not run_b:
            print(f"ERROR: Run B not found: {args.run_b}", file=sys.stderr)
            sys.exit(1)
        
        if args.exit_style:
            exit_style = args.exit_style
        else:
            exit_style_a = run_a.primary_exit_style
            exit_style_b = run_b.primary_exit_style
            
            if exit_style_a != exit_style_b:
                print(
                    f"ERROR: Runs have different primary exit styles "
                    f"(A={exit_style_a}, B={exit_style_b}). "
                    f"Please specify --exit-style explicitly.",
                    file=sys.stderr,
                )
                sys.exit(1)
            
            exit_style = exit_style_a
        
        metrics_a = fetch_metrics(db, run_a.id, exit_style)
        if not metrics_a:
            print(
                f"ERROR: Metrics not found for run A ({args.run_a}) "
                f"with exit_style={exit_style}",
                file=sys.stderr,
            )
            sys.exit(1)
        
        metrics_b = fetch_metrics(db, run_b.id, exit_style)
        if not metrics_b:
            print(
                f"ERROR: Metrics not found for run B ({args.run_b}) "
                f"with exit_style={exit_style}",
                file=sys.stderr,
            )
            sys.exit(1)
        
        print_diff_report(run_a, run_b, metrics_a, metrics_b, exit_style)
    
    sys.exit(0)


if __name__ == "__main__":
    main()
