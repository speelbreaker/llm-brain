#!/usr/bin/env python3
"""
Strategy Health Check - Automated battery of synthetic vs live_deribit comparisons.

Runs a standard set of backtests comparing synthetic pricing against captured
Deribit data and prints a compact health report.

Usage:
    python scripts/strategy_health_check.py --days 7 --exit-style tp_and_roll
    python scripts/strategy_health_check.py --days 14 --underlyings BTC ETH
"""

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backtest.compare import run_synthetic_vs_live_pair
from src.backtest.diff import compute_diff_for_runs


@dataclass
class TestConfig:
    """Configuration for a single test in the battery."""
    underlying: str
    decision_interval_minutes: int
    label: str


STANDARD_TESTS = [
    TestConfig("BTC", 1440, "BTC daily decisions"),
    TestConfig("BTC", 240, "BTC 4h decisions"),
    TestConfig("ETH", 1440, "ETH daily decisions"),
]


@dataclass
class TestResult:
    """Result of a single health check test."""
    config: TestConfig
    status: str
    synth_run_id: Optional[str] = None
    live_run_id: Optional[str] = None
    diff_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    
    @property
    def optimism_factor(self) -> Optional[float]:
        """Calculate optimism factor (synthetic / live net profit %)."""
        if not self.diff_data:
            return None
        
        metrics = self.diff_data.get("metrics", {})
        net_profit = metrics.get("net_profit_pct", {})
        synth_val = net_profit.get("a", 0)
        live_val = net_profit.get("b", 0)
        
        if abs(live_val) > 1e-9:
            return synth_val / live_val
        return None


def run_health_check_test(
    test: TestConfig,
    start_ts: datetime,
    end_ts: datetime,
    exit_style: str,
) -> TestResult:
    """
    Run a single health check test.
    
    Returns TestResult with status "success", "skipped", or "failed".
    """
    try:
        synth_run_id, live_run_id = run_synthetic_vs_live_pair(
            underlying=test.underlying,
            start_ts=start_ts,
            end_ts=end_ts,
            decision_interval_minutes=test.decision_interval_minutes,
            exit_style=exit_style,
            verbose=False,
        )
        
        diff_data = compute_diff_for_runs(
            run_id_a=synth_run_id,
            run_id_b=live_run_id,
            exit_style=exit_style,
        )
        
        return TestResult(
            config=test,
            status="success",
            synth_run_id=synth_run_id,
            live_run_id=live_run_id,
            diff_data=diff_data,
        )
        
    except Exception as e:
        error_msg = str(e)
        
        if "no data" in error_msg.lower() or "no files" in error_msg.lower():
            return TestResult(
                config=test,
                status="skipped",
                error_message=f"Not enough live_deribit data: {error_msg}",
            )
        
        return TestResult(
            config=test,
            status="failed",
            error_message=error_msg,
        )


def print_test_result(
    result: TestResult,
    start_date: str,
    end_date: str,
    exit_style: str,
) -> None:
    """Print formatted output for a single test result."""
    print()
    print("=" * 80)
    print(f"STRATEGY HEALTH CHECK - {result.config.underlying}, decisions={result.config.decision_interval_minutes} min")
    
    if result.status == "skipped":
        print(f"SKIPPED: {result.error_message}")
        print("-" * 80)
        return
    
    if result.status == "failed":
        print(f"FAILED: {result.error_message}")
        print("-" * 80)
        return
    
    print(f"Exit style: {exit_style}")
    print(f"Period: {start_date} -> {end_date}")
    print("Data source A: synthetic")
    print("Data source B: live_deribit")
    print("-" * 80)
    
    metrics = result.diff_data.get("metrics", {})
    
    net_profit = metrics.get("net_profit_pct", {})
    synth_pct = net_profit.get("a", 0)
    live_pct = net_profit.get("b", 0)
    diff_pct = net_profit.get("diff", 0)
    optimism = result.optimism_factor
    
    print("net_profit_pct:")
    print(f"  synthetic:    {synth_pct:+.1f} %")
    print(f"  live_deribit: {live_pct:+.1f} %")
    print(f"  diff:         {diff_pct:+.1f} pp")
    if optimism is not None:
        print(f"  optimism:     {optimism:.2f}x")
    else:
        print("  optimism:     N/A (live_deribit = 0)")
    
    num_trades = metrics.get("num_trades", {})
    print("num_trades:")
    print(f"  synthetic:    {int(num_trades.get('a', 0))}")
    print(f"  live_deribit: {int(num_trades.get('b', 0))}")
    
    win_rate = metrics.get("win_rate", {})
    print("win_rate:")
    print(f"  synthetic:    {win_rate.get('a', 0):.1f} %")
    print(f"  live_deribit: {win_rate.get('b', 0):.1f} %")
    
    max_dd = metrics.get("max_drawdown_pct", {})
    print("max_drawdown_pct:")
    print(f"  synthetic:    {max_dd.get('a', 0):.1f} %")
    print(f"  live_deribit: {max_dd.get('b', 0):.1f} %")
    
    profit_factor = metrics.get("profit_factor", {})
    print("profit_factor:")
    print(f"  synthetic:    {profit_factor.get('a', 0):.2f}")
    print(f"  live_deribit: {profit_factor.get('b', 0):.2f}")
    
    sharpe = metrics.get("sharpe_ratio", {})
    print("sharpe_ratio:")
    print(f"  synthetic:    {sharpe.get('a', 0):.2f}")
    print(f"  live_deribit: {sharpe.get('b', 0):.2f}")
    
    print("-" * 80)


def print_summary(results: List[TestResult], days: int) -> None:
    """Print final summary of all tests."""
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    for result in results:
        label = result.config.label
        
        if result.status == "success":
            optimism = result.optimism_factor
            if optimism is not None:
                print(f"  {label} ({days}d): optimism {optimism:.2f}x")
            else:
                print(f"  {label} ({days}d): optimism N/A (no live profit)")
        elif result.status == "skipped":
            print(f"  (SKIPPED: {label} - no live data)")
        else:
            print(f"  (FAILED: {label} - {result.error_message})")
    
    print("=" * 80)


def save_json_report(
    results: List[TestResult],
    start_date: str,
    end_date: str,
    exit_style: str,
    days: int,
) -> Optional[str]:
    """Save JSON report to data/health_checks/ directory."""
    try:
        output_dir = Path("data/health_checks")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        output_path = output_dir / f"healthcheck_{today}.json"
        
        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "period": {
                "start_date": start_date,
                "end_date": end_date,
                "days": days,
            },
            "exit_style": exit_style,
            "tests": [],
        }
        
        for result in results:
            test_data = {
                "underlying": result.config.underlying,
                "decision_interval_minutes": result.config.decision_interval_minutes,
                "label": result.config.label,
                "status": result.status,
            }
            
            if result.status == "success":
                test_data["synth_run_id"] = result.synth_run_id
                test_data["live_run_id"] = result.live_run_id
                test_data["optimism_factor"] = result.optimism_factor
                test_data["metrics"] = result.diff_data.get("metrics", {})
            elif result.error_message:
                test_data["error"] = result.error_message
            
            report["tests"].append(test_data)
        
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        
        return str(output_path)
        
    except Exception as e:
        print(f"Warning: Could not save JSON report: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Run strategy health check comparing synthetic vs live_deribit backtests"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Number of days for lookback window (default: 7)",
    )
    parser.add_argument(
        "--underlyings",
        nargs="+",
        default=["BTC", "ETH"],
        help="Underlyings to test (default: BTC ETH)",
    )
    parser.add_argument(
        "--exit-style",
        default="tp_and_roll",
        choices=["hold_to_expiry", "tp_and_roll"],
        help="Exit style to use (default: tp_and_roll)",
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Save JSON report to data/health_checks/",
    )
    
    args = parser.parse_args()
    
    underlyings = [u.upper() for u in args.underlyings]
    exit_style = args.exit_style
    days = args.days
    
    today = datetime.now(timezone.utc).date()
    end_date = today - timedelta(days=1)
    start_date = end_date - timedelta(days=days - 1)
    
    start_ts = datetime.combine(start_date, datetime.min.time(), tzinfo=timezone.utc)
    end_ts = datetime.combine(end_date, datetime.max.time().replace(microsecond=0), tzinfo=timezone.utc)
    
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")
    
    tests_to_run = [t for t in STANDARD_TESTS if t.underlying in underlyings]
    
    print()
    print("=" * 80)
    print("STRATEGY HEALTH CHECK")
    print("=" * 80)
    print(f"Period: {start_date_str} -> {end_date_str} ({days} days)")
    print(f"Exit style: {exit_style}")
    print(f"Underlyings: {', '.join(underlyings)}")
    print(f"Tests to run: {len(tests_to_run)}")
    print("=" * 80)
    
    results: List[TestResult] = []
    
    for i, test in enumerate(tests_to_run, 1):
        print(f"\n[{i}/{len(tests_to_run)}] Running {test.label}...")
        
        result = run_health_check_test(
            test=test,
            start_ts=start_ts,
            end_ts=end_ts,
            exit_style=exit_style,
        )
        results.append(result)
        
        print_test_result(result, start_date_str, end_date_str, exit_style)
    
    print_summary(results, days)
    
    if args.save_json:
        json_path = save_json_report(
            results=results,
            start_date=start_date_str,
            end_date=end_date_str,
            exit_style=exit_style,
            days=days,
        )
        if json_path:
            print(f"\nJSON report saved to: {json_path}")
    
    success_count = sum(1 for r in results if r.status == "success")
    skipped_count = sum(1 for r in results if r.status == "skipped")
    failed_count = sum(1 for r in results if r.status == "failed")
    
    print(f"\nResults: {success_count} success, {skipped_count} skipped, {failed_count} failed")
    
    if success_count > 0 or skipped_count > 0:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
