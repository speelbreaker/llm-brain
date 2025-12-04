#!/usr/bin/env python3
"""
Example script demonstrating how to use the backtesting framework.

Run with:
    python -m src.backtest.backtest_example

This script shows how to:
1. Simulate a single "what if I sold this call?" trade
2. Run a policy-based backtest across multiple decision times
3. Generate training data for ML/RL
"""
from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta

from .deribit_client import DeribitPublicClient
from .deribit_data_source import DeribitDataSource
from .covered_call_simulator import CoveredCallSimulator, always_trade_policy
from .types import CallSimulationConfig
from .training_dataset import (
    generate_training_data,
    export_to_jsonl,
    compute_dataset_stats,
)


def example_single_trade():
    """
    Example 1: Simulate a single trade.
    'What if I sold a 7DTE ~0.25delta BTC call right now?'
    """
    print("\n" + "=" * 60)
    print("Example 1: Single Trade Simulation")
    print("=" * 60)

    client = DeribitPublicClient()
    ds = DeribitDataSource(client)

    now = datetime.now(timezone.utc)

    config = CallSimulationConfig(
        underlying="BTC",
        start=now - timedelta(days=14),
        end=now,
        timeframe="1h",
        decision_interval_bars=24,
        initial_spot_position=1.0,
        contract_size=1.0,
        fee_rate=0.0005,
        target_dte=7,
        dte_tolerance=2,
        target_delta=0.25,
        delta_tolerance=0.05,
    )

    sim = CoveredCallSimulator(ds, config)

    decision_time = now - timedelta(days=7)
    trade = sim.simulate_single_call(decision_time=decision_time, size=0.1)

    if trade:
        print(f"\nSimulated trade: {trade.instrument_name}")
        print(f"  Open:  {trade.open_time.isoformat()} at {trade.open_price:.4f}")
        print(f"  Close: {trade.close_time.isoformat()} at {trade.close_price:.4f}")
        print(f"  Size:  {trade.size} BTC")
        print(f"  PnL:   ${trade.pnl:,.2f}")
        print(f"  PnL vs HODL: ${trade.pnl_vs_hodl:,.2f}")
        print(f"  Max Drawdown: {trade.max_drawdown_pct:.2f}%")
        print(f"  Notes: {trade.notes}")
    else:
        print("\nNo suitable option found for simulation.")

    ds.close()


def example_policy_backtest():
    """
    Example 2: Run a policy-based backtest.
    Simulate selling a 7DTE ~0.25delta call every day for 30 days.
    """
    print("\n" + "=" * 60)
    print("Example 2: Policy Backtest")
    print("=" * 60)

    client = DeribitPublicClient()
    ds = DeribitDataSource(client)

    now = datetime.now(timezone.utc)

    config = CallSimulationConfig(
        underlying="BTC",
        start=now - timedelta(days=30),
        end=now,
        timeframe="1h",
        decision_interval_bars=24,
        initial_spot_position=1.0,
        contract_size=1.0,
        fee_rate=0.0005,
        target_dte=7,
        dte_tolerance=2,
        target_delta=0.25,
        delta_tolerance=0.05,
    )

    sim = CoveredCallSimulator(ds, config)

    result = sim.simulate_policy(always_trade_policy, size=0.1)

    print(f"\nBacktest Results:")
    print(f"  Total Trades: {result.metrics['num_trades']}")
    print(f"  Final PnL: ${result.metrics['final_pnl']:,.2f}")
    print(f"  Avg PnL per Trade: ${result.metrics.get('avg_pnl', 0):,.2f}")
    print(f"  Avg PnL vs HODL: ${result.metrics.get('avg_pnl_vs_hodl', 0):,.2f}")
    print(f"  Max Drawdown: {result.metrics['max_drawdown_pct']:.2f}%")
    print(f"  Win Rate: {result.metrics.get('win_rate', 0) * 100:.1f}%")

    if result.trades:
        print(f"\nFirst 3 trades:")
        for t in result.trades[:3]:
            print(f"  - {t.instrument_name}: PnL=${t.pnl:,.2f}, vs HODL=${t.pnl_vs_hodl:,.2f}")

    ds.close()


def example_training_data():
    """
    Example 3: Generate training data for ML/RL.
    """
    print("\n" + "=" * 60)
    print("Example 3: Training Data Generation")
    print("=" * 60)

    client = DeribitPublicClient()
    ds = DeribitDataSource(client)

    now = datetime.now(timezone.utc)

    config = CallSimulationConfig(
        underlying="BTC",
        start=now - timedelta(days=14),
        end=now,
        timeframe="4h",
        decision_interval_bars=6,
        initial_spot_position=0.1,
        contract_size=1.0,
        fee_rate=0.0005,
        target_dte=7,
        dte_tolerance=2,
        target_delta=0.25,
        delta_tolerance=0.05,
    )

    sim = CoveredCallSimulator(ds, config)

    examples = generate_training_data(sim, policy=always_trade_policy)

    stats = compute_dataset_stats(examples)
    print(f"\nDataset Statistics:")
    print(json.dumps(stats, indent=2))

    if examples:
        print(f"\nFirst 3 examples:")
        for ex in examples[:3]:
            print(f"  - {ex.decision_time.isoformat()}: "
                  f"action={ex.action}, reward=${ex.reward:,.2f}")

    ds.close()


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("Covered Call Backtesting Framework Examples")
    print("=" * 60)

    try:
        example_single_trade()
    except Exception as e:
        print(f"Example 1 failed: {e}")

    try:
        example_policy_backtest()
    except Exception as e:
        print(f"Example 2 failed: {e}")

    try:
        example_training_data()
    except Exception as e:
        print(f"Example 3 failed: {e}")


if __name__ == "__main__":
    main()
