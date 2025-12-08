#!/usr/bin/env python3
"""
Manual position reconciliation script.

Connects to Deribit testnet, fetches exchange positions,
compares with local position tracker, and optionally auto-heals.

Usage:
    python scripts/reconcile_positions_once.py [--heal]

Options:
    --heal    Auto-heal local positions from exchange (default: dry-run report only)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings
from src.deribit_client import DeribitClient
from src.position_tracker import position_tracker
from src.reconciliation import (
    reconcile_positions,
    format_reconciliation_summary,
    DivergenceAction,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Manual position reconciliation")
    parser.add_argument(
        "--heal",
        action="store_true",
        help="Auto-heal local positions from exchange",
    )
    args = parser.parse_args()

    action: DivergenceAction = "auto_heal" if args.heal else "halt"

    print("=" * 60)
    print("Position Reconciliation Tool")
    print("=" * 60)
    print(f"Deribit Environment: {settings.deribit_env.upper()}")
    print(f"Underlyings: {', '.join(settings.underlyings)}")
    print(f"Tolerance USD: ${settings.position_reconcile_tolerance_usd:.2f}")
    print(f"Action: {'AUTO-HEAL' if args.heal else 'DRY-RUN (report only)'}")
    print("=" * 60)

    print("\nConnecting to Deribit...")
    client = DeribitClient()

    print("Fetching exchange positions...")
    try:
        positions_btc = client.get_positions("BTC")
        positions_eth = client.get_positions("ETH")
        
        exchange_positions = []
        for p in positions_btc + positions_eth:
            if abs(float(p.get("size", 0))) > 0:
                exchange_positions.append({
                    "symbol": p.get("instrument_name", ""),
                    "size": abs(float(p.get("size", 0))),
                    "direction": p.get("direction", "sell"),
                    "average_price": float(p.get("average_price", 0) or 0),
                    "mark_price": float(p.get("mark_price", 0) or 0),
                    "unrealized_pnl": float(p.get("total_profit_loss", 0) or 0),
                    "delta": float(p.get("delta", 0) or 0),
                    "underlying": "BTC" if "BTC" in p.get("instrument_name", "") else "ETH",
                    "option_type": p.get("option_type", "call"),
                })
        
        print(f"Found {len(exchange_positions)} positions on exchange")
        
    except Exception as e:
        print(f"ERROR: Failed to fetch exchange positions: {e}")
        return 1

    print("Loading local positions...")
    local_payload = position_tracker.get_open_positions_payload()
    local_positions = local_payload.get("positions", [])
    print(f"Found {len(local_positions)} positions in local tracker")

    print("\nRunning reconciliation...")
    new_positions, stats = reconcile_positions(
        exchange_positions=exchange_positions,
        local_positions=local_positions,
        action=action,
        tolerance_usd=settings.position_reconcile_tolerance_usd,
    )

    print()
    print(format_reconciliation_summary(stats))
    print()

    if stats["divergent"]:
        if args.heal:
            print("Applying auto-heal...")
            rebuilt_count = position_tracker.rebuild_from_exchange(exchange_positions)
            print(f"Rebuilt {rebuilt_count} positions from exchange.")
            print("NOTE: Chain history has been reset. PnL tracking will restart from current state.")
            
            print("\nVerifying reconciliation...")
            local_payload = position_tracker.get_open_positions_payload()
            local_positions = local_payload.get("positions", [])
            _, verify_stats = reconcile_positions(
                exchange_positions=exchange_positions,
                local_positions=local_positions,
                action="halt",
            )
            if verify_stats["divergent"]:
                print(f"WARNING: Still divergent after heal: {verify_stats}")
            else:
                print(f"SUCCESS: Now in sync ({verify_stats['exchange_count']} positions)")
        else:
            print("To auto-heal, run with --heal flag:")
            print("  python scripts/reconcile_positions_once.py --heal")
    else:
        print("No action needed - positions are in sync.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
