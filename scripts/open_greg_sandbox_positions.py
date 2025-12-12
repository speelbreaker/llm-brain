#!/usr/bin/env python
"""
Open sandbox Greg positions on Deribit testnet/DRY_RUN.

Creates 14 sandbox positions (7 strategies x 2 underlyings) for testing
the Greg dashboard, position management, and hedge engine.

Usage:
    python scripts/open_greg_sandbox_positions.py
    python scripts/open_greg_sandbox_positions.py --underlyings BTC,ETH
    python scripts/open_greg_sandbox_positions.py --btc-size 0.01 --eth-size 0.1
    python scripts/open_greg_sandbox_positions.py --clear
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone


sys.path.insert(0, ".")

from src.config import settings
from src.position_tracker import PositionTracker
from src.greg_sandbox_entry import (
    GREG_STRATEGIES,
    open_greg_strategy_position,
    get_sandbox_positions,
    clear_sandbox_positions,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Open sandbox Greg positions on Deribit testnet"
    )
    parser.add_argument(
        "--underlyings",
        default="BTC,ETH",
        help="Comma-separated list of underlyings (default: BTC,ETH)",
    )
    parser.add_argument(
        "--btc-size",
        type=float,
        default=0.01,
        help="Position size for BTC (default: 0.01)",
    )
    parser.add_argument(
        "--eth-size",
        type=float,
        default=0.1,
        help="Position size for ETH (default: 0.1)",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing sandbox positions before creating new ones",
    )
    parser.add_argument(
        "--clear-only",
        action="store_true",
        help="Only clear sandbox positions, don't create new ones",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_positions",
        help="List existing sandbox positions",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Greg Sandbox Position Opener")
    print("=" * 60)
    print()

    if not settings.dry_run and settings.deribit_env != "testnet":
        print("ERROR: Refusing to run sandbox opener on non-testnet / non-DRY_RUN environment.")
        print()
        print(f"  Current environment: {settings.deribit_env}")
        print(f"  Dry run mode: {settings.dry_run}")
        print()
        print("Set DRY_RUN=true or use testnet to proceed.")
        return 1

    print(f"Environment: {settings.deribit_env}")
    print(f"Dry Run: {settings.dry_run}")
    print()

    tracker = PositionTracker()

    if args.list_positions:
        sandbox = get_sandbox_positions(tracker)
        if not sandbox:
            print("No sandbox positions found.")
            return 0
        
        print(f"Found {len(sandbox)} sandbox positions:")
        print()
        for pos in sandbox:
            status = "OPEN" if pos.is_open() else "CLOSED"
            print(f"  [{status}] {pos.position_id}")
            print(f"         underlying={pos.underlying} strategy={pos.strategy_type}")
            print(f"         legs={pos.num_legs} origin={pos.origin} run_id={pos.run_id}")
            print()
        return 0

    if args.clear or args.clear_only:
        cleared = clear_sandbox_positions(tracker)
        print(f"Cleared {cleared} existing sandbox positions.")
        print()
        if args.clear_only:
            return 0

    run_id = f"GREG_SANDBOX_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    underlyings = [u.strip().upper() for u in args.underlyings.split(",") if u.strip()]

    print(f"Run ID: {run_id}")
    print(f"Underlyings: {', '.join(underlyings)}")
    print(f"BTC size: {args.btc_size}")
    print(f"ETH size: {args.eth_size}")
    print(f"Strategies: {len(GREG_STRATEGIES)}")
    print()
    print("-" * 60)

    success_count = 0
    fail_count = 0

    for underlying in underlyings:
        size = args.btc_size if underlying == "BTC" else args.eth_size
        
        for strategy_type in GREG_STRATEGIES:
            print(f"  -> {underlying} / {strategy_type} / size={size}")
            
            result = open_greg_strategy_position(
                tracker=tracker,
                underlying=underlying,
                strategy_type=strategy_type,
                size=size,
                sandbox=True,
                origin="GREG_SANDBOX",
                run_id=run_id,
            )
            
            if result.success:
                print(f"     OK position_id={result.position_id} legs={result.num_legs}")
                success_count += 1
            else:
                print(f"     FAILED: {result.error}")
                fail_count += 1

    print()
    print("-" * 60)
    print(f"Summary: {success_count} succeeded, {fail_count} failed")
    print()
    
    if fail_count > 0:
        print("Some positions failed to create. Check the errors above.")
        return 1
    
    print(f"[GREG_SANDBOX] Done. Created {success_count} sandbox positions.")
    print()
    print("These positions can now be viewed in the Greg dashboard and tested")
    print("with the position management and hedge engine.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
