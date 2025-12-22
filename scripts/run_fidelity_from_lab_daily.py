#!/usr/bin/env python3
"""Daily automation helper for Lab-based Synthetic Fidelity.

This is intended for local cron / operator runs (not CI), since it relies on
harvested live_deribit data being present.

Default behavior:
- Runs BTC and ETH for the last 3 full days ending yesterday.
- Writes reports to the fidelity store (data/fidelity_runs by default).

Usage:
  PYTHONPATH=. ./.venv/bin/python scripts/run_fidelity_from_lab_daily.py
  PYTHONPATH=. ./.venv/bin/python scripts/run_fidelity_from_lab_daily.py --days 7
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone

from src.backtest.fidelity_store import write_fidelity_report
from src.backtest.fidelity_suite import run_fidelity_from_lab


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=3, help="Number of full days to include (ending yesterday)")
    p.add_argument("--min-trades-per-case", type=int, default=5)
    p.add_argument("--decision-interval-minutes", type=int, default=60)
    args = p.parse_args()

    if args.days <= 0:
        raise SystemExit("--days must be >= 1")

    today = datetime.now(timezone.utc).date()
    end_date = today - timedelta(days=1)
    start_date = end_date - timedelta(days=int(args.days) - 1)

    start_ts = datetime(start_date.year, start_date.month, start_date.day, 0, 0, 0, tzinfo=timezone.utc)
    end_ts = datetime(end_date.year, end_date.month, end_date.day, 23, 59, 59, tzinfo=timezone.utc)

    for underlying in ("BTC", "ETH"):
        print(f"Running lab fidelity: {underlying} {start_date}..{end_date}")
        report = run_fidelity_from_lab(
            underlying=underlying,
            start_ts=start_ts,
            end_ts=end_ts,
            decision_interval_minutes=int(args.decision_interval_minutes),
            min_trades_per_case=int(args.min_trades_per_case),
        )
        summary = write_fidelity_report(report)
        print(f"  -> run_id={summary.get('run_id')} gate={summary.get('gate_label')} score={summary.get('overall_score')}")


if __name__ == "__main__":
    main()
