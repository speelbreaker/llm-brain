#!/usr/bin/env python3
"""Run the Lab-based Fidelity orchestrator and persist the report.

Usage:
  PYTHONPATH=. ./.venv/bin/python scripts/run_fidelity_from_lab.py --underlying BTC --start 2025-12-10 --end 2025-12-13
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, time, timezone

from src.backtest.fidelity_store import write_fidelity_report
from src.backtest.fidelity_suite import run_fidelity_from_lab


def _parse_date(s: str) -> date:
    return date.fromisoformat(s)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--underlying", required=True, choices=["BTC", "ETH"], help="Underlying asset")
    p.add_argument("--start", required=True, help="YYYY-MM-DD (inclusive)")
    p.add_argument("--end", required=True, help="YYYY-MM-DD (inclusive)")
    p.add_argument("--decision-interval-minutes", type=int, default=60)
    p.add_argument("--min-trades-per-case", type=int, default=5)
    args = p.parse_args()

    start_d = _parse_date(args.start)
    end_d = _parse_date(args.end)

    start_ts = datetime.combine(start_d, time.min, tzinfo=timezone.utc)
    # inclusive end date
    end_ts = datetime.combine(end_d, time.max, tzinfo=timezone.utc)

    report = run_fidelity_from_lab(
        underlying=args.underlying,
        start_ts=start_ts,
        end_ts=end_ts,
        decision_interval_minutes=int(args.decision_interval_minutes),
        min_trades_per_case=int(args.min_trades_per_case),
    )

    summary = write_fidelity_report(report)

    run_id = summary.get("run_id")
    base_dir = "data/fidelity_runs"
    print(f"run_id={run_id}")
    print(f"overall_score={summary.get('overall_score')}")
    print(f"gate_label={summary.get('gate_label')}")
    print(f"report_path={base_dir}/{run_id}/report.json")


if __name__ == "__main__":
    main()
