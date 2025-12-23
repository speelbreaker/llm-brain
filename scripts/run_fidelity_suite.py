#!/usr/bin/env python3
"""Run the Strategy PnL Parity + Synthetic Fidelity suite (MVP).

Usage:
    python -m scripts.run_fidelity_suite --underlying BTC --start 2025-12-01 --end 2025-12-05

If harvested data isn't present for the requested window, the runner falls back to
fixtures under tests/fixtures/fidelity/.

Outputs (in the fidelity runs directory):
    - <RUN_ID>/fidelity_report.json
    - <RUN_ID>/fidelity_report.md
    - latest.json
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from src.fidelity.run_suite import run_fidelity_suite_from_cli


def main() -> None:
    p = argparse.ArgumentParser(description="Run synthetic fidelity suite")
    p.add_argument("--underlying", required=True, choices=["BTC", "ETH"], help="Underlying")
    p.add_argument("--start", required=True, help="Start (YYYY-MM-DD, ISO8601, or unix ts)")
    p.add_argument("--end", required=True, help="End (YYYY-MM-DD, ISO8601, or unix ts)")
    p.add_argument(
        "--out-dir",
        default=None,
        help="Base output directory (defaults to data/fidelity_runs or $FIDELITY_RUNS_DIR)",
    )
    p.add_argument("--seed", type=int, default=123, help="Deterministic seed")
    p.add_argument("--slippage-bps", type=float, default=0.0, help="Slippage bps used for fills")
    p.add_argument(
        "--mode",
        default="quick",
        choices=["preflight", "quick", "full"],
        help="preflight=checks only, quick=fast suite, full=comprehensive",
    )

    args = p.parse_args()

    report = run_fidelity_suite_from_cli(
        start=args.start,
        end=args.end,
        underlying=args.underlying,
        seed=int(args.seed),
        out_dir=args.out_dir,
        slippage_bps=float(args.slippage_bps),
        mode=args.mode,
    )

    if report.get("mode") == "preflight":
        print(f"Preflight {report['run_id']} overall_status={report['overall_status']} -> {report['summary']}")
        for check in report.get("checks", []):
            print(f"  - {check.get('name')}: {check.get('status')} ({check.get('details')})")
        return

    base_dir = Path(args.out_dir) if args.out_dir else Path(os.getenv("FIDELITY_RUNS_DIR", "data/fidelity_runs"))
    run_dir = base_dir / str(report["run_id"])

    print(f"Wrote fidelity run {report['run_id']}")
    print(f"Wrote {run_dir / 'fidelity_report.json'}")
    print(f"Wrote {run_dir / 'fidelity_report.md'}")
    print(f"Wrote {base_dir / 'latest.json'}")


if __name__ == "__main__":
    main()
