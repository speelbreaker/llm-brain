#!/usr/bin/env python3
"""Sabotage drill for Synthetic Fidelity scoring.

This does NOT run real backtests.
It creates synthetic diff payloads and demonstrates that the scoring function
moves in the expected direction as diffs get worse.

Usage:
  PYTHONPATH=. ./.venv/bin/python scripts/sabotage_fidelity_drill.py
"""

from __future__ import annotations

from src.backtest.fidelity_suite import score_case_from_diff


def _payload(net_profit_pp: float, dd_pp: float) -> dict:
    return {
        "metrics": {
            "net_profit_pct": {"diff": net_profit_pp},
            "max_drawdown_pct": {"diff": dd_pp},
            "win_rate": {"diff": 0.0},
            "profit_factor": {"diff": 0.0},
            "avg_trade_usd": {"diff": 0.0},
        }
    }


def main() -> None:
    good, _ = score_case_from_diff(_payload(net_profit_pp=1.0, dd_pp=1.0))
    bad, _ = score_case_from_diff(_payload(net_profit_pp=20.0, dd_pp=20.0))

    print("Sabotage drill")
    print(f"  score(good diffs)={good:.2f}")
    print(f"  score(bad diffs) ={bad:.2f}")

    if good <= bad:
        raise SystemExit("Expected sabotage to lower the score")


if __name__ == "__main__":
    main()
