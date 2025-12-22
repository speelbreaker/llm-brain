from __future__ import annotations

from src.backtest.fidelity_suite import score_case_from_diff


def _make_diff(net_profit_pp: float) -> dict:
    # Mimic the shape returned by src/backtest/diff.compute_diff_for_runs
    return {
        "metrics": {
            "net_profit_pct": {"a": 0.0, "b": 0.0, "diff": net_profit_pp, "fmt_type": "pct"},
            "max_drawdown_pct": {"a": 0.0, "b": 0.0, "diff": 0.0, "fmt_type": "pct"},
            "win_rate": {"a": 0.0, "b": 0.0, "diff": 0.0, "fmt_type": "pct"},
            "profit_factor": {"a": 1.0, "b": 1.0, "diff": 0.0, "fmt_type": "float"},
            "avg_trade_usd": {"a": 0.0, "b": 0.0, "diff": 0.0, "fmt_type": "usd"},
        }
    }


def test_score_decreases_as_diffs_increase() -> None:
    score_small, _ = score_case_from_diff(_make_diff(net_profit_pp=1.0))
    score_big, _ = score_case_from_diff(_make_diff(net_profit_pp=20.0))

    assert score_small > score_big
    assert 0.0 <= score_big <= 100.0
    assert 0.0 <= score_small <= 100.0
