from __future__ import annotations

from src.fidelity.ops_runner import _apply_strategy_parity_coverage_penalty, _weighted_overall_score


def test_ops_runner_penalizes_strategy_parity_when_coverage_090() -> None:
    component_scores = {
        "underlying_returns": 80.0,
        "iv_surface_level": 80.0,
        "spot_iv_coupling": 80.0,
        "strategy_pnl_parity": 100.0,
    }
    weights = {
        "underlying_returns": 0.25,
        "iv_surface_level": 0.25,
        "spot_iv_coupling": 0.25,
        "strategy_pnl_parity": 0.25,
    }

    overall_before = _weighted_overall_score(component_scores, weights=weights)

    penalty = _apply_strategy_parity_coverage_penalty(
        component_scores,
        parity_coverage={
            "coverage_ratio_cases": 0.90,
            "invalid_trades_missing_quote": 0,
            "invalid_trades_missing_close": 0,
        },
        penalty_threshold=0.95,
    )

    assert penalty["applied"] is True
    assert component_scores["strategy_pnl_parity"] == 90.0

    overall_after = _weighted_overall_score(component_scores, weights=weights)
    assert overall_after < overall_before
