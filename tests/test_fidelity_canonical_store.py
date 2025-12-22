from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_canonical_fidelity_store_write_and_load_latest(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    base = tmp_path / "fidelity_runs"
    monkeypatch.setenv("FIDELITY_RUNS_DIR", str(base))

    from src.backtest import fidelity_store

    report = {
        "run_id": "run_0001",
        "created_at": "2025-01-01T00:00:01+00:00",
        "underlying": "BTC",
        "overall_score": 88.5,
        "gate_label": "TRUSTED",
        "component_scores": {
            "underlying_returns": 90.0,
            "iv_surface_level": 85.0,
            "spot_iv_coupling": 80.0,
            "strategy_pnl_parity": 92.0,
        },
        "coverage": {"coverage_ratio_cases": 1.0, "valid_cases": 1, "total_cases": 1},
        "thresholds": {"trusted_threshold": 80.0, "warn_threshold": 65.0},
    }

    fidelity_store.write_fidelity_report(report, base_dir=base)

    latest_report = fidelity_store.load_latest_report(underlying="BTC", base_dir=base)
    assert latest_report is not None
    assert latest_report["run_id"] == "run_0001"
    assert latest_report["created_at"] == "2025-01-01T00:00:01+00:00"
    assert latest_report["gate_label"] == "TRUSTED"

    # latest.json is full report (copy/symlink semantics)
    on_disk = json.loads((base / "latest.json").read_text())
    assert on_disk["run_id"] == "run_0001"

    # Summary pointer stays stable for /api/fidelity/latest.
    latest_summary = fidelity_store.load_latest(base_dir=base)
    assert latest_summary is not None
    assert latest_summary["run_id"] == "run_0001"
    assert "component_scores" in latest_summary

    facts = fidelity_store.load_latest_facts(underlying="BTC", base_dir=base)
    assert facts["available"] is True
    assert facts["source"] == "lab_store"
    assert facts["path"] == str(base / "BTC" / "latest.json")
    # Types are stable JSON primitives.
    assert isinstance(facts["run_id"], str)
    assert isinstance(facts["gate_label"], str)


def test_parity_suite_fixture_mode_scores_and_enforces_min_trades() -> None:
    from datetime import datetime, timezone

    from src.backtest.fidelity_suite import ParityCaseSpec, run_strategy_pnl_parity_suite

    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    end = datetime(2025, 1, 2, tzinfo=timezone.utc)

    case = ParityCaseSpec(
        underlying="BTC",
        start_ts=start,
        end_ts=end,
        decision_interval_minutes=60,
        exit_style="tp_and_roll",
    )

    key = f"BTC|{start.isoformat()}|{end.isoformat()}|60|tp_and_roll"

    perfect_diff = {
        "exit_style": "tp_and_roll",
        "metrics": {
            "net_profit_pct": {"a": 10.0, "b": 10.0, "diff": 0.0},
            "net_profit_usd": {"a": 1000.0, "b": 1000.0, "diff": 0.0},
            "max_drawdown_pct": {"a": 5.0, "b": 5.0, "diff": 0.0},
            "max_drawdown_usd": {"a": 500.0, "b": 500.0, "diff": 0.0},
            "win_rate": {"a": 55.0, "b": 55.0, "diff": 0.0},
            "profit_factor": {"a": 1.5, "b": 1.5, "diff": 0.0},
            "avg_trade_usd": {"a": 50.0, "b": 50.0, "diff": 0.0},
            "num_trades": {"a": 5, "b": 5, "diff": 0},
        },
    }

    out = run_strategy_pnl_parity_suite(
        cases=[case],
        min_trades_per_case=5,
        fixture_diffs={key: perfect_diff},
    )

    # Coverage schema must be stable and complete.
    cov = out["coverage"]
    for k in [
        "coverage_ratio_cases",
        "invalid_trades_missing_quote",
        "invalid_trades_missing_close",
        "total_trades_live",
        "total_trades_synth",
        "valid_cases",
        "total_cases",
    ]:
        assert k in cov

    assert out["coverage"]["total_cases"] == 1
    assert out["coverage"]["valid_cases"] == 1
    assert out["component_score"] == 100.0

    # Now enforce min-trades failure.
    low_trades = dict(perfect_diff)
    low_trades["metrics"] = dict(perfect_diff["metrics"])
    low_trades["metrics"]["num_trades"] = {"a": 2, "b": 10, "diff": 8}

    out2 = run_strategy_pnl_parity_suite(
        cases=[case],
        min_trades_per_case=5,
        fixture_diffs={key: low_trades},
    )
    assert out2["coverage"]["total_cases"] == 1
    assert out2["coverage"]["valid_cases"] == 0
    assert out2["component_score"] == 0.0
