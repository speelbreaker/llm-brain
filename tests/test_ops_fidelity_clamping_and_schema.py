from __future__ import annotations

from datetime import datetime, timezone

import pytest


def test_ops_runner_emits_warning_when_clamping(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.fidelity import ops_runner

    # Stub out the market suite to produce out-of-range component scores.
    class DummyResult:
        component_scores = {
            "underlying_returns": 240.0,
            "iv_surface_level": -12.0,
            "spot_iv_coupling": 50.0,
        }
        components = {}
        component_status = {}

    def fake_run_fidelity_suite(*args, **kwargs):
        return DummyResult()

    monkeypatch.setattr("src.fidelity.run_suite.run_fidelity_suite", fake_run_fidelity_suite)

    start_ts = int(datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp())
    end_ts = int(datetime(2025, 1, 2, tzinfo=timezone.utc).timestamp())

    # Force parity to a safe value.
    def fake_parity(*args, **kwargs):
        return {
            "component_score": 80.0,
            "coverage": {
                "coverage_ratio_cases": 1.0,
                "invalid_trades_missing_quote": 0,
                "invalid_trades_missing_close": 0,
                "total_trades_live": 10,
                "total_trades_synth": 10,
                "valid_cases": 1,
                "total_cases": 1,
            },
            "cases": [],
            "metric_specs": {},
        }

    monkeypatch.setattr(ops_runner, "run_strategy_pnl_parity_suite", fake_parity)

    report = ops_runner.run_ops_fidelity_suite(
        underlying="BTC",
        start_ts=start_ts,
        end_ts=end_ts,
        base_dir=None,
    )

    assert report["raw_component_scores"]["underlying_returns"] == 240.0
    assert report["component_scores"]["underlying_returns"] == 100.0
    assert report["component_scores"]["iv_surface_level"] == 0.0
    assert any("component_score_clamped:underlying_returns" in w for w in report.get("warnings") or [])
    assert any("component_score_clamped:iv_surface_level" in w for w in report.get("warnings") or [])


def test_ops_runner_schema_missing_forces_untrusted(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.fidelity import ops_runner

    class DummyResult:
        component_scores = {
            "underlying_returns": 90.0,
            "iv_surface_level": 90.0,
            "spot_iv_coupling": 90.0,
        }
        components = {}
        component_status = {}

    def fake_run_fidelity_suite(*args, **kwargs):
        return DummyResult()

    monkeypatch.setattr("src.fidelity.run_suite.run_fidelity_suite", fake_run_fidelity_suite)

    # Return malformed parity coverage (missing required keys)
    def fake_parity(*args, **kwargs):
        return {
            "component_score": 100.0,
            "coverage": {"valid_cases": 1},
            "cases": [],
            "metric_specs": {},
        }

    monkeypatch.setattr(ops_runner, "run_strategy_pnl_parity_suite", fake_parity)

    start_ts = int(datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp())
    end_ts = int(datetime(2025, 1, 2, tzinfo=timezone.utc).timestamp())

    report = ops_runner.run_ops_fidelity_suite(
        underlying="BTC",
        start_ts=start_ts,
        end_ts=end_ts,
        base_dir=None,
    )

    assert report["gate_label"] == "UNTRUSTED"
    errors = report.get("errors") or []
    assert any(e.get("code") == "FIDELITY_PARITY_COVERAGE_SCHEMA_MISSING" for e in errors)
