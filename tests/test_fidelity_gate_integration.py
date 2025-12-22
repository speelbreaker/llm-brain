from __future__ import annotations

from pathlib import Path

import pytest

from src.healthcheck import CheckStatus, check_fidelity_gate


class DummyCfg:
    def __init__(self, underlyings: list[str]):
        self.underlyings = underlyings


def _write_report(base: Path, *, run_id: str, underlying: str, gate_label: str, overall_score: float) -> None:
    from src.backtest import fidelity_store

    fidelity_store.write_fidelity_report(
        {
            "run_id": run_id,
            "created_at": "2025-01-01T00:00:01+00:00",
            "underlying": underlying,
            "overall_score": overall_score,
            "gate_label": gate_label,
            "component_scores": {"strategy_pnl_parity": overall_score},
            "coverage": {"strategy_pnl_parity": {"coverage_ratio_cases": 1.0, "valid_cases": 1, "total_cases": 1}},
            "thresholds": {"trusted_threshold": 80.0, "warn_threshold": 65.0},
        },
        base_dir=base,
    )


def test_check_fidelity_gate_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    base = tmp_path / "fidelity_runs"
    monkeypatch.setenv("FIDELITY_RUNS_DIR", str(base))
    monkeypatch.delenv("HEALTH_STRICT_SYNTHETIC_GATE", raising=False)

    cfg = DummyCfg(["BTC"])
    res = check_fidelity_gate(cfg, base_dir=base)
    assert res.status == CheckStatus.WARN
    assert res.meta is not None
    assert res.meta.get("gate_label") == "MISSING"


def test_check_fidelity_gate_untrusted(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    base = tmp_path / "fidelity_runs"
    monkeypatch.setenv("FIDELITY_RUNS_DIR", str(base))
    monkeypatch.delenv("HEALTH_STRICT_SYNTHETIC_GATE", raising=False)

    _write_report(base, run_id="r1", underlying="BTC", gate_label="UNTRUSTED", overall_score=10.0)

    cfg = DummyCfg(["BTC"])
    res = check_fidelity_gate(cfg, base_dir=base)
    assert res.status == CheckStatus.WARN
    assert res.meta is not None
    assert res.meta.get("gate_label") == "UNTRUSTED"


def test_check_fidelity_gate_warning(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    base = tmp_path / "fidelity_runs"
    monkeypatch.setenv("FIDELITY_RUNS_DIR", str(base))
    monkeypatch.delenv("HEALTH_STRICT_SYNTHETIC_GATE", raising=False)

    _write_report(base, run_id="r2", underlying="BTC", gate_label="WARNING", overall_score=70.0)

    cfg = DummyCfg(["BTC"])
    res = check_fidelity_gate(cfg, base_dir=base)
    assert res.status == CheckStatus.WARN
    assert res.meta is not None
    assert res.meta.get("gate_label") == "WARNING"


def test_check_fidelity_gate_trusted(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    base = tmp_path / "fidelity_runs"
    monkeypatch.setenv("FIDELITY_RUNS_DIR", str(base))
    monkeypatch.delenv("HEALTH_STRICT_SYNTHETIC_GATE", raising=False)

    _write_report(base, run_id="r3", underlying="BTC", gate_label="TRUSTED", overall_score=90.0)

    cfg = DummyCfg(["BTC"])
    res = check_fidelity_gate(cfg, base_dir=base)
    assert res.status == CheckStatus.OK
    assert res.meta is not None
    assert res.meta.get("gate_label") == "TRUSTED"
