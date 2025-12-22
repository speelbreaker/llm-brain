from __future__ import annotations

from pathlib import Path

import pytest


def test_fidelity_facts_prefers_lab_store_over_legacy(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    base = tmp_path / "fidelity_runs"
    monkeypatch.setenv("FIDELITY_RUNS_DIR", str(base))

    # Create a Lab run (truth source).
    from src.backtest import fidelity_store as lab_store

    lab_store.write_fidelity_report(
        {
            "run_id": "lab_0001",
            "created_at": "2025-01-01T00:00:01+00:00",
            "underlying": "BTC",
            "overall_score": 99.0,
            "gate_label": "TRUSTED",
        },
        base_dir=base,
    )

    # Legacy store is intentionally ignored by ops facts.

    from src.ops.fidelity_status import get_fidelity_facts

    facts = get_fidelity_facts(underlying="BTC", base_dir=base)
    assert facts["available"] is True
    assert facts["source"] == "lab_store"
    assert facts["run_id"] == "lab_0001"
    assert facts["gate_label"] == "TRUSTED"
    assert facts["overall_score"] == 99.0
    assert facts["created_at"] == "2025-01-01T00:00:01+00:00"

    # Deterministic per-underlying latest pointer path.
    assert facts["path"] == str(base / "BTC" / "latest.json")

def test_fidelity_facts_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    base = tmp_path / "fidelity_runs"
    monkeypatch.setenv("FIDELITY_RUNS_DIR", str(base))

    from src.ops.fidelity_status import get_fidelity_facts

    facts = get_fidelity_facts(underlying="BTC", base_dir=base)
    assert facts["available"] is False
    assert facts["source"] == "lab_store"
    assert facts["path"] is None


def test_lab_store_writes_per_underlying_latest_independently(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    base = tmp_path / "fidelity_runs"
    monkeypatch.setenv("FIDELITY_RUNS_DIR", str(base))

    from src.backtest import fidelity_store as lab_store

    lab_store.write_fidelity_report(
        {
            "run_id": "lab_btc",
            "created_at": "2025-02-01T00:00:01+00:00",
            "underlying": "BTC",
            "overall_score": 80.0,
            "gate_label": "TRUSTED",
        },
        base_dir=base,
    )
    lab_store.write_fidelity_report(
        {
            "run_id": "lab_eth",
            "created_at": "2025-02-01T00:00:02+00:00",
            "underlying": "ETH",
            "overall_score": 70.0,
            "gate_label": "WARNING",
        },
        base_dir=base,
    )

    from src.ops.fidelity_status import get_fidelity_facts

    btc = get_fidelity_facts(underlying="BTC", base_dir=base)
    eth = get_fidelity_facts(underlying="ETH", base_dir=base)

    assert btc["run_id"] == "lab_btc"
    assert eth["run_id"] == "lab_eth"
    assert btc["path"] == str(base / "BTC" / "latest.json")
    assert eth["path"] == str(base / "ETH" / "latest.json")
