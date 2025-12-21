from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.web_app import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_api_fidelity_latest_404(monkeypatch, tmp_path: Path, client: TestClient) -> None:
    monkeypatch.setenv("FIDELITY_RUNS_DIR", str(tmp_path / "fidelity_runs"))

    resp = client.get("/api/fidelity/latest")
    assert resp.status_code == 404
    assert resp.json() == {"error": "no_fidelity_runs"}


def test_api_fidelity_history_empty(monkeypatch, tmp_path: Path, client: TestClient) -> None:
    monkeypatch.setenv("FIDELITY_RUNS_DIR", str(tmp_path / "fidelity_runs"))

    resp = client.get("/api/fidelity/history")
    assert resp.status_code == 200
    assert resp.json() == []


def test_api_fidelity_latest_and_history_after_write(monkeypatch, tmp_path: Path, client: TestClient) -> None:
    base = tmp_path / "fidelity_runs"
    monkeypatch.setenv("FIDELITY_RUNS_DIR", str(base))

    from src.backtest import fidelity_store

    report = {
        "run_id": "20250101_000001",
        "created_at": "2025-01-01T00:00:01+00:00",
        "underlying": "BTC",
        "overall_score": 72.5,
        "gate_label": "WARNING",
        "component_scores": {"strategy_pnl_parity": 60.0},
        "coverage": {"coverage_ratio_cases": 0.9, "valid_cases": 2, "total_cases": 2},
    }
    fidelity_store.write_fidelity_report(report)

    resp_latest = client.get("/api/fidelity/latest")
    assert resp_latest.status_code == 200
    latest = resp_latest.json()
    assert latest["run_id"] == report["run_id"]
    assert latest["created_at"] == report["created_at"]
    assert latest["overall_score"] == report["overall_score"]
    assert latest["gate_label"] == report["gate_label"]
    assert "component_scores" in latest
    assert "coverage" in latest

    resp_hist = client.get("/api/fidelity/history?limit=30")
    assert resp_hist.status_code == 200
    hist = resp_hist.json()
    assert isinstance(hist, list)
    assert len(hist) >= 1
    assert hist[0]["run_id"] == report["run_id"]
