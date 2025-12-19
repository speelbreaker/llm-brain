from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from src.web_app import app


client = TestClient(app)


def test_fidelity_latest_empty(monkeypatch, tmp_path: Path) -> None:
    from src.fidelity import fidelity_store

    monkeypatch.setattr(fidelity_store, "FIDELITY_RUNS_DIR", tmp_path / "fidelity_runs")

    resp = client.get("/api/calibration/fidelity/latest?underlying=BTC")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert data["underlying"] == "BTC"
    assert data["report"] is None


def test_fidelity_latest_and_history(monkeypatch, tmp_path: Path) -> None:
    from src.fidelity import fidelity_store

    base = tmp_path / "fidelity_runs"
    monkeypatch.setattr(fidelity_store, "FIDELITY_RUNS_DIR", base)

    latest_path = base / "BTC" / "latest" / "fidelity_report.json"
    latest_path.parent.mkdir(parents=True, exist_ok=True)

    report_1 = {
        "run_id": "20250101_000001",
        "timestamp": "2025-01-01T00:00:01+00:00",
        "market_live_meta": {"source": "harvested"},
        "market_synth_meta": {"pricing": "synthetic_bs"},
        "component_scores": {"strategy_pnl_parity": 90.0},
        "overall_score": 90.0,
        "gate": "TRUSTED",
        "strategy_parity": {"p0": True},
    }

    latest_path.write_text(__import__("json").dumps(report_1), encoding="utf-8")

    hist_1 = base / "BTC" / "history" / "20250101_000001" / "fidelity_report.json"
    hist_2 = base / "BTC" / "history" / "20250102_000001" / "fidelity_report.json"
    hist_1.parent.mkdir(parents=True, exist_ok=True)
    hist_2.parent.mkdir(parents=True, exist_ok=True)

    hist_1.write_text(__import__("json").dumps(report_1), encoding="utf-8")

    report_2 = dict(report_1)
    report_2["run_id"] = "20250102_000001"
    report_2["timestamp"] = "2025-01-02T00:00:01+00:00"
    report_2["overall_score"] = 80.0
    report_2["gate"] = "WARNING"
    hist_2.write_text(__import__("json").dumps(report_2), encoding="utf-8")

    resp_latest = client.get("/api/calibration/fidelity/latest?underlying=BTC")
    assert resp_latest.status_code == 200
    data_latest = resp_latest.json()
    assert data_latest["report"]["run_id"] == "20250101_000001"

    resp_hist = client.get("/api/calibration/fidelity/history?underlying=BTC&limit=30")
    assert resp_hist.status_code == 200
    data_hist = resp_hist.json()
    assert data_hist["ok"] is True
    assert data_hist["underlying"] == "BTC"
    assert len(data_hist["runs"]) == 2
    # Sorted by run_id descending (directory name)
    assert data_hist["runs"][0]["run_id"] == "20250102_000001"
    assert data_hist["runs"][1]["run_id"] == "20250101_000001"
