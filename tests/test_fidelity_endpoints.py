from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from src.web_app import app


client = TestClient(app)


def test_fidelity_mvp_endpoints_empty(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("FIDELITY_RUNS_DIR", str(tmp_path / "fidelity_runs"))

    resp = client.get("/calibration/fidelity/latest")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert data["run_id"] is None
    assert data["report"] is None

    resp2 = client.get("/calibration/fidelity/history?limit=10")
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert data2["ok"] is True
    assert data2["runs"] == []


def test_fidelity_mvp_endpoints_with_fixture_run(monkeypatch, tmp_path: Path) -> None:
    # Force fixture fallback by pointing the harvester root somewhere nonexistent.
    monkeypatch.setenv("HARVESTER_DATA_ROOT", str(tmp_path / "nope"))
    monkeypatch.setenv("FIDELITY_RUNS_DIR", str(tmp_path / "fidelity_runs"))

    from src.fidelity.run_suite import run_fidelity_suite

    # Fixture window matches tests/fixtures/fidelity/live_snapshots.jsonl
    report = run_fidelity_suite(
        start_ts=1735689600,
        end_ts=1736121600,
        underlying="BTC",
        seed=123,
    )

    resp_latest = client.get("/calibration/fidelity/latest")
    assert resp_latest.status_code == 200
    data_latest = resp_latest.json()
    assert data_latest["ok"] is True
    assert data_latest["run_id"] == report.run_id
    assert data_latest["report"]["run_id"] == report.run_id

    # Underlying-scoped latest should return BTC run and not leak into ETH.
    resp_latest_btc = client.get("/calibration/fidelity/latest?underlying=BTC")
    assert resp_latest_btc.status_code == 200
    data_latest_btc = resp_latest_btc.json()
    assert data_latest_btc["ok"] is True
    assert data_latest_btc["run_id"] == report.run_id
    assert data_latest_btc["report"]["underlying"] == "BTC"

    resp_latest_eth = client.get("/calibration/fidelity/latest?underlying=ETH")
    assert resp_latest_eth.status_code == 200
    data_latest_eth = resp_latest_eth.json()
    assert data_latest_eth["ok"] is True
    assert data_latest_eth["run_id"] is None
    assert data_latest_eth["report"] is None

    resp_hist = client.get("/calibration/fidelity/history?limit=10")
    assert resp_hist.status_code == 200
    data_hist = resp_hist.json()
    assert data_hist["ok"] is True
    assert len(data_hist["runs"]) >= 1
    assert data_hist["runs"][0]["run_id"] == report.run_id

    resp_hist_btc = client.get("/calibration/fidelity/history?underlying=BTC&limit=10")
    assert resp_hist_btc.status_code == 200
    data_hist_btc = resp_hist_btc.json()
    assert data_hist_btc["ok"] is True
    assert len(data_hist_btc["runs"]) >= 1
    assert data_hist_btc["runs"][0]["underlying"] == "BTC"

    resp_hist_eth = client.get("/calibration/fidelity/history?underlying=ETH&limit=10")
    assert resp_hist_eth.status_code == 200
    data_hist_eth = resp_hist_eth.json()
    assert data_hist_eth["ok"] is True
    assert data_hist_eth["runs"] == []

    resp_report = client.get(f"/calibration/fidelity/report/{report.run_id}")
    assert resp_report.status_code == 200
    data_report = resp_report.json()
    assert data_report["ok"] is True
    assert data_report["run_id"] == report.run_id
    assert data_report["report"]["run_id"] == report.run_id

    resp_spec = client.get("/calibration/fidelity/spec")
    assert resp_spec.status_code == 200
    data_spec = resp_spec.json()
    assert data_spec["ok"] is True
    assert "spec" in data_spec
    assert "strategies" in data_spec["spec"]
