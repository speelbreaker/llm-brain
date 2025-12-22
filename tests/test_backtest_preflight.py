import pytest
from fastapi.testclient import TestClient

from src.web_app import app


@pytest.fixture
def client():
    return TestClient(app)


def _write_dummy_harvest_file(tmp_path, *, underlying_dir: str, filename: str) -> None:
    base = tmp_path / "live_deribit" / underlying_dir
    base.mkdir(parents=True, exist_ok=True)
    (base / filename).write_text("")


def test_backtest_start_missing_harvest_fails_fast(client, monkeypatch, tmp_path):
    monkeypatch.setenv("HARVEST_DATA_DIR", str(tmp_path / "live_deribit"))

    from src.backtest.manager import backtest_manager

    called = {"start": 0}

    def _start_should_not_run(*args, **kwargs):
        called["start"] += 1
        raise AssertionError("worker start should not be called on preflight failure")

    monkeypatch.setattr(backtest_manager, "start", _start_should_not_run)

    resp = client.post(
        "/api/backtest/start",
        json={
            "underlying": "BTC",
            "start": "2024-01-01",
            "end": "2024-01-07",
            "backtest_type": "generic",
        },
    )

    assert resp.status_code == 400
    body = resp.json()
    assert body.get("ok") is False
    assert body.get("error", {}).get("code") == "NO_HARVESTED_FILES"
    details = body.get("error", {}).get("details") or {}
    assert "data_readiness" in details
    assert "gates" in details
    assert "gate_overall" in details
    assert called["start"] == 0


def test_backtest_start_fidelity_block_fails_fast(client, monkeypatch, tmp_path):
    monkeypatch.setenv("FIDELITY_GATE_MODE", "block")
    monkeypatch.setenv("FIDELITY_RUNS_DIR", str(tmp_path / "fidelity_runs"))
    monkeypatch.setenv("HARVEST_DATA_DIR", str(tmp_path / "live_deribit"))

    # Provide minimal harvest file so harvest preflight passes and fidelity gate is the blocker.
    _write_dummy_harvest_file(
        tmp_path,
        underlying_dir="BTC_USDC",
        filename="BTC_USDC_2024-01-02_0000.parquet",
    )

    from src.backtest import fidelity_store

    fidelity_store.write_fidelity_report(
        {
            "run_id": "20250101_000001",
            "created_at": "2025-01-01T00:00:01+00:00",
            "underlying": "BTC",
            "overall_score": 10.0,
            "gate_label": "UNTRUSTED",
        }
    )

    from src.backtest.manager import backtest_manager

    called = {"start": 0}

    def _start_should_not_run(*args, **kwargs):
        called["start"] += 1
        raise AssertionError("worker start should not be called on fidelity block")

    monkeypatch.setattr(backtest_manager, "start", _start_should_not_run)

    resp = client.post(
        "/api/backtest/start",
        json={
            "underlying": "BTC",
            "start": "2024-01-01",
            "end": "2024-01-07",
            "backtest_type": "generic",
        },
    )

    assert resp.status_code == 400
    body = resp.json()
    assert body.get("ok") is False
    assert body.get("error", {}).get("code") == "FIDELITY_UNTRUSTED"
    details = body.get("error", {}).get("details") or {}
    assert "gates" in details
    assert any((g.get("code") == "FIDELITY_UNTRUSTED") for g in (details.get("gates") or []))
    assert called["start"] == 0


def test_backtest_start_success_includes_data_readiness(client, monkeypatch, tmp_path):
    monkeypatch.setenv("HARVEST_DATA_DIR", str(tmp_path / "live_deribit"))

    _write_dummy_harvest_file(
        tmp_path,
        underlying_dir="BTC_USDC",
        filename="BTC_USDC_2024-01-02_0000.parquet",
    )

    from src.backtest.manager import backtest_manager

    called = {"start": 0}

    def _start_ok(*args, **kwargs):
        called["start"] += 1
        return True

    monkeypatch.setattr(backtest_manager, "start", _start_ok)

    resp = client.post(
        "/api/backtest/start",
        json={
            "underlying": "BTC",
            "start": "2024-01-01",
            "end": "2024-01-07",
            "backtest_type": "generic",
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body.get("ok") is True
    assert body.get("started") is True
    assert "data_readiness" in body
    assert "gates" in body
    assert "gate_overall" in body
    assert body["data_readiness"].get("harvest_required") is True
    assert called["start"] == 1


def test_backtest_start_eth_missing_does_not_block_btc(client, monkeypatch, tmp_path):
    # Ensure BTC harvest exists; ETH harvest is missing.
    monkeypatch.setenv("HARVEST_DATA_DIR", str(tmp_path / "live_deribit"))

    _write_dummy_harvest_file(
        tmp_path,
        underlying_dir="BTC_USDC",
        filename="BTC_USDC_2024-01-02_0000.parquet",
    )

    from src.backtest.manager import backtest_manager

    called = {"start": 0}

    def _start_ok(*args, **kwargs):
        called["start"] += 1
        return True

    monkeypatch.setattr(backtest_manager, "start", _start_ok)

    resp = client.post(
        "/api/backtest/start",
        json={
            "underlying": "BTC",
            "start": "2024-01-01",
            "end": "2024-01-07",
            "backtest_type": "generic",
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body.get("ok") is True
    assert body.get("started") is True
    assert called["start"] == 1


def test_backtest_start_harvest_key_mismatch_fails(client, monkeypatch, tmp_path):
    # When linear/USDC is required, selecting BTC instead of BTC_USDC must fail.
    monkeypatch.setenv("HARVEST_DATA_DIR", str(tmp_path / "live_deribit"))

    # Only provide BTC (inverse-style) directory.
    _write_dummy_harvest_file(
        tmp_path,
        underlying_dir="BTC",
        filename="BTC_2024-01-02_0000.parquet",
    )

    from src.backtest.manager import backtest_manager

    def _start_should_not_run(*args, **kwargs):
        raise AssertionError("worker start should not be called on harvest key mismatch")

    monkeypatch.setattr(backtest_manager, "start", _start_should_not_run)

    resp = client.post(
        "/api/backtest/start",
        json={
            "underlying": "BTC",
            "start": "2024-01-01",
            "end": "2024-01-07",
            "backtest_type": "generic",
        },
    )

    assert resp.status_code == 400
    body = resp.json()
    assert body.get("ok") is False
    assert body.get("error", {}).get("code") == "HARVEST_KEY_MISMATCH"
