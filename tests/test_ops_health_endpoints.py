from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.web_app import app
from src.healthcheck import CheckStatus, HealthCheckResult


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def _write_dummy_harvest_file(tmp_path: Path, *, underlying_dir: str, filename: str) -> None:
    base = tmp_path / "live_deribit" / underlying_dir
    base.mkdir(parents=True, exist_ok=True)
    (base / filename).write_text("")


def test_ops_health_run_returns_per_underlying_gate_overall_and_dirs(client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("LIVE_DERIBIT_DATA_DIR", str(tmp_path / "live_deribit"))
    monkeypatch.setenv("CALIBRATION_DIR", str(tmp_path / "calibration_runs"))
    monkeypatch.setenv("FIDELITY_DIR", str(tmp_path / "fidelity_runs"))
    monkeypatch.setenv("CALIBRATION_GATE_MODE", "block")
    monkeypatch.delenv("OPS_HEALTH_RUN_SECRET", raising=False)

    # Ensure BTC harvest exists and matches expected key.
    _write_dummy_harvest_file(
        tmp_path,
        underlying_dir="BTC_USDC",
        filename="BTC_USDC_2024-01-02_0000.parquet",
    )

    # Create a BTC calibration record; ETH intentionally missing.
    from src.calibration_update_policy import record_calibration_result

    record_calibration_result(
        underlying="BTC",
        source="live",
        recommended_iv_multiplier=1.1,
        recommended_band_multipliers=None,
        sample_size=100,
        vega_sum=200.0,
        applied=True,
        applied_reason="test",
        base_dir=str(tmp_path / "calibration_runs"),
    )

    res = client.post("/api/ops/health/run")
    assert res.status_code == 200
    body = res.json()

    gate_overall = body.get("gate_overall") or {}
    assert "by_underlying" in gate_overall
    by_u = gate_overall.get("by_underlying") or {}
    assert "BTC" in by_u
    assert "ETH" in by_u

    # Can trade should differ when ETH calibration is missing but BTC is present.
    assert bool(by_u["BTC"].get("can_trade")) is True
    assert bool(by_u["ETH"].get("can_trade")) is False

    # Base dir consistency: check harvest_freshness check meta uses LIVE_DERIBIT_DATA_DIR.
    checks = body.get("checks") or []
    harvest = next((c for c in checks if c.get("name") == "harvest_freshness"), None)
    assert harvest is not None
    meta = harvest.get("meta") or {}
    assert meta.get("base_dir") == str(tmp_path / "live_deribit")

    # Base dir consistency: harvest gate details should also reflect the same base_dir.
    gates = body.get("gates") or []
    harvest_gate = next((g for g in gates if g.get("name") == "harvest" and g.get("underlying") == "BTC"), None)
    assert harvest_gate is not None
    details = harvest_gate.get("details") or {}
    assert details.get("base_dir") == str(tmp_path / "live_deribit")


def test_ops_health_status_endpoint_serves_cache(client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("LIVE_DERIBIT_DATA_DIR", str(tmp_path / "live_deribit"))
    monkeypatch.setenv("CALIBRATION_DIR", str(tmp_path / "calibration_runs"))
    monkeypatch.setenv("CALIBRATION_GATE_MODE", "warn")
    monkeypatch.delenv("OPS_HEALTH_RUN_SECRET", raising=False)

    _write_dummy_harvest_file(
        tmp_path,
        underlying_dir="BTC_USDC",
        filename="BTC_USDC_2024-01-02_0000.parquet",
    )

    # Prime cache
    res = client.post("/api/ops/health/run")
    assert res.status_code == 200

    res2 = client.get("/api/ops/health/status")
    assert res2.status_code == 200
    body2 = res2.json()
    assert body2.get("checked_at") is not None
    assert "gate_overall" in body2


def _clear_health_cache() -> None:
    from src import healthcheck

    with healthcheck._health_cache_lock:
        healthcheck._cached_health_status = None


def test_ops_health_status_empty_returns_404(client: TestClient) -> None:
    _clear_health_cache()

    resp = client.get("/api/ops/health/status")
    assert resp.status_code == 404
    assert resp.json() == {"error": "no_healthcheck_cached"}


def test_ops_health_run_populates_cache(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from src import healthcheck

    _clear_health_cache()

    secret = "ops-health-secret"
    monkeypatch.setenv("OPS_HEALTH_RUN_SECRET", secret)

    base_dir = tmp_path / "live_deribit"
    monkeypatch.setenv("HARVEST_DATA_DIR", str(base_dir))
    monkeypatch.setenv("FIDELITY_RUNS_DIR", str(tmp_path / "fidelity_runs"))
    for underlying in ("BTC", "ETH"):
        snapshot_dir = base_dir / underlying / "2025" / "01" / "01"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        (snapshot_dir / "snapshot.parquet").write_text("test")

    class DummyClient:
        def __enter__(self) -> "DummyClient":
            return self

        def __exit__(self, *args: object) -> None:
            return None

    monkeypatch.setattr(healthcheck, "DeribitClient", DummyClient)
    monkeypatch.setattr(
        healthcheck,
        "check_deribit_public",
        lambda client: HealthCheckResult(
            name="deribit_public",
            status=CheckStatus.OK,
            detail="public API OK",
            severity="OK",
            can_trade=True,
        ),
    )
    monkeypatch.setattr(
        healthcheck,
        "check_deribit_private",
        lambda client, cfg: HealthCheckResult(
            name="deribit_private",
            status=CheckStatus.OK,
            detail="private API OK",
            severity="OK",
            can_trade=True,
        ),
    )
    monkeypatch.setattr(
        healthcheck,
        "check_state_builder",
        lambda client, cfg: HealthCheckResult(
            name="state_builder",
            status=CheckStatus.OK,
            detail="state builder OK",
            severity="OK",
            can_trade=True,
        ),
    )

    real_harvest = healthcheck.check_harvest_freshness
    monkeypatch.setattr(
        healthcheck,
        "check_harvest_freshness",
        lambda cfg: real_harvest(cfg, base_dir=base_dir),
    )
    monkeypatch.setattr(
        healthcheck,
        "check_calibration_freshness",
        lambda cfg: HealthCheckResult(
            name="calibration_freshness",
            status=CheckStatus.OK,
            detail="BTC=1h OK",
            severity="OK",
            can_trade=True,
            meta={
                "per_underlying": {
                    "BTC": {
                        "age_hours": 1.0,
                        "applied": True,
                    }
                }
            },
        ),
    )
    monkeypatch.setattr(
        healthcheck,
        "check_fidelity_gate",
        lambda cfg: HealthCheckResult(
            name="fidelity_gate",
            status=CheckStatus.OK,
            detail="gate=TRUSTED score=90",
            severity="OK",
            can_trade=True,
            meta={"gate_label": "TRUSTED", "overall_score": 90},
        ),
    )

    headers = {"X-OPS-HEALTH-SECRET": secret}
    resp = client.post("/api/ops/health/run", headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert "overall_status" in data
    assert "worst_severity" in data
    assert "can_trade" in data
    assert "checks" in data
    assert "gates" in data
    assert "gate_overall" in data

    names = {check["name"] for check in data["checks"]}
    assert {
        "deribit_public",
        "state_builder",
        "harvest_freshness",
        "calibration_freshness",
        "fidelity_gate",
    }.issubset(names)

    resp_cached = client.get("/api/ops/health/status")
    assert resp_cached.status_code == 200
    cached = resp_cached.json()
    assert "overall_status" in cached
    assert "checks" in cached
    assert "gates" in cached
    assert "gate_overall" in cached


def test_ops_health_run_requires_secret_when_configured(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_health_cache()
    monkeypatch.setenv("OPS_HEALTH_RUN_SECRET", "secret-key")

    # Unauthorized: no header provided
    resp = client.post("/api/ops/health/run")
    assert resp.status_code == 403
    assert resp.json() == {"error": "unauthorized"}
