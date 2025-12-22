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


def _patch_ops_health_baseline(monkeypatch: pytest.MonkeyPatch) -> None:
    from src import healthcheck
    from src.ops import facts_resolver

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

    monkeypatch.setattr(
        healthcheck,
        "check_risk_config",
        lambda cfg: HealthCheckResult(
            name="risk_config",
            status=CheckStatus.OK,
            detail="risk config OK",
            severity="OK",
            can_trade=True,
        ),
    )

    def _ok_harvest(cfg, base_dir=None):
        return HealthCheckResult(
            name="harvest_freshness",
            status=CheckStatus.OK,
            detail="harvest ok",
            severity="OK",
            can_trade=True,
        )

    def _ok_calibration(cfg, base_dir=None):
        return HealthCheckResult(
            name="calibration_freshness",
            status=CheckStatus.OK,
            detail="calibration ok",
            severity="OK",
            can_trade=True,
        )

    def _ok_fidelity(cfg, base_dir=None):
        return HealthCheckResult(
            name="fidelity_gate",
            status=CheckStatus.OK,
            detail="fidelity ok",
            severity="OK",
            can_trade=True,
        )

    monkeypatch.setattr(healthcheck, "check_harvest_freshness", _ok_harvest)
    monkeypatch.setattr(healthcheck, "check_calibration_freshness", _ok_calibration)
    monkeypatch.setattr(healthcheck, "check_fidelity_gate", _ok_fidelity)

    def _resolve_ops_facts(cfg, *, now=None):
        return {
            "now": "2025-01-01T00:00:00+00:00",
            "underlyings_active": ["BTC", "ETH"],
            "paths": {
                "live_deribit_data_dir": "data/live_deribit",
                "calibration_dir": "data/calibration_runs",
                "fidelity_dir": "data/fidelity_runs",
            },
            "harvest": {"BTC": {}, "ETH": {}},
            "calibration": {"BTC": {}, "ETH": {}},
            "fidelity": {"BTC": {}, "ETH": {}},
        }

    monkeypatch.setattr(facts_resolver, "resolve_ops_facts", _resolve_ops_facts)


def _assert_health_payload(payload: dict) -> None:
    assert isinstance(payload.get("checks"), list)
    assert isinstance(payload.get("gates"), list)
    assert "gate_overall" in payload
    assert "checks_overall" in payload
    assert "checks_summary" in payload
    assert isinstance(payload.get("checks_summary"), str)
    assert isinstance(payload.get("can_trade_by_underlying"), dict)
    can_trade_by_underlying = payload.get("can_trade_by_underlying") or {}
    assert payload.get("can_trade") == any(can_trade_by_underlying.values())


def test_ops_health_run_returns_per_underlying_gate_overall_and_dirs(client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from src import healthcheck

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
    from src import healthcheck

    monkeypatch.setenv("LIVE_DERIBIT_DATA_DIR", str(tmp_path / "live_deribit"))
    monkeypatch.setenv("CALIBRATION_DIR", str(tmp_path / "calibration_runs"))
    monkeypatch.setenv("CALIBRATION_GATE_MODE", "warn")
    monkeypatch.delenv("OPS_HEALTH_RUN_SECRET", raising=False)

    _write_dummy_harvest_file(
        tmp_path,
        underlying_dir="BTC_USDC",
        filename="BTC_USDC_2024-01-02_0000.parquet",
    )

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

    # Prime cache
    res = client.post("/api/ops/health/run")
    assert res.status_code == 200

    res2 = client.get("/api/ops/health/status")
    assert res2.status_code == 200
    body2 = res2.json()
    assert body2.get("checked_at") is not None
    assert "gate_overall" in body2
    assert "checks_overall" in body2


def test_ops_health_gates_are_authoritative_when_present(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_ops_health_baseline(monkeypatch)
    monkeypatch.delenv("OPS_HEALTH_RUN_SECRET", raising=False)

    # Force gates to BLOCK (calibration missing -> gate FAIL -> can_trade False)
    monkeypatch.setenv("CALIBRATION_GATE_MODE", "block")
    monkeypatch.setenv("FIDELITY_GATE_MODE", "off")

    res = client.post("/api/ops/health/run")
    assert res.status_code == 200
    body = res.json()

    # Checks are OK due to baseline patch, but gates should still dominate.
    assert body.get("checks_overall") == "OK"
    assert body.get("overall_status") == "FAIL"
    assert body.get("can_trade") is False
    assert isinstance(body.get("summary"), str)
    assert body.get("summary").startswith("gates")


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


def test_ops_health_status_prefers_gates_over_checks_when_blocking(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """If gates block trade, status must reflect gates even if checks are OK."""
    from src import healthcheck

    _clear_health_cache()

    # Configure gate policy to block on missing calibration.
    monkeypatch.setenv("LIVE_DERIBIT_DATA_DIR", str(tmp_path / "live_deribit"))
    monkeypatch.setenv("CALIBRATION_DIR", str(tmp_path / "calibration_runs"))
    monkeypatch.setenv("CALIBRATION_GATE_MODE", "block")
    monkeypatch.delenv("OPS_HEALTH_RUN_SECRET", raising=False)

    # Provide harvest data so harvest isn't the blocker.
    _write_dummy_harvest_file(
        tmp_path,
        underlying_dir="BTC_USDC",
        filename="BTC_USDC_2024-01-02_0000.parquet",
    )

    # Force the calibration *check* to look OK, while calibration *facts* remain missing.
    monkeypatch.setattr(
        healthcheck,
        "check_calibration_freshness",
        lambda cfg, base_dir=None: HealthCheckResult(
            name="calibration_freshness",
            status=CheckStatus.OK,
            detail="calibration OK (patched)",
            severity="OK",
            can_trade=True,
            meta={"per_underlying": {"BTC": {"status": "OK"}}},
        ),
    )

    res = client.post("/api/ops/health/run")
    assert res.status_code == 200

    res2 = client.get("/api/ops/health/status")
    assert res2.status_code == 200
    body = res2.json()

    assert body.get("can_trade") is False
    assert body.get("worst_severity") == "FATAL"
    assert body.get("overall_status") == "FAIL"

    # Ensure our patched check really is OK (proves gates are the source of truth).
    checks = body.get("checks") or []
    calib = next((c for c in checks if c.get("name") == "calibration_freshness"), None)
    assert calib is not None
    assert calib.get("status") == "OK"


def test_ops_health_can_trade_combines_checks_and_gates(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from src import healthcheck
    from src.ops import gates as ops_gates

    _clear_health_cache()
    _patch_ops_health_baseline(monkeypatch)
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

    def _run_gates(self, gate_fns):
        return {
            "gates": [],
            "gate_overall": {
                "global": {"status": "PASS", "severity": "OK", "can_trade": True},
                "by_underlying": {
                    "BTC": {"status": "PASS", "severity": "OK", "can_trade": True},
                    "ETH": {"status": "FAIL", "severity": "FATAL", "can_trade": False},
                },
            },
        }

    monkeypatch.setattr(ops_gates.GateRunner, "run", _run_gates)

    res = client.post("/api/ops/health/run")
    assert res.status_code == 200
    body = res.json()
    _assert_health_payload(body)
    assert body["can_trade_by_underlying"]["ETH"] is False
    assert body["can_trade_by_underlying"]["BTC"] is True
    assert body["can_trade"] is True

    res2 = client.get("/api/ops/health/status")
    assert res2.status_code == 200
    body2 = res2.json()
    _assert_health_payload(body2)


def test_ops_health_deribit_public_fail_blocks_all_underlyings(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from src import healthcheck
    from src.ops import gates as ops_gates

    _clear_health_cache()
    _patch_ops_health_baseline(monkeypatch)
    monkeypatch.setattr(
        healthcheck,
        "check_deribit_public",
        lambda client: HealthCheckResult(
            name="deribit_public",
            status=CheckStatus.FAIL,
            detail="public API failed",
            error_code="DERIBIT_NETWORK",
            severity="DEGRADED",
            can_trade=False,
        ),
    )

    def _run_gates(self, gate_fns):
        return {
            "gates": [],
            "gate_overall": {
                "global": {"status": "PASS", "severity": "OK", "can_trade": True},
                "by_underlying": {
                    "BTC": {"status": "PASS", "severity": "OK", "can_trade": True},
                    "ETH": {"status": "PASS", "severity": "OK", "can_trade": True},
                },
            },
        }

    monkeypatch.setattr(ops_gates.GateRunner, "run", _run_gates)

    res = client.post("/api/ops/health/run")
    assert res.status_code == 200
    body = res.json()
    _assert_health_payload(body)
    assert body["can_trade"] is False
    assert all(value is False for value in body["can_trade_by_underlying"].values())


def test_ops_health_gate_overall_drives_status(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from src import healthcheck
    from src.ops import gates as ops_gates

    _clear_health_cache()
    _patch_ops_health_baseline(monkeypatch)
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

    def _run_gates(self, gate_fns):
        return {
            "gates": [],
            "gate_overall": {
                "global": {
                    "status": "FAIL",
                    "severity": "FATAL",
                    "can_trade": False,
                    "message": "truth missing",
                    "code": "TRUTH_BLOCK",
                },
                "by_underlying": {
                    "BTC": {"status": "FAIL", "severity": "FATAL", "can_trade": False},
                    "ETH": {"status": "FAIL", "severity": "FATAL", "can_trade": False},
                },
            },
        }

    monkeypatch.setattr(ops_gates.GateRunner, "run", _run_gates)

    res = client.post("/api/ops/health/run")
    assert res.status_code == 200
    body = res.json()
    _assert_health_payload(body)
    assert body["overall_status"] == "FAIL"
    assert body["checks_overall"] == "OK"
    assert "gates FAIL" in body["summary"]
    assert "code=TRUTH_BLOCK" in body["summary"]
    assert "truth missing" in body["summary"]


def test_ops_health_check_failure_precedes_gate(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from src import healthcheck
    from src.ops import gates as ops_gates
    from src.ops import gate_factories

    _clear_health_cache()
    _patch_ops_health_baseline(monkeypatch)
    monkeypatch.setenv("FIDELITY_GATE_MODE", "off")
    monkeypatch.setenv("CALIBRATION_GATE_MODE", "off")

    monkeypatch.setattr(
        healthcheck,
        "check_config",
        lambda cfg: HealthCheckResult(
            name="check_config",
            status=CheckStatus.FAIL,
            detail="config invalid",
            severity="FATAL",
            can_trade=False,
        ),
    )
    monkeypatch.setattr(
        healthcheck,
        "check_deribit_public",
        lambda client: HealthCheckResult(
            name="deribit_public",
            status=CheckStatus.OK,
            detail="public OK",
            severity="OK",
            can_trade=True,
        ),
    )

    def _build_underlying_gate_fns(*, harvest_required, **kwargs):
        return []

    def _run_gates(self, gate_fns):
        return {
            "gates": [],
            "gate_overall": {
                "global": {"status": "PASS", "severity": "OK", "can_trade": True},
                "by_underlying": {},
            },
        }

    monkeypatch.setattr(gate_factories, "build_underlying_gate_fns", _build_underlying_gate_fns)
    monkeypatch.setattr(ops_gates.GateRunner, "run", _run_gates)

    res = client.post("/api/ops/health/run")
    assert res.status_code == 200
    body = res.json()
    _assert_health_payload(body)
    assert body["overall_status"] == "FAIL"
    assert "check_config FAIL" in body["summary"]


def test_ops_health_harvest_required_toggles_with_gate_modes(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.ops import gate_factories
    from src.ops import gates as ops_gates

    _clear_health_cache()
    _patch_ops_health_baseline(monkeypatch)
    monkeypatch.setenv("FIDELITY_GATE_MODE", "off")
    monkeypatch.setenv("CALIBRATION_GATE_MODE", "off")

    captured = []

    def _build_underlying_gate_fns(*, harvest_required, **kwargs):
        captured.append(harvest_required)
        return []

    def _run_gates(self, gate_fns):
        return {
            "gates": [],
            "gate_overall": {
                "global": {"status": "PASS", "severity": "OK", "can_trade": True},
                "by_underlying": {},
            },
        }

    monkeypatch.setattr(gate_factories, "build_underlying_gate_fns", _build_underlying_gate_fns)
    monkeypatch.setattr(ops_gates.GateRunner, "run", _run_gates)

    res = client.post("/api/ops/health/run")
    assert res.status_code == 200
    assert captured and all(value is False for value in captured)

    captured.clear()
    monkeypatch.setenv("FIDELITY_GATE_MODE", "block")
    monkeypatch.setenv("CALIBRATION_GATE_MODE", "off")
    res2 = client.post("/api/ops/health/run")
    assert res2.status_code == 200
    assert captured and all(value is True for value in captured)

    captured.clear()
    monkeypatch.setenv("FIDELITY_GATE_MODE", "off")
    monkeypatch.setenv("CALIBRATION_GATE_MODE", "warn")
    res3 = client.post("/api/ops/health/run")
    assert res3.status_code == 200
    assert captured and all(value is True for value in captured)


def test_ops_health_gate_eval_exception_fails_closed(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from src import healthcheck
    from src.ops import gates as ops_gates

    _clear_health_cache()
    _patch_ops_health_baseline(monkeypatch)
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

    def _run_gates(self, gate_fns):
        raise RuntimeError("boom")

    monkeypatch.setattr(ops_gates.GateRunner, "run", _run_gates)

    res = client.post("/api/ops/health/run")
    assert res.status_code == 200
    body = res.json()
    _assert_health_payload(body)
    assert body["can_trade"] is False
    assert any(check.get("name") == "gates_runner" for check in body.get("checks") or [])
