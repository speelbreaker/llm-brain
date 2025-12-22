from __future__ import annotations

from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from src.web_app import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_api_calibration_run_with_policy_returns_decision(monkeypatch, client: TestClient) -> None:
    from src.calibration_update_policy import CalibrationRunRecord, UpdateDecision

    def _fake_run_calibration_with_policy(*args, **kwargs):
        record = CalibrationRunRecord(
            timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
            underlying="BTC",
            source="live",
            recommended_iv_multiplier=1.10,
            sample_size=123,
            vega_sum=456.0,
            smoothed_global_multiplier=1.05,
            applied=True,
            applied_reason="global Δ=0.05; sample=123; vega=456",
        )
        decision = UpdateDecision(
            should_apply=True,
            reason="global Δ=0.05; sample=123; vega=456",
            details={"global_delta": 0.05, "sample_size": 123, "vega_sum": 456.0},
        )
        return record, decision

    import src.calibration_update_policy as policy_mod

    monkeypatch.setattr(policy_mod, "run_calibration_with_policy", _fake_run_calibration_with_policy)

    resp = client.post(
        "/api/calibration/run_with_policy",
        json={"underlying": "BTC", "source": "live", "min_dte": 3.0, "max_dte": 30.0},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["applied"] is True
    assert "decision" in data
    assert data["decision"]["should_apply"] is True
    assert data["decision"]["applied"] is True
    assert isinstance(data["decision"]["why"], list)
    assert len(data["decision"]["why"]) >= 1
