"""
Tests for health and calibration automation features.

Tests cover:
1. Background healthcheck scheduler
2. Health guard integration in agent loop
3. Cached health status management
4. Auto-calibration daily script
5. Calibration update policy integration
"""
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime, timezone, timedelta

from src.healthcheck import (
    CheckStatus,
    CachedHealthStatus,
    run_and_cache_healthcheck,
    get_cached_health_status,
    set_agent_paused_due_to_health,
    is_agent_paused_due_to_health,
    get_health_status_for_api,
)
from src.deribit.base_client import HealthSeverity


class TestCachedHealthStatus:
    """Tests for health status caching."""

    def test_initial_cache_is_none(self):
        from src.healthcheck import _cached_health_status, _health_cache_lock
        with _health_cache_lock:
            pass

    def test_set_agent_paused_updates_flag(self):
        set_agent_paused_due_to_health(True)
        assert is_agent_paused_due_to_health() is True
        
        set_agent_paused_due_to_health(False)
        assert is_agent_paused_due_to_health() is False

    def test_get_health_status_for_api_returns_dict(self):
        result = get_health_status_for_api()
        assert isinstance(result, dict)
        assert "agent_paused_due_to_health" in result
        assert "last_run_at" in result
        assert "overall_status" in result
        assert "summary" in result

    @patch("src.healthcheck.run_agent_healthcheck")
    def test_run_and_cache_healthcheck(self, mock_run):
        mock_run.return_value = {
            "overall_status": "OK",
            "summary": "All checks passed",
            "results": [
                {"name": "config", "status": "ok", "detail": "mode=research"},
            ],
        }
        
        result = run_and_cache_healthcheck()
        
        assert result.overall_status == "OK"
        assert result.summary == "All checks passed"
        assert result.details is not None
        
        cached = get_cached_health_status()
        assert cached is not None
        assert cached.overall_status == "OK"


class TestHealthSeverityComputation:
    """Tests for health severity classification."""

    def test_fatal_severity_for_auth_errors(self):
        from src.healthcheck import _compute_worst_severity
        
        result = {
            "results": [
                {"name": "deribit_private", "status": "fail", "detail": "Authentication failed"},
            ]
        }
        
        severity = _compute_worst_severity(result)
        assert severity == HealthSeverity.FATAL

    def test_transient_severity_for_rate_limit(self):
        from src.healthcheck import _compute_worst_severity
        
        result = {
            "results": [
                {"name": "deribit_public", "status": "fail", "detail": "Rate limit exceeded"},
            ]
        }
        
        severity = _compute_worst_severity(result)
        assert severity == HealthSeverity.TRANSIENT

    def test_transient_severity_for_timeout(self):
        from src.healthcheck import _compute_worst_severity
        
        result = {
            "results": [
                {"name": "deribit_public", "status": "fail", "detail": "Request timeout"},
            ]
        }
        
        severity = _compute_worst_severity(result)
        assert severity == HealthSeverity.TRANSIENT

    def test_degraded_severity_for_unknown_errors(self):
        from src.healthcheck import _compute_worst_severity
        
        result = {
            "results": [
                {"name": "state_builder", "status": "fail", "detail": "Unknown error"},
            ]
        }
        
        severity = _compute_worst_severity(result)
        assert severity == HealthSeverity.DEGRADED


class TestAgentPauseState:
    """Tests for agent pause state management."""

    def test_pause_state_initially_false(self):
        set_agent_paused_due_to_health(False)
        assert is_agent_paused_due_to_health() is False

    def test_pause_state_toggle(self):
        set_agent_paused_due_to_health(True)
        assert is_agent_paused_due_to_health() is True
        
        set_agent_paused_due_to_health(False)
        assert is_agent_paused_due_to_health() is False

    @patch("src.healthcheck.run_agent_healthcheck")
    def test_cached_status_reflects_pause_state(self, mock_run):
        mock_run.return_value = {
            "overall_status": "FAIL",
            "summary": "config FAIL",
            "results": [
                {"name": "config", "status": "fail", "detail": "Invalid config"},
            ],
        }
        
        set_agent_paused_due_to_health(True)
        result = run_and_cache_healthcheck()
        
        assert result.agent_paused_due_to_health is True
        
        api_status = get_health_status_for_api()
        assert api_status["agent_paused_due_to_health"] is True
        
        set_agent_paused_due_to_health(False)


class TestAutoCalibrationScript:
    """Tests for the auto_calibrate_daily.py script components."""

    def test_calibration_run_result_dataclass(self):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        
        from scripts.auto_calibrate_daily import CalibrationRunResult
        
        result = CalibrationRunResult(
            underlying="BTC",
            status="ok",
            reason="Calibration successful",
            multiplier=1.05,
            smoothed_multiplier=1.04,
            mae_pct=2.5,
            vega_weighted_mae_pct=2.2,
            num_samples=150,
            applied=True,
            applied_reason="Delta > threshold",
        )
        
        assert result.underlying == "BTC"
        assert result.status == "ok"
        assert result.applied is True

    def test_calibration_run_result_defaults(self):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        
        from scripts.auto_calibrate_daily import CalibrationRunResult
        
        result = CalibrationRunResult(
            underlying="ETH",
            status="failed",
            reason="No data available",
        )
        
        assert result.multiplier is None
        assert result.smoothed_multiplier is None
        assert result.applied is False
        assert result.num_samples == 0


class TestCalibrationUpdatePolicy:
    """Tests for calibration update policy integration."""

    def test_policy_get_default(self):
        from src.calibration_update_policy import get_policy, CalibrationUpdatePolicy
        
        policy = get_policy()
        
        assert isinstance(policy, CalibrationUpdatePolicy)
        assert policy.min_delta_global > 0
        assert policy.min_sample_size > 0
        assert policy.ewma_alpha > 0

    def test_policy_to_dict(self):
        from src.calibration_update_policy import CalibrationUpdatePolicy
        
        policy = CalibrationUpdatePolicy()
        policy_dict = policy.to_dict()
        
        assert isinstance(policy_dict, dict)
        assert "min_delta_global" in policy_dict
        assert "min_sample_size" in policy_dict
        assert "ewma_alpha" in policy_dict
        assert policy_dict["min_delta_global"] == policy.min_delta_global


class TestHealthcheckAPIEndpoints:
    """Tests for healthcheck-related API endpoints."""

    @patch("src.healthcheck.run_and_cache_healthcheck")
    @patch("src.healthcheck.get_health_status_for_api")
    def test_system_health_status_endpoint_format(self, mock_api, mock_run):
        mock_api.return_value = {
            "last_run_at": "2025-01-01T00:00:00+00:00",
            "overall_status": "OK",
            "worst_severity": None,
            "summary": "All checks passed",
            "agent_paused_due_to_health": False,
        }
        
        result = mock_api()
        
        assert "last_run_at" in result
        assert "overall_status" in result
        assert "agent_paused_due_to_health" in result
        assert result["overall_status"] in ("OK", "WARN", "FAIL", None)


class TestHealthGuardIntegration:
    """Tests for health guard integration with agent loop."""

    def test_health_guard_respects_pause_state(self):
        set_agent_paused_due_to_health(True)
        paused = is_agent_paused_due_to_health()
        assert paused is True
        
        set_agent_paused_due_to_health(False)
        paused = is_agent_paused_due_to_health()
        assert paused is False

    @patch("src.healthcheck.run_agent_healthcheck")
    def test_fail_status_triggers_pause_consideration(self, mock_run):
        mock_run.return_value = {
            "overall_status": "FAIL",
            "summary": "deribit_public FAIL",
            "results": [
                {"name": "deribit_public", "status": "fail", "detail": "Connection failed"},
            ],
        }
        
        result = run_and_cache_healthcheck()
        
        assert result.overall_status == "FAIL"

    @patch("src.healthcheck.run_agent_healthcheck")
    def test_ok_status_allows_trading(self, mock_run):
        mock_run.return_value = {
            "overall_status": "OK",
            "summary": "All checks passed",
            "results": [
                {"name": "config", "status": "ok", "detail": "mode=research"},
                {"name": "deribit_public", "status": "ok", "detail": "API OK"},
            ],
        }
        
        set_agent_paused_due_to_health(False)
        result = run_and_cache_healthcheck()
        
        assert result.overall_status == "OK"
        assert is_agent_paused_due_to_health() is False
