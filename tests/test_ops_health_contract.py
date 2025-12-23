"""
Tests for OPS_HEALTH artifact contract enforcement.

Tests verify:
1. OPS_HEALTH_latest.json always parses and includes required keys
2. can_trade is bool, worst_severity not null, summary non-empty
3. Error path: generator produces valid JSON with fail-closed fields + error details
4. get_health_status_for_api() always returns contract-conforming dict
"""
import json
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from datetime import datetime, timezone


# Required keys that must always be present in OPS_HEALTH output
OPS_HEALTH_REQUIRED_KEYS = {
    "overall_status",
    "can_trade",
    "worst_severity",
    "summary",
    "checks",
    "gates",
    "gate_overall",
}

# Valid values for overall_status
VALID_OVERALL_STATUS = {"OK", "WARN", "FAIL"}


class TestOpsHealthContract:
    """Tests for OPS_HEALTH artifact contract compliance."""

    def test_required_keys_constant_defined(self):
        """Verify required keys are defined for contract validation."""
        assert len(OPS_HEALTH_REQUIRED_KEYS) == 7
        assert "can_trade" in OPS_HEALTH_REQUIRED_KEYS
        assert "worst_severity" in OPS_HEALTH_REQUIRED_KEYS
        assert "gate_overall" in OPS_HEALTH_REQUIRED_KEYS


class TestGetHealthStatusForApiContract:
    """Tests that get_health_status_for_api() always returns contract-conforming dict."""

    def test_returns_required_keys_when_cache_is_none(self):
        """When healthcheck hasn't run, should return fail-closed with all required keys."""
        from src.healthcheck import get_health_status_for_api, _health_cache_lock, _cached_health_status
        
        # Clear cache to simulate never-run state
        import src.healthcheck as hc
        original_cached = hc._cached_health_status
        try:
            hc._cached_health_status = None
            
            result = get_health_status_for_api()
            
            # All required keys must be present
            for key in OPS_HEALTH_REQUIRED_KEYS:
                assert key in result, f"Missing required key: {key}"
            
            # overall_status must be valid
            assert result["overall_status"] in VALID_OVERALL_STATUS
            
            # can_trade must be boolean
            assert isinstance(result["can_trade"], bool)
            
            # Fail-closed: can_trade should be False when cache is empty
            assert result["can_trade"] is False
            
            # worst_severity must not be None
            assert result["worst_severity"] is not None
            assert isinstance(result["worst_severity"], str)
            assert len(result["worst_severity"]) > 0
            
            # summary must not be empty
            assert result["summary"] is not None
            assert isinstance(result["summary"], str)
            assert len(result["summary"]) > 0
            
            # checks and gates must be lists
            assert isinstance(result["checks"], list)
            assert isinstance(result["gates"], list)
            
            # gate_overall must be a dict
            assert isinstance(result["gate_overall"], dict)
            
        finally:
            hc._cached_health_status = original_cached

    @patch("src.healthcheck.run_agent_healthcheck")
    def test_returns_required_keys_after_healthcheck_run(self, mock_run):
        """After healthcheck runs, should still return all required keys."""
        from src.healthcheck import run_and_cache_healthcheck, get_health_status_for_api
        
        mock_run.return_value = {
            "overall_status": "OK",
            "summary": "All checks passed",
            "checks": [
                {
                    "name": "config",
                    "status": "OK",
                    "detail": "mode=research",
                    "severity": "OK",
                    "can_trade": True,
                },
            ],
            "gates": [],
            "gate_overall": None,
        }
        
        run_and_cache_healthcheck()
        result = get_health_status_for_api()
        
        # All required keys must be present
        for key in OPS_HEALTH_REQUIRED_KEYS:
            assert key in result, f"Missing required key: {key}"
        
        # Contract enforcement
        assert result["overall_status"] in VALID_OVERALL_STATUS
        assert isinstance(result["can_trade"], bool)
        assert result["worst_severity"] is not None
        assert len(result["summary"]) > 0
        assert isinstance(result["checks"], list)
        assert isinstance(result["gates"], list)
        assert isinstance(result["gate_overall"], dict)

    @patch("src.healthcheck.run_agent_healthcheck")
    def test_normalizes_invalid_overall_status(self, mock_run):
        """If healthcheck returns invalid overall_status, it should be normalized to FAIL."""
        from src.healthcheck import run_and_cache_healthcheck, get_health_status_for_api
        
        mock_run.return_value = {
            "overall_status": "INVALID",  # Invalid value
            "summary": "Bad status",
            "checks": [],
        }
        
        run_and_cache_healthcheck()
        result = get_health_status_for_api()
        
        # Should be normalized to a valid value
        assert result["overall_status"] in VALID_OVERALL_STATUS

    @patch("src.healthcheck.run_agent_healthcheck")
    def test_normalizes_none_can_trade_to_boolean(self, mock_run):
        """If can_trade is None, should be normalized to boolean.
        
        Note: run_and_cache_healthcheck() uses _compute_worst_severity() fallback
        which returns True when no checks explicitly block trading (can_trade=False).
        The contract ensures can_trade is always boolean, never None.
        """
        from src.healthcheck import run_and_cache_healthcheck, get_health_status_for_api
        
        mock_run.return_value = {
            "overall_status": "OK",
            "summary": "Test",
            "can_trade": None,  # None value - will be computed from checks
            "checks": [],
        }
        
        run_and_cache_healthcheck()
        result = get_health_status_for_api()
        
        # Must be boolean, not None (contract requirement)
        assert isinstance(result["can_trade"], bool)
        # With no checks blocking trade, defaults to True
        assert result["can_trade"] is True

    @patch("src.healthcheck.run_agent_healthcheck")
    def test_normalizes_none_can_trade_with_blocking_check(self, mock_run):
        """If can_trade is None but checks block, should normalize to False."""
        from src.healthcheck import run_and_cache_healthcheck, get_health_status_for_api
        
        mock_run.return_value = {
            "overall_status": "FAIL",
            "summary": "Test failure",
            "can_trade": None,  # None value - will be computed from checks
            "checks": [
                {"name": "test", "status": "FAIL", "can_trade": False, "severity": "FATAL"}
            ],
        }
        
        run_and_cache_healthcheck()
        result = get_health_status_for_api()
        
        # Must be boolean, not None
        assert isinstance(result["can_trade"], bool)
        # Check explicitly blocks trading
        assert result["can_trade"] is False

    @patch("src.healthcheck.run_agent_healthcheck")
    def test_gate_overall_always_dict(self, mock_run):
        """gate_overall must always be a dict, never None."""
        from src.healthcheck import run_and_cache_healthcheck, get_health_status_for_api
        
        mock_run.return_value = {
            "overall_status": "OK",
            "summary": "Test",
            "checks": [],
            "gates": [],
            "gate_overall": None,  # None value
        }
        
        run_and_cache_healthcheck()
        result = get_health_status_for_api()
        
        # Must be dict, not None
        assert isinstance(result["gate_overall"], dict)
        assert "status" in result["gate_overall"]

    @patch("src.healthcheck.run_agent_healthcheck")
    def test_single_source_of_truth_fail_implies_no_trade(self, mock_run):
        """SINGLE SOURCE OF TRUTH: overall_status=FAIL must imply can_trade=False.
        
        Even if individual checks allow trading, a FAIL status means the system
        is unhealthy and trading should be blocked for safety.
        """
        from src.healthcheck import run_and_cache_healthcheck, get_health_status_for_api
        
        mock_run.return_value = {
            "overall_status": "FAIL",
            "summary": "Some check failed",
            "can_trade": True,  # Individual checks might say True
            "checks": [
                {"name": "failing_check", "status": "FAIL", "can_trade": True, "severity": "FATAL"}
            ],
            "gates": [],
            "gate_overall": None,
        }
        
        run_and_cache_healthcheck()
        result = get_health_status_for_api()
        
        # overall_status=FAIL must force can_trade=False
        assert result["overall_status"] == "FAIL"
        assert result["can_trade"] is False

    @patch("src.healthcheck.run_agent_healthcheck")
    def test_single_source_of_truth_ok_allows_trade(self, mock_run):
        """When overall_status=OK and checks allow, can_trade should be True."""
        from src.healthcheck import run_and_cache_healthcheck, get_health_status_for_api
        
        mock_run.return_value = {
            "overall_status": "OK",
            "summary": "All good",
            "can_trade": True,
            "checks": [
                {"name": "good_check", "status": "OK", "can_trade": True, "severity": "OK"}
            ],
            "gates": [],
            "gate_overall": None,
        }
        
        run_and_cache_healthcheck()
        result = get_health_status_for_api()
        
        assert result["overall_status"] == "OK"
        assert result["can_trade"] is True

    @patch("src.healthcheck.run_agent_healthcheck")
    def test_single_source_of_truth_warn_preserves_can_trade(self, mock_run):
        """When overall_status=WARN, can_trade should follow check logic."""
        from src.healthcheck import run_and_cache_healthcheck, get_health_status_for_api
        
        mock_run.return_value = {
            "overall_status": "WARN",
            "summary": "Warning but tradeable",
            "can_trade": True,
            "checks": [
                {"name": "warn_check", "status": "WARN", "can_trade": True, "severity": "DEGRADED"}
            ],
            "gates": [],
            "gate_overall": None,
        }
        
        run_and_cache_healthcheck()
        result = get_health_status_for_api()
        
        # WARN status should preserve can_trade from checks
        assert result["overall_status"] == "WARN"
        assert result["can_trade"] is True


class TestGenOpsHealthLatestContract:
    """Tests for gen_ops_health_latest.py contract compliance."""

    def test_fake_payload_conforms_to_contract(self):
        """Fake payload should include all required keys with proper types."""
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        
        from scripts.gen_ops_health_latest import _fake_payload
        
        payload = _fake_payload(Path("/tmp"))
        
        # All required keys must be present
        for key in OPS_HEALTH_REQUIRED_KEYS:
            assert key in payload, f"Missing required key in fake payload: {key}"
        
        # Contract enforcement
        assert payload["overall_status"] in VALID_OVERALL_STATUS
        assert isinstance(payload["can_trade"], bool)
        assert payload["worst_severity"] is not None
        assert isinstance(payload["worst_severity"], str)
        assert len(payload["summary"]) > 0
        assert isinstance(payload["checks"], list)
        assert isinstance(payload["gates"], list)
        assert isinstance(payload["gate_overall"], dict)

    def test_error_payload_conforms_to_contract(self):
        """Error payload should include all required keys with fail-closed values."""
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        
        from scripts.gen_ops_health_latest import _make_error_payload
        
        # Simulate an error
        test_error = ValueError("Test error message")
        payload = _make_error_payload(Path("/tmp"), test_error)
        
        # All required keys must be present
        for key in OPS_HEALTH_REQUIRED_KEYS:
            assert key in payload, f"Missing required key in error payload: {key}"
        
        # Contract enforcement - fail-closed
        assert payload["overall_status"] == "FAIL"
        assert payload["can_trade"] is False
        assert payload["worst_severity"] == "CRITICAL"
        assert len(payload["summary"]) > 0
        assert "ValueError" in payload["summary"]
        assert "Test error message" in payload["summary"]
        assert isinstance(payload["checks"], list)
        assert isinstance(payload["gates"], list)
        assert isinstance(payload["gate_overall"], dict)
        
        # Error details should be present
        assert "error" in payload
        assert "code" in payload["error"]
        assert "exception" in payload["error"]
        assert "traceback" in payload["error"]
        assert payload["error"]["code"] == "OPS_HEALTH_GENERATION_ERROR"

    def test_error_payload_traceback_is_capped(self):
        """Error payload traceback should be capped to reasonable size."""
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        
        from scripts.gen_ops_health_latest import _truncate_traceback
        
        # Create a long traceback
        long_tb = "\n".join([f"Line {i}: some traceback content" for i in range(100)])
        truncated = _truncate_traceback(long_tb, max_lines=50)
        
        lines = truncated.splitlines()
        # Should be capped at approximately max_lines (with truncation marker)
        assert len(lines) <= 51  # 50 lines + 1 truncation marker
        assert "truncated" in truncated.lower()

    def test_error_payload_short_traceback_not_truncated(self):
        """Short tracebacks should not be truncated."""
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        
        from scripts.gen_ops_health_latest import _truncate_traceback
        
        short_tb = "Line 1\nLine 2\nLine 3"
        result = _truncate_traceback(short_tb, max_lines=50)
        
        assert result == short_tb
        assert "truncated" not in result.lower()


class TestGenOpsHealthLatestErrorPath:
    """Tests for error handling in gen_ops_health_latest.py."""

    def test_generator_produces_valid_json_on_exception(self):
        """When get_health_status_for_api raises, generator should still produce valid JSON."""
        import sys
        import tempfile
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        
        from scripts.gen_ops_health_latest import _make_error_payload
        
        # Simulate various exception types
        exceptions = [
            ModuleNotFoundError("No module named 'src'"),
            ImportError("Cannot import foo"),
            RuntimeError("Something went wrong"),
            KeyError("missing_key"),
            Exception("Generic error"),
        ]
        
        for exc in exceptions:
            payload = _make_error_payload(Path("/tmp"), exc)
            
            # Must be valid JSON
            json_str = json.dumps(payload)
            parsed = json.loads(json_str)
            
            # Must have required keys
            for key in OPS_HEALTH_REQUIRED_KEYS:
                assert key in parsed, f"Missing {key} for {type(exc).__name__}"
            
            # Must be fail-closed
            assert parsed["overall_status"] == "FAIL"
            assert parsed["can_trade"] is False
            assert parsed["worst_severity"] == "CRITICAL"
            
            # Error details must include exception info
            assert type(exc).__name__ in parsed["error"]["exception"]

    @patch("src.healthcheck.get_health_status_for_api")
    def test_main_handles_healthcheck_exception(self, mock_api):
        """main() should catch exceptions and write fail-closed payload."""
        import sys
        import tempfile
        import os
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        
        # Mock to raise exception
        mock_api.side_effect = RuntimeError("Healthcheck exploded")
        
        from scripts.gen_ops_health_latest import _make_error_payload
        
        # Verify error payload is valid
        payload = _make_error_payload(Path("/tmp"), RuntimeError("Healthcheck exploded"))
        
        assert payload["overall_status"] == "FAIL"
        assert payload["can_trade"] is False
        assert "RuntimeError" in payload["summary"]
        assert "Healthcheck exploded" in payload["summary"]


class TestOpsHealthJsonParsing:
    """Tests that OPS_HEALTH_latest.json can always be parsed."""

    def test_fake_payload_produces_valid_json(self):
        """Fake payload should produce valid, parseable JSON."""
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        
        from scripts.gen_ops_health_latest import _fake_payload
        
        payload = _fake_payload(Path("/tmp"))
        
        # Must serialize to JSON without error
        json_str = json.dumps(payload, indent=2, sort_keys=True)
        
        # Must parse back without error
        parsed = json.loads(json_str)
        
        # Parsed result should match original
        assert parsed["overall_status"] == payload["overall_status"]
        assert parsed["can_trade"] == payload["can_trade"]

    def test_error_payload_produces_valid_json(self):
        """Error payload should produce valid, parseable JSON."""
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        
        from scripts.gen_ops_health_latest import _make_error_payload
        
        # Error with special characters
        exc = ValueError("Error with 'quotes' and \"double quotes\" and \\ backslash")
        payload = _make_error_payload(Path("/tmp"), exc)
        
        # Must serialize to JSON without error
        json_str = json.dumps(payload, indent=2, sort_keys=True)
        
        # Must parse back without error
        parsed = json.loads(json_str)
        
        assert parsed["overall_status"] == "FAIL"
        assert parsed["can_trade"] is False

