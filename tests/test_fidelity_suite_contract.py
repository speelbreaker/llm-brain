"""Contract tests for fidelity suite output."""
import pytest
from pathlib import Path

REQUIRED_KEYS = {"underlying", "overall_status", "can_trade", "checks", "summary", "mode"}
VALID_STATUS = {"OK", "WARN", "FAIL"}


class TestFidelitySuiteContract:
    """Verify fidelity suite output conforms to contract."""

    def test_preflight_has_required_keys(self, tmp_path):
        """Preflight output must have all required keys."""
        from src.fidelity.run_suite import run_fidelity_suite_from_cli
        
        result = run_fidelity_suite_from_cli(
            start="2025-01-01",
            end="2025-01-02",
            underlying="BTC",
            mode="preflight",
            out_dir=str(tmp_path),
        )
        
        for key in REQUIRED_KEYS:
            assert key in result, f"Missing required key: {key}"

    def test_overall_status_is_valid(self, tmp_path):
        """overall_status must be OK, WARN, or FAIL."""
        from src.fidelity.run_suite import run_fidelity_suite_from_cli
        
        result = run_fidelity_suite_from_cli(
            start="2025-01-01",
            end="2025-01-02",
            underlying="BTC",
            mode="preflight",
            out_dir=str(tmp_path),
        )
        
        assert result["overall_status"] in VALID_STATUS

    def test_can_trade_is_boolean(self, tmp_path):
        """can_trade must be a boolean."""
        from src.fidelity.run_suite import run_fidelity_suite_from_cli
        
        result = run_fidelity_suite_from_cli(
            start="2025-01-01",
            end="2025-01-02",
            underlying="BTC",
            mode="preflight",
            out_dir=str(tmp_path),
        )
        
        assert isinstance(result["can_trade"], bool)

    def test_fail_closed_when_data_missing(self, tmp_path):
        """Missing data should result in can_trade=False (fail-closed)."""
        from src.fidelity.run_suite import run_fidelity_suite_from_cli
        
        result = run_fidelity_suite_from_cli(
            start="2025-01-01",
            end="2025-01-02",
            underlying="BTC",
            mode="preflight",
            out_dir=str(tmp_path),
        )
        
        data_check = next(
            (c for c in result["checks"] if c["name"] == "data_available"),
            None
        )
        assert data_check is not None
        assert data_check["status"] == "FAIL"
        assert result["can_trade"] is False

    def test_preflight_checks_have_stage_0(self, tmp_path):
        """All preflight checks should be stage 0."""
        from src.fidelity.run_suite import run_fidelity_suite_from_cli
        
        result = run_fidelity_suite_from_cli(
            start="2025-01-01",
            end="2025-01-02",
            underlying="BTC",
            mode="preflight",
            out_dir=str(tmp_path),
        )
        
        for check in result["checks"]:
            assert check.get("stage") == 0

    def test_checks_is_list(self, tmp_path):
        """checks must be a list."""
        from src.fidelity.run_suite import run_fidelity_suite_from_cli
        
        result = run_fidelity_suite_from_cli(
            start="2025-01-01",
            end="2025-01-02",
            underlying="BTC",
            mode="preflight",
            out_dir=str(tmp_path),
        )
        
        assert isinstance(result["checks"], list)
