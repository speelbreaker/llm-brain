"""Unit tests for PR Supervisor."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.supervisor.config import SupervisorSettings
from src.supervisor.github import format_pr_comment, parse_webhook_payload, verify_signature
from src.supervisor.models import (
    ArbiterDecision,
    CheckResult,
    DiffStats,
    JobStatus,
    SupervisorJob,
    VerificationReport,
)
from src.supervisor.runner import VerificationRunner


class TestWebhookSignature:
    """Tests for GitHub webhook signature verification."""
    
    def test_verify_signature_valid(self):
        payload = b'{"action": "opened"}'
        secret = "test_secret"
        import hashlib
        import hmac
        signature = "sha256=" + hmac.new(
            secret.encode(), payload, hashlib.sha256
        ).hexdigest()
        
        assert verify_signature(payload, signature, secret) is True
    
    def test_verify_signature_invalid(self):
        payload = b'{"action": "opened"}'
        secret = "test_secret"
        
        assert verify_signature(payload, "sha256=invalid", secret) is False
    
    def test_verify_signature_missing_prefix(self):
        payload = b'{"action": "opened"}'
        secret = "test_secret"
        
        assert verify_signature(payload, "invalid_signature", secret) is False
    
    def test_verify_signature_empty(self):
        payload = b'{"action": "opened"}'
        secret = "test_secret"
        
        assert verify_signature(payload, "", secret) is False


class TestWebhookPayloadParsing:
    """Tests for webhook payload parsing."""
    
    def test_parse_pr_opened_payload(self):
        payload = {
            "action": "opened",
            "pull_request": {
                "number": 42,
                "html_url": "https://github.com/owner/repo/pull/42",
                "head": {
                    "sha": "abc123def456",
                    "ref": "feature-branch",
                    "repo": {
                        "full_name": "owner/repo",
                        "fork": False,
                    }
                },
                "base": {
                    "ref": "main",
                }
            },
            "repository": {
                "full_name": "owner/repo",
            },
            "sender": {
                "login": "developer",
            }
        }
        
        result = parse_webhook_payload(payload)
        
        assert result is not None
        assert result.action == "opened"
        assert result.pr_number == 42
        assert result.head_sha == "abc123def456"
        assert result.head_ref == "feature-branch"
        assert result.base_ref == "main"
        assert result.is_fork is False
        assert result.sender == "developer"
    
    def test_parse_fork_pr_payload(self):
        payload = {
            "action": "opened",
            "pull_request": {
                "number": 42,
                "html_url": "https://github.com/owner/repo/pull/42",
                "head": {
                    "sha": "abc123def456",
                    "ref": "feature-branch",
                    "repo": {
                        "full_name": "forker/repo",
                        "fork": True,
                    }
                },
                "base": {
                    "ref": "main",
                }
            },
            "repository": {
                "full_name": "owner/repo",
            },
            "sender": {
                "login": "forker",
            }
        }
        
        result = parse_webhook_payload(payload)
        
        assert result is not None
        assert result.is_fork is True
    
    def test_parse_invalid_payload(self):
        payload = {"action": "opened"}
        
        result = parse_webhook_payload(payload)
        assert result is None


class TestPRCommentFormatting:
    """Tests for PR comment formatting."""
    
    def test_format_basic_comment(self):
        checks = [
            {"command": "pytest", "passed": True, "duration_seconds": 5.2},
            {"command": "ruff check .", "passed": False, "duration_seconds": 1.1},
        ]
        
        comment = format_pr_comment(
            run_number=1,
            commit_sha="abc123def456789",
            checks=checks,
        )
        
        assert "## 🤖 Supervisor Run #1" in comment
        assert "`abc123de`" in comment
        assert "✅ Pass" in comment
        assert "❌ Fail" in comment
        assert "pytest" in comment
    
    def test_format_comment_with_failure_summary(self):
        checks = [
            {"command": "pytest", "passed": False, "duration_seconds": 10.0},
        ]
        
        comment = format_pr_comment(
            run_number=2,
            commit_sha="abc123",
            checks=checks,
            failure_summary="AssertionError: expected 1, got 2",
        )
        
        assert "### Failure Summary" in comment
        assert "AssertionError" in comment
    
    def test_format_comment_with_arbiter_decision(self):
        checks = [
            {"command": "pytest", "passed": False, "duration_seconds": 5.0},
        ]
        arbiter = {
            "auto_fix_allowed": True,
            "fix_objectives": ["Fix assertion error", "Update test"],
            "risk_level": "low",
        }
        
        comment = format_pr_comment(
            run_number=1,
            commit_sha="abc123",
            checks=checks,
            arbiter_decision=arbiter,
        )
        
        assert "### Arbiter Decision" in comment
        assert "✅ Yes" in comment
        assert "Fix assertion error" in comment
    
    def test_format_comment_with_final_status(self):
        checks = [
            {"command": "pytest", "passed": True, "duration_seconds": 5.0},
        ]
        
        comment = format_pr_comment(
            run_number=1,
            commit_sha="abc123",
            checks=checks,
            final_status="✅ All checks passed - Ready to merge",
        )
        
        assert "### Final Status" in comment
        assert "Ready to merge" in comment


class TestDiffStats:
    """Tests for diff stats and threshold enforcement."""
    
    def test_within_thresholds_pass(self):
        stats = DiffStats(
            files_changed=5,
            lines_added=100,
            lines_removed=50,
            total_loc_changed=150,
        )
        
        assert stats.within_thresholds(max_files=10, max_loc=300) is True
    
    def test_within_thresholds_files_exceeded(self):
        stats = DiffStats(
            files_changed=15,
            lines_added=100,
            lines_removed=50,
            total_loc_changed=150,
        )
        
        assert stats.within_thresholds(max_files=10, max_loc=300) is False
    
    def test_within_thresholds_loc_exceeded(self):
        stats = DiffStats(
            files_changed=5,
            lines_added=200,
            lines_removed=200,
            total_loc_changed=400,
        )
        
        assert stats.within_thresholds(max_files=10, max_loc=300) is False
    
    def test_within_thresholds_both_exceeded(self):
        stats = DiffStats(
            files_changed=15,
            lines_added=200,
            lines_removed=200,
            total_loc_changed=400,
        )
        
        assert stats.within_thresholds(max_files=10, max_loc=300) is False


class TestVerificationReport:
    """Tests for verification report generation."""
    
    def test_report_all_passed(self):
        checks = [
            CheckResult(command="pytest", exit_code=0, passed=True),
            CheckResult(command="ruff", exit_code=0, passed=True),
        ]
        
        report = VerificationReport(
            commit_sha="abc123",
            checks=checks,
            all_passed=True,
        )
        
        assert report.all_passed is True
        assert len(report.checks) == 2
    
    def test_report_with_failures(self):
        checks = [
            CheckResult(command="pytest", exit_code=1, passed=False, stderr="1 failed"),
            CheckResult(command="ruff", exit_code=0, passed=True),
        ]
        
        report = VerificationReport(
            commit_sha="abc123",
            checks=checks,
            all_passed=False,
            failure_summary="1 test failed",
            failing_tests=["test_example.py::test_one"],
        )
        
        assert report.all_passed is False
        assert len(report.failing_tests) == 1


class TestJobStatus:
    """Tests for job status management."""
    
    def test_job_status_update(self):
        job = SupervisorJob(
            job_id="test-123",
            repo_full_name="owner/repo",
            pr_number=42,
            head_sha="abc123",
            head_ref="feature",
            base_ref="main",
            pr_url="https://github.com/owner/repo/pull/42",
        )
        
        assert job.status == JobStatus.PENDING
        
        old_updated = job.updated_at
        job.update_status(JobStatus.RUNNING)
        
        assert job.status == JobStatus.RUNNING
        assert job.updated_at >= old_updated


class TestArbiterDecision:
    """Tests for arbiter decision model."""
    
    def test_arbiter_allows_fix(self):
        decision = ArbiterDecision(
            auto_fix_allowed=True,
            fix_objectives=["Fix type error", "Add missing import"],
            risk_level="low",
        )
        
        assert decision.auto_fix_allowed is True
        assert len(decision.fix_objectives) == 2
    
    def test_arbiter_denies_fix(self):
        decision = ArbiterDecision(
            auto_fix_allowed=False,
            risk_level="high",
            stop_reason="Security-sensitive changes require human review",
        )
        
        assert decision.auto_fix_allowed is False
        assert decision.stop_reason is not None


class TestSettings:
    """Tests for configuration settings."""
    
    def test_default_settings(self):
        with patch.dict("os.environ", {}, clear=True):
            settings = SupervisorSettings()
            
            assert settings.enabled is False
            assert settings.max_loops == 3
            assert settings.max_files_changed == 10
            assert settings.max_loc_changed == 300
            assert settings.allow_forks is False
    
    def test_get_check_commands(self):
        with patch.dict("os.environ", {
            "CHECK_CMD_1": "pytest",
            "CHECK_CMD_2": "ruff check .",
            "CHECK_CMD_3": "mypy .",
        }):
            settings = SupervisorSettings()
            commands = settings.get_check_commands()
            
            assert len(commands) == 3
            assert "pytest" in commands
            assert "ruff check ." in commands
            assert "mypy ." in commands
    
    def test_get_check_commands_without_optional(self):
        with patch.dict("os.environ", {
            "CHECK_CMD_1": "pytest",
            "CHECK_CMD_2": "",
        }):
            settings = SupervisorSettings()
            commands = settings.get_check_commands()
            
            assert len(commands) == 1
            assert "pytest" in commands
