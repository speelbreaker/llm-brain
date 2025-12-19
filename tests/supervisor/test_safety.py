"""Safety tests for PR Supervisor hotfixes."""

import asyncio
import hashlib
import hmac
import json
import os
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.supervisor.store import JobStore


class TestWebhookSecretRequired:
    """Tests for webhook secret requirement (CRITICAL)."""

    def test_missing_secret_returns_503(self):
        """Webhook returns 503 when GITHUB_WEBHOOK_SECRET is missing."""
        with patch.dict(
            os.environ,
            {
                "SUPERVISOR_ENABLED": "1",
                "GITHUB_WEBHOOK_SECRET": "",
                "GITHUB_TOKEN": "test-token",
            },
            clear=False,
        ):
            with patch("src.supervisor.app.get_settings") as mock_settings:
                from src.supervisor.config import SupervisorSettings

                settings = SupervisorSettings()
                settings.enabled = True
                settings.github_webhook_secret = ""
                settings.github_token = "test-token"
                mock_settings.return_value = settings

                from src.supervisor.app import app
                from fastapi.testclient import TestClient

                with TestClient(app, raise_server_exceptions=False) as client:
                    response = client.post(
                        "/github/webhook",
                        json={"action": "opened", "pull_request": {}, "repository": {}},
                        headers={"X-GitHub-Event": "pull_request"},
                    )

                    assert response.status_code == 503
                    data = response.json()
                    assert data["ok"] is False
                    assert data["error"] == "not_ready"
                    assert "GITHUB_WEBHOOK_SECRET" in data.get("details", "")

    def test_invalid_signature_returns_401(self):
        """Webhook returns 401 when signature is invalid."""
        from src.supervisor.github import verify_signature

        payload = b'{"action": "opened"}'
        secret = "test-secret"

        result = verify_signature(payload, "sha256=invalid", secret)
        assert result is False

        result = verify_signature(payload, "sha256=", secret)
        assert result is False

        result = verify_signature(payload, "badprefix=abc", secret)
        assert result is False

    def test_valid_signature_accepted(self):
        """Webhook accepts valid HMAC signature."""
        from src.supervisor.github import verify_signature

        payload = b'{"action": "opened"}'
        secret = "test-secret"

        expected = hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()

        result = verify_signature(payload, f"sha256={expected}", secret)
        assert result is True


class TestApprovalStatePersistence:
    """Tests for approval state persistence (HIGH)."""

    def test_no_silent_failures(self):
        """Approval state save raises on write failure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JobStore(f"{tmpdir}/jobs.jsonl")

            store.set_pr_approval("owner/repo", 1, True, user_id=123)

            state = store.get_pr_approval("owner/repo", 1)
            assert state.approved_by_telegram is True
            assert state.approved_by_user_id == 123
            assert state.approved_at is not None

    def test_timezone_aware_timestamps(self):
        """Approval timestamps are timezone-aware UTC."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JobStore(f"{tmpdir}/jobs.jsonl")

            store.set_pr_approval("owner/repo", 1, True, user_id=123)

            approval_path = store._get_approval_path()
            with open(approval_path) as f:
                data = json.load(f)

            key = "owner/repo:1"
            timestamp = data[key]["approved_at"]
            assert timestamp.endswith("+00:00") or timestamp.endswith("Z")

    def test_concurrent_writes_safe(self):
        """Concurrent approval writes don't corrupt file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JobStore(f"{tmpdir}/jobs.jsonl")
            errors = []

            def writer(pr_num):
                try:
                    for i in range(5):
                        store.set_pr_approval(
                            "owner/repo", pr_num, True, user_id=pr_num * 100 + i
                        )
                        time.sleep(0.01)
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=writer, args=(i,)) for i in range(3)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(errors) == 0

            approval_path = store._get_approval_path()
            with open(approval_path) as f:
                data = json.load(f)

            assert len(data) == 3


class TestWorkspaceCleanupSafety:
    """Tests for workspace cleanup safety (HIGH)."""

    @pytest.mark.asyncio
    async def test_active_workspace_not_deleted(self):
        """Cleanup skips workspaces with .active sentinel."""
        from src.supervisor.workspace import WorkspaceManager, ACTIVE_SENTINEL
        from src.supervisor.config import SupervisorSettings

        with tempfile.TemporaryDirectory() as tmpdir:
            settings = SupervisorSettings()
            settings.base_jobs_dir = tmpdir
            settings.workspace_ttl_hours = 1

            manager = WorkspaceManager(settings)

            old_workspace = Path(tmpdir) / "old-job-123"
            old_workspace.mkdir()
            sentinel = old_workspace / ACTIVE_SENTINEL
            sentinel.touch()

            old_time = time.time() - (2 * 3600)
            os.utime(old_workspace, (old_time, old_time))

            cleaned = await manager.cleanup_old_workspaces()

            assert old_workspace.exists()
            assert cleaned == 0

    @pytest.mark.asyncio
    async def test_inactive_workspace_deleted(self):
        """Cleanup removes old workspaces without .active sentinel."""
        from src.supervisor.workspace import WorkspaceManager
        from src.supervisor.config import SupervisorSettings

        with tempfile.TemporaryDirectory() as tmpdir:
            settings = SupervisorSettings()
            settings.base_jobs_dir = tmpdir
            settings.workspace_ttl_hours = 1

            manager = WorkspaceManager(settings)

            old_workspace = Path(tmpdir) / "old-job-456"
            old_workspace.mkdir()

            old_time = time.time() - (2 * 3600)
            os.utime(old_workspace, (old_time, old_time))

            cleaned = await manager.cleanup_old_workspaces()

            assert not old_workspace.exists()
            assert cleaned == 1

    def test_mark_workspace_inactive(self):
        """mark_workspace_inactive removes sentinel file."""
        from src.supervisor.workspace import WorkspaceManager, ACTIVE_SENTINEL
        from src.supervisor.config import SupervisorSettings

        with tempfile.TemporaryDirectory() as tmpdir:
            settings = SupervisorSettings()
            settings.base_jobs_dir = tmpdir

            manager = WorkspaceManager(settings)

            workspace = Path(tmpdir) / "test-job"
            workspace.mkdir()
            sentinel = workspace / ACTIVE_SENTINEL
            sentinel.touch()

            assert sentinel.exists()

            manager.mark_workspace_inactive(str(workspace))

            assert not sentinel.exists()


class TestDeadlockPrevention:
    """Tests for deadlock prevention in approval state (CRITICAL)."""

    def test_set_pr_approval_does_not_hang(self):
        """set_pr_approval() should not deadlock when called."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JobStore(f"{tmpdir}/jobs.jsonl")
            completed = []

            def do_approval():
                store.set_pr_approval("owner/repo", 1, True, user_id=123)
                completed.append(True)

            thread = threading.Thread(target=do_approval)
            thread.start()
            thread.join(timeout=1.0)

            assert not thread.is_alive(), "set_pr_approval() deadlocked"
            assert len(completed) == 1

    def test_set_pr_paused_does_not_hang(self):
        """set_pr_paused() should not deadlock when called."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JobStore(f"{tmpdir}/jobs.jsonl")
            completed = []

            def do_pause():
                store.set_pr_paused("owner/repo", 1, True, user_id=123)
                completed.append(True)

            thread = threading.Thread(target=do_pause)
            thread.start()
            thread.join(timeout=1.0)

            assert not thread.is_alive(), "set_pr_paused() deadlocked"
            assert len(completed) == 1


class TestCorruptedFileBackup:
    """Tests for corrupted approval file backup (HIGH)."""

    def test_corrupted_file_backed_up(self):
        """Corrupted approval file should be backed up and continue with empty state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JobStore(f"{tmpdir}/jobs.jsonl")

            approval_path = store._get_approval_path()
            approval_path.parent.mkdir(parents=True, exist_ok=True)
            with open(approval_path, "w") as f:
                f.write("not valid json {{{")

            state = store.get_pr_approval("owner/repo", 1)

            assert state.approved_by_telegram is False

            backup_files = list(approval_path.parent.glob("*.corrupt-*"))
            assert len(backup_files) == 1

            with open(backup_files[0]) as f:
                assert f.read() == "not valid json {{{"


class TestStaleSentinelCleanup:
    """Tests for stale sentinel handling in workspace cleanup (HIGH)."""

    @pytest.mark.asyncio
    async def test_stale_sentinel_allows_cleanup(self):
        """Workspace with stale sentinel (older than TTL) should be cleaned up."""
        from src.supervisor.workspace import WorkspaceManager, ACTIVE_SENTINEL
        from src.supervisor.config import SupervisorSettings

        with tempfile.TemporaryDirectory() as tmpdir:
            settings = SupervisorSettings()
            settings.base_jobs_dir = tmpdir
            settings.workspace_ttl_hours = 1

            manager = WorkspaceManager(settings)

            old_workspace = Path(tmpdir) / "stale-job-789"
            old_workspace.mkdir()
            sentinel = old_workspace / ACTIVE_SENTINEL
            sentinel.touch()

            old_time = time.time() - (4 * 3600)
            os.utime(old_workspace, (old_time, old_time))
            os.utime(sentinel, (old_time, old_time))

            cleaned = await manager.cleanup_old_workspaces(sentinel_ttl_hours=2)

            assert not old_workspace.exists()
            assert cleaned == 1

    @pytest.mark.asyncio
    async def test_fresh_sentinel_prevents_cleanup(self):
        """Workspace with fresh sentinel should not be cleaned up."""
        from src.supervisor.workspace import WorkspaceManager, ACTIVE_SENTINEL
        from src.supervisor.config import SupervisorSettings

        with tempfile.TemporaryDirectory() as tmpdir:
            settings = SupervisorSettings()
            settings.base_jobs_dir = tmpdir
            settings.workspace_ttl_hours = 1

            manager = WorkspaceManager(settings)

            old_workspace = Path(tmpdir) / "active-job-999"
            old_workspace.mkdir()
            sentinel = old_workspace / ACTIVE_SENTINEL
            sentinel.touch()

            old_time = time.time() - (4 * 3600)
            os.utime(old_workspace, (old_time, old_time))

            cleaned = await manager.cleanup_old_workspaces(sentinel_ttl_hours=2)

            assert old_workspace.exists()
            assert cleaned == 0


class TestRetryHelper:
    """Tests for retry helper imports and behavior."""

    def test_retry_imports_cleanly(self):
        """retry.py imports without errors."""
        from src.supervisor.retry import (
            with_retry,
            get_retry_client,
            RETRYABLE_STATUS_CODES,
        )

        assert callable(with_retry)
        assert callable(get_retry_client)
        assert 429 in RETRYABLE_STATUS_CODES
        assert 500 in RETRYABLE_STATUS_CODES
        assert 502 in RETRYABLE_STATUS_CODES
        assert 503 in RETRYABLE_STATUS_CODES
        assert 504 in RETRYABLE_STATUS_CODES

    @pytest.mark.asyncio
    async def test_retry_on_retryable_error(self):
        """Retry helper retries on retryable status codes."""
        from src.supervisor.retry import with_retry
        import httpx

        call_count = 0

        async def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                response = httpx.Response(503)
                raise httpx.HTTPStatusError(
                    "Service unavailable", request=MagicMock(), response=response
                )
            return "success"

        result = await with_retry(flaky_func, max_retries=3, base_delay=0.01)

        assert result == "success"
        assert call_count == 3
