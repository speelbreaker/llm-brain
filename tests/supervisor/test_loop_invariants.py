"""End-to-end loop invariant test."""

import asyncio
import hashlib
import hmac
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from src.supervisor.app import app as supervisor_app
from src.supervisor.app import run_supervisor_job
from src.supervisor.config import SupervisorSettings
from src.supervisor.loop.fixers import FixResult
from src.supervisor.loop.types import FixPlan, LoopDecision, SkepticReport
from src.supervisor.models import CheckResult, DiffStats, JobStage, JobStatus, VerificationReport
from src.supervisor.store import JobStore


def make_signature(payload: bytes, secret: str) -> str:
    """Create a valid HMAC SHA-256 signature."""
    return "sha256=" + hmac.new(
        secret.encode("utf-8"),
        payload,
        hashlib.sha256,
    ).hexdigest()


class FakeNotifier:
    def __init__(self, *_args, **_kwargs):
        pass

    async def notify_job_start(self, *_args, **_kwargs):
        return None

    async def notify_checks_result(self, *_args, **_kwargs):
        return None

    async def notify_arbiter_decision(self, *_args, **_kwargs):
        return None

    async def notify_fix_started(self, *_args, **_kwargs):
        return None

    async def notify_fix_pushed(self, *_args, **_kwargs):
        return None

    async def notify_final_result(self, *_args, **_kwargs):
        return None


class FakeWorkspaceManager:
    def __init__(self, workspace_path: str):
        self.workspace_path = workspace_path
        self.commit_and_push = AsyncMock(return_value=None)

    async def cleanup_old_workspaces(self):
        return None

    async def setup_workspace(self, **_kwargs):
        return self.workspace_path

    async def get_diff_stats(self, _workspace_path: str) -> DiffStats:
        return DiffStats(files_changed=1, lines_added=1, lines_removed=0, total_loc_changed=1)

    async def cleanup_workspace(self, *_args, **_kwargs):
        return None


class FakeVerificationRunner:
    def __init__(self, secret: str):
        self.secret = secret
        self.calls = 0

    async def run_checks(self, _workspace_path: str, head_sha: str) -> VerificationReport:
        self.calls += 1
        if self.calls == 1:
            return VerificationReport(
                commit_sha=head_sha,
                checks=[
                    CheckResult(
                        command="python -m ruff check .",
                        exit_code=1,
                        passed=False,
                        stdout=f"lint fail {self.secret}",
                        stderr="",
                    )
                ],
                all_passed=False,
                failure_summary=f"lint fail {self.secret}",
                failing_tests=[],
            )
        return VerificationReport(
            commit_sha=head_sha,
            checks=[
                CheckResult(
                    command="python -m ruff check .",
                    exit_code=0,
                    passed=True,
                    stdout="",
                    stderr="",
                )
            ],
            all_passed=True,
            failure_summary="",
            failing_tests=[],
        )


class FakeCodexFixer:
    async def apply_fix(self, *_args, **_kwargs):
        raise AssertionError("Codex fixer should not be invoked in lint-only path")

    def build_fix_prompt(self, *_args, **_kwargs):
        return "prompt"


def test_loop_invariants_lint_only(tmp_path, monkeypatch):
    settings = SupervisorSettings()
    settings.enabled = True
    settings.github_webhook_secret = "test_secret"
    settings.github_token = "ghp_testsecret1234567890"
    settings.autofix_push = False
    settings.enable_codex = False
    settings.telegram_enabled = False
    settings.base_jobs_dir = str(tmp_path)

    comment_bodies: list[str] = []

    github_client = SimpleNamespace(
        get_repo_clone_url=AsyncMock(return_value="https://example.com/repo.git"),
        get_pr_files=AsyncMock(return_value=[{"filename": "sample.py"}]),
        get_pr_info=AsyncMock(return_value={"labels": []}),
        post_pr_comment=AsyncMock(side_effect=lambda repo, pr_number, body: comment_bodies.append(body) or {"id": 1}),
        update_pr_comment=AsyncMock(return_value={"id": 1}),
    )

    fake_workspace = FakeWorkspaceManager(str(tmp_path))
    fake_runner = FakeVerificationRunner(settings.github_token)

    async def fake_apply_fix_plan(*_args, **_kwargs):
        return FixResult(applied=True, fixer="ruff_fix", changed_files=["sample.py"], notes=["ok"])

    def fake_propose_fix_plan(*_args, **_kwargs):
        return FixPlan(category="lint_only", objectives=[], approach="", estimated_risk="low")

    def fake_review_fix_plan(*_args, **_kwargs):
        return SkepticReport(risk_level="low", blockers=[], warnings=[])

    def fake_arbitrate(*_args, **_kwargs):
        return LoopDecision(decision="dry_run", reason="ok", fix_objectives=[], allowed_to_modify=[], risk_level="low")

    import src.supervisor.app as app_module

    monkeypatch.setattr(app_module, "WorkspaceManager", lambda _settings: fake_workspace)
    monkeypatch.setattr(app_module, "VerificationRunner", lambda _settings: fake_runner)
    monkeypatch.setattr(app_module, "CodexFixer", lambda _settings: FakeCodexFixer())
    monkeypatch.setattr(app_module, "TelegramNotifier", FakeNotifier)
    monkeypatch.setattr(app_module, "apply_fix_plan", fake_apply_fix_plan)
    monkeypatch.setattr(app_module, "propose_fix_plan", fake_propose_fix_plan)
    monkeypatch.setattr(app_module, "review_fix_plan", fake_review_fix_plan)
    monkeypatch.setattr(app_module, "arbitrate", fake_arbitrate)

    store = JobStore(str(tmp_path / "job_history.jsonl"))
    store.save = store._save_job_sync

    supervisor_app.state.ready = True
    supervisor_app.state.startup_errors = []
    supervisor_app.state.settings = settings
    supervisor_app.state.store = store
    supervisor_app.state.github_client = github_client
    supervisor_app.state.telegram_http = None
    supervisor_app.state.job_queue = SimpleNamespace(put=AsyncMock())

    payload = {
        "action": "opened",
        "pull_request": {
            "number": 42,
            "html_url": "https://github.com/owner/repo/pull/42",
            "head": {
                "sha": "abc123def456",
                "ref": "feature-branch",
                "repo": {"full_name": "owner/repo", "fork": False},
            },
            "base": {"ref": "main"},
        },
        "repository": {"full_name": "owner/repo"},
        "sender": {"login": "testuser"},
    }

    with TestClient(supervisor_app, raise_server_exceptions=False) as client:
        body = json.dumps(payload).encode()
        signature = make_signature(body, settings.github_webhook_secret)
        response = client.post(
            "/github/webhook",
            content=body,
            headers={
                "X-Hub-Signature-256": signature,
                "X-GitHub-Event": "pull_request",
                "Content-Type": "application/json",
            },
        )

        assert response.status_code == 200
        job_id = response.json()["job_id"]

        job = store.get(job_id)
        assert job is not None

        asyncio.run(run_supervisor_job(job, supervisor_app))

        assert job.status == JobStatus.FIXED
        assert "DRY RUN" in job.final_message

        stage_names = [entry.stage for entry in job.stage_history]
        assert stage_names == [
            JobStage.RECEIVED,
            JobStage.ANALYZING,
            JobStage.DEBATING,
            JobStage.FIXING,
            JobStage.VERIFYING,
            JobStage.COMMENTING,
            JobStage.DONE,
        ]

        assert comment_bodies
        comment_body = comment_bodies[-1]
        assert "Check Results" in comment_body
        assert "Arbiter" in comment_body
        assert "Auto-fix" in comment_body

        job_response = client.get(f"/jobs/{job_id}")
        assert job_response.status_code == 200
        job_payload = json.dumps(job_response.json())

        assert settings.github_token not in job_payload
        assert settings.github_token not in comment_body
