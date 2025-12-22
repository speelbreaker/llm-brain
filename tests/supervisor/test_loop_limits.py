"""Tests for loop stop conditions."""

from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
import sys

import pytest
from fastapi import FastAPI

from src.supervisor.app import run_supervisor_job
from src.supervisor.config import SupervisorSettings
from src.supervisor.loop.fixers import FixResult
from src.supervisor.loop.types import FixPlan, LoopDecision, SkepticReport
from src.supervisor.models import CheckResult, DiffStats, JobStatus, SupervisorJob, VerificationReport
from src.supervisor.store import JobStore


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
    def __init__(self, report: VerificationReport):
        self.report = report

    async def run_checks(self, *_args, **_kwargs) -> VerificationReport:
        return self.report


class FakeCodexFixer:
    async def apply_fix(self, *_args, **_kwargs):
        return False, ""

    def build_fix_prompt(self, *_args, **_kwargs):
        return "prompt"


@pytest.mark.asyncio
async def test_loop_halts_at_max_fix_attempts(tmp_path, monkeypatch):
    settings = SupervisorSettings()
    settings.enable_codex = True
    settings.autofix_push = False
    settings.max_loops = 2
    settings.max_fix_attempts = 1
    settings.max_total_runtime_seconds = 3600
    settings.fix_backoff_base_seconds = 0
    settings.fix_backoff_factor = 1
    settings.fix_backoff_max_seconds = 0
    settings.codex_bin = sys.executable

    verification = VerificationReport(
        commit_sha="deadbeef",
        checks=[
            CheckResult(
                command="pytest",
                exit_code=1,
                passed=False,
                stdout="fail",
                stderr="",
            )
        ],
        all_passed=False,
        failure_summary="lint fail",
        failing_tests=[],
    )

    fake_workspace = FakeWorkspaceManager(str(tmp_path))
    fake_runner = FakeVerificationRunner(verification)

    async def fake_apply_fix_plan(*_args, **_kwargs):
        return FixResult(applied=False, fixer="noop", changed_files=[], notes=[])

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

    github_client = SimpleNamespace(
        get_repo_clone_url=AsyncMock(return_value="https://example.com/repo.git"),
        get_pr_files=AsyncMock(return_value=[{"filename": "sample.py"}]),
        get_pr_info=AsyncMock(return_value={"labels": []}),
        post_pr_comment=AsyncMock(return_value={"id": 1}),
        update_pr_comment=AsyncMock(return_value={"id": 1}),
    )

    app = FastAPI()
    app.state.settings = settings
    app.state.store = JobStore(str(tmp_path / "job_history.jsonl"))
    app.state.store.save = app.state.store._save_job_sync
    app.state.github_client = github_client
    app.state.telegram_http = None

    job = SupervisorJob(
        job_id="job-max-fix",
        repo_full_name="owner/repo",
        pr_number=1,
        head_sha="deadbeef",
        head_ref="main",
        base_ref="main",
        pr_url="https://github.com/owner/repo/pull/1",
    )

    await run_supervisor_job(job, app)

    assert job.status == JobStatus.NEEDS_HUMAN
    assert job.reason_code == "LOOP_LIMIT"
    assert "Max fix attempts" in job.final_message
    assert fake_workspace.commit_and_push.await_count == 0


@pytest.mark.asyncio
async def test_loop_halts_on_timeout(tmp_path, monkeypatch):
    settings = SupervisorSettings()
    settings.enable_codex = False
    settings.autofix_push = False
    settings.max_total_runtime_seconds = 1

    fake_workspace = FakeWorkspaceManager(str(tmp_path))
    fake_runner = FakeVerificationRunner(
        VerificationReport(
            commit_sha="deadbeef",
            checks=[],
            all_passed=False,
            failure_summary="",
            failing_tests=[],
        )
    )

    import src.supervisor.app as app_module

    monkeypatch.setattr(app_module, "WorkspaceManager", lambda _settings: fake_workspace)
    monkeypatch.setattr(app_module, "VerificationRunner", lambda _settings: fake_runner)
    monkeypatch.setattr(app_module, "CodexFixer", lambda _settings: FakeCodexFixer())
    monkeypatch.setattr(app_module, "TelegramNotifier", FakeNotifier)

    github_client = SimpleNamespace(
        get_repo_clone_url=AsyncMock(return_value="https://example.com/repo.git"),
        get_pr_files=AsyncMock(return_value=[]),
        get_pr_info=AsyncMock(return_value={"labels": []}),
        post_pr_comment=AsyncMock(return_value={"id": 1}),
        update_pr_comment=AsyncMock(return_value={"id": 1}),
    )

    app = FastAPI()
    app.state.settings = settings
    app.state.store = JobStore(str(tmp_path / "job_history.jsonl"))
    app.state.store.save = app.state.store._save_job_sync
    app.state.github_client = github_client
    app.state.telegram_http = None

    job = SupervisorJob(
        job_id="job-timeout",
        repo_full_name="owner/repo",
        pr_number=2,
        head_sha="deadbeef",
        head_ref="main",
        base_ref="main",
        pr_url="https://github.com/owner/repo/pull/2",
    )
    job.created_at = datetime.utcnow() - timedelta(seconds=10)

    await run_supervisor_job(job, app)

    assert job.status == JobStatus.NEEDS_HUMAN
    assert job.reason_code == "LOOP_LIMIT"
    assert "Max runtime" in job.final_message
