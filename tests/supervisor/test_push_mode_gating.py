import asyncio
import pytest

from src.supervisor.app import run_supervisor_job
from src.supervisor.config import SupervisorSettings
from src.supervisor.models import (
    ArbiterDecision,
    CheckResult,
    DiffStats,
    JobStatus,
    SupervisorJob,
    VerificationReport,
)
from src.supervisor.store import JobStore
from fastapi import FastAPI


class DummyGithub:
    async def get_repo_clone_url(self, repo):
        return "https://example.com/repo.git"

    async def get_pr_files(self, repo, pr_number):
        return []

    async def get_pr_info(self, repo, pr_number):
        return {
            "title": "Test PR",
            "body": "",
            "labels": [{"name": "autofix-ok"}],
        }

    async def post_pr_comment(self, repo, pr_number, body):
        return {}

    async def update_pr_comment(self, repo, comment_id, body):
        return {}


def _build_settings(push_enabled: bool) -> SupervisorSettings:
    settings = SupervisorSettings()
    settings.enabled = True
    settings.enable_codex = True
    settings.autofix_push = push_enabled
    settings.autofix_dry_run = not push_enabled
    settings.autofix_policy = "label"
    settings.autofix_label = "autofix-ok"
    settings.github_token = "token"
    settings.github_webhook_secret = "secret"
    settings.max_loops = 1
    return settings


def _build_verifications(head_sha: str):
    failing = VerificationReport(
        commit_sha=head_sha,
        checks=[
            CheckResult(
                command="ruff",
                exit_code=1,
                passed=False,
                stdout="lint fail",
                stderr="",
            ),
            CheckResult(
                command="pytest",
                exit_code=0,
                passed=True,
                stdout="ok",
                stderr="",
            ),
        ],
        all_passed=False,
        failure_summary="lint fail",
    )

    passing = VerificationReport(
        commit_sha=head_sha,
        checks=[
            CheckResult(
                command="ruff",
                exit_code=0,
                passed=True,
                stdout="ok",
                stderr="",
            ),
            CheckResult(
                command="pytest",
                exit_code=0,
                passed=True,
                stdout="ok",
                stderr="",
            ),
        ],
        all_passed=True,
        failure_summary="",
    )

    return failing, passing


def _build_app(settings: SupervisorSettings, store_path) -> FastAPI:
    app = FastAPI()
    app.state.settings = settings
    app.state.ready = True
    app.state.startup_errors = []
    app.state.job_queue = asyncio.Queue()
    app.state.store = JobStore(str(store_path))
    app.state.github_client = DummyGithub()
    app.state.telegram_http = None
    return app


@pytest.mark.asyncio
async def test_push_disabled_uses_dry_run(monkeypatch, tmp_path):
    head_sha = "deadbeef"
    settings = _build_settings(push_enabled=False)
    app = _build_app(settings, tmp_path / "jobs.jsonl")

    failing, passing = _build_verifications(head_sha)
    call_count = {"count": 0}

    async def fake_run_checks(self, workspace_path, commit_sha, commands=None):
        call_count["count"] += 1
        return failing if call_count["count"] == 1 else passing

    commit_calls = {"count": 0}

    async def fake_commit_and_push(self, workspace_path, message, branch):
        commit_calls["count"] += 1
        return "pushed-sha"

    async def fake_setup_workspace(self, job_id, repo_url, head_sha, head_ref, base_ref="main", pr_number=None):
        path = tmp_path / "workspace"
        path.mkdir(exist_ok=True)
        return str(path)

    async def fake_cleanup_old(self):
        return 0

    async def fake_get_diff(self, workspace_path):
        return DiffStats(files_changed=1, lines_added=1, lines_removed=0, total_loc_changed=1)

    async def fake_apply_fix(self, workspace_path, arbiter_decision, verification, changed_files):
        return True, "fixed"

    async def fake_run_debate(self, verification, changed_files, pr_title="", pr_body=""):
        return ArbiterDecision(
            auto_fix_allowed=True,
            fix_objectives=["fix lint"],
            risk_level="low",
        )

    monkeypatch.setattr(
        "src.supervisor.app.VerificationRunner.run_checks", fake_run_checks, raising=True
    )
    monkeypatch.setattr(
        "src.supervisor.app.WorkspaceManager.setup_workspace",
        fake_setup_workspace,
        raising=True,
    )
    monkeypatch.setattr(
        "src.supervisor.app.WorkspaceManager.cleanup_old_workspaces",
        fake_cleanup_old,
        raising=True,
    )
    monkeypatch.setattr(
        "src.supervisor.app.WorkspaceManager.get_diff_stats",
        fake_get_diff,
        raising=True,
    )
    monkeypatch.setattr(
        "src.supervisor.app.WorkspaceManager.commit_and_push",
        fake_commit_and_push,
        raising=True,
    )
    monkeypatch.setattr(
        "src.supervisor.app.CodexFixer.apply_fix",
        fake_apply_fix,
        raising=True,
    )
    monkeypatch.setattr(
        "src.supervisor.app.DebateSystem.run_debate",
        fake_run_debate,
        raising=True,
    )

    job = SupervisorJob(
        job_id="job-1",
        repo_full_name="owner/repo",
        pr_number=1,
        head_sha=head_sha,
        head_ref="feature",
        base_ref="main",
        pr_url="https://example.com/pr/1",
        is_fork=False,
    )

    await run_supervisor_job(job, app)

    assert job.status == JobStatus.FIXED
    assert "DRY RUN" in job.final_message or "DRY RUN" in job.final_message.upper()
    assert commit_calls["count"] == 0
    assert job.fix_attempts and job.fix_attempts[0].committed is False


@pytest.mark.asyncio
async def test_push_enabled_invokes_commit(monkeypatch, tmp_path):
    head_sha = "feedface"
    settings = _build_settings(push_enabled=True)
    app = _build_app(settings, tmp_path / "jobs.jsonl")

    failing, passing = _build_verifications(head_sha)
    call_count = {"count": 0}

    async def fake_run_checks(self, workspace_path, commit_sha, commands=None):
        call_count["count"] += 1
        return failing if call_count["count"] == 1 else passing

    commit_calls = {"count": 0}

    async def fake_commit_and_push(self, workspace_path, message, branch):
        commit_calls["count"] += 1
        return "pushed-sha"

    async def fake_setup_workspace(self, job_id, repo_url, head_sha, head_ref, base_ref="main", pr_number=None):
        path = tmp_path / "workspace2"
        path.mkdir(exist_ok=True)
        return str(path)

    async def fake_cleanup_old(self):
        return 0

    async def fake_get_diff(self, workspace_path):
        return DiffStats(files_changed=1, lines_added=1, lines_removed=0, total_loc_changed=1)

    async def fake_apply_fix(self, workspace_path, arbiter_decision, verification, changed_files):
        return True, "fixed"

    async def fake_run_debate(self, verification, changed_files, pr_title="", pr_body=""):
        return ArbiterDecision(
            auto_fix_allowed=True,
            fix_objectives=["fix lint"],
            risk_level="low",
        )

    monkeypatch.setattr(
        "src.supervisor.app.VerificationRunner.run_checks", fake_run_checks, raising=True
    )
    monkeypatch.setattr(
        "src.supervisor.app.WorkspaceManager.setup_workspace",
        fake_setup_workspace,
        raising=True,
    )
    monkeypatch.setattr(
        "src.supervisor.app.WorkspaceManager.cleanup_old_workspaces",
        fake_cleanup_old,
        raising=True,
    )
    monkeypatch.setattr(
        "src.supervisor.app.WorkspaceManager.get_diff_stats",
        fake_get_diff,
        raising=True,
    )
    monkeypatch.setattr(
        "src.supervisor.app.WorkspaceManager.commit_and_push",
        fake_commit_and_push,
        raising=True,
    )
    monkeypatch.setattr(
        "src.supervisor.app.CodexFixer.apply_fix",
        fake_apply_fix,
        raising=True,
    )
    monkeypatch.setattr(
        "src.supervisor.app.DebateSystem.run_debate",
        fake_run_debate,
        raising=True,
    )

    job = SupervisorJob(
        job_id="job-2",
        repo_full_name="owner/repo",
        pr_number=1,
        head_sha=head_sha,
        head_ref="feature",
        base_ref="main",
        pr_url="https://example.com/pr/1",
        is_fork=False,
    )

    await run_supervisor_job(job, app)

    assert job.status == JobStatus.FIXED
    assert commit_calls["count"] == 1
    assert job.fix_attempts
    assert job.fix_attempts[0].committed is True
    assert job.fix_attempts[0].commit_sha == "pushed-sha"
