import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI

from src.supervisor.app import run_supervisor_job
from src.supervisor.config import get_settings
from src.supervisor.models import (
    ArbiterDecision,
    CheckResult,
    DiffStats,
    JobStatus,
    SupervisorJob,
    VerificationReport,
)
from src.supervisor.store import JobStore
import src.supervisor.app as app_module


class FakeWorkspaceManager:
    def __init__(self, settings):
        self.settings = settings

    async def cleanup_old_workspaces(self):
        return None

    async def setup_workspace(self, job_id, clone_url, head_sha, head_ref):
        return "/tmp/fake-ws"

    async def get_diff_stats(self, workspace_path):
        return DiffStats(files_changed=0, total_loc_changed=0)

    async def commit_and_push(self, *args, **kwargs):
        return "deadbeef"

    async def cleanup_workspace(self, *args, **kwargs):
        return None


class FakeRunner:
    def __init__(self, settings):
        self.settings = settings
        self.calls = 0

    async def run_checks(self, workspace_path, head_sha):
        self.calls += 1
        if self.calls == 1:
            return VerificationReport(
                commit_sha=head_sha,
                checks=[
                    CheckResult(
                        command="python -m pytest -q",
                        exit_code=0,
                        passed=True,
                        stdout="",
                        stderr="",
                        duration_seconds=0.1,
                    ),
                    CheckResult(
                        command="python -m ruff check .",
                        exit_code=1,
                        passed=False,
                        stdout="F401 unused import",
                        stderr="",
                        duration_seconds=0.1,
                    ),
                ],
                all_passed=False,
                failure_summary="ruff failed",
                failing_tests=[],
            )
        return VerificationReport(
            commit_sha=head_sha,
            checks=[
                CheckResult(
                    command="python -m pytest -q",
                    exit_code=0,
                    passed=True,
                    stdout="",
                    stderr="",
                    duration_seconds=0.1,
                ),
                CheckResult(
                    command="python -m ruff check .",
                    exit_code=0,
                    passed=True,
                    stdout="",
                    stderr="",
                    duration_seconds=0.1,
                ),
            ],
            all_passed=True,
            failure_summary="",
            failing_tests=[],
        )


class FakeDebateSystem:
    def __init__(self, settings):
        self.settings = settings

    async def run_debate(self, **kwargs):
        return ArbiterDecision(
            auto_fix_allowed=True,
            fix_objectives=["fix the boom"],
            risk_level="low",
        )


@pytest.mark.asyncio
async def test_autofix_dry_run_posts_comment_and_skips_push(monkeypatch, tmp_path):
    settings = get_settings()
    settings.enable_codex = True
    settings.autofix_policy = "label"
    settings.autofix_dry_run = True
    settings.autofix_push = False
    settings.base_jobs_dir = str(tmp_path)

    store = JobStore(str(tmp_path / "jobs.jsonl"))

    post_comments = []

    async def fake_post_comment(repo, pr_num, comment):
        post_comments.append(comment)

    github_client = SimpleNamespace(
        get_repo_clone_url=AsyncMock(return_value="https://example.com/repo.git"),
        get_pr_files=AsyncMock(return_value=[{"filename": "src/foo.py"}]),
        get_pr_info=AsyncMock(
            return_value={
                "title": "Test PR",
                "body": "Body",
                "labels": [{"name": settings.autofix_label}],
            }
        ),
        post_pr_comment=fake_post_comment,
    )

    app = FastAPI()
    app.state.settings = settings
    app.state.store = store
    app.state.github_client = github_client
    app.state.ready = True
    app.state.job_queue = asyncio.Queue()
    app.state.telegram_http = None

    monkeypatch.setattr(app_module, "WorkspaceManager", FakeWorkspaceManager)
    monkeypatch.setattr(app_module, "VerificationRunner", FakeRunner)
    monkeypatch.setattr(app_module, "DebateSystem", FakeDebateSystem)

    apply_fix_mock = AsyncMock(return_value=(True, "ok"))
    commit_and_push_mock = AsyncMock(return_value="deadbeef")
    def build_prompt_mock(self, *args, **kwargs):  # noqa: ANN001
        return "PROMPT CONTENT WITH FIX"
    monkeypatch.setattr(app_module.CodexFixer, "apply_fix", apply_fix_mock)
    monkeypatch.setattr(app_module.CodexFixer, "build_fix_prompt", build_prompt_mock)
    monkeypatch.setattr(
        FakeWorkspaceManager,
        "commit_and_push",
        commit_and_push_mock,
    )

    # Stub notifier to no-op
    class NoopNotifier:
        def __init__(self, *args, **kwargs):
            pass

        async def notify_job_start(self, *args, **kwargs):
            return None

        async def notify_checks_result(self, *args, **kwargs):
            return None

        async def notify_arbiter_decision(self, *args, **kwargs):
            return None

        async def notify_final_result(self, *args, **kwargs):
            return None

        async def notify_fix_started(self, *args, **kwargs):
            return None

    monkeypatch.setattr(app_module, "TelegramNotifier", NoopNotifier)

    job = SupervisorJob(
        job_id="job-1",
        repo_full_name="speelbreaker/llm-brain",
        pr_number=1,
        head_sha="abcdef1234567890",
        head_ref="pr-supervisor-smoke",
        base_ref="main",
        pr_url="https://github.com/speelbreaker/llm-brain/pull/1",
        is_fork=False,
    )

    await run_supervisor_job(job, app)

    assert job.status == JobStatus.FIXED
    assert "DRY RUN" in job.final_message
    assert len(job.fix_attempts) == 1
    assert job.fix_attempts[0].committed is False

    apply_fix_mock.assert_awaited()
    commit_and_push_mock.assert_not_awaited()
    assert post_comments, "Expected a GitHub comment to be posted"
    assert any("DRY RUN" in c for c in post_comments)
