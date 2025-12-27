"Unit tests for loop safety guards (T006, T009, T010)."

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from src.supervisor.models import SupervisorJob, DiffStats, JobStatus, JobStage
from src.supervisor.app import run_supervisor_job
from src.supervisor.config import SupervisorSettings

@pytest.mark.anyio
async def test_fix_too_large_halt():
    """Verify job halts if diff thresholds are exceeded (T009)."""
    settings = SupervisorSettings()
    settings.openai_api_key = "sk-test"
    settings.enable_codex = True
    settings.codex_bin = "ls"
    settings.max_files_changed = 5
    settings.max_loc_changed = 100
    
    job = SupervisorJob(
        job_id="test_large_fix",
        repo_full_name="org/repo",
        pr_number=1,
        head_sha="sha1",
        head_ref="main",
        base_ref="main",
        pr_url="url"
    )
    
    app = MagicMock()
    app.state.settings = settings
    app.state.ready = True
    
    # Setup GitHub Client Mock
    github_client_instance = MagicMock()
    app.state.github_client = github_client_instance
    github_client_instance.get_pr_info = AsyncMock(return_value={"title": "title", "labels": [{"name": "autofix-ok"}]})
    github_client_instance.get_pr_files = AsyncMock(return_value=[{"filename": "test.py", "status": "modified"}])
    github_client_instance.get_repo_clone_url = AsyncMock(return_value="https://github.com/org/repo")
    
    # Setup Store Mock
    store_mock = MagicMock()
    app.state.store = store_mock
    approval = MagicMock(paused=False, approved_by_telegram=False)
    store_mock.get_pr_approval.return_value = approval
    
    workspace_manager_instance = MagicMock()
    workspace_manager_instance.setup_workspace = AsyncMock(return_value="/tmp/work")
    workspace_manager_instance.cleanup_old_workspaces = AsyncMock()
    workspace_manager_instance.cleanup_workspace = AsyncMock()
    
    # Mock excessive diff
    large_diff = DiffStats(files_changed=10, total_loc_changed=500)
    workspace_manager_instance.get_diff_stats = AsyncMock(return_value=large_diff)
    
    runner_instance = MagicMock()
    failed_report = MagicMock()
    failed_report.all_passed = False
    failed_report.failing_tests = ["test_a"]
    failed_report.failure_summary = "failure"
    failed_report.checks = []
    runner_instance.run_checks = AsyncMock(return_value=failed_report)
    runner_instance._run_command = AsyncMock(return_value=MagicMock(passed=False))
    
    deterministic_fixer_instance = MagicMock()
    deterministic_fixer_instance.run_fix = AsyncMock(return_value=(True, "fixed"))
    
    debate_system_instance = MagicMock()
    debate_system_instance.run_debate = AsyncMock(return_value=MagicMock(auto_fix_allowed=True, risk_level="low"))
    
    codex_fixer_instance = MagicMock()
    codex_fixer_instance.apply_fix = AsyncMock(return_value=(True, "fixed by codex"))
    
    with patch("src.supervisor.app.WorkspaceManager", return_value=workspace_manager_instance), \
         patch("src.supervisor.app.VerificationRunner", return_value=runner_instance), \
         patch("src.supervisor.app.DeterministicFixer", return_value=deterministic_fixer_instance), \
         patch("src.supervisor.app.DebateSystem", return_value=debate_system_instance), \
         patch("src.supervisor.app.CodexFixer", return_value=codex_fixer_instance), \
         patch("src.supervisor.app.JobStore", return_value=store_mock), \
         patch("src.supervisor.app.TelegramNotifier", return_value=MagicMock()):
        
        await run_supervisor_job(job, app)
    
    assert job.status == JobStatus.NEEDS_HUMAN
    assert job.final_message == "Fix too large"

@pytest.mark.anyio
async def test_job_timeout_check():
    """Verify job halts if MAX_TOTAL_RUNTIME is exceeded (T010)."""
    settings = SupervisorSettings()
    settings.max_total_runtime_seconds = 1 # 1 second
    
    job = SupervisorJob(
        job_id="test_timeout",
        repo_full_name="org/repo",
        pr_number=1,
        head_sha="sha1",
        head_ref="main",
        base_ref="main",
        pr_url="url"
    )
    
    app = MagicMock()
    app.state.settings = settings
    app.state.ready = True
    
    github_client_instance = MagicMock()
    app.state.github_client = github_client_instance
    github_client_instance.get_repo_clone_url = AsyncMock(return_value="url")
    
    app.state.store = MagicMock()
    
    workspace_manager_instance = MagicMock()
    workspace_manager_instance.setup_workspace = AsyncMock(return_value="/tmp/work")
    workspace_manager_instance.cleanup_old_workspaces = AsyncMock()
    workspace_manager_instance.cleanup_workspace = AsyncMock()
    
    runner_instance = MagicMock()
    # Mock long running check to trigger timeout
    async def long_check(*args, **kwargs):
        import asyncio
        await asyncio.sleep(2)
        return MagicMock(all_passed=False)
    
    runner_instance.run_checks = long_check
    
    with patch("src.supervisor.app.WorkspaceManager", return_value=workspace_manager_instance), \
         patch("src.supervisor.app.VerificationRunner", return_value=runner_instance), \
         patch("src.supervisor.app.TelegramNotifier", return_value=MagicMock()):
        
        await run_supervisor_job(job, app)
    
    assert "exceeded" in job.final_message