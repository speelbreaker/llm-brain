"""FastAPI application for PR Supervisor."""

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel

from .codex_fixer import CodexFixer
from .config import SupervisorSettings, get_settings
from .debate import DebateSystem
from .github import GitHubClient, format_pr_comment, parse_webhook_payload, verify_signature
from .models import (
    ArbiterDecision,
    FixAttempt,
    JobStatus,
    SupervisorJob,
    VerificationReport,
)
from .runner import VerificationRunner
from .store import JobStore
from .telegram_notify import TelegramNotifier
from .workspace import WorkspaceManager

logger = logging.getLogger(__name__)


class HealthResponse(BaseModel):
    ok: bool
    enabled: bool
    version: str = "0.1.0"


class JobResponse(BaseModel):
    job_id: str
    status: str
    message: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.settings = settings
    app.state.store = JobStore(f"{settings.base_jobs_dir}/job_history.jsonl")
    app.state.github_client = None
    if settings.github_token:
        app.state.github_client = GitHubClient(settings.github_token)
    yield
    if app.state.github_client:
        await app.state.github_client.close()


app = FastAPI(
    title="PR Supervisor",
    description="Automated PR verification and auto-fix service",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    settings = get_settings()
    return HealthResponse(ok=True, enabled=settings.enabled)


@app.post("/github/webhook", response_model=JobResponse)
async def github_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    x_hub_signature_256: str = Header(None, alias="X-Hub-Signature-256"),
    x_github_event: str = Header(None, alias="X-GitHub-Event"),
):
    """Handle GitHub PR webhooks."""
    settings: SupervisorSettings = request.app.state.settings
    
    if not settings.enabled:
        return JobResponse(
            job_id="",
            status="disabled",
            message="Supervisor is disabled. Set SUPERVISOR_ENABLED=1 to enable.",
        )
    
    if not settings.github_webhook_secret:
        raise HTTPException(status_code=500, detail="GITHUB_WEBHOOK_SECRET not configured")
    
    body = await request.body()
    
    if not verify_signature(body, x_hub_signature_256 or "", settings.github_webhook_secret):
        raise HTTPException(status_code=401, detail="Invalid signature")
    
    if x_github_event != "pull_request":
        return JobResponse(
            job_id="",
            status="ignored",
            message=f"Ignoring event type: {x_github_event}",
        )
    
    data = await request.json()
    payload = parse_webhook_payload(data)
    
    if not payload:
        raise HTTPException(status_code=400, detail="Failed to parse webhook payload")
    
    if payload.action not in ("opened", "reopened", "synchronize"):
        return JobResponse(
            job_id="",
            status="ignored",
            message=f"Ignoring action: {payload.action}",
        )
    
    if payload.is_fork and not settings.allow_forks:
        return JobResponse(
            job_id="",
            status="skipped",
            message="PR from fork - skipped (SUPERVISOR_ALLOW_FORKS=0)",
        )
    
    store: JobStore = request.app.state.store
    existing = store.get_by_sha(payload.repo_full_name, payload.pr_number, payload.head_sha)
    if existing:
        return JobResponse(
            job_id=existing.job_id,
            status="duplicate",
            message=f"Job already exists for SHA {payload.head_sha[:8]}",
        )
    
    job_id = f"pr-{payload.pr_number}-{payload.head_sha[:8]}-{uuid.uuid4().hex[:6]}"
    job = SupervisorJob(
        job_id=job_id,
        repo_full_name=payload.repo_full_name,
        pr_number=payload.pr_number,
        head_sha=payload.head_sha,
        head_ref=payload.head_ref,
        base_ref=payload.base_ref,
        pr_url=payload.pr_url,
        is_fork=payload.is_fork,
    )
    
    store.save(job)
    
    background_tasks.add_task(run_supervisor_job, job, request.app)
    
    return JobResponse(
        job_id=job_id,
        status="queued",
        message=f"Job queued for PR #{payload.pr_number}",
    )


@app.get("/jobs")
async def list_jobs(request: Request, limit: int = 50):
    """List recent supervisor jobs."""
    store: JobStore = request.app.state.store
    jobs = store.list_recent(limit)
    return {"jobs": [j.model_dump() for j in jobs]}


@app.get("/jobs/{job_id}")
async def get_job(request: Request, job_id: str):
    """Get a specific job by ID."""
    store: JobStore = request.app.state.store
    job = store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.model_dump()


async def run_supervisor_job(job: SupervisorJob, app: FastAPI) -> None:
    """Main supervisor job orchestrator."""
    settings: SupervisorSettings = app.state.settings
    store: JobStore = app.state.store
    github_client: GitHubClient = app.state.github_client
    
    notifier = TelegramNotifier(settings)
    workspace_manager = WorkspaceManager(settings)
    runner = VerificationRunner(settings)
    debate_system = DebateSystem(settings)
    codex_fixer = CodexFixer(settings)
    
    try:
        job.update_status(JobStatus.RUNNING)
        store.save(job)
        
        await notifier.notify_job_start(job)
        
        clone_url = await github_client.get_repo_clone_url(job.repo_full_name)
        workspace_path = await workspace_manager.setup_workspace(
            job.job_id,
            clone_url,
            job.head_sha,
            job.head_ref,
        )
        job.workspace_path = workspace_path
        store.save(job)
        
        verification = await runner.run_checks(workspace_path, job.head_sha)
        job.verification = verification
        
        run_number = store.get_run_count(job.repo_full_name, job.pr_number)
        
        if verification.all_passed:
            job.update_status(JobStatus.CHECKS_PASSED)
            job.final_message = "All checks passed"
            store.save(job)
            
            comment = format_pr_comment(
                run_number=run_number,
                commit_sha=job.head_sha,
                checks=[c.model_dump() for c in verification.checks],
                final_status="✅ All checks passed - Ready to merge",
                telegram_enabled=settings.telegram_enabled,
            )
            await github_client.post_pr_comment(job.repo_full_name, job.pr_number, comment)
            await notifier.notify_checks_result(job, passed=True, checks=verification.checks)
            await notifier.notify_final_result(job, success=True)
            return
        
        job.update_status(JobStatus.CHECKS_FAILED)
        store.save(job)
        
        await notifier.notify_checks_result(
            job, 
            passed=False, 
            checks=verification.checks,
            failure_excerpt=verification.failure_summary[:500],
        )
        
        pr_files = await github_client.get_pr_files(job.repo_full_name, job.pr_number)
        changed_files = [f.get("filename", "") for f in pr_files]
        
        pr_info = await github_client.get_pr_info(job.repo_full_name, job.pr_number)
        pr_title = pr_info.get("title", "")
        pr_body = pr_info.get("body", "") or ""
        
        job.update_status(JobStatus.DEBATING)
        store.save(job)
        
        arbiter_decision = await debate_system.run_debate(
            verification=verification,
            changed_files=changed_files,
            pr_title=pr_title,
            pr_body=pr_body,
        )
        job.arbiter_decision = arbiter_decision
        store.save(job)
        
        await notifier.notify_arbiter_decision(job, arbiter_decision)
        
        comment = format_pr_comment(
            run_number=run_number,
            commit_sha=job.head_sha,
            checks=[c.model_dump() for c in verification.checks],
            failure_summary=verification.failure_summary,
            arbiter_decision=arbiter_decision.model_dump(),
            fix_started=arbiter_decision.auto_fix_allowed and settings.enable_codex,
            telegram_enabled=settings.telegram_enabled,
        )
        await github_client.post_pr_comment(job.repo_full_name, job.pr_number, comment)
        
        if not arbiter_decision.auto_fix_allowed:
            job.update_status(JobStatus.NEEDS_HUMAN)
            job.final_message = f"Auto-fix denied: {arbiter_decision.stop_reason}"
            store.save(job)
            await notifier.notify_final_result(job, success=False, message=arbiter_decision.stop_reason or "")
            return
        
        if not settings.enable_codex:
            job.update_status(JobStatus.NEEDS_HUMAN)
            job.final_message = "Codex auto-fix disabled (SUPERVISOR_ENABLE_CODEX=0)"
            store.save(job)
            await notifier.notify_final_result(job, success=False, message="Codex disabled")
            return
        
        job.update_status(JobStatus.FIXING)
        store.save(job)
        
        for loop_num in range(1, settings.max_loops + 1):
            await notifier.notify_fix_started(job, loop_num, settings.max_loops)
            
            success, codex_output = await codex_fixer.apply_fix(
                workspace_path=workspace_path,
                arbiter_decision=arbiter_decision,
                verification=verification,
                changed_files=changed_files,
            )
            
            diff_stats = await workspace_manager.get_diff_stats(workspace_path)
            
            fix_attempt = FixAttempt(
                loop_number=loop_num,
                codex_prompt=codex_fixer.build_fix_prompt(arbiter_decision, verification, changed_files)[:500],
                codex_output=codex_output[:1000],
                diff_stats=diff_stats,
            )
            
            if not success:
                fix_attempt.committed = False
                job.fix_attempts.append(fix_attempt)
                store.save(job)
                continue
            
            if not diff_stats.within_thresholds(settings.max_files_changed, settings.max_loc_changed):
                job.update_status(JobStatus.NEEDS_HUMAN)
                job.final_message = (
                    f"Fix too large: {diff_stats.files_changed} files, "
                    f"{diff_stats.total_loc_changed} LOC (max: {settings.max_files_changed} files, "
                    f"{settings.max_loc_changed} LOC)"
                )
                fix_attempt.committed = False
                job.fix_attempts.append(fix_attempt)
                store.save(job)
                
                await github_client.post_pr_comment(
                    job.repo_full_name,
                    job.pr_number,
                    f"🛑 **Fix too large - needs human review**\n\n"
                    f"Changes: {diff_stats.files_changed} files, {diff_stats.total_loc_changed} LOC\n"
                    f"Thresholds: {settings.max_files_changed} files, {settings.max_loc_changed} LOC"
                )
                await notifier.notify_final_result(job, success=False, message=job.final_message)
                return
            
            new_verification = await runner.run_checks(workspace_path, job.head_sha)
            fix_attempt.verification = new_verification
            
            if new_verification.all_passed:
                commit_sha = await workspace_manager.commit_and_push(
                    workspace_path=workspace_path,
                    message=f"fix: auto-fix by PR Supervisor (loop {loop_num})",
                    branch=job.head_ref,
                )
                
                if commit_sha:
                    fix_attempt.committed = True
                    fix_attempt.commit_sha = commit_sha
                    job.fix_attempts.append(fix_attempt)
                    
                    job.update_status(JobStatus.FIXED)
                    job.final_message = f"Fixed and pushed: {commit_sha[:8]}"
                    store.save(job)
                    
                    await notifier.notify_fix_pushed(job, commit_sha)
                    
                    await github_client.post_pr_comment(
                        job.repo_full_name,
                        job.pr_number,
                        f"✅ **Auto-fix successful**\n\n"
                        f"Pushed commit `{commit_sha[:8]}` with fixes.\n"
                        f"All checks now pass."
                    )
                    await notifier.notify_final_result(job, success=True)
                    return
            
            verification = new_verification
            job.fix_attempts.append(fix_attempt)
            store.save(job)
        
        job.update_status(JobStatus.NEEDS_HUMAN)
        job.final_message = f"Max loops ({settings.max_loops}) reached without fixing all issues"
        store.save(job)
        
        await github_client.post_pr_comment(
            job.repo_full_name,
            job.pr_number,
            f"🛑 **Needs human review**\n\n"
            f"Attempted {settings.max_loops} fix loops but couldn't resolve all issues."
        )
        await notifier.notify_final_result(job, success=False, message=job.final_message)
    
    except Exception as e:
        logger.exception(f"Job {job.job_id} failed with error")
        job.update_status(JobStatus.ERROR)
        job.error_message = str(e)[:500]
        store.save(job)
        
        await notifier.notify_final_result(job, success=False, message=f"Error: {str(e)[:100]}")
    
    finally:
        if job.workspace_path:
            repo_name = job.repo_full_name.split("/")[-1]
            await workspace_manager.cleanup_workspace(job.job_id, repo_name)
