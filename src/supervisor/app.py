"""FastAPI application for PR Supervisor with hardened job queue."""

import asyncio
import logging
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
import inspect
from typing import Optional

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.healthcheck import get_health_status_for_api
from .codex_fixer import CodexFixer
from .config import SupervisorSettings, get_settings
from .github import GitHubClient, format_pr_comment, parse_webhook_payload, verify_signature
from .loop.arbiter import arbitrate
from .loop.fixers import apply_fix_plan
from .loop.optimist import propose_fix_plan
from .loop.policy import load_policy
from .loop.skeptic import review_fix_plan
from .models import (
    ArbiterDecision,
    FixAttempt,
    JobStage,
    JobStatus,
    SupervisorJob,
)
from .redact import redact_job_for_api, redact_secrets
from .runner import VerificationRunner
from .store import JobStore
from .telegram_notify import TelegramNotifier
from .workspace import WorkspaceManager

logger = logging.getLogger(__name__)

MAX_TRUNCATE_CHARS = 5000


_original_get_event_loop = asyncio.get_event_loop


def _patched_get_event_loop():
    """Wrap asyncio.get_event_loop so callers always get a loop."""
    try:
        return _original_get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop


def _ensure_event_loop() -> None:
    """Ensure a default event loop exists for sync test contexts."""
    asyncio.get_event_loop = _patched_get_event_loop
    policy = asyncio.get_event_loop_policy()
    try:
        loop = policy.get_event_loop()
    except RuntimeError:
        loop = policy.new_event_loop()
        policy.set_event_loop(loop)
        return

    if loop.is_closed():
        loop = policy.new_event_loop()
    policy.set_event_loop(loop)


_ensure_event_loop()


class HealthResponse(BaseModel):
    ok: bool
    enabled: bool
    ready: bool
    version: str = "0.2.0"
    error: Optional[str] = None


class JobResponse(BaseModel):
    job_id: str
    status: str
    message: str


class ConfigErrorResponse(BaseModel):
    ok: bool = False
    error: str
    missing: list[str] = []


def truncate_field(value: Optional[str], max_chars: int = MAX_TRUNCATE_CHARS) -> dict:
    """Truncate a field and return metadata."""
    if not value:
        return {"value": None, "truncated": False}
    if len(value) <= max_chars:
        return {"value": value, "truncated": False}
    return {
        "value": value[:max_chars],
        "truncated": True,
        "original_length": len(value),
        "max_chars": max_chars,
    }


def truncate_job_for_api(job_dict: dict) -> dict:
    """Apply truncation to all large text fields in a job dict."""
    if job_dict.get("verification") and isinstance(job_dict["verification"], dict):
        verification = job_dict["verification"]
        
        if "failure_summary" in verification:
            truncated = truncate_field(verification.get("failure_summary"))
            verification["failure_summary"] = truncated["value"]
            verification["failure_summary_truncated"] = truncated["truncated"]
        
        if "checks" in verification and isinstance(verification["checks"], list):
            for check in verification["checks"]:
                if "stdout" in check:
                    truncated = truncate_field(check.get("stdout"))
                    check["stdout"] = truncated["value"]
                    check["stdout_truncated"] = truncated["truncated"]
                if "stderr" in check:
                    truncated = truncate_field(check.get("stderr"))
                    check["stderr"] = truncated["value"]
                    check["stderr_truncated"] = truncated["truncated"]
    
    attempts_key = "fix_attempt_history"
    fallback_key = "fix_attempts"
    attempts = job_dict.get(attempts_key)
    key_used = attempts_key if isinstance(attempts, list) else None
    if not key_used:
        attempts = job_dict.get(fallback_key)
        if isinstance(attempts, list):
            key_used = fallback_key
        else:
            attempts = None
            key_used = None

    if attempts and isinstance(attempts, list):
        for attempt in attempts:
            if "codex_output" in attempt:
                truncated = truncate_field(attempt.get("codex_output"))
                attempt["codex_output"] = truncated["value"]
                attempt["codex_output_truncated"] = truncated["truncated"]
            if "codex_prompt" in attempt:
                truncated = truncate_field(attempt.get("codex_prompt"))
                attempt["codex_prompt"] = truncated["value"]
                attempt["codex_prompt_truncated"] = truncated["truncated"]
        if key_used != attempts_key:
            job_dict[attempts_key] = attempts
    return job_dict


def _compute_fix_backoff(attempt: int, settings: SupervisorSettings) -> float:
    """Compute backoff delay for fix attempts."""
    if attempt <= 0:
        return 0.0
    base = max(settings.fix_backoff_base_seconds, 0.0)
    factor = max(settings.fix_backoff_factor, 1.0)
    delay = base * (factor ** (attempt - 1))
    return min(delay, max(settings.fix_backoff_max_seconds, 0.0))


def _runtime_exceeded(job: SupervisorJob, settings: SupervisorSettings) -> bool:
    """Check if the job runtime has exceeded the configured maximum."""
    if settings.max_total_runtime_seconds <= 0:
        return False
    elapsed = (datetime.utcnow() - job.created_at).total_seconds()
    return elapsed > settings.max_total_runtime_seconds


def _apply_loop_limit(job: SupervisorJob, store: JobStore, reason: str) -> None:
    """Mark the job as halted due to loop limits."""
    job.update_status(JobStatus.NEEDS_HUMAN)
    job.reason_code = "LOOP_LIMIT"
    job.final_message = reason
    store.save(job)


async def _finalize_with_limit(
    job: SupervisorJob,
    store: JobStore,
    settings: SupervisorSettings,
    notifier: TelegramNotifier,
    github_client: GitHubClient,
    run_number: int,
    verification: Optional[object],
    arbiter_decision: Optional[ArbiterDecision],
    reason: str,
) -> None:
    """Finalize a job with a loop limit message and optional comment."""
    _apply_loop_limit(job, store, reason)
    if job.stage not in (JobStage.VERIFYING, JobStage.COMMENTING, JobStage.DONE):
        job.transition_stage(JobStage.VERIFYING)
        store.save(job)

    if verification:
        failure_summary = getattr(verification, "failure_summary", "")
        failure_summary_redacted = redact_secrets(failure_summary, settings)
        comment = format_pr_comment(
            run_number=run_number,
            commit_sha=job.head_sha,
            checks=[c.model_dump() for c in getattr(verification, "checks", [])],
            failure_summary=failure_summary_redacted,
            arbiter_decision=arbiter_decision.model_dump() if arbiter_decision else None,
            final_status=f"🛑 Loop halted: {reason}",
            telegram_enabled=settings.telegram_enabled,
        )
        job.transition_stage(JobStage.COMMENTING)
        store.save(job)
        await upsert_pr_comment(job, github_client, store, settings, comment)
        await notifier.notify_final_result(job, success=False, message=reason)

    job.transition_stage(JobStage.DONE)
    store.save(job)


async def upsert_pr_comment(
    job: SupervisorJob,
    github_client: GitHubClient,
    store: JobStore,
    settings: SupervisorSettings,
    body: str,
) -> None:
    """Post or update a PR comment for a job, idempotent by job_id."""
    comment_body = redact_secrets(body, settings)
    if job.pr_comment_id:
        await github_client.update_pr_comment(
            job.repo_full_name,
            job.pr_comment_id,
            comment_body,
        )
        return

    response = await github_client.post_pr_comment(
        job.repo_full_name,
        job.pr_number,
        comment_body,
    )
    comment_id = response.get("id") if isinstance(response, dict) else None
    if comment_id:
        job.pr_comment_id = int(comment_id)
        store.save(job)


async def job_worker(app: FastAPI) -> None:
    """Background worker that processes jobs from the queue.
    
    This worker runs continuously and pulls jobs from app.state.job_queue.
    It ensures jobs are always executed while the app is running.
    """
    logger.info("Job worker started")
    
    while True:
        try:
            job = await app.state.job_queue.get()
            
            job_app = app
            if isinstance(job, tuple):
                if len(job) >= 2 and hasattr(job[1], 'state'):
                    logger.warning("Legacy tuple queue payload received, using provided app context")
                    job, job_app = job[0], job[1]
                else:
                    logger.warning("Legacy tuple queue payload received, app context discarded")
                    job = job[0]
            
            try:
                logger.info("Worker processing job: %s", job.job_id)
                await run_supervisor_job(job, job_app)
            except Exception:
                logger.error("Job %s failed in worker", job.job_id, exc_info=False)
            finally:
                app.state.job_queue.task_done()
                
        except asyncio.CancelledError:
            logger.info("Job worker cancelled, shutting down")
            break
        except Exception:
            logger.error("Job worker error", exc_info=False)
            await asyncio.sleep(1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan with startup validation and worker management."""
    import httpx
    use_preconfigured_settings = getattr(app.state, "use_preconfigured_settings", False)
    if hasattr(app.state, "use_preconfigured_settings"):
        app.state.use_preconfigured_settings = False
    preconfigured_settings = getattr(app.state, "settings", None)
    if use_preconfigured_settings and isinstance(preconfigured_settings, SupervisorSettings):
        settings = preconfigured_settings
    else:
        settings = get_settings()
        app.state.settings = settings
    app.state.ready = False
    app.state.startup_errors = []  # list[str]

    use_preconfigured_job_queue = getattr(app.state, "use_preconfigured_job_queue", False)
    if hasattr(app.state, "use_preconfigured_job_queue"):
        app.state.use_preconfigured_job_queue = False
    job_queue = getattr(app.state, "job_queue", None)
    if job_queue is None or not use_preconfigured_job_queue:
        job_queue = asyncio.Queue()
        app.state.job_queue = job_queue
    app.state.supervisor_worker_task = None  # Optional[asyncio.Task]

    use_preconfigured_store = getattr(app.state, "use_preconfigured_store", False)
    if hasattr(app.state, "use_preconfigured_store"):
        app.state.use_preconfigured_store = False
    if not use_preconfigured_store or not getattr(app.state, "store", None):
        app.state.store = JobStore(f"{settings.base_jobs_dir}/job_history.jsonl")

    use_preconfigured_github_client = getattr(app.state, "use_preconfigured_github_client", False)
    if hasattr(app.state, "use_preconfigured_github_client"):
        app.state.use_preconfigured_github_client = False
    if use_preconfigured_github_client and getattr(app.state, "github_client", None):
        pass
    else:
        app.state.github_client = None
    if not hasattr(app.state, "telegram_http"):
        app.state.telegram_http = None

    if settings.enabled:
        missing = []
        if not settings.github_webhook_secret or not settings.github_webhook_secret.strip():
            missing.append("GITHUB_WEBHOOK_SECRET")
        if not settings.github_token or not settings.github_token.strip():
            missing.append("GITHUB_TOKEN")
        
        if missing:
            error_msg = f"Supervisor enabled but missing required settings: {', '.join(missing)}"
            logger.error(error_msg)
            app.state.startup_errors = missing
            app.state.ready = False
        else:
            if not getattr(app.state, "github_client", None):
                app.state.github_client = GitHubClient(settings.github_token)  # type: ignore[arg-type]
            if settings.telegram_enabled and settings.telegram_bot_token and settings.telegram_chat_id:
                if not getattr(app.state, "telegram_http", None):
                    app.state.telegram_http = httpx.AsyncClient(timeout=httpx.Timeout(20.0))
            app.state.ready = True

            if hasattr(job_queue, "get"):
                app.state.supervisor_worker_task = asyncio.create_task(job_worker(app))
                logger.info("Supervisor ready with job worker started")
            else:
                logger.info("Supervisor ready (custom job queue, worker not started)")
    else:
        logger.info("Supervisor disabled (SUPERVISOR_ENABLED=0)")
        app.state.ready = True
    
    yield
    
    if app.state.supervisor_worker_task:
        logger.info("Cancelling job worker...")
        app.state.supervisor_worker_task.cancel()
        try:
            await asyncio.wait_for(app.state.supervisor_worker_task, timeout=5.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass
        logger.info("Job worker stopped")
    
    github_client = getattr(app.state, "github_client", None)
    if github_client:
        close_method = getattr(github_client, "close", None)
        if close_method:
            close_result = close_method()
            if inspect.isawaitable(close_result):
                await close_result
    
    if app.state.telegram_http:
        await app.state.telegram_http.aclose()
        logger.info("Telegram HTTP client closed")


app = FastAPI(
    title="PR Supervisor",
    description="Automated PR verification and auto-fix service",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health(request: Request):
    """Health check endpoint."""
    settings: SupervisorSettings = request.app.state.settings
    ready = request.app.state.ready
    errors = request.app.state.startup_errors
    
    return HealthResponse(
        ok=ready,
        enabled=settings.enabled,
        ready=ready,
        error=f"Missing: {', '.join(errors)}" if errors else None,
    )


@app.get("/api/diag")
async def diag(request: Request):
    """Runtime diagnostics for the supervisor."""
    settings: SupervisorSettings = request.app.state.settings
    worker_task = getattr(request.app.state, "supervisor_worker_task", None)
    worker_alive = bool(worker_task and not worker_task.done())
    build_id = os.getenv("BUILD_ID") or os.getenv("BUILD_NUMBER") or None
    provider_health = get_health_status_for_api()
    ok = bool(settings.enabled and request.app.state.ready and not request.app.state.startup_errors)

    return {
        "ok": ok,
        "enabled": settings.enabled,
        "ready": request.app.state.ready,
        "build_id": build_id,
        "worker_alive": worker_alive,
        "debug_enabled": settings.debug,
        "push_enabled": settings.autofix_push,
        "dry_run": settings.autofix_dry_run,
        "codex_available": settings.is_codex_available(),
        "provider_health": provider_health,
    }


@app.post("/github/webhook")
async def github_webhook(
    request: Request,
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
    
    startup_errors = getattr(request.app.state, "startup_errors", []) or []
    missing_secret = (
        not settings.github_webhook_secret or not settings.github_webhook_secret.strip()
        or "GITHUB_WEBHOOK_SECRET" in startup_errors
    )

    if missing_secret:
        return JSONResponse(
            status_code=503,
            content={
                "ok": False,
                "error": "not_ready",
                "details": "GITHUB_WEBHOOK_SECRET missing",
            }
        )
    
    if not request.app.state.ready:
        return JSONResponse(
            status_code=503,
            content={
                "ok": False,
                "error": "not_ready",
                "details": "Supervisor startup incomplete",
            }
        )
    
    body = await request.body()
    
    if not x_hub_signature_256:
        return JSONResponse(
            status_code=401,
            content={"ok": False, "error": "invalid_signature", "detail": "Missing X-Hub-Signature-256 header"}
        )
    
    if not verify_signature(body, x_hub_signature_256, settings.github_webhook_secret):
        return JSONResponse(
            status_code=401,
            content={"ok": False, "error": "invalid_signature", "detail": "Signature verification failed"}
        )
    
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
    
    approval = store.get_pr_approval(payload.repo_full_name, payload.pr_number)
    if approval.paused:
        return JobResponse(
            job_id="",
            status="paused",
            message=f"PR #{payload.pr_number} is paused",
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
    
    await request.app.state.job_queue.put(job)
    
    return JobResponse(
        job_id=job_id,
        status="queued",
        message=f"Job queued for PR #{payload.pr_number}",
    )


class SimulatePRRequest(BaseModel):
    """Request body for PR simulation endpoint."""
    repo: str
    pr_number: int
    action: str = "synchronize"


async def simulate_pr_event_handler(
    request: Request,
    body: SimulatePRRequest,
    x_debug_token: Optional[str] = Header(None, alias="X-Debug-Token"),
):
    """
    Simulate a PR webhook event for testing (debug mode only).
    
    Enabled only when SUPERVISOR_DEBUG=1.
    Requires X-Debug-Token header if SUPERVISOR_DEBUG_TOKEN is configured.
    """
    settings: SupervisorSettings = request.app.state.settings
    
    if settings.debug_token and settings.debug_token.strip():
        if not x_debug_token or x_debug_token != settings.debug_token:
            raise HTTPException(
                status_code=401,
                detail="Invalid or missing debug token"
            )
    
    if not settings.enabled:
        raise HTTPException(
            status_code=400,
            detail="Supervisor disabled"
        )
    
    if not request.app.state.ready:
        raise HTTPException(
            status_code=503,
            detail="Supervisor not ready"
        )
    
    github_client: GitHubClient = request.app.state.github_client
    if not github_client:
        raise HTTPException(
            status_code=500,
            detail="GitHub client not configured"
        )
    
    try:
        pr_info = await github_client.get_pr_info(body.repo, body.pr_number)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to fetch PR info: {str(e)}"
        )
    
    head_sha = pr_info.get("head", {}).get("sha", "")
    head_ref = pr_info.get("head", {}).get("ref", "")
    base_ref = pr_info.get("base", {}).get("ref", "")
    is_fork = pr_info.get("head", {}).get("repo", {}).get("fork", False)
    
    store: JobStore = request.app.state.store
    existing = store.get_by_sha(body.repo, body.pr_number, head_sha)
    if existing:
        return {
            "ok": True,
            "job_id": existing.job_id,
            "status": "duplicate",
            "message": f"Job already exists for SHA {head_sha[:8]}"
        }
    
    approval = store.get_pr_approval(body.repo, body.pr_number)
    if approval.paused:
        return {
            "ok": False,
            "error": "pr_paused",
            "message": f"PR #{body.pr_number} is paused"
        }
    
    job_id = f"pr-{body.pr_number}-{head_sha[:8]}-{uuid.uuid4().hex[:6]}"
    job = SupervisorJob(
        job_id=job_id,
        repo_full_name=body.repo,
        pr_number=body.pr_number,
        head_sha=head_sha,
        head_ref=head_ref,
        base_ref=base_ref,
        pr_url=pr_info.get("html_url", ""),
        is_fork=is_fork,
    )
    
    store.save(job)
    await request.app.state.job_queue.put(job)
    
    return {
        "ok": True,
        "job_id": job_id,
        "status": "queued",
        "message": f"Simulated job queued for PR #{body.pr_number}"
    }


def register_debug_routes(app_instance: FastAPI) -> None:
    """Register debug routes only if SUPERVISOR_DEBUG=1."""
    settings = get_settings()
    if settings.debug:
        app_instance.post("/debug/simulate_pr_event")(simulate_pr_event_handler)


register_debug_routes(app)


@app.get("/api/jobs")
async def list_jobs_api(request: Request, limit: int = 50):
    """List recent supervisor jobs (API route)."""
    settings: SupervisorSettings = request.app.state.settings
    store: JobStore = request.app.state.store
    jobs = store.list_recent(limit)
    
    result = []
    for job in jobs:
        job_dict = job.model_dump()
        job_dict = redact_job_for_api(job_dict, settings)
        job_dict = truncate_job_for_api(job_dict)
        result.append(job_dict)
    
    return {"jobs": result, "count": len(result)}


@app.get("/api/jobs/{job_id}")
async def get_job_api(request: Request, job_id: str):
    """Get a specific job by ID (API route)."""
    settings: SupervisorSettings = request.app.state.settings
    store: JobStore = request.app.state.store
    job = store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_dict = job.model_dump()
    job_dict = redact_job_for_api(job_dict, settings)
    job_dict = truncate_job_for_api(job_dict)
    
    return {"ok": True, "job": job_dict}


@app.get("/jobs")
async def list_jobs(request: Request, limit: int = 50):
    """List recent supervisor jobs with truncation and redaction."""
    settings: SupervisorSettings = request.app.state.settings
    store: JobStore = request.app.state.store
    jobs = store.list_recent(limit)
    
    result = []
    for job in jobs:
        job_dict = job.model_dump()
        job_dict = redact_job_for_api(job_dict, settings)
        job_dict = truncate_job_for_api(job_dict)
        result.append(job_dict)
    
    return {"jobs": result}


@app.get("/jobs/{job_id}")
async def get_job(request: Request, job_id: str):
    """Get a specific job by ID with truncation and redaction."""
    settings: SupervisorSettings = request.app.state.settings
    store: JobStore = request.app.state.store
    job = store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_dict = job.model_dump()
    job_dict = redact_job_for_api(job_dict, settings)
    job_dict = truncate_job_for_api(job_dict)
    
    return job_dict


async def run_supervisor_job(job: SupervisorJob, app: FastAPI) -> None:
    """Main supervisor job orchestrator."""
    settings: SupervisorSettings = app.state.settings
    codex_available: bool = settings.is_codex_available()
    store: JobStore = app.state.store
    github_client: GitHubClient = app.state.github_client
    
    if not github_client:
        logger.error(f"Job {job.job_id}: GitHub client not available")
        job.update_status(JobStatus.ERROR)
        job.error_message = "GitHub client not configured"
        store.save(job)
        return
    
    notifier = TelegramNotifier(settings, http_client=app.state.telegram_http)
    workspace_manager = WorkspaceManager(settings)
    runner = VerificationRunner(settings)
    codex_fixer = CodexFixer(settings)
    loop_policy = load_policy()
    
    try:
        await workspace_manager.cleanup_old_workspaces()
        
        job.update_status(JobStatus.RUNNING)
        job.transition_stage(JobStage.ANALYZING)
        store.save(job)

        run_number = store.get_run_count(job.repo_full_name, job.pr_number)

        if _runtime_exceeded(job, settings):
            await _finalize_with_limit(
                job=job,
                store=store,
                settings=settings,
                notifier=notifier,
                github_client=github_client,
                run_number=run_number,
                verification=job.verification,
                arbiter_decision=job.arbiter_decision,
                reason=(
                    f"Max runtime {settings.max_total_runtime_seconds}s exceeded"
                ),
            )
            return
        
        await notifier.notify_job_start(job)
        
        clone_url = await github_client.get_repo_clone_url(job.repo_full_name)
        setup_sig = inspect.signature(workspace_manager.setup_workspace)
        setup_params = setup_sig.parameters
        setup_kwargs: dict[str, object] = {
            "job_id": job.job_id,
            "head_sha": job.head_sha,
            "head_ref": job.head_ref,
        }
        if "repo_url" in setup_params:
            setup_kwargs["repo_url"] = clone_url
        elif "clone_url" in setup_params:
            setup_kwargs["clone_url"] = clone_url
        if "base_ref" in setup_params:
            setup_kwargs["base_ref"] = job.base_ref
        if "pr_number" in setup_params:
            setup_kwargs["pr_number"] = job.pr_number
        workspace_path = await workspace_manager.setup_workspace(**setup_kwargs)
        job.workspace_path = workspace_path
        store.save(job)
        
        verification = await runner.run_checks(workspace_path, job.head_sha)
        job.increment_verify_attempt()
        job.verification = verification
        
        if verification.all_passed:
            job.update_status(JobStatus.CHECKS_PASSED)
            job.final_message = "All checks passed"
            job.transition_stage(JobStage.BYPASSED)
            job.transition_stage(JobStage.SKIPPED)
            job.transition_stage(JobStage.VERIFYING)
            store.save(job)
            
            comment = format_pr_comment(
                run_number=run_number,
                commit_sha=job.head_sha,
                checks=[c.model_dump() for c in verification.checks],
                final_status="✅ All checks passed - Ready to merge",
                telegram_enabled=settings.telegram_enabled,
            )
            job.transition_stage(JobStage.COMMENTING)
            store.save(job)
            await upsert_pr_comment(job, github_client, store, settings, comment)
            await notifier.notify_checks_result(job, passed=True, checks=verification.checks)
            await notifier.notify_final_result(job, success=True)
            job.transition_stage(JobStage.DONE)
            store.save(job)
            return
        
        job.update_status(JobStatus.CHECKS_FAILED)
        store.save(job)
        
        failure_excerpt = redact_secrets(verification.failure_summary[:500], settings)
        await notifier.notify_checks_result(
            job, 
            passed=False, 
            checks=verification.checks,
            failure_excerpt=failure_excerpt,
        )

        if _runtime_exceeded(job, settings):
            await _finalize_with_limit(
                job=job,
                store=store,
                settings=settings,
                notifier=notifier,
                github_client=github_client,
                run_number=run_number,
                verification=verification,
                arbiter_decision=job.arbiter_decision,
                reason=(
                    f"Max runtime {settings.max_total_runtime_seconds}s exceeded"
                ),
            )
            return
        
        changed_files: list[str] = []
        try:
            pr_files = await github_client.get_pr_files(job.repo_full_name, job.pr_number)
            for entry in pr_files:
                filename = entry.get("filename")
                status = entry.get("status", "").lower()
                if not filename or status == "removed":
                    continue
                if filename.endswith(".py"):
                    changed_files.append(filename)
        except Exception as exc:  # pragma: no cover - best-effort, not core to tests
            logger.warning("Failed to fetch PR files for lint targeting", exc_info=exc)
        
        pr_info = await github_client.get_pr_info(job.repo_full_name, job.pr_number)
        
        job.update_status(JobStatus.DEBATING)
        job.transition_stage(JobStage.DEBATING)
        job.increment_debate_attempt()
        store.save(job)

        fix_plan = propose_fix_plan(verification, verification.failure_summary)
        skeptic_report = review_fix_plan(
            plan=fix_plan,
            verification=verification,
            changed_files=changed_files,
        )
        pr_labels = [lbl.get("name", "") for lbl in pr_info.get("labels", [])]
        loop_decision = arbitrate(
            plan=fix_plan,
            skeptic=skeptic_report,
            policy=loop_policy,
            changed_files=changed_files,
            pr_labels=pr_labels,
            push_env_enabled=settings.autofix_push,
        )
        arbiter_decision = ArbiterDecision(
            auto_fix_allowed=loop_decision.decision in ("dry_run", "push"),
            decision=loop_decision.decision,
            reason=loop_decision.reason,
            fix_objectives=loop_decision.fix_objectives,
            risk_level=loop_decision.risk_level,
            stop_reason=loop_decision.reason if loop_decision.decision == "deny" else None,
            allowed_to_modify=loop_decision.allowed_to_modify,
            optimist_summary=fix_plan.rationale[:200],
            skeptic_summary="; ".join(skeptic_report.warnings)[:200],
        )
        job.fix_plan = fix_plan.model_dump()
        job.skeptic_report = skeptic_report.model_dump()
        job.loop_decision = loop_decision.model_dump()
        job.arbiter_decision = arbiter_decision
        store.save(job)

        await notifier.notify_arbiter_decision(job, arbiter_decision)

        failure_summary_redacted = redact_secrets(verification.failure_summary, settings)
        comment = format_pr_comment(
            run_number=run_number,
            commit_sha=job.head_sha,
            checks=[c.model_dump() for c in verification.checks],
            failure_summary=failure_summary_redacted,
            arbiter_decision=arbiter_decision.model_dump(),
            fix_started=arbiter_decision.auto_fix_allowed,
            telegram_enabled=settings.telegram_enabled,
        )
        await upsert_pr_comment(job, github_client, store, settings, comment)

        if not arbiter_decision.auto_fix_allowed:
            job.update_status(JobStatus.NEEDS_HUMAN)
            job.final_message = f"Auto-fix denied: {arbiter_decision.stop_reason}"
            job.transition_stage(JobStage.SKIPPED)
            job.transition_stage(JobStage.VERIFYING)
            store.save(job)

            denied_comment = format_pr_comment(
                run_number=run_number,
                commit_sha=job.head_sha,
                checks=[c.model_dump() for c in verification.checks],
                failure_summary=failure_summary_redacted,
                arbiter_decision=arbiter_decision.model_dump(),
                final_status=f"🛑 Auto-fix denied: {arbiter_decision.stop_reason}. Please review manually.",
                telegram_enabled=settings.telegram_enabled,
            )
            job.transition_stage(JobStage.COMMENTING)
            store.save(job)
            await upsert_pr_comment(job, github_client, store, settings, denied_comment)
            await notifier.notify_final_result(job, success=False, message=arbiter_decision.stop_reason or "")
            job.transition_stage(JobStage.DONE)
            store.save(job)
            return
        
        job.update_status(JobStatus.FIXING)
        job.transition_stage(JobStage.FIXING)
        store.save(job)

        if job.fix_attempts >= settings.max_fix_attempts:
            await _finalize_with_limit(
                job=job,
                store=store,
                settings=settings,
                notifier=notifier,
                github_client=github_client,
                run_number=run_number,
                verification=verification,
                arbiter_decision=arbiter_decision,
                reason=f"Max fix attempts ({settings.max_fix_attempts}) reached",
            )
            return

        total_loops = 1 + (settings.max_loops if settings.enable_codex else 0)
        await notifier.notify_fix_started(job, 1, total_loops)

        if _runtime_exceeded(job, settings):
            await _finalize_with_limit(
                job=job,
                store=store,
                settings=settings,
                notifier=notifier,
                github_client=github_client,
                run_number=run_number,
                verification=verification,
                arbiter_decision=arbiter_decision,
                reason=(
                    f"Max runtime {settings.max_total_runtime_seconds}s exceeded"
                ),
            )
            return

        if job.fix_attempts >= settings.max_fix_attempts:
            await _finalize_with_limit(
                job=job,
                store=store,
                settings=settings,
                notifier=notifier,
                github_client=github_client,
                run_number=run_number,
                verification=verification,
                arbiter_decision=arbiter_decision,
                reason=f"Max fix attempts ({settings.max_fix_attempts}) reached",
            )
            return

        job.increment_fix_attempt()
        deterministic_result = await apply_fix_plan(
            workspace_path=workspace_path,
            plan=fix_plan,
            verification=verification,
            changed_files=changed_files,
        )
        deterministic_notes = [
            redact_secrets(note, settings) for note in deterministic_result.notes
        ]
        diff_stats = await workspace_manager.get_diff_stats(workspace_path)
        fix_attempt = FixAttempt(
            loop_number=1,
            fixer=deterministic_result.fixer,
            notes=deterministic_notes,
            diff_stats=diff_stats,
        )

        if deterministic_result.applied:
            too_many_files = (
                loop_policy.max_files_touched
                and diff_stats.files_changed > loop_policy.max_files_touched
            )
            too_many_loc = (
                loop_policy.max_loc_changed
                and diff_stats.total_loc_changed > loop_policy.max_loc_changed
            )
            if too_many_files or too_many_loc:
                job.update_status(JobStatus.NEEDS_HUMAN)
                job.final_message = (
                    f"Fix too large: {diff_stats.files_changed} files, "
                    f"{diff_stats.total_loc_changed} LOC (max: {loop_policy.max_files_touched} files, "
                    f"{loop_policy.max_loc_changed} LOC)"
                )
                fix_attempt.committed = False
                job.fix_attempt_history.append(fix_attempt)
                job.transition_stage(JobStage.VERIFYING)
                store.save(job)

                too_large_comment = format_pr_comment(
                    run_number=run_number,
                    commit_sha=job.head_sha,
                    checks=[c.model_dump() for c in verification.checks],
                    failure_summary=failure_summary_redacted,
                    arbiter_decision=arbiter_decision.model_dump(),
                    final_status=f"🛑 Fix too large: {job.final_message}",
                    telegram_enabled=settings.telegram_enabled,
                )
                job.transition_stage(JobStage.COMMENTING)
                store.save(job)
                await upsert_pr_comment(job, github_client, store, settings, too_large_comment)
                await notifier.notify_final_result(job, success=False, message=job.final_message)
                job.transition_stage(JobStage.DONE)
                store.save(job)
                return

            new_verification = await runner.run_checks(workspace_path, job.head_sha)
            job.increment_verify_attempt()
            fix_attempt.verification = new_verification
            job.fix_attempt_history.append(fix_attempt)
            store.save(job)

            if new_verification.all_passed:
                if arbiter_decision.decision == "push":
                    commit_sha = await workspace_manager.commit_and_push(
                        workspace_path=workspace_path,
                        message="fix: auto-fix by PR Supervisor (deterministic)",
                        branch=job.head_ref,
                    )
                    if commit_sha:
                        fix_attempt.committed = True
                        fix_attempt.commit_sha = commit_sha
                        job.update_status(JobStatus.FIXED)
                        job.final_message = f"Fixed and pushed: {commit_sha[:8]}"
                        job.transition_stage(JobStage.VERIFYING)
                        store.save(job)

                        await notifier.notify_fix_pushed(job, commit_sha)
                        pushed_comment = format_pr_comment(
                            run_number=run_number,
                            commit_sha=job.head_sha,
                            checks=[c.model_dump() for c in new_verification.checks],
                            arbiter_decision=arbiter_decision.model_dump(),
                            final_status=(
                                f"✅ Auto-fix successful. Pushed `{commit_sha[:8]}`."
                            ),
                            telegram_enabled=settings.telegram_enabled,
                        )
                        job.transition_stage(JobStage.COMMENTING)
                        store.save(job)
                        await upsert_pr_comment(job, github_client, store, settings, pushed_comment)
                        await notifier.notify_final_result(job, success=True)
                        job.transition_stage(JobStage.DONE)
                        store.save(job)
                        return
                job.update_status(JobStatus.FIXED)
                job.final_message = "Fixed in DRY RUN; not pushed"
                job.transition_stage(JobStage.VERIFYING)
                store.save(job)

                dry_run_comment = format_pr_comment(
                    run_number=run_number,
                    commit_sha=job.head_sha,
                    checks=[c.model_dump() for c in new_verification.checks],
                    arbiter_decision=arbiter_decision.model_dump(),
                    final_status="✅ Auto-fix successful (DRY RUN). Not pushed.",
                    telegram_enabled=settings.telegram_enabled,
                )
                job.transition_stage(JobStage.COMMENTING)
                store.save(job)
                await upsert_pr_comment(job, github_client, store, settings, dry_run_comment)
                await notifier.notify_final_result(job, success=True)
                job.transition_stage(JobStage.DONE)
                store.save(job)
                return

        if not codex_available:
            job.update_status(JobStatus.NEEDS_HUMAN)
            job.reason_code = "CODEX_UNAVAILABLE"
            job.final_message = (
                "Codex auto-fixes require the configured binary, but it is unavailable. "
                "Install Codex or disable SUPERVISOR_ENABLE_CODEX."
            )
            job.transition_stage(JobStage.VERIFYING)
            store.save(job)

            codex_missing_comment = format_pr_comment(
                run_number=run_number,
                commit_sha=job.head_sha,
                checks=[c.model_dump() for c in verification.checks],
                failure_summary=failure_summary_redacted,
                arbiter_decision=arbiter_decision.model_dump(),
                final_status="🛑 Auto-fix aborted: Codex unavailable. Manual review required.",
                telegram_enabled=settings.telegram_enabled,
            )
            job.transition_stage(JobStage.COMMENTING)
            store.save(job)
            await upsert_pr_comment(job, github_client, store, settings, codex_missing_comment)
            await notifier.notify_final_result(job, success=False, message=job.final_message)
            job.transition_stage(JobStage.DONE)
            store.save(job)
            return

        for loop_num in range(1, settings.max_loops + 1):
            await notifier.notify_fix_started(job, loop_num + 1, total_loops)

            if _runtime_exceeded(job, settings):
                await _finalize_with_limit(
                    job=job,
                    store=store,
                    settings=settings,
                    notifier=notifier,
                    github_client=github_client,
                    run_number=run_number,
                    verification=verification,
                    arbiter_decision=arbiter_decision,
                    reason=(
                        f"Max runtime {settings.max_total_runtime_seconds}s exceeded"
                    ),
                )
                return

            if job.fix_attempts >= settings.max_fix_attempts:
                await _finalize_with_limit(
                    job=job,
                    store=store,
                    settings=settings,
                    notifier=notifier,
                    github_client=github_client,
                    run_number=run_number,
                    verification=verification,
                    arbiter_decision=arbiter_decision,
                    reason=f"Max fix attempts ({settings.max_fix_attempts}) reached",
                )
                return

            job.increment_fix_attempt()
            success, codex_output = await codex_fixer.apply_fix(
                workspace_path=workspace_path,
                arbiter_decision=arbiter_decision,
                verification=verification,
                changed_files=changed_files,
            )

            diff_stats = await workspace_manager.get_diff_stats(workspace_path)

            fix_attempt = FixAttempt(
                loop_number=loop_num + 1,
                fixer="codex",
                codex_prompt=codex_fixer.build_fix_prompt(arbiter_decision, verification, changed_files)[:500],
                codex_output=redact_secrets(codex_output[:1000], settings),
                diff_stats=diff_stats,
            )

            if not success:
                fix_attempt.committed = False
                job.fix_attempt_history.append(fix_attempt)
                store.save(job)
                backoff = _compute_fix_backoff(job.fix_attempts, settings)
                if backoff > 0:
                    await asyncio.sleep(backoff)
                continue

            too_many_files = (
                loop_policy.max_files_touched
                and diff_stats.files_changed > loop_policy.max_files_touched
            )
            too_many_loc = (
                loop_policy.max_loc_changed
                and diff_stats.total_loc_changed > loop_policy.max_loc_changed
            )
            if too_many_files or too_many_loc:
                job.update_status(JobStatus.NEEDS_HUMAN)
                job.final_message = (
                    f"Fix too large: {diff_stats.files_changed} files, "
                    f"{diff_stats.total_loc_changed} LOC (max: {loop_policy.max_files_touched} files, "
                    f"{loop_policy.max_loc_changed} LOC)"
                )
                fix_attempt.committed = False
                job.fix_attempt_history.append(fix_attempt)
                job.transition_stage(JobStage.VERIFYING)
                store.save(job)

                too_large_comment = format_pr_comment(
                    run_number=run_number,
                    commit_sha=job.head_sha,
                    checks=[c.model_dump() for c in verification.checks],
                    failure_summary=failure_summary_redacted,
                    arbiter_decision=arbiter_decision.model_dump(),
                    final_status=f"🛑 Fix too large: {job.final_message}",
                    telegram_enabled=settings.telegram_enabled,
                )
                job.transition_stage(JobStage.COMMENTING)
                store.save(job)
                await upsert_pr_comment(job, github_client, store, settings, too_large_comment)
                await notifier.notify_final_result(job, success=False, message=job.final_message)
                job.transition_stage(JobStage.DONE)
                store.save(job)
                return

            new_verification = await runner.run_checks(workspace_path, job.head_sha)
            job.increment_verify_attempt()
            fix_attempt.verification = new_verification

        if new_verification.all_passed:
            commit_sha: Optional[str] = None
            if arbiter_decision.decision == "push":
                commit_sha = await workspace_manager.commit_and_push(
                    workspace_path=workspace_path,
                    message=f"fix: auto-fix by PR Supervisor (loop {loop_num})",
                    branch=job.head_ref,
                )

            if commit_sha:
                fix_attempt.committed = True
                fix_attempt.commit_sha = commit_sha
                job.fix_attempt_history.append(fix_attempt)

                job.update_status(JobStatus.FIXED)
                job.final_message = f"Fixed and pushed: {commit_sha[:8]}"
                job.transition_stage(JobStage.VERIFYING)
                store.save(job)

                await notifier.notify_fix_pushed(job, commit_sha)

                pushed_comment = format_pr_comment(
                    run_number=run_number,
                    commit_sha=job.head_sha,
                    checks=[c.model_dump() for c in new_verification.checks],
                    arbiter_decision=arbiter_decision.model_dump(),
                    final_status=(
                        f"✅ Auto-fix successful. Pushed `{commit_sha[:8]}`."
                    ),
                    telegram_enabled=settings.telegram_enabled,
                )
                job.transition_stage(JobStage.COMMENTING)
                store.save(job)
                await upsert_pr_comment(job, github_client, store, settings, pushed_comment)
                await notifier.notify_final_result(job, success=True)
                job.transition_stage(JobStage.DONE)
                store.save(job)
                return

            job.update_status(JobStatus.FIXED)
            job.final_message = "Fixed in DRY RUN; not pushed"
            job.fix_attempt_history.append(fix_attempt)
            job.transition_stage(JobStage.VERIFYING)
            store.save(job)

            dry_run_comment = format_pr_comment(
                run_number=run_number,
                commit_sha=job.head_sha,
                checks=[c.model_dump() for c in new_verification.checks],
                arbiter_decision=arbiter_decision.model_dump(),
                final_status="✅ Auto-fix successful (DRY RUN). Not pushed.",
                telegram_enabled=settings.telegram_enabled,
            )
            job.transition_stage(JobStage.COMMENTING)
            store.save(job)
            await upsert_pr_comment(job, github_client, store, settings, dry_run_comment)
            await notifier.notify_final_result(job, success=True)
            job.transition_stage(JobStage.DONE)
            store.save(job)
            return

        verification = new_verification
        job.fix_attempt_history.append(fix_attempt)
        store.save(job)
        backoff = _compute_fix_backoff(job.fix_attempts, settings)
        if backoff > 0:
            await asyncio.sleep(backoff)

        job.update_status(JobStatus.NEEDS_HUMAN)
        job.final_message = f"Max loops ({settings.max_loops}) reached without fixing all issues"
        job.transition_stage(JobStage.VERIFYING)
        store.save(job)

        needs_human_comment = format_pr_comment(
            run_number=run_number,
            commit_sha=job.head_sha,
            checks=[c.model_dump() for c in verification.checks],
            failure_summary=failure_summary_redacted,
            arbiter_decision=arbiter_decision.model_dump(),
            final_status=(
                f"🛑 Attempted {settings.max_loops} fix loops but couldn't resolve all issues."
            ),
            telegram_enabled=settings.telegram_enabled,
        )
        job.transition_stage(JobStage.COMMENTING)
        store.save(job)
        await upsert_pr_comment(job, github_client, store, settings, needs_human_comment)
        await notifier.notify_final_result(job, success=False, message=job.final_message)
        job.transition_stage(JobStage.DONE)
        store.save(job)
    
    except Exception as e:
        logger.error(f"Job {job.job_id} failed with error: {type(e).__name__}", exc_info=False)
        job.update_status(JobStatus.ERROR)
        error_msg = redact_secrets(str(e)[:500], settings)
        job.error_message = error_msg
        store.save(job)
        
        await notifier.notify_final_result(job, success=False, message=f"Error: {error_msg[:100]}")
    
    finally:
        if job.workspace_path:
            repo_name = job.repo_full_name.split("/")[-1]
            await workspace_manager.cleanup_workspace(job.job_id, repo_name)
