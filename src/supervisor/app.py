"""FastAPI application for PR Supervisor with hardened job queue."""

import asyncio
import logging
import os
import time
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
from .loop.fixers import DeterministicFixer, FixMode
from .config import SupervisorSettings, get_settings
from .github import GitHubClient, format_pr_comment, format_fallback_comment, parse_webhook_payload, verify_signature
from .loop.arbiter import arbitrate
from .loop.fixers import apply_fix_plan
from .loop.optimist import propose_fix_plan
from .loop.policy import load_policy
from .loop.skeptic import review_fix_plan
from .loop.types import FixPlan
from .models import (
    ArbiterDecision,
    FixAttempt,
    JobStage,
    JobStatus,
    SupervisorJob,
)
from .debate import LLMFailure
from .policy import check_autofix_policy
from .redact import redact_job_for_api, redact_secrets
from .runner import VerificationRunner
from .store import JobStore
from .telegram_notify import TelegramNotifier
from .workspace import WorkspaceManager
from .debate import DebateSystem

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
        cancel = getattr(app.state.supervisor_worker_task, "cancel", None)
        if callable(cancel):
            cancel()
        try:
            await asyncio.wait_for(app.state.supervisor_worker_task, timeout=5.0)
        except (asyncio.CancelledError, asyncio.TimeoutError, TypeError):
            # TypeError can occur for dummy tasks that aren't awaitable in tests.
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


def _get_runtime_settings(request: Request) -> SupervisorSettings:
    """Read supervisor settings from app.state without relying on app.state.settings.

    Rules:
    - In unit tests (and in the standalone supervisor app), handlers often set
      `app.state.use_preconfigured_settings=True` + `app.state.settings=...`.
      Honor that.
    - In the integrated dashboard app, supervisor settings live under
      `app.state.supervisor_settings` to avoid clobbering `app.state.settings`.
    """
    if getattr(request.app.state, "use_preconfigured_settings", False):
        s = getattr(request.app.state, "settings", None)
        if isinstance(s, SupervisorSettings):
            return s
        raise RuntimeError("Supervisor preconfigured settings missing or invalid")

    sup = getattr(request.app.state, "supervisor_settings", None)
    if sup is not None:
        if isinstance(sup, SupervisorSettings):
            return sup
        raise RuntimeError("app.state.supervisor_settings is not a SupervisorSettings")

    # Backward-compat for the standalone supervisor app.
    s = getattr(request.app.state, "settings", None)
    if isinstance(s, SupervisorSettings):
        return s

    # Integrated app without supervisor config: fail closed (disabled).
    return SupervisorSettings(enabled=False)


@app.get("/health", response_model=HealthResponse)
async def health(request: Request):
    """Health check endpoint."""
    settings: SupervisorSettings = _get_runtime_settings(request)
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
    settings: SupervisorSettings = _get_runtime_settings(request)
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
    settings: SupervisorSettings = _get_runtime_settings(request)
    
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
    settings: SupervisorSettings = _get_runtime_settings(request)
    
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
    settings: SupervisorSettings = _get_runtime_settings(request)
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
    settings: SupervisorSettings = _get_runtime_settings(request)
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
    settings: SupervisorSettings = _get_runtime_settings(request)
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
    settings: SupervisorSettings = _get_runtime_settings(request)
    store: JobStore = request.app.state.store
    job = store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_dict = job.model_dump()
    job_dict = redact_job_for_api(job_dict, settings)
    job_dict = truncate_job_for_api(job_dict)
    
    return job_dict


async def run_supervisor_job(job: SupervisorJob, app: FastAPI) -> None:
    """Main supervisor job orchestrator with hardened loop."""
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
    debate_system = DebateSystem(settings)
    codex_fixer = CodexFixer(settings)
    deterministic_fixer = DeterministicFixer(runner)
    loop_policy = load_policy()
    
    # Loop Limits
    MAX_TOTAL_RUNTIME = settings.max_total_runtime_seconds or 600  # 10 minutes default if not set
    MAX_ATTEMPTS = {
        JobStatus.FIX_LINT: settings.max_loops,
        JobStatus.FIX_FORMAT: 3,
        JobStatus.FIX_IMPORT: 3,
        JobStatus.FIX_TESTS: 1,
    }
    
    start_time = time.time()
    
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
        
        if not settings.autofix_dry_run:
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
        
        # Initial Verification
        verification = await runner.run_checks(workspace_path, job.head_sha)
        job.increment_verify_attempt()
        job.verification = verification

        # Fail-fast: if checks indicate a non-lint failure and Codex is unavailable,
        # do not enter the fix loop (tests expect CODEX_UNAVAILABLE).
        if settings.enable_codex and not codex_available:
            failing_cmds = [getattr(c, "command", "") for c in (verification.checks or []) if not getattr(c, "passed", True)]
            looks_like_tests = bool(getattr(verification, "failing_tests", None)) or any("pytest" in (cmd or "") for cmd in failing_cmds)
            if looks_like_tests:
                job.update_status(JobStatus.NEEDS_HUMAN)
                job.reason_code = "CODEX_UNAVAILABLE"
                job.final_message = "Codex auto-fixes require Codex to be available (CODEX_BIN). Codex unavailable; manual review required"
                store.save(job)
                # Post comment with check results (no fixes attempted)
                comment = format_pr_comment(
                    run_number=run_number,
                    commit_sha=job.head_sha,
                    checks=[c.model_dump() for c in verification.checks],
                    failure_summary=redact_secrets(verification.failure_summary, settings),
                    final_status="🛑 Needs human: Codex unavailable",
                    telegram_enabled=settings.telegram_enabled,
                )
                job.transition_stage(JobStage.COMMENTING)
                store.save(job)
                await upsert_pr_comment(job, github_client, store, settings, comment)
                job.transition_stage(JobStage.DONE)
                store.save(job)
                return
        
        # Helper for deterministic probe
        async def run_probe(cmd: str) -> bool:
            """Run a lightweight probe command to classify failures.

            In unit tests, VerificationRunner may be replaced by a fake that doesn't expose
            private helpers like _run_command. In that case, fail the probe closed (return False)
            so we pick a conservative fix stage instead of crashing.
            """
            run_cmd = getattr(runner, "_run_command", None)
            if not callable(run_cmd):
                return False
            res = await run_cmd(cmd, workspace_path)
            return bool(getattr(res, "passed", False))

        while True:
            # 1. Check Runtime Limit
            if time.time() - start_time > MAX_TOTAL_RUNTIME:
                await _finalize_with_limit(
                    job=job,
                    store=store,
                    settings=settings,
                    notifier=notifier,
                    github_client=github_client,
                    run_number=run_number,
                    verification=verification,
                    arbiter_decision=job.arbiter_decision,
                    reason=f"Max runtime {MAX_TOTAL_RUNTIME}s exceeded",
                )
                return

            # 2. Check Verification Status
            if verification.all_passed:
                had_fixes = bool(getattr(job, "fix_attempt_history", None))
                if had_fixes:
                    job.update_status(JobStatus.FIXED)
                    job.final_message = "DRY RUN: Fix applied and checks passed" if settings.autofix_dry_run else "Fix applied and checks passed"
                else:
                    job.update_status(JobStatus.CHECKS_PASSED)
                    job.final_message = "All checks passed"

                job.transition_stage(JobStage.VERIFYING)
                store.save(job)
                
                comment = format_pr_comment(
                    run_number=run_number,
                    commit_sha=job.head_sha,
                    checks=[c.model_dump() for c in verification.checks],
                    arbiter_decision=(job.arbiter_decision.model_dump() if getattr(job, "arbiter_decision", None) else None),
                    final_status=(
                        "✅ Auto-fix complete (DRY RUN) — checks passed" if (had_fixes and settings.autofix_dry_run)
                        else "✅ Auto-fix complete — checks passed" if had_fixes
                        else "✅ All checks passed - Ready to merge"
                    ),
                    telegram_enabled=settings.telegram_enabled,
                )
                job.transition_stage(JobStage.COMMENTING)
                store.save(job)
                await upsert_pr_comment(job, github_client, store, settings, comment)
                
                if not settings.autofix_dry_run:
                    await notifier.notify_checks_result(job, passed=True, checks=verification.checks)
                    await notifier.notify_final_result(job, success=True)
                else:
                    logger.info("DRY RUN: Checks passed. Would post comment and notify success.")
                
                job.transition_stage(JobStage.DONE)
                store.save(job)
                return

            # 3. Failed -> Classify
            changed_files: list[str] = []
            try:
                pr_files = await github_client.get_pr_files(job.repo_full_name, job.pr_number)
                for entry in pr_files:
                    filename = entry.get("filename")
                    status = entry.get("status", "").lower()
                    if not filename or status == "removed":
                        continue
                    changed_files.append(filename)
            except Exception as exc:
                logger.warning("Failed to fetch PR files", exc_info=exc)

            py_files = [f for f in changed_files if f.endswith(".py")]
            files_str = " ".join(f"'{f}'" for f in py_files)
            
            next_stage = None
            
            # Deterministic Fixer Classification
            if py_files:
                # Probe 1: General Lint (Standard ruff check)
                if not await run_probe(f"python3 -m ruff check {files_str}"):
                    next_stage = JobStatus.FIX_LINT
                # Probe 2: Imports
                elif not await run_probe(f"python3 -m ruff check --select I {files_str}"):
                     next_stage = JobStatus.FIX_IMPORT
                # Probe 3: Format
                elif not await run_probe(f"python3 -m ruff format --check {files_str}"):
                     next_stage = JobStatus.FIX_FORMAT
                else:
                     next_stage = JobStatus.FIX_TESTS
            else:
                next_stage = JobStatus.FIX_TESTS

            # 4. Check Stage Limits
            # We map JobStatus to attempt keys or similar. 
            # Existing code uses specific attempt fields. 
            # We need to adapt or standardize.
            # Using generic counters:
            current_attempts = job.attempt_counters.get(next_stage, 0)
            max_attempts = MAX_ATTEMPTS.get(next_stage, 3)
            
            if current_attempts >= max_attempts:
                await _finalize_with_limit(
                    job=job,
                    store=store,
                    settings=settings,
                    notifier=notifier,
                    github_client=github_client,
                    run_number=run_number,
                    verification=verification,
                    arbiter_decision=job.arbiter_decision,
                    reason=f"Loop limit hit: {next_stage} attempts={current_attempts}",
                )
                return
            
            # 5. Execute Stage
            job.update_status(next_stage)
            job.attempt_counters[next_stage] = current_attempts + 1
            store.save(job)
            
            pr_info = await github_client.get_pr_info(job.repo_full_name, job.pr_number)
            pr_labels = [lbl.get("name", "") for lbl in pr_info.get("labels", [])]
            
            # Arbiter for LINT (LLM)
            arbiter_decision = ArbiterDecision(auto_fix_allowed=True)
            risk_level = "unknown"

            if next_stage == JobStatus.FIX_LINT:
                 # LINT stage can use the LLM arbiter debate, but must be optional.
                 # In CI/unit tests (and in minimal deployments), LLM keys may be absent.
                 job.transition_stage(JobStage.DEBATING)
                 store.save(job)

                 if (not settings.enable_codex) or (not settings.is_llm_available()):
                     arbiter_decision = ArbiterDecision(
                         auto_fix_allowed=True,
                         risk_level="unknown",
                         stop_reason="llm_unconfigured",
                         arbiter_reasoning="LLM not configured; proceeding with deterministic dry-run flow.",
                     )
                     job.arbiter_decision = arbiter_decision
                     store.save(job)
                 else:
                     try:
                        pr_title = pr_info.get("title", "")
                        pr_body = pr_info.get("body", "") or ""
                        arbiter_decision = await debate_system.run_debate(
                            verification=verification,
                            changed_files=changed_files,
                            pr_title=pr_title,
                            pr_body=pr_body,
                        )
                        job.arbiter_decision = arbiter_decision
                        risk_level = arbiter_decision.risk_level
                        store.save(job)
                        if not settings.autofix_dry_run:
                            await notifier.notify_arbiter_decision(job, arbiter_decision)
                     except LLMFailure as llm_err:
                         logger.warning(f"Job {job.job_id}: LLM failed - {llm_err.failure_reason}")
                         
                         if not settings.autofix_dry_run:
                             failure_summary_redacted = redact_secrets(verification.failure_summary, settings)
                             comment = format_fallback_comment(
                                run_number=run_number,
                                commit_sha=job.head_sha,
                                checks=[c.model_dump() for c in verification.checks],
                                failure_summary=failure_summary_redacted,
                                llm_error=llm_err.failure_reason,
                                telegram_enabled=settings.telegram_enabled,
                            )
                             await upsert_pr_comment(job, github_client, store, settings, comment)
                             
                             job.final_message = f"LLM unavailable: {llm_err.failure_reason}"
                             job.update_status(JobStatus.CHECKS_FAILED) 
                             store.save(job)
                             await notifier.notify_final_result(job, success=False, message=job.final_message)
                         return

                 if not arbiter_decision.auto_fix_allowed:
                      job.update_status(JobStatus.NEEDS_HUMAN)
                      job.final_message = f"Auto-fix denied: {arbiter_decision.stop_reason}"
                      store.save(job)
                      if not settings.autofix_dry_run:
                          await notifier.notify_final_result(job, success=False, message=arbiter_decision.stop_reason or "")
                      return
                 
                 job.update_status(JobStatus.FIX_LINT) 

            # Check Auto-fix Policy
            # NOTE: lint-only fixes are deterministic and do not require Codex approval.
            if next_stage != JobStatus.FIX_LINT:
                autofix_decision = check_autofix_policy(
                    settings=settings,
                    store=store,
                    repo=job.repo_full_name,
                    pr_number=job.pr_number,
                    pr_labels=pr_labels,
                    arbiter_risk_level=risk_level,
                )
                
                if not autofix_decision.allowed:
                    job.update_status(JobStatus.NEEDS_HUMAN)
                    job.final_message = autofix_decision.reason
                    store.save(job)
                    if not settings.autofix_dry_run:
                        await notifier.notify_final_result(job, success=False, message=autofix_decision.reason)
                    return

            # Execute Fixer
            job.transition_stage(JobStage.FIXING)
            store.save(job)

            fix_success = False
            fix_msg = ""
            fix_attempt = FixAttempt(loop_number=current_attempts + 1)
            
            if next_stage in (JobStatus.FIX_FORMAT, JobStatus.FIX_IMPORT, JobStatus.FIX_TESTS):
                mode_map = {
                    JobStatus.FIX_FORMAT: FixMode.FORMAT,
                    JobStatus.FIX_IMPORT: FixMode.IMPORT,
                    JobStatus.FIX_TESTS: FixMode.TESTS
                }
                fix_success, fix_msg = await deterministic_fixer.run_fix(
                    mode_map[next_stage], 
                    workspace_path, 
                    changed_files,
                    job.head_sha,
                    verification_report=verification
                )
                fix_attempt.codex_prompt = f"Deterministic fix: {next_stage}"
                fix_attempt.codex_output = fix_msg
                
            elif next_stage == JobStatus.FIX_LINT:
                # Lint-only path: deterministic fixer (ruff) via the shared fix-plan executor.
                fix_result = await apply_fix_plan(
                    fix_plan=FixPlan(category="lint_only", objectives=[], approach="", estimated_risk="low"),
                    workspace_path=workspace_path,
                    changed_files=changed_files,
                    verification=verification,
                    settings=settings,
                )
                fix_success = bool(getattr(fix_result, "applied", False))
                fix_attempt.codex_prompt = "Deterministic lint fix"
                fix_attempt.codex_output = redact_secrets(str(getattr(fix_result, "notes", ""))[:1000], settings)

            # Check for changes & Commit
            diff_stats = await workspace_manager.get_diff_stats(workspace_path)
            fix_attempt.diff_stats = diff_stats
            
            if diff_stats.files_changed > 0:
                 if not diff_stats.within_thresholds(settings.max_files_changed, settings.max_loc_changed):
                      job.update_status(JobStatus.NEEDS_HUMAN)
                      job.final_message = "Fix too large"
                      job.fix_attempt_history.append(fix_attempt)
                      store.save(job)
                      if not settings.autofix_dry_run:
                          await notifier.notify_final_result(job, success=False, message=job.final_message)
                      return
                 
                 commit_sha = None
                 if settings.autofix_dry_run:
                      logger.info("DRY RUN: Skipping commit and push.")
                      # Verify locally with current changes
                      verification = await runner.run_checks(workspace_path, job.head_sha)
                      fix_attempt.verification = verification
                 elif not settings.autofix_push:
                      logger.info("Push disabled: Skipping commit and push.")
                      verification = await runner.run_checks(workspace_path, job.head_sha)
                      fix_attempt.verification = verification
                 else:
                      commit_sha = await workspace_manager.commit_and_push(
                        workspace_path=workspace_path,
                        message=f"fix: auto-fix {next_stage} (loop {current_attempts + 1})",
                        branch=job.head_ref,
                    )
                 
                 if commit_sha or settings.autofix_dry_run or not settings.autofix_push:
                      if commit_sha:
                          fix_attempt.committed = True
                          fix_attempt.commit_sha = commit_sha
                          job.head_sha = commit_sha # Update HEAD
                          # Re-verify
                          verification = await runner.run_checks(workspace_path, job.head_sha)
                          fix_attempt.verification = verification
                          
                          if verification.all_passed and not settings.autofix_dry_run:
                               await notifier.notify_fix_pushed(job, commit_sha)
                      
                      # If dry_run, verification is already updated above.
                 else:
                      fix_success = False
            else:
                 # No changes
                 if next_stage == JobStatus.FIX_TESTS and fix_success:
                      # TESTS_ONLY verified success?
                      verification = await runner.run_checks(workspace_path, job.head_sha)
                      job.verification = verification

            job.fix_attempt_history.append(fix_attempt)
            store.save(job)
            
            if diff_stats.files_changed == 0 and not verification.all_passed:
                 # Fix didn't change anything and we are still failing
                 # Logic continues to next iteration where it might pick a different stage or hit limits
                 pass
            
            # Loop continues...
            
    except Exception as e:
        logger.error(f"Job {job.job_id} failed with error: {type(e).__name__}", exc_info=False)
        job.update_status(JobStatus.ERROR)
        error_msg = redact_secrets(str(e)[:500], settings)
        job.error_message = error_msg
        store.save(job)
        if not settings.autofix_dry_run:
            await notifier.notify_final_result(job, success=False, message=f"Error: {error_msg[:100]}")
    
    finally:
        if job.workspace_path:
            repo_name = job.repo_full_name.split("/")[-1]
            await workspace_manager.cleanup_workspace(job.job_id, repo_name)