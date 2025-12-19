"""FastAPI application for PR Supervisor with hardened job queue."""

import asyncio
import logging
import os
import re
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .codex_fixer import CodexFixer
from .config import SupervisorSettings, get_settings
from .debate import DebateSystem
from .github import (
    GitHubClient,
    format_pr_comment,
    parse_webhook_payload,
    verify_signature,
)
from .models import (
    FixAttempt,
    JobStatus,
    SupervisorJob,
)
from .policy import check_autofix_policy
from .redact import redact_job_for_api, redact_secrets
from .runner import VerificationRunner
from .store import JobStore
from .telegram_notify import TelegramNotifier
from .workspace import WorkspaceManager

logger = logging.getLogger(__name__)

MAX_TRUNCATE_CHARS = 5000


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

    if job_dict.get("fix_attempts") and isinstance(job_dict["fix_attempts"], list):
        for attempt in job_dict["fix_attempts"]:
            if "codex_output" in attempt:
                truncated = truncate_field(attempt.get("codex_output"))
                attempt["codex_output"] = truncated["value"]
                attempt["codex_output_truncated"] = truncated["truncated"]
            if "codex_prompt" in attempt:
                truncated = truncate_field(attempt.get("codex_prompt"))
                attempt["codex_prompt"] = truncated["value"]
                attempt["codex_prompt_truncated"] = truncated["truncated"]

    return job_dict


FORBIDDEN_ENV_PATTERN = re.compile(r"(key|token|secret|password)", re.IGNORECASE)
FORBIDDEN_DIAG_PATTERN = re.compile(
    r"(key|token|secret|password|api_key|private_key|openai|github)", re.IGNORECASE
)
ALLOWED_ENV_KEYS = {
    "SUPERVISOR_ENABLED",
    "SUPERVISOR_AUTOFIX_DRY_RUN",
    "MODEL_OPTIMIST",
    "MODEL_SKEPTIC",
    "MODEL_ARBITER",
    "CODEX_MODEL",
    "GEMINI_MODEL",
}


def build_safe_env_snapshot(settings: SupervisorSettings) -> dict[str, str | None]:
    """Return a minimal, allowlisted env snapshot with denylist enforcement."""
    env_snapshot: dict[str, str | None] = {}

    for key in ALLOWED_ENV_KEYS:
        if key in os.environ and not FORBIDDEN_ENV_PATTERN.search(key):
            env_snapshot[key] = os.environ.get(key)

    # Ensure enabled flag is always present
    if "SUPERVISOR_ENABLED" not in env_snapshot:
        env_snapshot["SUPERVISOR_ENABLED"] = "1" if settings.enabled else "0"

    # Final guardrail: drop anything that accidentally matches the forbidden pattern
    return {
        k: v
        for k, v in env_snapshot.items()
        if not FORBIDDEN_ENV_PATTERN.search(k or "")
    }


def record_worker_error(app: FastAPI, err: Exception) -> None:
    """Track worker error counts by exception type on app state."""
    if not hasattr(app.state, "worker_errors"):
        app.state.worker_errors = {}
    key = err.__class__.__name__
    app.state.worker_errors[key] = app.state.worker_errors.get(key, 0) + 1


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
                if len(job) >= 2 and hasattr(job[1], "state"):
                    logger.warning(
                        "Legacy tuple queue payload received, using provided app context"
                    )
                    job, job_app = job[0], job[1]
                else:
                    logger.warning(
                        "Legacy tuple queue payload received, app context discarded"
                    )
                    job = job[0]

            try:
                logger.info("Worker processing job: %s", job.job_id)
                await run_supervisor_job(job, job_app)
            except Exception as exc:
                logger.exception("Job %s failed in worker (phase=worker_execute)", job.job_id)
                record_worker_error(app, exc)
            finally:
                app.state.job_queue.task_done()

        except asyncio.CancelledError:
            logger.info("Job worker cancelled, shutting down")
            break
        except Exception as exc:
            logger.exception("Job worker error (phase=worker_loop)")
            record_worker_error(app, exc)
            await asyncio.sleep(1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan with startup validation and worker management."""
    import httpx

    settings = get_settings()
    app.state.settings = settings
    app.state.ready = False
    app.state.startup_errors = []  # list[str]
    app.state.build_id = os.environ.get("BUILD_ID") or os.environ.get("GIT_SHA")

    app.state.job_queue = asyncio.Queue()
    app.state.supervisor_worker_task = None  # Optional[asyncio.Task]
    app.state.worker_errors = {}

    app.state.store = JobStore(f"{settings.base_jobs_dir}/job_history.jsonl")
    app.state.github_client = None
    app.state.telegram_http = None

    if settings.enabled:
        missing = []
        if (
            not settings.github_webhook_secret
            or not settings.github_webhook_secret.strip()
        ):
            missing.append("GITHUB_WEBHOOK_SECRET")
        if not settings.github_token or not settings.github_token.strip():
            missing.append("GITHUB_TOKEN")

        if missing:
            error_msg = f"Supervisor enabled but missing required settings: {', '.join(missing)}"
            logger.error(error_msg)
            app.state.startup_errors = missing
            app.state.ready = False
        else:
            app.state.github_client = GitHubClient(settings.github_token)  # type: ignore[arg-type]
            if (
                settings.telegram_enabled
                and settings.telegram_bot_token
                and settings.telegram_chat_id
            ):
                app.state.telegram_http = httpx.AsyncClient(timeout=httpx.Timeout(20.0))
            app.state.ready = True

            app.state.supervisor_worker_task = asyncio.create_task(job_worker(app))
            logger.info("Supervisor ready with job worker started")
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

    if app.state.github_client:
        await app.state.github_client.close()

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



class DiagResponse(BaseModel):
    ok: bool = True
    version: str = "0.2.1"
    build_id: Optional[str] = None
    code_paths: dict[str, str]
    models: dict[str, str | None]
    env_flags: dict[str, str | None]
    jobs: dict[str, object]
    notes: list[str] = []


@app.get("/api/diag", response_model=DiagResponse)
async def diag(request: Request):
    settings: SupervisorSettings = request.app.state.settings

    build_id = getattr(request.app.state, "build_id", None)
    # Import inside handler so module paths reflect the running container mount
    from src.supervisor import workspace as ws
    from src.supervisor.llm import openai_provider as oai

    def present(v: str | None) -> str | None:
        return v if v else None

    def present_bool(v: str | None) -> str:
        return "1" if (v and v.strip()) else "0"

    code_paths_raw = {
        "app_py": __file__,
        "workspace_py": getattr(ws, "__file__", "unknown"),
        "llm_provider_py": getattr(oai, "__file__", "unknown"),
    }
    code_paths = {
        k: v
        for k, v in code_paths_raw.items()
        if not FORBIDDEN_DIAG_PATTERN.search(f"{k}{v}")
    }

    models = {
        "MODEL_OPTIMIST": present(getattr(settings, "model_optimist", None)),
        "MODEL_SKEPTIC": present(getattr(settings, "model_skeptic", None)),
        "MODEL_ARBITER": present(getattr(settings, "model_arbiter", None)),
        "CODEX_MODEL": present(getattr(settings, "codex_model", None)),
    }

    env_flags = {
        k: v
        for k, v in build_safe_env_snapshot(settings).items()
        if not FORBIDDEN_DIAG_PATTERN.search(f"{k}{v}")
    }

    job_queue = getattr(request.app.state, "job_queue", None)
    worker_task = getattr(request.app.state, "supervisor_worker_task", None)
    error_counts = getattr(request.app.state, "worker_errors", {})
    jobs_stats = {
        "queue_depth": job_queue.qsize() if job_queue else 0,
        "worker_alive": bool(worker_task) and not worker_task.done(),
        "error_counts": error_counts,
    }

    notes = []
    notes.append(f"ready={getattr(request.app.state, 'ready', None)}")
    notes.append(f"queue_size={request.app.state.job_queue.qsize() if hasattr(request.app.state,'job_queue') else 'na'}")

    return DiagResponse(
        build_id=build_id,
        code_paths=code_paths,
        models=models,
        env_flags=env_flags,
        jobs=jobs_stats,
        notes=notes,
    )



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

    if not settings.github_webhook_secret or not settings.github_webhook_secret.strip():
        return JSONResponse(
            status_code=503,
            content={
                "ok": False,
                "error": "not_ready",
                "details": "GITHUB_WEBHOOK_SECRET missing",
            },
        )

    if not request.app.state.ready:
        return JSONResponse(
            status_code=503,
            content={
                "ok": False,
                "error": "not_ready",
                "details": "Supervisor startup incomplete",
            },
        )

    body = await request.body()

    if not x_hub_signature_256:
        return JSONResponse(
            status_code=401,
            content={
                "ok": False,
                "error": "invalid_signature",
                "detail": "Missing X-Hub-Signature-256 header",
            },
        )

    if not verify_signature(body, x_hub_signature_256, settings.github_webhook_secret):
        return JSONResponse(
            status_code=401,
            content={
                "ok": False,
                "error": "invalid_signature",
                "detail": "Signature verification failed",
            },
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
    existing = store.get_by_sha(
        payload.repo_full_name, payload.pr_number, payload.head_sha
    )
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
                status_code=401, detail="Invalid or missing debug token"
            )

    if not settings.enabled:
        raise HTTPException(status_code=400, detail="Supervisor disabled")

    if not request.app.state.ready:
        raise HTTPException(status_code=503, detail="Supervisor not ready")

    github_client: GitHubClient = request.app.state.github_client
    if not github_client:
        raise HTTPException(status_code=500, detail="GitHub client not configured")

    try:
        pr_info = await github_client.get_pr_info(body.repo, body.pr_number)
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Failed to fetch PR info: {str(e)}"
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
            "message": f"Job already exists for SHA {head_sha[:8]}",
        }

    approval = store.get_pr_approval(body.repo, body.pr_number)
    if approval.paused:
        return {
            "ok": False,
            "error": "pr_paused",
            "message": f"PR #{body.pr_number} is paused",
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
        "message": f"Simulated job queued for PR #{body.pr_number}",
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

    try:
        await workspace_manager.cleanup_old_workspaces()

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
            comment = redact_secrets(comment, settings)
            await github_client.post_pr_comment(
                job.repo_full_name, job.pr_number, comment
            )
            await notifier.notify_checks_result(
                job, passed=True, checks=verification.checks
            )
            await notifier.notify_final_result(job, success=True)
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

        failure_summary_redacted = redact_secrets(
            verification.failure_summary, settings
        )
        comment = format_pr_comment(
            run_number=run_number,
            commit_sha=job.head_sha,
            checks=[c.model_dump() for c in verification.checks],
            failure_summary=failure_summary_redacted,
            arbiter_decision=arbiter_decision.model_dump(),
            fix_started=arbiter_decision.auto_fix_allowed and settings.enable_codex,
            telegram_enabled=settings.telegram_enabled,
        )
        comment = redact_secrets(comment, settings)
        await github_client.post_pr_comment(job.repo_full_name, job.pr_number, comment)

        if not arbiter_decision.auto_fix_allowed:
            job.update_status(JobStatus.NEEDS_HUMAN)
            job.final_message = f"Auto-fix denied: {arbiter_decision.stop_reason}"
            store.save(job)
            await notifier.notify_final_result(
                job, success=False, message=arbiter_decision.stop_reason or ""
            )
            return

        pr_labels = [lbl.get("name", "") for lbl in pr_info.get("labels", [])]
        risk_level = (
            arbiter_decision.risk_level
            if hasattr(arbiter_decision, "risk_level")
            else None
        )

        autofix_decision = check_autofix_policy(
            settings=settings,
            store=store,
            repo=job.repo_full_name,
            pr_number=job.pr_number,
            pr_labels=pr_labels,
            arbiter_risk_level=risk_level,
        )

        if not autofix_decision.allowed:
            status = (
                JobStatus.NEEDS_HUMAN
                if autofix_decision.needs_human
                else JobStatus.NEEDS_HUMAN
            )
            job.update_status(status)
            job.final_message = autofix_decision.reason
            store.save(job)
            await notifier.notify_final_result(
                job, success=False, message=autofix_decision.reason
            )
            return

        dry_run_enabled = settings.autofix_dry_run and not settings.autofix_push

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
                codex_prompt=codex_fixer.build_fix_prompt(
                    arbiter_decision, verification, changed_files
                )[:500],
                codex_output=redact_secrets(codex_output[:1000], settings),
                diff_stats=diff_stats,
            )

            if not success:
                fix_attempt.committed = False
                job.fix_attempts.append(fix_attempt)
                store.save(job)
                continue

            if not diff_stats.within_thresholds(
                settings.max_files_changed, settings.max_loc_changed
            ):
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
                    f"Thresholds: {settings.max_files_changed} files, {settings.max_loc_changed} LOC",
                )
                await notifier.notify_final_result(
                    job, success=False, message=job.final_message
                )
                return

            new_verification = await runner.run_checks(workspace_path, job.head_sha)
            fix_attempt.verification = new_verification

            if new_verification.all_passed:
                if dry_run_enabled:
                    fix_attempt.committed = False
                    job.fix_attempts.append(fix_attempt)

                    job.update_status(JobStatus.FIXED)
                    job.final_message = "DRY RUN: fix ready; not pushed."
                    store.save(job)

                    check_lines = []
                    for check in new_verification.checks:
                        status = "✅" if check.passed else "❌"
                        cmd_lower = check.command.lower()
                        if "pytest" in cmd_lower:
                            cmd = "pytest"
                        elif "ruff" in cmd_lower:
                            cmd = "ruff"
                        else:
                            cmd = check.command.split()[0].split("/")[-1]
                        result_text = "Pass" if check.passed else "Fail"
                        check_lines.append(f"- {status} {result_text} `{cmd}`")
                    if not check_lines:
                        check_lines = ["- (no checks)"]

                    diff_summary = (
                        f"{diff_stats.files_changed} files, +{diff_stats.lines_added}/"
                        f"-{diff_stats.lines_removed} (LOC {diff_stats.total_loc_changed})"
                    )
                    dry_comment = (
                        "### ✅ DRY RUN: Fix validated locally; not pushed\n\n"
                        f"Would push to `{job.head_ref}`.\n\n"
                        f"Diff summary: {diff_summary}\n\n"
                        "Checks after fix:\n"
                        f"{chr(10).join(check_lines)}\n"
                    )
                    dry_comment = redact_secrets(dry_comment, settings)
                    await github_client.post_pr_comment(
                        job.repo_full_name,
                        job.pr_number,
                        dry_comment,
                    )
                    await notifier.notify_final_result(job, success=True)
                    return

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
                        f"All checks now pass.",
                    )
                    await notifier.notify_final_result(job, success=True)
                    return

            verification = new_verification
            job.fix_attempts.append(fix_attempt)
            store.save(job)

        job.update_status(JobStatus.NEEDS_HUMAN)
        job.final_message = (
            f"Max loops ({settings.max_loops}) reached without fixing all issues"
        )
        store.save(job)

        await github_client.post_pr_comment(
            job.repo_full_name,
            job.pr_number,
            f"🛑 **Needs human review**\n\n"
            f"Attempted {settings.max_loops} fix loops but couldn't resolve all issues.",
        )
        await notifier.notify_final_result(
            job, success=False, message=job.final_message
        )

    except Exception as e:
        logger.error(
            f"Job {job.job_id} failed with error: {type(e).__name__}", exc_info=False
        )
        job.update_status(JobStatus.ERROR)
        error_msg = redact_secrets(str(e)[:500], settings)
        job.error_message = error_msg
        store.save(job)

        await notifier.notify_final_result(
            job, success=False, message=f"Error: {error_msg[:100]}"
        )

    finally:
        if job.workspace_path:
            repo_name = job.repo_full_name.split("/")[-1]
            await workspace_manager.cleanup_workspace(job.job_id, repo_name)
