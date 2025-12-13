"""
Codex Remote Runner for Telegram Bot integration.

Calls a remote Codex runner service via HTTP POST.
Supports multiple output modes: normal, short, debug.

Async job support:
- start_codex_job: Start a long-running job
- poll_codex_job: Check job status
- run_codex_job_async: Start and poll with timeout
"""
from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Literal, Optional

import httpx

logger = logging.getLogger(__name__)

TIMEOUT_SECONDS = 180
MAX_STDOUT_NORMAL = 12000
MAX_STDERR_NORMAL = 1000
MAX_STDOUT_DEBUG = 12000
MAX_STDERR_DEBUG = 6000
MAX_SHORT_OUTPUT = 3000

OutputMode = Literal["normal", "short", "debug", "review", "audit", "fix_prompt"]

REVIEW_TASK_PREFIX = """You are a code reviewer. Perform these checks:

1. Print source: echo "SOURCE: github/main"
2. Get current commit: echo "COMMIT: $(git rev-parse --short HEAD)"
3. Get origin/main: echo "ORIGIN_MAIN: $(git rev-parse --short origin/main)"
4. Run smoke tests: python -m pytest -q --tb=no 2>&1 | head -30
5. Scan for issues: grep -rn "TODO\\|FIXME\\|XXX" src/ agent/ --include="*.py" 2>/dev/null | head -10
6. Check for missing imports or obvious errors

Return a CONCISE summary in this format:
SOURCE: github/main
COMMIT: <hash>
ORIGIN_MAIN: <hash>
TEST STATUS: PASS/FAIL (X passed, Y failed)
TOP ISSUES:
- issue 1
- issue 2
SUGGESTED FIXES:
- fix 1
- fix 2

Now review: """

AUDIT_TASK_PREFIX = """You are a security auditor. Perform these checks:

1. Print source: echo "SOURCE: github/main"
2. Get commit: git rev-parse --short HEAD
3. Run tests: python -m pytest -q --tb=no 2>&1 | head -20
4. Run pip-audit: python -m pip_audit -r requirements.txt 2>&1 | head -20 (if pip_audit is installed and requirements.txt exists, otherwise note "pip-audit not available")
5. Run bandit: python -m bandit -r src agent -q 2>&1 | head -30 (if bandit is installed, otherwise note "bandit not available")
6. Check for hardcoded secrets, exposed keys, SQL injection risks

Return CONCISE format:
SOURCE: github/main
COMMIT: <hash>
TEST STATUS: PASS/FAIL
SECURITY FINDINGS:
- [SEVERITY] finding (or "pip-audit/bandit not available" if tools missing)
RECOMMENDATIONS:
- action

Now audit: """

FIX_PROMPT_PREFIX = """You are a senior engineer creating a handoff prompt for a junior developer.

Analyze the codebase and produce a SINGLE, COMPLETE Builder-ready prompt that:
1. Lists exact files to edit (full paths)
2. Shows precise code changes (before/after or patch-style)
3. Includes acceptance criteria
4. Shows how to verify (test commands using python -m pytest)

Format as a clean, paste-ready prompt starting with "BUILDER PROMPT:".

Now create fix prompt for: """


def _get_runner_url() -> Optional[str]:
    """Get the Codex runner URL from environment."""
    return os.environ.get("CODEX_RUNNER_URL")


def _get_runner_token() -> Optional[str]:
    """Get the Codex runner token from environment."""
    token = (os.getenv("CODEX_RUNNER_TOKEN") or "").strip()
    return token if token else None


def _truncate(text: str, max_len: int) -> str:
    """Truncate text with ellipsis if too long."""
    if len(text) <= max_len:
        return text
    return text[:max_len - 20] + "\n... [truncated]"


def _format_normal(data: dict) -> str:
    """Format response for normal mode - stdout only, stderr on error."""
    stdout = data.get("stdout", "").strip()
    stderr = data.get("stderr", "").strip()
    ok = data.get("ok", True)
    exit_code = data.get("exit_code", 0)
    
    if stdout:
        return _truncate(stdout, MAX_STDOUT_NORMAL)
    
    if not ok or exit_code != 0:
        snippet = _truncate(stderr, MAX_STDERR_NORMAL) if stderr else "No details available"
        return f"Codex failed (exit_code={exit_code}).\n{snippet}"
    
    return "No output."


def _format_short(data: dict) -> str:
    """Format response for short mode - concise output."""
    result = _format_normal(data)
    return _truncate(result, MAX_SHORT_OUTPUT)


def _format_debug(data: dict) -> str:
    """Format response for debug mode - full details."""
    ok = data.get("ok", False)
    exit_code = data.get("exit_code", -1)
    duration_ms = data.get("duration_ms", 0)
    stdout = data.get("stdout", "").strip()
    stderr = data.get("stderr", "").strip()
    
    parts = [
        f"ok: {ok}",
        f"exit_code: {exit_code}",
        f"duration_ms: {duration_ms}",
        "",
        "=== STDOUT ===",
        _truncate(stdout, MAX_STDOUT_DEBUG) if stdout else "(empty)",
        "",
        "=== STDERR ===",
        _truncate(stderr, MAX_STDERR_DEBUG) if stderr else "(empty)",
    ]
    return "\n".join(parts)


def _format_review(data: dict, job_id: Optional[str] = None) -> str:
    """Format response for review mode - includes sync_status, exit_code, duration if present."""
    stdout = data.get("stdout", "").strip()
    stderr = data.get("stderr", "").strip()
    sync_status = data.get("sync_status")
    exit_code = data.get("exit_code")
    duration_ms = data.get("duration_ms")
    
    parts = []
    
    if job_id:
        parts.append(f"JOB: {job_id}")
    
    if sync_status:
        parts.append(f"SYNC: {sync_status}")
    
    if exit_code is not None:
        parts.append(f"EXIT_CODE: {exit_code}")
    
    if duration_ms is not None:
        parts.append(f"DURATION_MS: {duration_ms}")
    
    if parts:
        parts.append("")
    
    if stdout:
        parts.append(_truncate(stdout, MAX_STDOUT_NORMAL))
    elif stderr:
        parts.append(_truncate(stderr, 2000))
    else:
        ok = data.get("ok", True)
        if not ok or (exit_code is not None and exit_code != 0):
            parts.append(f"Codex failed (exit_code={exit_code}). No output available.")
        else:
            parts.append("No output.")
    
    return "\n".join(parts)


async def run_codex_remote(task: str, *, mode: OutputMode = "normal") -> str:
    """
    Run a Codex task on the remote runner service.
    
    Args:
        task: The task description to send to Codex
        mode: Output mode - "normal", "short", or "debug"
        
    Returns:
        Formatted plain-text result based on mode
    """
    runner_url = _get_runner_url()
    runner_token = _get_runner_token()
    
    if not runner_url:
        return "Error: CODEX_RUNNER_URL not configured"
    
    if not runner_token:
        return "Error: CODEX_RUNNER_TOKEN not configured"
    
    endpoint = f"{runner_url.rstrip('/')}/v1/codex"
    
    headers = {"Content-Type": "application/json"}
    if runner_token:
        headers["X-RUNNER-TOKEN"] = runner_token
    
    if mode == "short":
        task = (
            "Answer in <=10 lines. Quote at most 2 snippets (<=20 lines each). "
            "No extra commentary.\n\n" + task
        )
    elif mode == "review":
        task = REVIEW_TASK_PREFIX + task
    elif mode == "audit":
        task = AUDIT_TASK_PREFIX + task
    elif mode == "fix_prompt":
        task = FIX_PROMPT_PREFIX + task
    
    payload = {"task": task}
    
    try:
        logger.info(
            "codex_remote: url=%s token_set=%s token_len=%s",
            endpoint, bool(runner_token), len(runner_token or "")
        )
        async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
            response = await client.post(endpoint, json=payload, headers=headers)
            logger.info("codex_remote: status=%s", response.status_code)
            
            if response.status_code != 200:
                return f"Error: Remote runner returned status {response.status_code}"
            
            data = response.json()
            
            if mode == "debug":
                return _format_debug(data)
            elif mode == "short":
                return _format_short(data)
            elif mode == "review":
                return _format_review(data)
            else:
                return _format_normal(data)
            
    except httpx.TimeoutException:
        return "Error: Request timed out after 180 seconds"
    except httpx.ConnectError:
        return "Error: Could not connect to remote runner"
    except httpx.RequestError as e:
        logger.error(f"Codex remote request error: {e}")
        return f"Error: Network error - {type(e).__name__}"
    except Exception as e:
        logger.error(f"Codex remote unexpected error: {e}")
        return f"Error: {str(e)[:200]}"


async def check_runner_health() -> str:
    """
    Check if the remote Codex runner is reachable.
    
    Returns:
        "OK" if reachable, "Not reachable" otherwise
    """
    runner_url = _get_runner_url()
    
    if not runner_url:
        return "Not reachable (CODEX_RUNNER_URL not configured)"
    
    endpoint = f"{runner_url.rstrip('/')}/health"
    
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(endpoint)
            
            if response.status_code == 200:
                return "OK"
            else:
                return f"Not reachable (status {response.status_code})"
                
    except httpx.TimeoutException:
        return "Not reachable (timeout)"
    except httpx.ConnectError:
        return "Not reachable (connection failed)"
    except httpx.RequestError:
        return "Not reachable (network error)"
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return "Not reachable (error)"


@dataclass
class CodexJobResult:
    """Result from polling a Codex job."""
    job_id: str
    status: str
    data: Optional[dict] = None
    error: Optional[str] = None


async def start_codex_job(task: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Start a Codex job on the remote runner service.
    
    Args:
        task: The task description to send to Codex
        
    Returns:
        Tuple of (job_id, sync_status, error). If successful, error is None.
    """
    runner_url = _get_runner_url()
    runner_token = _get_runner_token()
    
    if not runner_url:
        return None, None, "CODEX_RUNNER_URL not configured"
    
    if not runner_token:
        return None, None, "CODEX_RUNNER_TOKEN not configured"
    
    endpoint = f"{runner_url.rstrip('/')}/v1/codex_jobs"
    
    headers = {"Content-Type": "application/json"}
    if runner_token:
        headers["X-RUNNER-TOKEN"] = runner_token
    
    payload = {"task": task}
    
    try:
        logger.info(
            "codex_job_start: url=%s token_set=%s token_len=%s",
            endpoint, bool(runner_token), len(runner_token or "")
        )
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(endpoint, json=payload, headers=headers)
            logger.info("codex_job_start: status=%s", response.status_code)
            
            if response.status_code not in (200, 201, 202):
                return None, None, f"Remote runner returned status {response.status_code}"
            
            data = response.json()
            job_id = data.get("job_id") or data.get("id")
            sync_status = data.get("sync_status")
            
            if not job_id:
                return None, None, "No job_id in response"
            
            return job_id, sync_status, None
            
    except httpx.TimeoutException:
        return None, None, "Request timed out"
    except httpx.ConnectError:
        return None, None, "Could not connect to remote runner"
    except httpx.RequestError as e:
        logger.error(f"Codex job start error: {e}")
        return None, None, f"Network error - {type(e).__name__}"
    except Exception as e:
        logger.error(f"Codex job start unexpected error: {e}")
        return None, None, str(e)[:200]


async def poll_codex_job(job_id: str) -> CodexJobResult:
    """
    Poll a Codex job for status/results.
    
    Args:
        job_id: The job ID to poll
        
    Returns:
        CodexJobResult with status and data
    """
    runner_url = _get_runner_url()
    runner_token = _get_runner_token()
    
    if not runner_url:
        return CodexJobResult(job_id=job_id, status="error", error="CODEX_RUNNER_URL not configured")
    
    endpoint = f"{runner_url.rstrip('/')}/v1/codex_jobs/{job_id}"
    
    headers = {}
    if runner_token:
        headers["X-RUNNER-TOKEN"] = runner_token
    
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(endpoint, headers=headers)
            logger.info("codex_job_poll: job_id=%s status=%s", job_id, response.status_code)
            
            if response.status_code == 404:
                return CodexJobResult(job_id=job_id, status="not_found", error="Job not found")
            
            if response.status_code != 200:
                return CodexJobResult(
                    job_id=job_id, 
                    status="error", 
                    error=f"Remote runner returned status {response.status_code}"
                )
            
            payload = response.json()
            status = payload.get("status", "unknown")
            
            if status in ("done", "completed", "complete", "finished"):
                res = payload.get("result") or {}
                data = {
                    "stdout": res.get("stdout", ""),
                    "stderr": res.get("stderr", ""),
                    "exit_code": res.get("exit_code"),
                    "duration_ms": res.get("duration_ms"),
                    "ok": res.get("ok", True),
                    "sync_status": res.get("sync_status") or payload.get("sync_status"),
                }
            else:
                data = payload
            
            return CodexJobResult(job_id=job_id, status=status, data=data)
            
    except httpx.TimeoutException:
        return CodexJobResult(job_id=job_id, status="error", error="Poll request timed out")
    except httpx.ConnectError:
        return CodexJobResult(job_id=job_id, status="error", error="Could not connect to remote runner")
    except httpx.RequestError as e:
        logger.error(f"Codex job poll error: {e}")
        return CodexJobResult(job_id=job_id, status="error", error=f"Network error - {type(e).__name__}")
    except Exception as e:
        logger.error(f"Codex job poll unexpected error: {e}")
        return CodexJobResult(job_id=job_id, status="error", error=str(e)[:200])


async def run_codex_job_async(
    task: str,
    *,
    mode: OutputMode = "normal",
    poll_interval: float = 2.0,
    timeout_seconds: float = 600.0,
) -> tuple[str, Optional[str]]:
    """
    Start a Codex job and poll until completion or timeout.
    
    Args:
        task: The task description to send to Codex
        mode: Output mode for formatting the result
        poll_interval: Seconds between polls (default 2.0)
        timeout_seconds: Max wait time in seconds (default 600 = 10 minutes)
        
    Returns:
        Tuple of (formatted_result, job_id). job_id is included if still running.
    """
    if mode == "short":
        task = (
            "Answer in <=10 lines. Quote at most 2 snippets (<=20 lines each). "
            "No extra commentary.\n\n" + task
        )
    elif mode == "review":
        task = REVIEW_TASK_PREFIX + task
    elif mode == "audit":
        task = AUDIT_TASK_PREFIX + task
    elif mode == "fix_prompt":
        task = FIX_PROMPT_PREFIX + task
    
    job_id, sync_status, error = await start_codex_job(task)
    
    if error:
        return f"Error starting job: {error}", None
    
    if not job_id:
        return "Error: No job ID returned", None
    
    logger.info("codex_job_async: started job_id=%s sync_status=%s", job_id, sync_status)
    
    elapsed = 0.0
    while elapsed < timeout_seconds:
        await asyncio.sleep(poll_interval)
        elapsed += poll_interval
        
        result = await poll_codex_job(job_id)
        
        if result.error:
            return f"Error polling job: {result.error}", job_id
        
        if result.status in ("completed", "complete", "done", "finished"):
            data = result.data or {}
            if sync_status and not data.get("sync_status"):
                data["sync_status"] = sync_status
            
            if mode == "debug":
                return _format_debug(data), job_id
            elif mode == "short":
                return _format_short(data), job_id
            elif mode == "review":
                return _format_review(data, job_id=job_id), job_id
            elif mode == "audit":
                return _format_review(data, job_id=job_id), job_id
            else:
                return _format_normal(data), job_id
        
        if result.status in ("failed", "error"):
            data = result.data or {}
            error_msg = data.get("error") or data.get("stderr") or "Unknown error"
            return f"Job {job_id} failed: {error_msg[:500]}\nUse /codex_job {job_id} to check logs.", job_id
        
        if result.status in ("cancelled", "canceled"):
            return f"Job {job_id} was cancelled.", job_id
    
    msg = f"Still running. JOB: {job_id}\nTry /codex_job {job_id} to check status later."
    if sync_status:
        msg = f"SYNC: {sync_status}\n\n{msg}"
    return msg, job_id


async def fetch_codex_job(job_id: str, mode: OutputMode = "normal") -> str:
    """
    Fetch the current status or result of a Codex job.
    
    Args:
        job_id: The job ID to fetch
        mode: Output mode for formatting
        
    Returns:
        Formatted result or status message
    """
    result = await poll_codex_job(job_id)
    
    if result.error:
        return f"Error: {result.error}"
    
    if result.status in ("completed", "complete", "done", "finished"):
        data = result.data or {}
        
        if mode == "debug":
            return _format_debug(data)
        elif mode == "short":
            return _format_short(data)
        elif mode in ("review", "audit"):
            return _format_review(data, job_id=job_id)
        else:
            return _format_normal(data)
    
    if result.status in ("failed", "error"):
        data = result.data or {}
        error_msg = data.get("error") or data.get("stderr") or "Unknown error"
        return f"Job {job_id} failed: {error_msg[:500]}"
    
    if result.status in ("cancelled", "canceled"):
        return f"Job {job_id} was cancelled."
    
    if result.status in ("pending", "queued", "running", "in_progress"):
        return f"Job {job_id} is still {result.status}. Try again later."
    
    return f"Job {job_id} status: {result.status}"
