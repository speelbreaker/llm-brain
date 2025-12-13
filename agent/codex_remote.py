"""
Codex Remote Runner for Telegram Bot integration.

Calls a remote Codex runner service via HTTP POST.
Supports multiple output modes: normal, short, debug.
"""
from __future__ import annotations

import logging
import os
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
    return os.environ.get("CODEX_RUNNER_TOKEN")


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


def _format_review(data: dict) -> str:
    """Format response for review mode - includes sync_status if present."""
    stdout = data.get("stdout", "").strip()
    sync_status = data.get("sync_status")
    
    parts = []
    if sync_status:
        parts.append(f"SYNC: {sync_status}")
        parts.append("")
    
    if stdout:
        parts.append(_truncate(stdout, MAX_STDOUT_NORMAL))
    else:
        stderr = data.get("stderr", "").strip()
        ok = data.get("ok", True)
        exit_code = data.get("exit_code", 0)
        if not ok or exit_code != 0:
            snippet = _truncate(stderr, MAX_STDERR_NORMAL) if stderr else "No details available"
            parts.append(f"Codex failed (exit_code={exit_code}).\n{snippet}")
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
    
    headers = {
        "Content-Type": "application/json",
        "X-RUNNER-TOKEN": runner_token,
    }
    
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
        async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
            response = await client.post(endpoint, json=payload, headers=headers)
            
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
