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

OutputMode = Literal["normal", "short", "debug"]


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
