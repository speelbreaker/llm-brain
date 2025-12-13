"""
Codex Remote Runner for Telegram Bot integration.

Calls a remote Codex runner service via HTTP POST.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

TIMEOUT_SECONDS = 180


def _get_runner_url() -> Optional[str]:
    """Get the Codex runner URL from environment."""
    return os.environ.get("CODEX_RUNNER_URL")


def _get_runner_token() -> Optional[str]:
    """Get the Codex runner token from environment."""
    return os.environ.get("CODEX_RUNNER_TOKEN")


async def run_codex_remote(task: str) -> str:
    """
    Run a Codex task on the remote runner service.
    
    Args:
        task: The task description to send to Codex
        
    Returns:
        Formatted plain-text result (stdout first, then stderr if present)
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
    
    payload = {"task": task}
    
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
            response = await client.post(endpoint, json=payload, headers=headers)
            
            if response.status_code != 200:
                return f"Error: Remote runner returned status {response.status_code}"
            
            data = response.json()
            
            stdout = data.get("stdout", "")
            stderr = data.get("stderr", "")
            
            result_parts = []
            if stdout:
                result_parts.append(stdout)
            if stderr:
                result_parts.append(f"\n--- stderr ---\n{stderr}")
            
            if not result_parts:
                return "Codex completed with no output."
            
            return "".join(result_parts)
            
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
