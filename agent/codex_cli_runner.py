"""
Codex CLI Runner for Telegram Bot integration.

IMPORTANT: The OpenAI Codex CLI (`@openai/codex`) uses an internal sandbox (Landlock LSM)
that conflicts with Replit's sandbox environment. This causes "Permission denied" errors.

As a workaround, this module provides:
1. A status check that explains the limitation
2. An alternative implementation using the OpenAI API directly for code-related queries

For full Codex CLI functionality, run it locally or in a non-sandboxed environment.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import subprocess
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

MAX_OUTPUT_SIZE = 4000
TIMEOUT_DEFAULT = 180

SECRET_PATTERNS = [
    r'sk-[a-zA-Z0-9]{20,}',
    r'sk_live_[a-zA-Z0-9]+',
    r'sk_test_[a-zA-Z0-9]+',
    r'OPENAI_API_KEY=[^\s]+',
    r'Bearer\s+[a-zA-Z0-9._-]+',
]


def redact_secrets(text: str) -> str:
    """Redact known secret patterns from output."""
    result = text
    for pattern in SECRET_PATTERNS:
        result = re.sub(pattern, '[REDACTED]', result, flags=re.IGNORECASE)
    return result


def check_codex_available() -> Tuple[bool, str]:
    """
    Check if Codex CLI is available and functional.
    
    Returns:
        Tuple of (is_available, message)
    """
    if not os.path.exists('node_modules/@openai/codex'):
        return False, "Codex CLI not installed. Run 'npm install' to install."
    
    try:
        result = subprocess.run(
            ['node', '-e', 'console.log(require("@openai/codex/package.json").version)'],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=os.getcwd(),
        )
        if result.returncode == 0:
            version = result.stdout.strip()
            return True, f"Codex CLI v{version} installed (but sandbox conflicts with Replit)"
        else:
            return False, f"Codex CLI check failed: {result.stderr.strip()}"
    except subprocess.TimeoutExpired:
        return False, "Codex CLI check timed out"
    except Exception as e:
        return False, f"Codex CLI check error: {str(e)}"


def get_codex_status() -> str:
    """Get a formatted status message about Codex CLI availability."""
    lines = []
    
    try:
        node_result = subprocess.run(['node', '-v'], capture_output=True, text=True, timeout=5)
        node_version = node_result.stdout.strip() if node_result.returncode == 0 else "not found"
    except Exception:
        node_version = "error"
    
    try:
        npm_result = subprocess.run(['npm', '-v'], capture_output=True, text=True, timeout=5)
        npm_version = npm_result.stdout.strip() if npm_result.returncode == 0 else "not found"
    except Exception:
        npm_version = "error"
    
    lines.append(f"Node.js: {node_version}")
    lines.append(f"npm: {npm_version}")
    
    available, msg = check_codex_available()
    lines.append(f"Codex CLI: {msg}")
    
    api_key_set = bool(os.environ.get('OPENAI_API_KEY'))
    lines.append(f"OPENAI_API_KEY: {'configured' if api_key_set else 'NOT SET'}")
    
    if available:
        lines.append("")
        lines.append("NOTE: Codex CLI sandbox conflicts with Replit environment.")
        lines.append("Use /ask for code questions via OpenAI API instead.")
    
    return "\n".join(lines)


async def codex_exec(task: str, timeout_sec: int = TIMEOUT_DEFAULT) -> str:
    """
    Attempt to execute a Codex CLI task.
    
    Due to sandbox conflicts with Replit, this will likely fail.
    Use the OpenAI API integration (/ask command) instead.
    
    Args:
        task: The task/prompt for Codex
        timeout_sec: Maximum execution time in seconds
        
    Returns:
        Output from Codex or error message
    """
    if not task or not task.strip():
        return "Error: No task provided"
    
    if len(task) > 2000:
        return "Error: Task too long (max 2000 characters)"
    
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        return "Error: OPENAI_API_KEY not configured in Replit Secrets"
    
    available, msg = check_codex_available()
    if not available:
        return f"Error: {msg}"
    
    sanitized_task = task.replace('"', '\\"').replace('$', '\\$').replace('`', '\\`')
    
    cmd = [
        'npx', '@openai/codex', 'exec',
        '--dangerously-bypass-approvals-and-sandbox',
        '--json',
        sanitized_task
    ]
    
    try:
        env = os.environ.copy()
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            cwd=os.getcwd(),
        )
        
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout_sec
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            return f"Error: Codex execution timed out after {timeout_sec}s"
        
        output = stdout.decode('utf-8', errors='replace')
        error_output = stderr.decode('utf-8', errors='replace')
        
        if process.returncode != 0:
            if 'Permission denied' in error_output or 'run_parent' in error_output:
                return (
                    "Error: Codex CLI sandbox conflicts with Replit environment.\n\n"
                    "The Codex CLI uses Landlock LSM for sandboxing, which is incompatible "
                    "with Replit's container sandbox.\n\n"
                    "Alternative: Use /ask <question> to query the codebase via OpenAI API."
                )
            return f"Error: Codex failed (code {process.returncode})\n{redact_secrets(error_output)[:500]}"
        
        output = redact_secrets(output)
        if len(output) > MAX_OUTPUT_SIZE:
            output = output[:MAX_OUTPUT_SIZE] + f"\n\n[Truncated, {len(output) - MAX_OUTPUT_SIZE} chars omitted]"
        
        return output if output.strip() else "Codex completed with no output"
        
    except Exception as e:
        logger.exception("Codex execution error")
        return f"Error: {str(e)}"


async def codex_via_api(task: str) -> str:
    """
    Alternative: Use OpenAI API directly for code-related queries.
    
    This bypasses the Codex CLI entirely and uses the same OpenAI API
    that powers the /ask command.
    """
    try:
        from openai import OpenAI
        
        client = OpenAI()
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful coding assistant. Answer questions about code, "
                        "explain concepts, and provide code examples. Be concise and practical."
                    )
                },
                {"role": "user", "content": task}
            ],
            max_tokens=2000,
            temperature=0.7,
        )
        
        return response.choices[0].message.content or "No response generated"
        
    except Exception as e:
        logger.exception("OpenAI API error")
        return f"Error using OpenAI API: {str(e)}"
