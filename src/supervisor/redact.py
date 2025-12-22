"""Secret redaction utilities for PR Supervisor.

Redacts sensitive tokens and keys from text before:
- Posting PR comments
- Sending Telegram messages
- Returning API payloads
"""

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import SupervisorSettings

REDACTED = "***REDACTED***"

TOKEN_PATTERNS = [
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"gho_[A-Za-z0-9]{20,}"),
    re.compile(r"ghr_[A-Za-z0-9]{20,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"Bearer\s+[A-Za-z0-9\-_\.]{15,}", re.IGNORECASE),
    re.compile(r"x-goog-api-key[:=]\s*\S+", re.IGNORECASE),
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"sk-proj-[A-Za-z0-9\-_]{20,}"),
    re.compile(r"AIza[A-Za-z0-9\-_]{35}"),
    re.compile(r"[0-9]{9,12}:[A-Za-z0-9_\-]{35}"),
    re.compile(r"xoxb-[A-Za-z0-9\-]{50,}"),
    re.compile(r"xoxp-[A-Za-z0-9\-]{50,}"),
    re.compile(r"npm_[A-Za-z0-9]{36}"),
    re.compile(r"pypi-[A-Za-z0-9]{40,}"),
    re.compile(r"AKIA[A-Z0-9]{16}"),
    re.compile(r"(?:secret|token|key|password|apikey|api_key|auth)[\s:=]+['\"]?[A-Za-z0-9\-_\.]{16,}['\"]?", re.IGNORECASE),
    re.compile(r"[A-Za-z0-9+/]{40,}={0,2}"),
]


def redact_secrets(text: str, settings: "SupervisorSettings") -> str:
    """Redact secrets from text.
    
    Args:
        text: The text to redact secrets from
        settings: SupervisorSettings containing configured secrets
        
    Returns:
        Text with secrets replaced by ***REDACTED***
    """
    if not text:
        return text
    
    result = text
    
    configured_secrets = [
        settings.github_token,
        settings.github_webhook_secret,
        settings.openai_api_key,
        settings.telegram_bot_token,
        settings.gemini_api_key,
    ]
    
    for secret in configured_secrets:
        if secret and len(secret) > 8:
            result = result.replace(secret, REDACTED)
    
    for pattern in TOKEN_PATTERNS:
        result = pattern.sub(REDACTED, result)
    
    return result


def redact_job_for_api(job_dict: dict, settings: "SupervisorSettings") -> dict:
    """Redact secrets from a job dict before returning via API.
    
    Args:
        job_dict: The job data as a dictionary
        settings: SupervisorSettings containing configured secrets
        
    Returns:
        Job dict with secrets redacted from text fields
    """
    sensitive_fields = [
        "error_message",
        "final_message",
        "workspace_path",
    ]
    
    result = job_dict.copy()
    
    for field in sensitive_fields:
        if field in result and isinstance(result[field], str):
            result[field] = redact_secrets(result[field], settings)
    
    if "verification" in result and isinstance(result["verification"], dict):
        verification = result["verification"].copy()
        if "failure_summary" in verification and isinstance(verification["failure_summary"], str):
            verification["failure_summary"] = redact_secrets(verification["failure_summary"], settings)
        if "checks" in verification and isinstance(verification["checks"], list):
            checks = []
            for check in verification["checks"]:
                check_copy = check.copy()
                if "stdout" in check_copy and isinstance(check_copy["stdout"], str):
                    check_copy["stdout"] = redact_secrets(check_copy["stdout"], settings)
                if "stderr" in check_copy and isinstance(check_copy["stderr"], str):
                    check_copy["stderr"] = redact_secrets(check_copy["stderr"], settings)
                checks.append(check_copy)
            verification["checks"] = checks
        result["verification"] = verification
    
    def _redact_fix_attempts(attempts: list[dict]) -> list[dict]:
        redacted = []
        for attempt in attempts:
            attempt_copy = attempt.copy()
            if "codex_output" in attempt_copy and isinstance(attempt_copy["codex_output"], str):
                attempt_copy["codex_output"] = redact_secrets(attempt_copy["codex_output"], settings)
            if "codex_prompt" in attempt_copy and isinstance(attempt_copy["codex_prompt"], str):
                attempt_copy["codex_prompt"] = redact_secrets(attempt_copy["codex_prompt"], settings)
            redacted.append(attempt_copy)
        return redacted

    if "fix_attempt_history" in result and isinstance(result["fix_attempt_history"], list):
        result["fix_attempt_history"] = _redact_fix_attempts(result["fix_attempt_history"])

    if "fix_attempts" in result and isinstance(result["fix_attempts"], list):
        result["fix_attempts"] = _redact_fix_attempts(result["fix_attempts"])

    if "fix_attempt_history" not in result and "fix_attempts" in result:
        result["fix_attempt_history"] = result["fix_attempts"]
    
    return result
