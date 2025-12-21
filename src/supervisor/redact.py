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

SENSITIVE_KEYWORDS = [
    "token",
    "secret",
    "password",
    "apikey",
    "api_key",
    "auth",
    "bearer",
    "cookie",
    "key",
    "credential",
    "session",
]

TOKEN_PATTERNS = [
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"gho_[A-Za-z0-9]{20,}"),
    re.compile(r"ghr_[A-Za-z0-9]{20,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"Bearer\s+[A-Za-z0-9\-_\.]{15,}", re.IGNORECASE),
    re.compile(r"x-goog-api-key[:=]\s*\S+", re.IGNORECASE),
    re.compile(r"bearer\s+[A-Za-z0-9\-_\.]{15,}", re.IGNORECASE),
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"sk-proj-[A-Za-z0-9\-_]{20,}"),
    re.compile(r"sk-or-[A-Za-z0-9\-_]{20,}"),
    re.compile(r"AIza[A-Za-z0-9\-_]{35}"),
    re.compile(r"[0-9]{9,12}:[A-Za-z0-9_\-]{35}"),
    re.compile(r"xoxb-[A-Za-z0-9\-]{50,}"),
    re.compile(r"xoxp-[A-Za-z0-9\-]{50,}"),
    re.compile(r"npm_[A-Za-z0-9]{36}"),
    re.compile(r"pypi-[A-Za-z0-9]{40,}"),
    re.compile(r"AKIA[A-Z0-9]{16}"),
    re.compile(
        r"(?:secret|token|key|password|apikey|api_key|auth)[\s:=]+['\"]?[A-Za-z0-9\-_\.]{16,}['\"]?",
        re.IGNORECASE,
    ),
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


def _is_sensitive_key(key: str) -> bool:
    lowered = key.lower()
    return any(keyword in lowered for keyword in SENSITIVE_KEYWORDS)


def _redact_value(value, settings: "SupervisorSettings", key: str | None = None):
    if isinstance(value, dict):
        return {
            k: _redact_value(v, settings, k) if not _is_sensitive_key(str(k)) else REDACTED
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_redact_value(item, settings) for item in value]
    if isinstance(value, str):
        redacted_value = redact_secrets(value, settings)
        if key and _is_sensitive_key(key):
            return REDACTED
        return redacted_value
    if key and _is_sensitive_key(str(key)):
        return REDACTED
    return value


def redact_job_for_api(job_dict: dict, settings: "SupervisorSettings") -> dict:
    """Redact secrets from a job dict before returning via API or logging."""
    return _redact_value(job_dict, settings)
