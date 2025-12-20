"""LLM fallback orchestration with retries and circuit breaker."""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from pydantic import ValidationError

from ..config import SupervisorSettings
from .router import get_provider

logger = logging.getLogger(__name__)


RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}
PERMANENT_STATUS_CODES = {400, 401, 403, 404}


def _extract_json(text: str) -> dict:
    """Extract and parse JSON from model output."""
    content = text.strip()
    if content.startswith("```"):
        lines = content.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        content = "\n".join(lines).strip()
    if content.startswith("{") and content.endswith("}"):
        return json.loads(content)
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or start >= end:
        raise json.JSONDecodeError("No JSON object found", content, 0)
    return json.loads(content[start : end + 1])


def _status_from_error(exc: Exception) -> Optional[int]:
    status = getattr(exc, "status_code", None)
    if status is not None:
        return status
    response = getattr(exc, "response", None)
    if response is not None:
        return getattr(response, "status_code", None)
    return None


def _is_retryable_error(exc: Exception) -> bool:
    status = _status_from_error(exc)
    if status in RETRYABLE_STATUS_CODES:
        return True
    if status in PERMANENT_STATUS_CODES:
        return False
    msg = str(exc).lower()
    if "timeout" in msg or "timed out" in msg:
        return True
    if "rate limit" in msg or "rate_limit" in msg or "429" in msg:
        return True
    if "unauthorized" in msg or "forbidden" in msg or "invalid api key" in msg:
        return False
    return True


@dataclass
class ProviderCircuitBreaker:
    failure_threshold: int = 3
    window_seconds: int = 300
    cooldown_seconds: int = 300
    _failures: dict[str, list[float]] = field(default_factory=dict)
    _open_until: dict[str, float] = field(default_factory=dict)
    _clock: Callable[[], float] = time.time

    def record_failure(self, provider: str) -> None:
        now = self._clock()
        failures = [t for t in self._failures.get(provider, []) if now - t <= self.window_seconds]
        failures.append(now)
        self._failures[provider] = failures
        if len(failures) >= self.failure_threshold:
            self._open_until[provider] = now + self.cooldown_seconds

    def record_success(self, provider: str) -> None:
        self._failures.pop(provider, None)
        self._open_until.pop(provider, None)

    def is_open(self, provider: str) -> bool:
        now = self._clock()
        open_until = self._open_until.get(provider, 0)
        return open_until > now

    def summary(self) -> dict[str, dict[str, object]]:
        now = self._clock()
        summary: dict[str, dict[str, object]] = {}
        for provider in set(self._failures.keys()) | set(self._open_until.keys()):
            failures = [t for t in self._failures.get(provider, []) if now - t <= self.window_seconds]
            open_until = self._open_until.get(provider)
            summary[provider] = {
                "failures_recent": len(failures),
                "breaker_open": bool(open_until and open_until > now),
                "cooldown_seconds": max(0, int(open_until - now)) if open_until else 0,
            }
        return summary


_breaker = ProviderCircuitBreaker()


async def generate_json_with_fallback(
    settings: SupervisorSettings,
    provider_chain: list[str],
    model_for_provider: dict[str, str],
    prompt: str,
    schema_hint: str,
    validator: Callable[[dict], None],
    max_tokens: int = 600,
    temperature: float = 0.3,
    max_retries: int = 2,
) -> tuple[dict, str]:
    """Generate JSON with fallback chain and circuit breaker."""
    for provider_name in provider_chain:
        if _breaker.is_open(provider_name):
            logger.warning("LLM provider %s breaker open; skipping.", provider_name)
            continue
        try:
            provider = get_provider(provider_name, settings)
        except Exception as exc:
            logger.warning("LLM provider %s unavailable: %s", provider_name, type(exc).__name__)
            _breaker.record_failure(provider_name)
            continue
        model = model_for_provider[provider_name]
        attempts = 0
        while attempts < max_retries:
            attempts += 1
            try:
                full_prompt = (
                    f"{prompt}\n\nRespond with ONLY valid JSON matching this schema:\n{schema_hint}"
                )
                text = await provider.generate(
                    prompt=full_prompt,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature if attempts == 1 else 0.2,
                )
                parsed = _extract_json(text)
                validator(parsed)
                _breaker.record_success(provider_name)
                return parsed, provider_name
            except (json.JSONDecodeError, ValidationError, ValueError) as exc:
                logger.warning("LLM %s produced invalid JSON (attempt %s).", provider_name, attempts)
                if attempts >= max_retries:
                    _breaker.record_failure(provider_name)
                    break
                await _sleep_backoff(attempts)
            except Exception as exc:
                retryable = _is_retryable_error(exc)
                logger.warning("LLM %s error (attempt %s, retryable=%s).", provider_name, attempts, retryable)
                if not retryable:
                    _breaker.record_failure(provider_name)
                    break
                if attempts >= max_retries:
                    _breaker.record_failure(provider_name)
                    break
                await _sleep_backoff(attempts)

    raise LLMUnavailable("All providers unavailable")


async def _sleep_backoff(attempt: int) -> None:
    base_delay = 0.5
    max_delay = 6.0
    delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
    delay += random.uniform(0, delay * 0.2)
    await asyncio.sleep(delay)


class LLMUnavailable(Exception):
    """Raised when all LLM providers are unavailable."""


def get_provider_health() -> dict[str, dict[str, object]]:
    """Get circuit breaker health summary."""
    return _breaker.summary()
