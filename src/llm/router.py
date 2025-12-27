"""LLM provider router with ordered fallback chains."""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

DEFAULT_MODEL_CHAINS: Dict[str, List[str]] = {
    "trading_optimist": ["zhipu:glm-4.7", "minimax:minimax-m2.1", "openai:gpt-5.2"],
    "trading_skeptic": ["google:gemini-3-pro", "zhipu:glm-4.7", "minimax:minimax-m2.1"],
    "trading_arbiter": [
        "openai:gpt-5.2",
        "google:gemini-3-pro",
        "zhipu:glm-4.7",
        "minimax:minimax-m2.1",
    ],
    "codex_coder": ["openai:codex", "google:gemini-3-pro"],
}

DEFAULT_CONFIG_PATH = Path("config/models.yaml")


class LLMRouterError(RuntimeError):
    """Base error for router failures."""


class LLMRetryableError(LLMRouterError):
    """Errors that allow fallback to the next provider."""


class LLMAuthError(LLMRouterError):
    """Authentication or permission errors."""


class LLMInvalidRequestError(LLMRouterError):
    """Invalid request or configuration errors."""


@dataclass(frozen=True)
class ModelTarget:
    provider: str
    model: str


class LLMRouter:
    """Route LLM calls across configured provider/model chains."""

    def __init__(
        self,
        *,
        config_path: Optional[Path] = None,
        model_chains: Optional[Dict[str, List[str]]] = None,
        providers: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._logger = logger or logging.getLogger(__name__)
        self._chains = model_chains or load_model_chains(config_path)
        self._providers = providers or _default_provider_registry()

    def call(
        self,
        role: str,
        messages: List[Dict[str, str]],
        json_schema: Optional[Dict[str, Any]] = None,
        tool_schema: Optional[Dict[str, Any]] = None,
    ) -> str:
        chain = self._chains.get(role) or []
        if not chain:
            raise LLMInvalidRequestError(f"No model chain configured for role '{role}'")

        targets = [_parse_target(item) for item in chain]
        last_retryable: Optional[Exception] = None

        for attempt, target in enumerate(targets, start=1):
            provider = self._providers.get(target.provider)
            if provider is None:
                raise LLMInvalidRequestError(f"Provider '{target.provider}' is not configured")

            self._log_event(
                "llm_router_attempt",
                role=role,
                provider=target.provider,
                model=target.model,
                attempt=attempt,
                attempts=len(targets),
            )
            try:
                result = provider.call(
                    messages,
                    model=target.model,
                    json_schema=json_schema,
                    tool_schema=tool_schema,
                )
                self._log_event(
                    "llm_router_success",
                    role=role,
                    provider=target.provider,
                    model=target.model,
                    attempt=attempt,
                    attempts=len(targets),
                )
                return result
            except Exception as exc:  # noqa: BLE001 - explicit classification below
                retryable = _is_retryable_error(exc)
                self._log_event(
                    "llm_router_failure",
                    role=role,
                    provider=target.provider,
                    model=target.model,
                    attempt=attempt,
                    attempts=len(targets),
                    retryable=retryable,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
                if retryable:
                    last_retryable = exc
                    continue
                raise exc

        raise LLMRetryableError("All providers failed") from last_retryable

    def _log_event(self, event: str, **fields: Any) -> None:
        payload = {"event": event, "timestamp": datetime.now(timezone.utc).isoformat(), **fields}
        self._logger.info(json.dumps(payload, sort_keys=True, ensure_ascii=True))


def load_model_chains(config_path: Optional[Path] = None) -> Dict[str, List[str]]:
    path = config_path or DEFAULT_CONFIG_PATH
    overrides: Dict[str, List[str]] = {}
    if path and path.exists():
        raw = path.read_text(encoding="utf-8")
        overrides = _parse_config(raw)

    merged = dict(DEFAULT_MODEL_CHAINS)
    for role, chain in overrides.items():
        if chain:
            merged[role] = chain
    return merged


def _parse_target(entry: str) -> ModelTarget:
    if ":" not in entry:
        raise LLMInvalidRequestError(f"Invalid model target '{entry}'. Expected provider:model.")
    provider, model = entry.split(":", 1)
    provider = provider.strip().lower()
    model = model.strip()
    if not provider or not model:
        raise LLMInvalidRequestError(f"Invalid model target '{entry}'.")
    return ModelTarget(provider=provider, model=model)


def _parse_config(raw: str) -> Dict[str, List[str]]:
    content = raw.strip()
    if not content:
        return {}
    if content.startswith("{"):
        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            raise LLMInvalidRequestError("Invalid JSON in model config") from exc
        return _normalize_chains(data)
    return _parse_simple_yaml(content)


def _normalize_chains(data: Any) -> Dict[str, List[str]]:
    if not isinstance(data, dict):
        raise LLMInvalidRequestError("Model config must be an object of role -> list")
    normalized: Dict[str, List[str]] = {}
    for key, value in data.items():
        if isinstance(value, list):
            normalized[key] = [str(item) for item in value]
        elif isinstance(value, str):
            normalized[key] = [value]
    return normalized


def _parse_simple_yaml(content: str) -> Dict[str, List[str]]:
    chains: Dict[str, List[str]] = {}
    current_key: Optional[str] = None
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.endswith(":"):
            current_key = line[:-1].strip()
            if current_key:
                chains[current_key] = []
            continue
        if line.startswith("-") and current_key:
            value = line[1:].strip().strip("'\"")
            if value:
                chains[current_key].append(value)
            continue
        if ":" in line and not line.startswith("-"):
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if value.startswith("[") and value.endswith("]"):
                inner = value[1:-1].strip()
                items = [item.strip().strip("'\"") for item in inner.split(",") if item.strip()]
                chains[key] = items
            elif value:
                chains[key] = [value.strip("'\"")]
            current_key = key
    return chains


def _default_provider_registry() -> Dict[str, Any]:
    return {
        "openai": _OpenAIProvider(),
        "google": _GeminiProvider(),
        "gemini": _GeminiProvider(),
        "zhipu": _OpenAIProvider(provider_name="zhipu"),
        "minimax": _OpenAIProvider(provider_name="minimax"),
    }


def _build_schema_suffix(
    json_schema: Optional[Dict[str, Any]],
    tool_schema: Optional[Dict[str, Any]],
) -> str:
    parts: List[str] = []
    if json_schema:
        parts.append(
            "Return ONLY valid JSON matching this schema:\n"
            + json.dumps(json_schema, indent=2, sort_keys=True, ensure_ascii=True)
        )
    if tool_schema:
        parts.append(
            "If tool usage is required, follow this tool schema:\n"
            + json.dumps(tool_schema, indent=2, sort_keys=True, ensure_ascii=True)
        )
    if not parts:
        return ""
    return "\n\n" + "\n\n".join(parts)


def _messages_to_prompt(messages: List[Dict[str, str]]) -> str:
    lines = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        lines.append(f"{role.upper()}: {content}")
    return "\n".join(lines)


def _is_retryable_error(exc: Exception) -> bool:
    if isinstance(exc, LLMRetryableError):
        return True
    if isinstance(exc, (LLMAuthError, LLMInvalidRequestError)):
        return False
    status = _extract_status_code(exc)
    if status is not None:
        if status in {408, 429} or 500 <= status <= 599:
            return True
        if status in {400, 401, 403}:
            return False
    message = str(exc).lower()
    if any(token in message for token in ("unauthorized", "forbidden", "invalid request", "authentication")):
        return False
    if any(
        token in message
        for token in (
            "timeout",
            "timed out",
            "rate limit",
            "too many requests",
            "temporarily unavailable",
            "connection reset",
            "connection refused",
        )
    ):
        return True
    if isinstance(exc, (TimeoutError, ConnectionError, OSError)):
        return True
    return False


def _extract_status_code(exc: Exception) -> Optional[int]:
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status
    response = getattr(exc, "response", None)
    if response is not None:
        resp_status = getattr(response, "status_code", None)
        if isinstance(resp_status, int):
            return resp_status
    return None


class _OpenAIProvider:
    def __init__(self, provider_name: str = "openai") -> None:
        self._provider_name = provider_name
        self._client = None

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        api_key, base_url = _get_openai_credentials(self._provider_name)
        if not api_key:
            raise LLMAuthError(f"Missing API key for provider '{self._provider_name}'")
        try:
            from openai import OpenAI  # type: ignore
        except Exception as exc:  # pragma: no cover - import failure
            raise LLMInvalidRequestError("OpenAI client is unavailable") from exc
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        return self._client

    def call(
        self,
        messages: List[Dict[str, str]],
        *,
        model: str,
        json_schema: Optional[Dict[str, Any]] = None,
        tool_schema: Optional[Dict[str, Any]] = None,
    ) -> str:
        client = self._ensure_client()
        payload = list(messages)
        schema_suffix = _build_schema_suffix(json_schema, tool_schema)
        if schema_suffix:
            payload.append({"role": "system", "content": schema_suffix})
        response = client.chat.completions.create(
            model=model,
            messages=payload,
            max_tokens=1200,
        )
        try:
            return response.choices[0].message.content or ""
        except Exception as exc:
            raise LLMInvalidRequestError("Malformed OpenAI response") from exc


class _GeminiProvider:
    def __init__(self) -> None:
        self._client = None

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import httpx
        except Exception as exc:  # pragma: no cover - import failure
            raise LLMInvalidRequestError("httpx is required for Gemini provider") from exc
        self._client = httpx.Client(timeout=60.0)
        return self._client

    def call(
        self,
        messages: List[Dict[str, str]],
        *,
        model: str,
        json_schema: Optional[Dict[str, Any]] = None,
        tool_schema: Optional[Dict[str, Any]] = None,
    ) -> str:
        api_key = os.environ.get("GEMINI_API_KEY")
        base_url = os.environ.get("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta")
        if not api_key:
            raise LLMAuthError("Missing GEMINI_API_KEY")
        prompt = _messages_to_prompt(messages) + _build_schema_suffix(json_schema, tool_schema)
        client = self._ensure_client()
        url = f"{base_url.rstrip('/')}/models/{model}:generateContent"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": 1200, "temperature": 0.3},
        }
        response = client.post(url, json=payload, headers={"x-goog-api-key": api_key})
        response.raise_for_status()
        data = response.json()
        try:
            candidates = data.get("candidates", [])
            if not candidates:
                return ""
            parts = candidates[0].get("content", {}).get("parts", [])
            if not parts:
                return ""
            return parts[0].get("text", "")
        except Exception as exc:
            raise LLMInvalidRequestError("Malformed Gemini response") from exc


def _get_openai_credentials(provider_name: str) -> tuple[Optional[str], Optional[str]]:
    provider_upper = provider_name.upper()
    if provider_name == "openai":
        api_key = os.environ.get("AI_INTEGRATIONS_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
        base_url = os.environ.get("AI_INTEGRATIONS_OPENAI_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
        return api_key, base_url
    api_key = os.environ.get(f"{provider_upper}_API_KEY")
    base_url = os.environ.get(f"{provider_upper}_BASE_URL")
    return api_key, base_url
