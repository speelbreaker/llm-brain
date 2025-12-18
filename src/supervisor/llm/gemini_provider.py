from __future__ import annotations

import json
import re
from typing import Any, Optional

import httpx

from .base import LLMProvider

_GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"


def _extract_first_json_object(text: str) -> dict[str, Any]:
    if not text:
        return {}
    text = text.strip()

    # pure JSON
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {"value": obj}
    except Exception:
        pass

    # find first {...}
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return {}
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else {"value": obj}
    except Exception:
        return {}


class GeminiProvider(LLMProvider):
    """
    Gemini REST API provider using:
    POST https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent
    Header: x-goog-api-key
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        http: Optional[httpx.AsyncClient] = None,
        api_base: str = _GEMINI_API_BASE,
    ):
        if not api_key:
            raise ValueError("GeminiProvider requires api_key")
        self.api_key = api_key
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.http = http or httpx.AsyncClient(timeout=httpx.Timeout(60.0))
        self._owns_client = http is None

    async def generate(
        self,
        prompt: str,
        model: str | None = None,
        max_tokens: int = 500,
        temperature: float = 0.7,
        system: str | None = None,
    ) -> str:
        target_model = model or self.model
        url = f"{self.api_base}/models/{target_model}:generateContent"
        text = f"{system}\n\n{prompt}" if system else prompt

        payload = {
            "contents": [{"parts": [{"text": text}]}],
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": temperature,
            },
        }
        r = await self.http.post(
            url,
            headers={
                "x-goog-api-key": self.api_key,
                "content-type": "application/json",
            },
            json=payload,
        )
        r.raise_for_status()
        data = r.json()

        candidates = data.get("candidates") or []
        if not candidates:
            return ""
        content = (candidates[0] or {}).get("content") or {}
        parts = content.get("parts") or []
        if not parts:
            return ""
        return (parts[0] or {}).get("text") or ""

    async def generate_json(
        self,
        prompt: str,
        model: str | None = None,
        schema_hint: str | None = None,
        max_tokens: int = 500,
        temperature: float = 0.3,
        system: str | None = None,
    ) -> dict[str, Any]:
        system_msg = system or "Return ONLY valid JSON. No markdown. No prose."
        prompt2 = (
            f"{prompt}\n\nJSON schema hint:\n{schema_hint}" if schema_hint else prompt
        )
        txt = await self.generate(
            prompt2,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_msg,
        )
        return _extract_first_json_object(txt)

    async def close(self) -> None:
        if self._owns_client:
            await self.http.aclose()
