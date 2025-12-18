from __future__ import annotations

import json
import re
from typing import Any, Optional

from openai import AsyncOpenAI

from .base import LLMProvider


def _strip_code_fences(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        t = "\n".join(lines).strip()
    return t


def _extract_first_json_object(text: str) -> dict[str, Any]:
    t = _strip_code_fences(text)
    if not t:
        return {}
    # Fast path: pure JSON
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass
    # Best-effort: first {...} block
    m = re.search(r"\{.*\}", t, flags=re.DOTALL)
    if not m:
        return {}
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _looks_like_response_format_not_supported(err: Exception) -> bool:
    s = (getattr(err, "message", None) or str(err) or "").lower()
    return ("response_format" in s) and ("json_object" in s) and ("not supported" in s)


class OpenAIProvider(LLMProvider):
    """LLM provider using OpenAI Chat Completions."""

    def __init__(self, api_key: str | None = None):
        self.client = AsyncOpenAI(api_key=api_key) if api_key else AsyncOpenAI()

    async def generate(
        self,
        prompt: str,
        model: str,
        max_tokens: int = 500,
        temperature: float = 0.7,
        system: str | None = None,
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        resp = await self.client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return (resp.choices[0].message.content or "").strip()

    async def generate_json(
        self,
        prompt: str,
        model: str,
        schema_hint: str | None = None,
        max_tokens: int = 500,
        temperature: float = 0.3,
        system: str | None = None,
    ) -> dict[str, Any]:
        # We try strict JSON mode first; if model rejects it, we retry without response_format.
        sys_msg = (
            (system + "\n\n") if system else ""
        ) + "Return ONLY valid JSON. Do not include markdown, code fences, or commentary."

        if schema_hint:
            prompt2 = f"{prompt}\n\nJSON schema hint:\n{schema_hint}"
        else:
            prompt2 = prompt

        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": prompt2},
        ]

        try:
            resp = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                response_format={"type": "json_object"},
                max_tokens=max_tokens,
                temperature=temperature,
            )
            txt = resp.choices[0].message.content or "{}"
            return _extract_first_json_object(txt)

        except Exception as e:
            if not _looks_like_response_format_not_supported(e):
                raise

            # Fallback: same prompt, no response_format
            resp2 = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            txt2 = resp2.choices[0].message.content or "{}"
            return _extract_first_json_object(txt2)

    async def close(self) -> None:
        close_fn = getattr(self.client, "close", None)
        if close_fn:
            await close_fn()
