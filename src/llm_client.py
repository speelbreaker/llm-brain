"""Shared helpers for calling LLM providers safely.

Goal: avoid duplicated timeout/fallback logic and avoid double-executing requests.
"""

from __future__ import annotations

from typing import Any, Dict


def chat_completions_create_with_timeout(
    client: Any,
    *,
    timeout_s: float,
    req_kwargs: Dict[str, Any],
) -> Any:
    """Call OpenAI-compatible chat.completions.create with best-effort timeout.

    Rules:
    - Prefer `client.with_options(timeout=...)` when available.
    - Fall back to passing `timeout=` to `.create(...)` only when supported.
    - Avoid double-executing the request: only retry on clear timeout-parameter incompatibility.
    """

    # Preferred path: per-request options (OpenAI SDK v1/v2)
    if hasattr(client, "with_options"):
        try:
            return client.with_options(timeout=timeout_s).chat.completions.create(**req_kwargs)
        except TypeError as e:
            # Only fall back if timeout/with_options is the issue.
            if "timeout" not in str(e):
                raise
        except Exception:
            # Do not fall back on generic errors (would cause duplicate calls).
            raise

    # Fallback: try passing timeout kwarg to create (some SDK variants)
    try:
        return client.chat.completions.create(**req_kwargs, timeout=timeout_s)
    except TypeError as e:
        msg = str(e)
        if "unexpected keyword argument" in msg and "timeout" in msg:
            return client.chat.completions.create(**req_kwargs)
        raise
