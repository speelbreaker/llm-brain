"""
AI Steward - Project Brain module.

Reads project meta-docs (ROADMAP, BACKLOG, UI gaps, HEALTHCHECK, replit.md)
and uses LLM to summarize and propose next actions.

Never interacts with Deribit or trading - purely a project planning/QA helper.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from src.config import settings


def _load_file_snippet(rel_path: str, max_chars: int = 4000) -> str:
    """
    Load up to max_chars from a text file relative to the project root.
    Returns "" if the file does not exist or cannot be read.
    """
    try:
        root = Path(__file__).resolve().parent.parent
        path = root / rel_path
        if not path.exists():
            return ""
        text = path.read_text(encoding="utf-8", errors="ignore")
        return text[:max_chars]
    except Exception:
        return ""


class StewardItem(BaseModel):
    """A single actionable item from the steward report."""
    id: str = Field(..., description="Short identifier, e.g. 'P1-ui-bots-status-badges'")
    area: str = Field(..., description="Area of the project, e.g. 'UI / Bots', 'Backtesting', 'Healthcheck'")
    priority: str = Field(..., description="Priority label like P0/P1/P2/info")
    reason: str = Field(..., description="Why this item matters now")
    suggested_change: str = Field(..., description="One-sentence description of the concrete change")


class StewardReport(BaseModel):
    """The full steward report."""
    ok: bool = True
    generated_at: str
    llm_used: bool = False
    summary: str
    top_items: List[StewardItem] = Field(default_factory=list)
    builder_prompt: str = ""


_last_report: Optional[StewardReport] = None


def get_last_report() -> Optional[StewardReport]:
    """Return the last generated steward report, or None if never run."""
    return _last_report


def generate_steward_report() -> StewardReport:
    """
    Build a StewardReport by reading project meta-docs and (if possible)
    calling the OpenAI client with a strict JSON schema.

    If LLM access is unavailable, returns a fallback report without raising.
    """
    global _last_report

    now = datetime.now(timezone.utc).isoformat()

    corpus: Dict[str, str] = {
        "roadmap": _load_file_snippet("ROADMAP.md"),
        "backlog": _load_file_snippet("ROADMAP_BACKLOG.md"),
        "ui_gaps": _load_file_snippet("UI_FEATURE_GAPS.md"),
        "healthcheck": _load_file_snippet("HEALTHCHECK.md"),
        "replit": _load_file_snippet("replit.md"),
    }

    fallback = StewardReport(
        ok=True,
        generated_at=now,
        llm_used=False,
        summary=(
            "AI Steward fallback report – LLM not configured or failed. "
            "Docs were loaded, but no AI summary is available."
        ),
        top_items=[],
        builder_prompt=(
            "LLM not available inside the app. Configure Replit/OpenAI integrations "
            "and re-run the steward to get AI-generated next steps."
        ),
    )

    try:
        from src.agent_brain_llm import _get_openai_client
        client = _get_openai_client()

        system_prompt = (
            "You are an AI 'project steward' for an options-trading bot repo. "
            "You see a subset of the ROADMAP, ROADMAP_BACKLOG, UI_FEATURE_GAPS, "
            "HEALTHCHECK, and replit.md. "
            "Your job is to:\n"
            "1) Summarize the current state in 2-4 sentences.\n"
            "2) Identify the next 3-5 concrete tasks that are both important and actionable.\n"
            "3) Prepare a short builder prompt that the user can paste into another AI "
            "   (the 'AI Builder') to implement those tasks.\n\n"
            "Respond ONLY with JSON matching this schema:\n"
            "{\n"
            '  "summary": "short paragraph",\n'
            '  "top_items": [\n'
            "    {\n"
            '      "id": "short-id",\n'
            '      "area": "area of the project",\n'
            '      "priority": "P0|P1|P2|info",\n'
            '      "reason": "why this matters now",\n'
            '      "suggested_change": "one-sentence change description"\n'
            "    }\n"
            "  ],\n"
            '  "builder_prompt": "prompt text for the AI builder"\n'
            "}\n"
            "Do not add any extra fields."
        )

        user_payload = {
            "settings": {
                "mode": settings.mode,
                "deribit_env": getattr(settings, "deribit_env", "testnet"),
            },
            "corpus": corpus,
        }

        response = client.chat.completions.create(
            model=settings.llm_model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload)},
            ],
            response_format={"type": "json_object"},
            max_completion_tokens=1024,
        )

        content = response.choices[0].message.content or "{}"
        raw = json.loads(content)

        summary = raw.get("summary", fallback.summary)
        items_raw = raw.get("top_items") or []
        builder_prompt = raw.get("builder_prompt", fallback.builder_prompt)

        items: List[StewardItem] = []
        for item in items_raw:
            try:
                items.append(
                    StewardItem(
                        id=str(item.get("id", "")) or "unnamed",
                        area=str(item.get("area", "unspecified")),
                        priority=str(item.get("priority", "info")),
                        reason=str(item.get("reason", "")),
                        suggested_change=str(item.get("suggested_change", "")),
                    )
                )
            except Exception:
                continue

        report = StewardReport(
            ok=True,
            generated_at=now,
            llm_used=True,
            summary=summary,
            top_items=items,
            builder_prompt=builder_prompt,
        )
    except Exception:
        report = fallback

    _last_report = report
    return report
