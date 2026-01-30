#!/usr/bin/env python3
"""Codex-based diff reviewer for local pre-push gating.

Reads a git diff from stdin (or via --diff-file) and asks the configured LLM
for a structured review. Exits non-zero if severity >= threshold.

This is intentionally conservative: it is used as a safety gate.

Usage:
  git diff --cached | python3 tools/codex_review_diff.py
  python3 tools/codex_review_diff.py --diff-file /tmp/diff.patch

Env:
  - OPENAI_API_KEY (or source /etc/llmagentbrain/platform.env before running)
  - CODE_REVIEW_MODEL (optional, default: gpt-5.2)
  - CODE_REVIEW_THRESHOLD (optional, default: HIGH)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Literal


Severity = Literal["INFO", "LOW", "MEDIUM", "HIGH", "CRITICAL"]


def _read_diff(args: argparse.Namespace) -> str:
    if args.diff_file:
        return open(args.diff_file, "r", encoding="utf-8", errors="ignore").read()
    return sys.stdin.read()


def _severity_rank(s: str) -> int:
    order = {"INFO": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
    return order.get((s or "").upper().strip(), 0)


def _call_openai(prompt: str) -> dict[str, Any]:
    # Use the project dependency (already used elsewhere in the repo)
    from openai import OpenAI

    api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing")

    model = (os.environ.get("CODE_REVIEW_MODEL") or "gpt-5.2").strip()
    timeout_s = float(os.environ.get("CODE_REVIEW_TIMEOUT_S") or "45")

    client = OpenAI(api_key=api_key)

    # Prefer the shared timeout helper if available
    try:
        from src.llm_client import chat_completions_create_with_timeout

        resp = chat_completions_create_with_timeout(
            client,
            timeout_s=timeout_s,
            req_kwargs={
                "model": model,
                "messages": [
                    {"role": "system", "content": "Return ONLY valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                "response_format": {"type": "json_object"},
                "max_completion_tokens": 1200,
            },
        )
        content = resp.choices[0].message.content or "{}"
    except Exception:
        # Fallback to raw call
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Return ONLY valid JSON."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            max_completion_tokens=1200,
            timeout=timeout_s,
        )
        content = resp.choices[0].message.content or "{}"

    data = json.loads(content)
    if not isinstance(data, dict):
        raise ValueError("model did not return a JSON object")
    return data


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--diff-file", default="", help="Read diff from file instead of stdin")
    ap.add_argument(
        "--threshold",
        default=os.environ.get("CODE_REVIEW_THRESHOLD") or "HIGH",
        help="Fail if severity >= threshold (default: HIGH)",
    )
    args = ap.parse_args()

    diff = _read_diff(args)
    if not diff.strip():
        print("[codex-review] No diff; skipping.")
        return 0

    prompt = (
        "You are a senior code reviewer. Review the following git diff. "
        "Find correctness bugs, security issues, concurrency hazards, missing guards, "
        "broken feature flags, and platform portability issues. "
        "Be strict.\n\n"
        "Return JSON with EXACT shape:\n"
        "{\n"
        "  \"overall\": {\"severity\": \"INFO|LOW|MEDIUM|HIGH|CRITICAL\", \"summary\": string},\n"
        "  \"issues\": [\n"
        "    {\"severity\": \"INFO|LOW|MEDIUM|HIGH|CRITICAL\", \"title\": string, \"details\": string, \"files\": [string], \"suggested_fix\": string}\n"
        "  ]\n"
        "}\n\n"
        "DIFF:\n" + diff[:180_000]
    )

    try:
        review = _call_openai(prompt)
    except Exception as e:
        print(f"[codex-review] ERROR calling model: {e}")
        # Fail closed: we don't want silent skips when the gate is expected.
        return 2

    overall = (review.get("overall") or {}) if isinstance(review, dict) else {}
    severity = str(overall.get("severity") or "INFO").upper()
    summary = str(overall.get("summary") or "")

    print(f"[codex-review] severity={severity} summary={summary}")

    issues = review.get("issues")
    if isinstance(issues, list) and issues:
        for it in issues[:30]:
            if not isinstance(it, dict):
                continue
            sev = str(it.get("severity") or "INFO").upper()
            title = str(it.get("title") or "(no title)")
            files = it.get("files") or []
            files_s = ", ".join(files) if isinstance(files, list) else ""
            print(f"- [{sev}] {title}" + (f" ({files_s})" if files_s else ""))

    if _severity_rank(severity) >= _severity_rank(args.threshold):
        print(f"[codex-review] FAIL: severity {severity} >= threshold {args.threshold}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
