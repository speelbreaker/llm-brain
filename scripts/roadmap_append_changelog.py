#!/usr/bin/env python3

import argparse
import os
import re
from datetime import datetime, timezone
from pathlib import Path


CHANGELOG_HEADER = "## Changelog (auto)"
CHANGELOG_HINT = "- (entries appended newest-first)"


def _now_utc_minute() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")


def _normalize_agent(value: str) -> str:
    v = value.strip()
    if v.lower() == "copilot":
        return "COPILOT"
    if v.lower() == "codex":
        return "CODEx"
    raise ValueError("--agent must be CODEx or COPILOT")


def _normalize_summary(summary: str) -> str:
    # Allow callers to pass literal "\n" sequences (common when shell-escaping)
    summary = summary.replace("\\n", "\n")
    raw_lines = [
        line.strip()
        for line in summary.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    ]

    cleaned: list[str] = []
    for line in raw_lines:
        if not line:
            continue
        line = re.sub(r"^[-*•]+\s*", "", line)
        if line:
            cleaned.append(line)

    if not cleaned:
        return summary.strip()

    cleaned = cleaned[:3]
    return "; ".join(cleaned)


def _normalize_endpoints(endpoints: str) -> str:
    if not endpoints or endpoints.strip().lower() == "none":
        return "none"
    parts = [p.strip() for p in endpoints.split(",") if p.strip()]
    return ", ".join(parts) if parts else "none"


def _ensure_changelog_section(text: str) -> str:
    if CHANGELOG_HEADER in text:
        return text

    suffix = "\n" if text.endswith("\n") else "\n\n"
    return text + suffix + CHANGELOG_HEADER + "\n" + CHANGELOG_HINT + "\n"


def _find_insert_index(lines: list[str]) -> int:
    header_idx = None
    for i, line in enumerate(lines):
        if line.rstrip("\n") == CHANGELOG_HEADER:
            header_idx = i

    if header_idx is None:
        return -1

    insert_idx = header_idx + 1
    if insert_idx < len(lines) and lines[insert_idx].rstrip("\n") == CHANGELOG_HINT:
        insert_idx += 1

    # If there are blank lines after the hint/header, insert after them.
    while insert_idx < len(lines) and lines[insert_idx].strip() == "":
        insert_idx += 1

    return insert_idx


def _render_entry(agent: str, sha: str, summary: str, tests: str, endpoints: str, context_pack_uploaded: str) -> str:
    return (
        f"- {_now_utc_minute()} [{agent}] sha={sha}\n"
        f"  - Summary: {summary}\n"
        f"  - Tests: {tests}\n"
        f"  - Endpoints: {endpoints}\n"
        f"  - Context-pack: uploaded ({context_pack_uploaded})\n"
    )


def _context_pack_uploaded() -> str:
    raw = (os.environ.get("CONTEXT_PACK_UPLOADED") or "").strip().lower()
    if raw in {"1", "true", "yes", "y"}:
        return "yes"
    if raw in {"0", "false", "no", "n", ""}:
        return "no"
    return "no"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Inserts a formatted entry at the top of 'Changelog (auto)' in ROADMAP_BACKLOG.md."
        )
    )
    parser.add_argument("--agent", required=True, help="CODEx|COPILOT")
    parser.add_argument("--sha", required=True, help="git sha (short or full)")
    parser.add_argument("--summary", required=True, help='"<1-3 bullets>"')
    parser.add_argument("--tests", required=True, help='"<pytest summary line>"')
    parser.add_argument("--endpoints", required=True, help='"<comma-separated paths or none>"')

    args = parser.parse_args()
    try:
        agent = _normalize_agent(args.agent)
    except ValueError as exc:
        parser.error(str(exc))

    repo_root = Path(__file__).resolve().parents[1]
    roadmap_path = repo_root / "ROADMAP_BACKLOG.md"
    latest_path = repo_root / "docs" / "ROADMAP_BACKLOG_latest.md"
    text = roadmap_path.read_text(encoding="utf-8")
    text = _ensure_changelog_section(text)

    # Keep newline characters so insertion index matches file layout
    lines = [line if line.endswith("\n") else line + "\n" for line in text.splitlines(keepends=True)]
    insert_idx = _find_insert_index(lines)
    if insert_idx < 0:
        raise RuntimeError("Changelog (auto) section not found")

    entry = _render_entry(
        agent=agent,
        sha=args.sha.strip(),
        summary=_normalize_summary(args.summary),
        tests=args.tests.strip(),
        endpoints=_normalize_endpoints(args.endpoints),
        context_pack_uploaded=_context_pack_uploaded(),
    )

    lines.insert(insert_idx, entry)
    roadmap_path.write_text("".join(lines), encoding="utf-8")
    latest_path.write_text("".join(lines), encoding="utf-8")
    print(f"Inserted changelog entry into {roadmap_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
