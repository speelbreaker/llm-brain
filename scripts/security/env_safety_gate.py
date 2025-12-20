#!/usr/bin/env python3
"""Reject unsafe supervisor env toggles and secret assignments in tracked files."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


UNSAFE_TOGGLES = {
    "SUPERVISOR_DEBUG": re.compile(r"^\s*SUPERVISOR_DEBUG\s*=\s*1\b"),
    "SUPERVISOR_AUTOFIX_PUSH": re.compile(r"^\s*SUPERVISOR_AUTOFIX_PUSH\s*=\s*1\b"),
    "SUPERVISOR_ENABLE_CODEX": re.compile(r"^\s*SUPERVISOR_ENABLE_CODEX\s*=\s*1\b"),
}

SECRET_KEYS = {
    "OPENAI_API_KEY": re.compile(r"^\s*OPENAI_API_KEY\s*="),
    "GITHUB_TOKEN": re.compile(r"^\s*GITHUB_TOKEN\s*="),
    "TELEGRAM_BOT_TOKEN": re.compile(r"^\s*TELEGRAM_BOT_TOKEN\s*="),
    "GEMINI_API_KEY": re.compile(r"^\s*GEMINI_API_KEY\s*="),
    "WEBHOOK_SECRET": re.compile(r"^\s*WEBHOOK_SECRET\s*="),
    "GITHUB_WEBHOOK_SECRET": re.compile(r"^\s*GITHUB_WEBHOOK_SECRET\s*="),
}


def list_tracked_files() -> list[Path]:
    output = subprocess.check_output(["git", "ls-files"], text=True)
    return [Path(line.strip()) for line in output.splitlines() if line.strip()]


def scan_files(paths: list[Path]) -> list[tuple[str, str]]:
    findings: list[tuple[str, str]] = []
    for path in paths:
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for line in content.splitlines():
            for key, pattern in UNSAFE_TOGGLES.items():
                if pattern.search(line):
                    findings.append((str(path), key))
            for key, pattern in SECRET_KEYS.items():
                if pattern.search(line):
                    findings.append((str(path), key))
    return findings


def main() -> int:
    findings = scan_files(list_tracked_files())
    if not findings:
        print("OK: no unsafe supervisor env toggles or secret assignments found.")
        return 0

    print("UNSAFE_ENV_CONFIG detected:")
    for path, key in findings:
        print(f"- {path}: {key}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
