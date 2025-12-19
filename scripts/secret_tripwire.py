"""
Lightweight secret tripwire for CI/pre-commit.

Scans tracked source files for suspicious secret-like patterns.
Excludes common docs/assets to reduce false positives.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable


SUSPICIOUS_PATTERNS = [
    re.compile(
        r"(api_key|token|secret|password|private_key)[\\s:=]{1,6}['\"]?[A-Za-z0-9_-]{16,}",
        re.IGNORECASE,
    ),
]

SKIP_DIRS = {
    "docs",
    "attached_assets",
    ".git",
    ".github",
    "tests",
    "logs",
}

SKIP_SUFFIXES = {
    ".md",
    ".rst",
    ".txt",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".svg",
    ".csv",
    ".json",
    ".lock",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".pdf",
}


def iter_tracked_files() -> list[Path]:
    out = subprocess.check_output(["git", "ls-files"], text=True)
    return [Path(line.strip()) for line in out.splitlines() if line.strip()]


def should_skip(path: Path) -> bool:
    parts = set(path.parts)
    if parts & SKIP_DIRS:
        return True
    if path.suffix.lower() in SKIP_SUFFIXES:
        return True
    return False


def _redact(text: str) -> str:
    if len(text) <= 4:
        return "***REDACTED***"
    return "***REDACTED***" + text[-4:]


def scan_paths(paths: Iterable[Path]) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    for path in paths:
        if should_skip(path):
            continue
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for pat in SUSPICIOUS_PATTERNS:
            for match in pat.finditer(content):
                line_no = content.count("\n", 0, match.start()) + 1
                findings.append(
                    {
                        "path": str(path),
                        "line": str(line_no),
                        "pattern": pat.pattern,
                        "value": _redact(match.group(0)),
                    }
                )
                break
    return findings


def main() -> int:
    files = iter_tracked_files()
    findings = scan_paths(files)

    if findings:
        print("🚨 Potential secrets detected (values redacted):")
        for f in findings:
            print(f"- {f['path']}: line {f['line']} pattern={f['pattern']}")
        return 1

    print("✅ No obvious secrets detected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
