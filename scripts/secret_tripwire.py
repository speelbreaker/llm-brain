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


SUSPICIOUS_PATTERNS = [
    re.compile(r"(api_key|token|secret|password|private_key)[\\s:=]{1,6}['\"]?[A-Za-z0-9]{20,}", re.IGNORECASE),
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


def main() -> int:
    failures: list[str] = []
    for path in iter_tracked_files():
        if should_skip(path):
            continue
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for pat in SUSPICIOUS_PATTERNS:
            for match in pat.finditer(content):
                snippet = content[max(0, match.start() - 10) : match.end() + 10]
                failures.append(f"{path}: {snippet.strip()}")
                break  # one hit is enough per file

    if failures:
        print("🚨 Potential secrets detected:")
        for item in failures:
            print(f"- {item}")
        return 1

    print("✅ No obvious secrets detected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
