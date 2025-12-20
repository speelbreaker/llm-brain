#!/usr/bin/env python3
"""Reject unsafe supervisor env toggles and secret assignments in env-like files."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Iterable


UNSAFE_TOGGLES = {
    "SUPERVISOR_DEBUG",
    "SUPERVISOR_AUTOFIX_PUSH",
    "SUPERVISOR_ENABLE_CODEX",
}

SECRET_KEYS = {
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "GITHUB_TOKEN",
    "TELEGRAM_BOT_TOKEN",
    "GITHUB_WEBHOOK_SECRET",
}

DEFAULT_ENV_PATHS = {
    "docker/pr-supervisor.env",
    "docker/.env.supervisor",
}

IGNORE_PREFIXES = ("docs/", "attached_assets/")
IGNORE_SUFFIXES = (".md",)
ENV_GLOBS = ("docker/*.env",)
TRUTHY = {"1", "true", "yes", "on"}
PLACEHOLDERS = {"<REDACTED>", "<YOUR_TOKEN>", "CHANGE_ME", "REPLACE_ME", "YOUR_TOKEN_HERE"}


def list_tracked_files() -> list[Path]:
    output = subprocess.check_output(["git", "ls-files"], text=True)
    return [Path(line.strip()) for line in output.splitlines() if line.strip()]

def is_ignored(path: Path) -> bool:
    path_str = path.as_posix()
    if path_str.startswith(IGNORE_PREFIXES):
        return True
    return path_str.lower().endswith(IGNORE_SUFFIXES)


def is_example_env(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".example") or name.endswith(".env.example")


def is_env_like(path: Path) -> bool:
    path_str = path.as_posix()
    if path_str in DEFAULT_ENV_PATHS:
        return True
    if path.name.startswith(".env"):
        return True
    for pattern in ENV_GLOBS:
        if path.match(pattern):
            return True
    return False


def select_default_paths(paths: Iterable[Path]) -> list[Path]:
    selected: list[Path] = []
    for path in paths:
        if is_ignored(path):
            continue
        if not is_env_like(path):
            continue
        if is_example_env(path):
            continue
        selected.append(path)
    return selected


def parse_env_line(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if "=" not in stripped:
        return None
    key, value = stripped.split("=", 1)
    key = key.strip()
    value = value.strip()
    if not key:
        return None
    if value.startswith(("'", '"')) and value.endswith(value[:1]) and len(value) >= 2:
        value = value[1:-1]
    return key, value


def _is_truthy(value: str) -> bool:
    return value.strip().lower() in TRUTHY


def _is_placeholder(value: str) -> bool:
    return value.strip().upper() in PLACEHOLDERS


def scan_files(paths: list[Path]) -> list[tuple[str, str]]:
    findings: list[tuple[str, str]] = []
    for path in paths:
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for line in content.splitlines():
            parsed = parse_env_line(line)
            if not parsed:
                continue
            key, value = parsed
            if key in UNSAFE_TOGGLES and _is_truthy(value):
                findings.append((str(path), key))
                continue
            if key in SECRET_KEYS:
                if value == "" or _is_placeholder(value):
                    continue
                findings.append((str(path), key))
    return findings


def main() -> int:
    args = [Path(p) for p in sys.argv[1:]]
    if args:
        paths = [p for p in args if p.exists()]
        if not paths:
            print("OK: no env files to scan.")
            return 0
    else:
        paths = select_default_paths(list_tracked_files())
    findings = scan_files(paths)
    if not findings:
        print("OK: no unsafe supervisor env toggles or secret assignments found.")
        return 0

    print("UNSAFE_ENV_CONFIG detected:")
    for path, key in findings:
        print(f"- {path}: {key}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
