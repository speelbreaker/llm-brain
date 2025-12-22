"""Deterministic fixers for the supervisor loop."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Iterable

from .types import FixPlan
from ..models import VerificationReport


class FixResult:
    """Simple fix result container."""

    def __init__(self, applied: bool, fixer: str, changed_files: list[str], notes: list[str]):
        self.applied = applied
        self.fixer = fixer
        self.changed_files = changed_files
        self.notes = notes


async def _run_command(command: str, cwd: str) -> tuple[int, str]:
    try:
        process = await asyncio.create_subprocess_shell(
            command,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
    except (FileNotFoundError, OSError) as exc:
        return 1, f"Command failed to start: {exc}"

    stdout, _ = await process.communicate()
    output = stdout.decode(errors="replace") if stdout else ""
    return process.returncode or 0, output


async def _git_changed_files(workspace_path: str) -> list[str]:
    code, output = await _run_command("git diff --name-only", workspace_path)
    if code != 0:
        return []
    return [line.strip() for line in output.splitlines() if line.strip()]


async def _fix_lint_only(workspace_path: str) -> FixResult:
    notes: list[str] = []
    code, _ = await _run_command("python -m ruff check . --fix", workspace_path)
    if code != 0:
        return FixResult(False, "ruff_fix", [], ["ruff --fix failed"])

    code, _ = await _run_command("python -m ruff check .", workspace_path)
    if code == 0:
        notes.append("ruff clean after --fix")
    else:
        notes.append("ruff still failing after --fix")

    changed = await _git_changed_files(workspace_path)
    return FixResult(True, "ruff_fix", changed, notes)


def _find_failing_test_file(failing_tests: Iterable[str]) -> str | None:
    for test_name in failing_tests:
        if "::" in test_name:
            return test_name.split("::", 1)[0]
    return None


def _patch_env_leak_test(file_path: Path) -> bool:
    text = file_path.read_text()
    pattern = r"(patch\\.dict\\([^\\n]*?,\\s*[^\\n]*?clear=)False"
    updated, count = re.subn(pattern, r"\\1True", text)
    if count:
        file_path.write_text(updated)
        return True
    return False


async def _fix_single_test_env_leak(
    workspace_path: str,
    verification: VerificationReport,
) -> FixResult:
    test_file = _find_failing_test_file(verification.failing_tests)
    if not test_file:
        return FixResult(False, "env_leak_fix", [], ["No failing test file identified"])

    file_path = Path(workspace_path) / test_file
    if not file_path.exists():
        return FixResult(False, "env_leak_fix", [], [f"Missing test file {test_file}"])

    changed = _patch_env_leak_test(file_path)
    if not changed:
        return FixResult(False, "env_leak_fix", [], ["No patch.dict(clear=False) found"])

    return FixResult(True, "env_leak_fix", [test_file], ["Set clear=True for patch.dict"])


async def apply_fix_plan(
    workspace_path: str,
    plan: FixPlan,
    verification: VerificationReport,
) -> FixResult:
    """Apply the deterministic fixer for the given plan."""
    if plan.category == "lint_only":
        return await _fix_lint_only(workspace_path)

    if plan.category == "single_test_env_leak":
        return await _fix_single_test_env_leak(workspace_path, verification)

    return FixResult(False, "unsupported", [], ["No deterministic fixer for plan category"])
