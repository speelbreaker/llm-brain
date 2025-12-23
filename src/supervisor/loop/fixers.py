"""Deterministic fixers for the supervisor loop."""

from __future__ import annotations

import asyncio
import os
import re
import shlex
import logging
from enum import Enum
from pathlib import Path
from typing import Iterable, Optional

from .types import FixPlan
from ..runner import VerificationRunner, get_sanitized_env
from ..models import VerificationReport

logger = logging.getLogger(__name__)

class FixMode(str, Enum):
    FORMAT = "format"
    IMPORT = "import"
    TESTS = "tests"

class DeterministicFixer:
    def __init__(self, runner: VerificationRunner):
        self.runner = runner

    async def run_fix(
        self,
        mode: FixMode,
        workspace_path: str,
        changed_files: list[str],
        head_sha: str,
        verification_report: Optional[VerificationReport] = None
    ) -> tuple[bool, str]:
        """
        Run deterministic fix logic.
        
        Args:
            mode: The fix mode (FORMAT, IMPORT, TESTS).
            workspace_path: Path to the git workspace.
            changed_files: List of changed files.
            head_sha: Current head SHA (for verification reporting).
            verification_report: Optional previous verification report (for targeted fixes).
            
        Returns:
            tuple[bool, str]: (success, message_or_output)
        """
        py_files = [f for f in changed_files if f.endswith(".py")]
        
        # Helper for running commands
        env = get_sanitized_env()
        async def run_cmd(cmd_str: str) -> tuple[int, str, str]:
            proc = await asyncio.create_subprocess_shell(
                cmd_str,
                cwd=workspace_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env
            )
            stdout, stderr = await proc.communicate()
            return proc.returncode or 0, stdout.decode("utf-8", errors="replace"), stderr.decode("utf-8", errors="replace")

        if mode == FixMode.FORMAT:
            if not py_files:
                return True, "No Python files to format"
            files_str = " ".join(f"'{f}'" for f in py_files)
            
            # 1. ruff format
            code, out, err = await run_cmd(f"python3 -m ruff format {files_str}")
            if code != 0:
                return False, f"Format failed: {err or out}"
            
            # 2. ruff check
            code, out, err = await run_cmd(f"python3 -m ruff check {files_str}")
            if code != 0:
                return False, f"Lint check failed after format: {out}\n{err}"
                
            return True, "Formatted and linted successfully"

        elif mode == FixMode.IMPORT:
            if not py_files:
                return True, "No Python files to fix imports"
            files_str = " ".join(f"'{f}'" for f in py_files)
            
            # 1. ruff check --select I --fix
            code, out, err = await run_cmd(f"python3 -m ruff check --select I --fix {files_str}")
            if code != 0:
                return False, f"Import fix failed: {err or out}"
                
            # 2. ruff check
            code, out, err = await run_cmd(f"python3 -m ruff check {files_str}")
            if code != 0:
                return False, f"Lint check failed after import fix: {out}\n{err}"
                
            return True, "Imports fixed and linted successfully"

        elif mode == FixMode.TESTS:
            # Targeted rerun if we know what failed
            if verification_report and verification_report.failing_tests:
                tests_to_run = " ".join(f"'{t}'" for t in verification_report.failing_tests)
                code, out, err = await run_cmd(f"python3 -m pytest {tests_to_run}")
                if code == 0:
                    return True, "Tests passed on targeted rerun (flake mitigation)"
            
            # Fallback: simple cleanup and full rerun
            if py_files:
                files_str = " ".join(f"'{f}'" for f in py_files)
                await run_cmd(f"python3 -m ruff format {files_str}")
            
            report = await self.runner.run_checks(workspace_path, head_sha)
            
            if report.all_passed:
                return True, "Tests passed after cleanup/retry"
            else:
                return False, f"Tests failed: {report.failure_summary}"

        return False, f"Unknown mode: {mode}"


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


def _normalize_candidate(candidate: str) -> str | None:
    if not candidate:
        return None
    candidate = candidate.strip().replace("\\", "/")
    if candidate.startswith("./"):
        candidate = candidate[2:]
    if candidate.startswith("../") or candidate.startswith("/"):
        return None
    if not candidate.lower().endswith(".py"):
        return None
    if os.path.isabs(candidate):
        return None
    return candidate


def _filter_existing_python_targets(workspace_path: str, candidates: Iterable[str]) -> list[str]:
    workspace = Path(workspace_path)
    result: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = _normalize_candidate(candidate)
        if not normalized or normalized in seen:
            continue
        target_path = workspace / normalized
        if not target_path.exists() or not target_path.is_file():
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def _extract_targets_from_checks(workspace_path: str, verification: VerificationReport) -> list[str]:
    failures: list[str] = []
    path_pattern = re.compile(r"-->\s*(?P<path>[^:\s]+\.py):")
    for check in verification.checks:
        for text in (check.stdout, check.stderr):
            if not text:
                continue
            failures.extend(path_pattern.findall(text))
    if verification.failure_summary:
        failures.extend(path_pattern.findall(verification.failure_summary))
    return _filter_existing_python_targets(workspace_path, failures)


def _extract_targets_from_commands(workspace_path: str, verification: VerificationReport) -> list[str]:
    candidates: list[str] = []
    for check in verification.checks:
        tokens = shlex.split(check.command)
        collecting = False
        for token in tokens:
            if not collecting:
                if token == "check":
                    collecting = True
                continue
            if token.startswith("-"):
                break
            candidates.append(token)
    return _filter_existing_python_targets(workspace_path, candidates)


def _determine_lint_targets(
    workspace_path: str,
    changed_files: list[str] | None,
    verification: VerificationReport,
) -> list[str]:
    candidates: list[str] = []
    if changed_files:
        candidates.extend(_filter_existing_python_targets(workspace_path, changed_files))
    if not candidates:
        candidates.extend(_extract_targets_from_checks(workspace_path, verification))
    if not candidates:
        candidates.extend(_extract_targets_from_commands(workspace_path, verification))
    sightings: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            sightings.append(candidate)
    return sightings


async def _fix_lint_only(
    workspace_path: str,
    changed_files: list[str] | None,
    verification: VerificationReport,
) -> FixResult:
    notes: list[str] = []
    targets = _determine_lint_targets(workspace_path, changed_files, verification)
    if not targets:
        return FixResult(False, "ruff_fix", [], ["No lint-only targets determined"])
    quoted_targets = " ".join(shlex.quote(target) for target in targets)
    fix_cmd = f"python -m ruff check {quoted_targets} --fix"
    code, _ = await _run_command(fix_cmd, workspace_path)
    if code != 0:
        return FixResult(False, "ruff_fix", [], ["ruff --fix failed"])

    check_cmd = f"python -m ruff check {quoted_targets}"
    code, _ = await _run_command(check_cmd, workspace_path)
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
    pattern = r"(patch\.dict\([^\\n]*?,\\s*[^\\n]*?clear=)False"
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
    changed_files: list[str] | None = None,
) -> FixResult:
    """Apply the deterministic fixer for the given plan."""
    if plan.category == "lint_only":
        return await _fix_lint_only(workspace_path, changed_files, verification)

    if plan.category == "single_test_env_leak":
        return await _fix_single_test_env_leak(workspace_path, verification)

    return FixResult(False, "unsupported", [], ["No deterministic fixer for plan category"])