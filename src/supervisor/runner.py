"""Verification runner - executes checks and captures output."""

import asyncio
import os
import re
import time
from typing import Optional

from .config import SupervisorSettings
from .models import CheckResult, VerificationReport


SENSITIVE_ENV_PREFIXES = ("SUPERVISOR_", "GITHUB_", "TELEGRAM_")
SENSITIVE_ENV_KEYS = ("OPENAI_API_KEY", "GEMINI_API_KEY", "DATABASE_URL", "SESSION_SECRET")
SAFE_ENV_KEYS = ("PATH", "HOME", "LANG", "LC_ALL", "USER", "SHELL", "TERM", "PYTHONPATH", "VIRTUAL_ENV")


def get_sanitized_env() -> dict[str, str]:
    """Create a sanitized environment for subprocess execution.
    
    Removes sensitive keys to prevent test/lint processes from being
    influenced by supervisor configuration or accessing secrets.
    """
    env = os.environ.copy()
    
    keys_to_remove = set()
    for key in env:
        if key.startswith(SENSITIVE_ENV_PREFIXES):
            keys_to_remove.add(key)
        elif key in SENSITIVE_ENV_KEYS:
            keys_to_remove.add(key)
    
    for key in keys_to_remove:
        del env[key]
    
    for key in SAFE_ENV_KEYS:
        if key in os.environ:
            env[key] = os.environ[key]
    
    return env


class VerificationRunner:
    """Runs verification commands and produces structured reports."""
    
    def __init__(self, settings: SupervisorSettings):
        self.settings = settings
        self.max_output_lines = 100
        self.max_output_chars = 5000
    
    async def run_checks(
        self,
        workspace_path: str,
        commit_sha: str,
        commands: Optional[list[str]] = None,
    ) -> VerificationReport:
        """Run all configured check commands."""
        if commands is None:
            commands = self.settings.get_check_commands()
        
        checks: list[CheckResult] = []
        failing_tests: list[str] = []
        
        for cmd in commands:
            result = await self._run_command(cmd, workspace_path)
            checks.append(result)
            
            if not result.passed:
                failing_tests.extend(self._extract_failing_tests(result.stdout + result.stderr))
        
        all_passed = all(c.passed for c in checks)
        failure_summary = self._build_failure_summary(checks) if not all_passed else ""
        
        return VerificationReport(
            commit_sha=commit_sha,
            checks=checks,
            all_passed=all_passed,
            failure_summary=failure_summary,
            failing_tests=failing_tests[:20],
        )
    
    async def _run_command(self, command: str, cwd: str) -> CheckResult:
        """Run a single command with timeout and output capture.
        
        Uses a sanitized environment to prevent subprocess from being
        influenced by supervisor secrets or configuration.
        """
        start_time = time.time()
        sanitized_env = get_sanitized_env()
        
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=sanitized_env,
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.settings.command_timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return CheckResult(
                    command=command,
                    exit_code=-1,
                    passed=False,
                    stdout="",
                    stderr=f"Command timed out after {self.settings.command_timeout}s",
                    duration_seconds=self.settings.command_timeout,
                    truncated=False,
                )
            
            duration = time.time() - start_time
            stdout_str, stdout_truncated = self._truncate_output(stdout.decode(errors="replace"))
            stderr_str, stderr_truncated = self._truncate_output(stderr.decode(errors="replace"))
            
            exit_code = process.returncode if process.returncode is not None else -1
            
            return CheckResult(
                command=command,
                exit_code=exit_code,
                passed=(exit_code == 0),
                stdout=stdout_str,
                stderr=stderr_str,
                duration_seconds=duration,
                truncated=stdout_truncated or stderr_truncated,
            )
        
        except FileNotFoundError:
            return CheckResult(
                command=command,
                exit_code=-1,
                passed=True,
                stdout="",
                stderr="Command not found (skipped gracefully)",
                duration_seconds=0.0,
                truncated=False,
            )
        except Exception as e:
            return CheckResult(
                command=command,
                exit_code=-1,
                passed=False,
                stdout="",
                stderr=f"Error running command: {str(e)}",
                duration_seconds=time.time() - start_time,
                truncated=False,
            )
    
    def _truncate_output(self, output: str) -> tuple[str, bool]:
        """Truncate output to last N lines and max chars."""
        lines = output.split("\n")
        truncated = False
        
        if len(lines) > self.max_output_lines:
            lines = lines[-self.max_output_lines:]
            truncated = True
        
        result = "\n".join(lines)
        
        if len(result) > self.max_output_chars:
            result = result[-self.max_output_chars:]
            truncated = True
        
        return result, truncated
    
    def _extract_failing_tests(self, output: str) -> list[str]:
        """Extract failing test names from output."""
        failing = []
        
        pytest_pattern = r"(?:FAILED|ERROR)\s+([^\s:]+(?:::[^\s]+)?)"
        for match in re.finditer(pytest_pattern, output):
            failing.append(match.group(1))
        
        return failing
    
    def _build_failure_summary(self, checks: list[CheckResult]) -> str:
        """Build a summary of failures for the PR comment."""
        summary_parts = []
        
        for check in checks:
            if not check.passed:
                output = check.stderr if check.stderr else check.stdout
                lines = output.strip().split("\n")[-10:]
                summary_parts.append(f"### {check.command}\n" + "\n".join(lines))
        
        return "\n\n".join(summary_parts)[:2000]
