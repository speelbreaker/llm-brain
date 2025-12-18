"""Verification runner - executes checks and captures output."""

import asyncio
import os
import re
import time
from typing import Optional


def _sanitized_check_env() -> dict[str, str]:
    """
    Build a clean environment for running repo checks inside a PR workspace.

    Why:
      - The supervisor container runs with SUPERVISOR_ENABLED=1 and LLM secrets/models.
      - If we leak those into `pytest`/`ruff`, the PR's own tests can behave differently
        than they would in CI, and tests that expect default settings will fail.
    """
    env = os.environ.copy()

    # Remove supervisor + model/provider env that should NEVER influence the repo under test.
    kill_prefixes = ("SUPERVISOR_", "OPENAI_", "GEMINI_", "GITHUB_")
    kill_exact = {
        "MODEL_OPTIMIST",
        "MODEL_SKEPTIC",
        "MODEL_ARBITER",
        "CODEX_MODEL",
        # common accidental leaks
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "OTEL_SERVICE_NAME",
    }

    for key in list(env.keys()):
        if key.startswith(kill_prefixes) or key in kill_exact:
            env.pop(key, None)

    return env

from .config import SupervisorSettings
from .models import CheckResult, VerificationReport


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
        """Run a single command with timeout and output capture."""
        start_time = time.time()
        
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=_sanitized_check_env(),
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
