import asyncio
import logging
from enum import Enum
from typing import Optional

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
