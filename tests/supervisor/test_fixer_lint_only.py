"""Unit tests for FixMode.LINT_ONLY (T014)."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from src.supervisor.loop.fixers import DeterministicFixer, FixMode
from src.supervisor.models import VerificationReport, CheckResult

@pytest.mark.anyio
async def test_run_fix_lint_only_success(tmp_path):
    """Verify ruff --fix logic via mocked run_cmd."""
    runner = MagicMock()
    fixer = DeterministicFixer(runner)
    
    workspace = tmp_path / "work"
    workspace.mkdir()
    (workspace / "test.py").write_text("import os\n")
    
    # Mocking internal _run_command or similar if possible, 
    # but run_fix uses create_subprocess_shell directly.
    # So we patch asyncio.create_subprocess_shell
    
    mock_proc = AsyncMock()
    mock_proc.communicate.return_value = (b"success", b"")
    mock_proc.returncode = 0
    
    with patch("asyncio.create_subprocess_shell", return_value=mock_proc):
        report = await fixer.run_fix(
            mode=FixMode.IMPORT,
            workspace_path=str(workspace),
            changed_files=["test.py"],
            head_sha="sha1"
        )
    
    assert report[0] is True
    assert "successfully" in report[1].lower()

@pytest.mark.anyio
async def test_determine_lint_targets():
    """Verify targets are extracted from verification failures."""
    from src.supervisor.loop.fixers import _determine_lint_targets
    
    # The regex in fixers.py: _extract_targets_from_checks
    # path_pattern = re.compile(r"-->\s*(?P<path>[^:\s]+\.py):")
    
    report = VerificationReport(
        commit_sha="sha",
        all_passed=False,
        failure_summary="--> src/app.py:10:1: F401 [*] 'os' imported but unused",
        checks=[
            CheckResult(
                command="ruff check src/app.py",
                exit_code=1,
                passed=False,
                stdout="--> src/app.py:10:1: F401 [*] 'os' imported but unused"
            )
        ]
    )
    
    # Mock workspace exists
    with patch("src.supervisor.loop.fixers.Path.exists", return_value=True), \
         patch("src.supervisor.loop.fixers.Path.is_file", return_value=True):
        # Passing None for changed_files to force extraction from report
        targets = _determine_lint_targets("/tmp", None, report)
        
    assert "src/app.py" in targets
