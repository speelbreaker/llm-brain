import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.supervisor.loop.fixers import DeterministicFixer, FixMode
from src.supervisor.models import VerificationReport

@pytest.mark.asyncio
async def test_format_only_targets_changed_files():
    runner = AsyncMock()
    fixer = DeterministicFixer(runner)
    
    with patch("asyncio.create_subprocess_shell") as mock_shell:
        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b"", b"")
        mock_proc.returncode = 0
        mock_shell.return_value = mock_proc
        
        success, msg = await fixer.run_fix(
            FixMode.FORMAT, 
            "/tmp/ws", 
            ["a.py", "b.txt", "c.py"],
            "sha123"
        )
        
        assert success
        # Check that only .py files were targeted
        # The command should contain "a.py" and "c.py" but not "b.txt"
        args = mock_shell.call_args_list[0][0][0]
        assert "a.py" in args
        assert "c.py" in args
        assert "b.txt" not in args
        assert "ruff format" in args

@pytest.mark.asyncio
async def test_import_only_targets_changed_files():
    runner = AsyncMock()
    fixer = DeterministicFixer(runner)
    
    with patch("asyncio.create_subprocess_shell") as mock_shell:
        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b"", b"")
        mock_proc.returncode = 0
        mock_shell.return_value = mock_proc
        
        success, msg = await fixer.run_fix(
            FixMode.IMPORT, 
            "/tmp/ws", 
            ["a.py"],
            "sha123"
        )
        
        assert success
        # Should check ruff check --select I --fix
        calls = [c[0][0] for c in mock_shell.call_args_list]
        assert any("ruff check --select I --fix" in cmd for cmd in calls)

@pytest.mark.asyncio
async def test_tests_only_escalates_without_ai_edits_when_still_failing():
    runner = AsyncMock()
    # Mock verification failure
    runner.run_checks.return_value = VerificationReport(
        commit_sha="sha", 
        all_passed=False, 
        failure_summary="Fail"
    )
    
    fixer = DeterministicFixer(runner)
    
    with patch("asyncio.create_subprocess_shell") as mock_shell:
        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b"", b"")
        mock_proc.returncode = 0
        mock_shell.return_value = mock_proc
        
        success, msg = await fixer.run_fix(
            FixMode.TESTS, 
            "/tmp/ws", 
            ["a.py"],
            "sha123"
        )
        
        # Should return False because tests failed
        assert not success
        assert "Tests failed" in msg
        # Should have run format (cleanup)
        assert any("ruff format" in c[0][0] for c in mock_shell.call_args_list)
