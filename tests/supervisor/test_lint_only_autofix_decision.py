from unittest.mock import AsyncMock

import pytest

from src.supervisor.config import SupervisorSettings
from src.supervisor.debate import DebateSystem
from src.supervisor.models import CheckResult, VerificationReport


@pytest.mark.asyncio
async def test_lint_only_decision_bypasses_llm(monkeypatch):
    settings = SupervisorSettings()
    system = DebateSystem(settings)

    verification = VerificationReport(
        commit_sha="deadbeef",
        checks=[
            CheckResult(
                command="python -m pytest -q",
                exit_code=0,
                passed=True,
                stdout="",
                stderr="",
                duration_seconds=0.1,
            ),
            CheckResult(
                command="python -m ruff check .",
                exit_code=1,
                passed=False,
                stdout="F401 unused import",
                stderr="",
                duration_seconds=0.1,
            ),
        ],
        all_passed=False,
        failure_summary="ruff failed",
        failing_tests=[],
    )

    monkeypatch.setattr(
        system,
        "_call_agent",
        AsyncMock(side_effect=AssertionError("LLM should not be called")),
    )
    monkeypatch.setattr(
        system,
        "_call_arbiter",
        AsyncMock(side_effect=AssertionError("LLM should not be called")),
    )

    decision = await system.run_debate(
        verification=verification,
        changed_files=[],
        pr_title="",
        pr_body="",
    )

    assert decision.auto_fix_allowed is True
    assert decision.risk_level == "low"
    assert decision.fix_objectives
