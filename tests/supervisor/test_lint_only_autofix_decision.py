from unittest.mock import AsyncMock

import pytest

from src.supervisor.config import SupervisorSettings
from src.supervisor.debate import DebateSystem
from src.supervisor.models import ArbiterDecision, CheckResult, VerificationReport


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


@pytest.mark.asyncio
async def test_lint_only_decision_requires_pytest_and_only_ruff_fail(monkeypatch):
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
            CheckResult(
                command="python -m mypy .",
                exit_code=1,
                passed=False,
                stdout="mypy error",
                stderr="",
                duration_seconds=0.1,
            ),
        ],
        all_passed=False,
        failure_summary="ruff + mypy failed",
        failing_tests=[],
    )

    call_agent = AsyncMock(
        return_value={"role": "optimist", "summary": "", "bullets": []}
    )
    call_arbiter = AsyncMock(
        return_value=ArbiterDecision(
            auto_fix_allowed=False,
            fix_objectives=[],
            risk_level="med",
            stop_reason="needs_review",
        )
    )

    monkeypatch.setattr(system, "_call_agent", call_agent)
    monkeypatch.setattr(system, "_call_arbiter", call_arbiter)

    await system.run_debate(
        verification=verification,
        changed_files=[],
        pr_title="",
        pr_body="",
    )

    assert call_agent.await_count > 0
