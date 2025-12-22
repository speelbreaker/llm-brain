"""Optimist helper for planning fixes."""

from __future__ import annotations

from typing import Iterable

from .types import FixPlan
from ..models import VerificationReport


def _guess_category(verification: VerificationReport, failure_summary: str) -> str:
    lint_detected = any(
        "ruff" in check.command.lower() for check in verification.checks
    ) or "lint" in failure_summary.lower()
    if lint_detected:
        return "lint_only"

    if verification.failing_tests:
        return "single_test_env_leak"

    return "unknown"


def propose_fix_plan(verification: VerificationReport, failure_summary: str) -> FixPlan:
    category = _guess_category(verification, failure_summary)
    return FixPlan(
        category=category,
        objectives=["Apply safe deterministic fix"],
        approach="Deterministic fixer" if category != "unknown" else "Manual review",
        estimated_risk="low",
        rationale=failure_summary[:400],
    )
