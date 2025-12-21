"""Skeptic guardrail evaluator."""

from __future__ import annotations

from .types import SkepticReport


def review_fix_plan(
    plan: object,
    verification: object | None = None,
    changed_files: list[str] | None = None,
) -> SkepticReport:
    return SkepticReport(
        risk_level="low",
        blockers=[],
        warnings=[],
        classification="deterministic",
        file_risk_score=len(changed_files or []),
        loc_risk_score=0,
    )
