"""Policy logic for loop decisions."""

import json
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from .types import LoopDecision

DEFAULTS_PATH = Path(__file__).parent / "policy_defaults.json"


class PolicyContext(BaseModel):
    category: str
    files_touched: int
    loc_changed: int
    changed_files: list[str]
    pr_labels: list[str] = Field(default_factory=list)
    push_env_enabled: bool = False
    requested_mode: str = "dry_run"


class PolicyDefinition(BaseModel):
    max_files_touched: Optional[int] = 10
    max_loc_changed: Optional[int] = 300
    denylist_paths: list[str] = Field(default_factory=list)
    push_label: str = "autofix-ok"


def load_policy(path: Path | None = None) -> PolicyDefinition:
    target = path or DEFAULTS_PATH
    if not target.exists():
        return PolicyDefinition()
    data = json.loads(target.read_text())
    return PolicyDefinition(**data)


def evaluate_policy(context: PolicyContext, policy: PolicyDefinition) -> LoopDecision:
    normalized = context.requested_mode.lower()

    for root in policy.denylist_paths:
        for changed in context.changed_files:
            if changed.startswith(root):
                return LoopDecision(
                    decision="DENY",
                    reason=f"File in denylist: {changed}",
                    risk_level="high",
                )

    if policy.max_files_touched and context.files_touched > policy.max_files_touched:
        return LoopDecision(
            decision="DENY",
            reason="Too many files touched",
            risk_level="medium",
        )

    if policy.max_loc_changed and context.loc_changed > policy.max_loc_changed:
        return LoopDecision(
            decision="DENY",
            reason="Too many LOC changed",
            risk_level="high",
        )

    if normalized == "push":
        if not context.push_env_enabled:
            return LoopDecision(
                decision="DRY_RUN",
                reason="Push env missing",
                risk_level="medium",
            )
        if policy.push_label not in context.pr_labels:
            return LoopDecision(
                decision="DRY_RUN",
                reason="Missing push label",
                risk_level="medium",
            )
        return LoopDecision(
            decision="PUSH",
            reason="Push approved",
            risk_level="low",
        )

    return LoopDecision(
        decision="DRY_RUN",
        reason="Dry-run fallback",
        risk_level="low",
    )
