"""Arbiter decision maker for the loop."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .policy import PolicyContext, PolicyDefinition, evaluate_policy, load_policy
from .types import FixPlan, LoopDecision

if TYPE_CHECKING:
    from .types import SkepticReport


DEFAULT_POLICY = load_policy()


def arbitrate(
    plan: FixPlan,
    skeptic: "SkepticReport",
    policy: PolicyDefinition,
    changed_files: list[str],
    pr_labels: list[str],
    push_env_enabled: bool,
    requested_mode: str = "dry_run",
) -> LoopDecision:
    context = PolicyContext(
        category=plan.category,
        files_touched=len(changed_files),
        loc_changed=len(changed_files) * 10,
        changed_files=changed_files,
        pr_labels=pr_labels,
        push_env_enabled=push_env_enabled,
        requested_mode=requested_mode,
    )
    policy_decision = evaluate_policy(context, policy)

    return LoopDecision(
        decision=policy_decision.decision.lower(),
        reason=policy_decision.reason,
        fix_objectives=plan.objectives,
        allowed_to_modify=plan.objectives,
        risk_level=policy_decision.risk_level,
    )
