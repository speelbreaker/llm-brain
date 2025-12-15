"""Autofix policy enforcement for PR Supervisor."""

from dataclasses import dataclass
from typing import Optional

from .config import SupervisorSettings
from .store import JobStore, PRApprovalState


@dataclass
class AutofixDecision:
    """Result of autofix policy check."""
    allowed: bool
    reason: str
    needs_human: bool = False


def check_autofix_policy(
    settings: SupervisorSettings,
    store: JobStore,
    repo: str,
    pr_number: int,
    pr_labels: list[str],
    arbiter_risk_level: Optional[str] = None,
) -> AutofixDecision:
    """
    Check if autofix (Codex) is allowed for this PR based on policy.
    
    Args:
        settings: Supervisor settings
        store: Job store for approval state
        repo: Repository full name
        pr_number: PR number
        pr_labels: Labels on the PR
        arbiter_risk_level: Risk level from arbiter debate ("low", "medium", "high")
    
    Returns:
        AutofixDecision with allowed status and reason
    """
    if not settings.enable_codex:
        return AutofixDecision(
            allowed=False,
            reason="Codex is disabled (SUPERVISOR_ENABLE_CODEX=0)"
        )
    
    approval = store.get_pr_approval(repo, pr_number)
    
    if approval.paused:
        return AutofixDecision(
            allowed=False,
            reason="PR is paused"
        )
    
    policy = settings.autofix_policy
    has_label = settings.autofix_label in pr_labels
    has_telegram_approval = approval.approved_by_telegram
    
    policy_satisfied = False
    policy_reason = ""
    
    if policy == "label":
        policy_satisfied = has_label
        if not has_label:
            policy_reason = f"PR needs label '{settings.autofix_label}'"
    elif policy == "telegram":
        policy_satisfied = has_telegram_approval
        if not has_telegram_approval:
            policy_reason = "PR needs Telegram approval (/autofix)"
    elif policy == "both":
        policy_satisfied = has_label and has_telegram_approval
        if not policy_satisfied:
            missing = []
            if not has_label:
                missing.append(f"label '{settings.autofix_label}'")
            if not has_telegram_approval:
                missing.append("Telegram approval")
            policy_reason = f"PR needs: {' and '.join(missing)}"
    else:
        return AutofixDecision(
            allowed=False,
            reason=f"Invalid autofix policy: {policy}"
        )
    
    if not policy_satisfied:
        return AutofixDecision(
            allowed=False,
            reason=policy_reason
        )
    
    if arbiter_risk_level == "high" and settings.require_human_for_high_risk:
        return AutofixDecision(
            allowed=False,
            reason="High-risk changes require human review",
            needs_human=True
        )
    
    return AutofixDecision(
        allowed=True,
        reason="Autofix policy satisfied"
    )


def get_policy_status_text(
    settings: SupervisorSettings,
    store: JobStore,
    repo: str,
    pr_number: int,
    pr_labels: list[str],
) -> str:
    """Get human-readable policy status for Telegram messages."""
    if not settings.enable_codex:
        return "Codex disabled"
    
    approval = store.get_pr_approval(repo, pr_number)
    policy = settings.autofix_policy
    has_label = settings.autofix_label in pr_labels
    has_telegram = approval.approved_by_telegram
    
    parts = []
    
    if policy in ("label", "both"):
        label_status = "yes" if has_label else "no"
        parts.append(f"Label: {label_status}")
    
    if policy in ("telegram", "both"):
        tg_status = "yes" if has_telegram else "no"
        parts.append(f"Telegram: {tg_status}")
    
    if approval.paused:
        parts.append("PAUSED")
    
    return " | ".join(parts) if parts else "Ready"
