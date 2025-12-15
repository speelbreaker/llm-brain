"""Tests for autofix policy enforcement."""

import pytest
from unittest.mock import MagicMock, patch

from src.supervisor.policy import check_autofix_policy, AutofixDecision
from src.supervisor.store import PRApprovalState


class MockSettings:
    """Mock settings for testing."""
    def __init__(
        self,
        enable_codex: bool = True,
        autofix_policy: str = "label",
        autofix_label: str = "autofix-ok",
        require_human_for_high_risk: bool = True,
    ):
        self.enable_codex = enable_codex
        self.autofix_policy = autofix_policy
        self.autofix_label = autofix_label
        self.require_human_for_high_risk = require_human_for_high_risk


class MockStore:
    """Mock store for testing."""
    def __init__(self, approval: PRApprovalState = None):
        self._approval = approval or PRApprovalState(repo="owner/repo", pr_number=1)
    
    def get_pr_approval(self, repo: str, pr_number: int) -> PRApprovalState:
        return self._approval


class TestCheckAutofixPolicy:
    """Tests for check_autofix_policy function."""
    
    def test_codex_disabled_returns_not_allowed(self):
        """When SUPERVISOR_ENABLE_CODEX=0, autofix is not allowed."""
        settings = MockSettings(enable_codex=False)
        store = MockStore()
        
        decision = check_autofix_policy(
            settings=settings,
            store=store,
            repo="owner/repo",
            pr_number=1,
            pr_labels=["autofix-ok"],
        )
        
        assert not decision.allowed
        assert "disabled" in decision.reason.lower()
    
    def test_paused_pr_returns_not_allowed(self):
        """When PR is paused, autofix is not allowed."""
        settings = MockSettings(enable_codex=True)
        approval = PRApprovalState(repo="owner/repo", pr_number=1, paused=True)
        store = MockStore(approval=approval)
        
        decision = check_autofix_policy(
            settings=settings,
            store=store,
            repo="owner/repo",
            pr_number=1,
            pr_labels=["autofix-ok"],
        )
        
        assert not decision.allowed
        assert "paused" in decision.reason.lower()
    
    def test_label_policy_requires_label(self):
        """Label policy requires the autofix label."""
        settings = MockSettings(enable_codex=True, autofix_policy="label")
        store = MockStore()
        
        decision = check_autofix_policy(
            settings=settings,
            store=store,
            repo="owner/repo",
            pr_number=1,
            pr_labels=[],
        )
        
        assert not decision.allowed
        assert "label" in decision.reason.lower()
    
    def test_label_policy_allowed_with_label(self):
        """Label policy allows when label is present."""
        settings = MockSettings(enable_codex=True, autofix_policy="label")
        store = MockStore()
        
        decision = check_autofix_policy(
            settings=settings,
            store=store,
            repo="owner/repo",
            pr_number=1,
            pr_labels=["autofix-ok"],
        )
        
        assert decision.allowed
    
    def test_telegram_policy_requires_approval(self):
        """Telegram policy requires Telegram approval."""
        settings = MockSettings(enable_codex=True, autofix_policy="telegram")
        approval = PRApprovalState(repo="owner/repo", pr_number=1, approved_by_telegram=False)
        store = MockStore(approval=approval)
        
        decision = check_autofix_policy(
            settings=settings,
            store=store,
            repo="owner/repo",
            pr_number=1,
            pr_labels=[],
        )
        
        assert not decision.allowed
        assert "telegram" in decision.reason.lower()
    
    def test_telegram_policy_allowed_with_approval(self):
        """Telegram policy allows when approved."""
        settings = MockSettings(enable_codex=True, autofix_policy="telegram")
        approval = PRApprovalState(repo="owner/repo", pr_number=1, approved_by_telegram=True)
        store = MockStore(approval=approval)
        
        decision = check_autofix_policy(
            settings=settings,
            store=store,
            repo="owner/repo",
            pr_number=1,
            pr_labels=[],
        )
        
        assert decision.allowed
    
    def test_both_policy_requires_both(self):
        """Both policy requires label AND Telegram approval."""
        settings = MockSettings(enable_codex=True, autofix_policy="both")
        approval = PRApprovalState(repo="owner/repo", pr_number=1, approved_by_telegram=True)
        store = MockStore(approval=approval)
        
        decision = check_autofix_policy(
            settings=settings,
            store=store,
            repo="owner/repo",
            pr_number=1,
            pr_labels=[],
        )
        
        assert not decision.allowed
        assert "label" in decision.reason.lower()
    
    def test_both_policy_allowed_when_both_present(self):
        """Both policy allows when both requirements are met."""
        settings = MockSettings(enable_codex=True, autofix_policy="both")
        approval = PRApprovalState(repo="owner/repo", pr_number=1, approved_by_telegram=True)
        store = MockStore(approval=approval)
        
        decision = check_autofix_policy(
            settings=settings,
            store=store,
            repo="owner/repo",
            pr_number=1,
            pr_labels=["autofix-ok"],
        )
        
        assert decision.allowed
    
    def test_high_risk_blocks_autofix(self):
        """High risk level blocks autofix when require_human_for_high_risk=True."""
        settings = MockSettings(
            enable_codex=True,
            autofix_policy="label",
            require_human_for_high_risk=True,
        )
        store = MockStore()
        
        decision = check_autofix_policy(
            settings=settings,
            store=store,
            repo="owner/repo",
            pr_number=1,
            pr_labels=["autofix-ok"],
            arbiter_risk_level="high",
        )
        
        assert not decision.allowed
        assert decision.needs_human
        assert "human review" in decision.reason.lower()
    
    def test_high_risk_allowed_when_disabled(self):
        """High risk doesn't block when require_human_for_high_risk=False."""
        settings = MockSettings(
            enable_codex=True,
            autofix_policy="label",
            require_human_for_high_risk=False,
        )
        store = MockStore()
        
        decision = check_autofix_policy(
            settings=settings,
            store=store,
            repo="owner/repo",
            pr_number=1,
            pr_labels=["autofix-ok"],
            arbiter_risk_level="high",
        )
        
        assert decision.allowed


class TestPRApprovalState:
    """Tests for PR approval state storage."""
    
    def test_default_state(self):
        """Default state has no approvals."""
        state = PRApprovalState(repo="owner/repo", pr_number=1)
        
        assert not state.approved_by_telegram
        assert not state.paused
        assert state.approved_at is None
        assert state.approved_by_user_id is None
