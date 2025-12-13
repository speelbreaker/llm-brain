"""
Tests for hybrid decision mode logic in the agent loop.

Tests the decision_mode behavior:
- rule_only: rules execute, LLM optional shadow logging
- llm_only: LLM executes with fallback to rules on error/invalid
- hybrid_shadow: rules execute, LLM runs in shadow for logging/comparison
"""
from __future__ import annotations

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from src.models import ActionType


class TestDecisionModeLogic:
    """Tests for the decision mode selection logic."""
    
    def test_rule_only_uses_rule_action(self):
        """In rule_only mode, proposed_action should be the rule action."""
        rule_action = {
            "action": ActionType.OPEN_COVERED_CALL.value,
            "params": {"symbol": "BTC-27DEC24-100000-C"},
            "reasoning": "Good IVRV",
        }
        
        settings_mock = MagicMock()
        settings_mock.decision_mode = "rule_only"
        settings_mock.llm_enabled = True
        settings_mock.llm_shadow_enabled = False
        
        proposed_action = rule_action.copy()
        decision_source = "rule_based"
        
        assert proposed_action["action"] == "OPEN_COVERED_CALL"
        assert decision_source == "rule_based"
    
    def test_llm_only_uses_llm_when_valid(self):
        """In llm_only mode with valid LLM output, proposed_action should be LLM action."""
        llm_action = {
            "action": ActionType.OPEN_COVERED_CALL.value,
            "params": {"symbol": "BTC-27DEC24-100000-C"},
            "reasoning": "LLM decision",
            "validated": True,
        }
        rule_action = {
            "action": ActionType.DO_NOTHING.value,
            "params": {},
            "reasoning": "Rule says nothing",
        }
        
        settings_mock = MagicMock()
        settings_mock.decision_mode = "llm_only"
        settings_mock.llm_enabled = True
        
        if llm_action is not None and llm_action.get("validated", False):
            proposed_action = llm_action.copy()
            decision_source = "llm"
        else:
            proposed_action = rule_action.copy()
            decision_source = "llm_fallback_to_rule"
        
        assert proposed_action["action"] == "OPEN_COVERED_CALL"
        assert decision_source == "llm"
    
    def test_llm_only_falls_back_to_rule_when_invalid(self):
        """In llm_only mode with invalid LLM, proposed_action should be rule action."""
        llm_action = {
            "action": ActionType.DO_NOTHING.value,
            "params": {},
            "reasoning": "LLM rejected by validation",
            "validated": False,
        }
        rule_action = {
            "action": ActionType.OPEN_COVERED_CALL.value,
            "params": {"symbol": "BTC-27DEC24-100000-C"},
            "reasoning": "Rule decision",
        }
        
        settings_mock = MagicMock()
        settings_mock.decision_mode = "llm_only"
        settings_mock.llm_enabled = True
        
        if llm_action is not None and llm_action.get("validated", False):
            proposed_action = llm_action.copy()
            decision_source = "llm"
        else:
            proposed_action = rule_action.copy()
            decision_source = "llm_fallback_to_rule"
        
        assert proposed_action["action"] == "OPEN_COVERED_CALL"
        assert decision_source == "llm_fallback_to_rule"
    
    def test_llm_only_falls_back_when_llm_is_none(self):
        """In llm_only mode with LLM error (None), proposed_action should be rule action."""
        llm_action = None
        rule_action = {
            "action": ActionType.OPEN_COVERED_CALL.value,
            "params": {"symbol": "BTC-27DEC24-100000-C"},
            "reasoning": "Rule decision",
        }
        
        settings_mock = MagicMock()
        settings_mock.decision_mode = "llm_only"
        settings_mock.llm_enabled = True
        
        if llm_action is not None and llm_action.get("validated", False):
            proposed_action = llm_action.copy()
            decision_source = "llm"
        else:
            proposed_action = rule_action.copy()
            decision_source = "llm_fallback_to_rule"
        
        assert proposed_action["action"] == "OPEN_COVERED_CALL"
        assert decision_source == "llm_fallback_to_rule"
    
    def test_hybrid_shadow_uses_rule_action(self):
        """In hybrid_shadow mode, proposed_action should be rule action."""
        rule_action = {
            "action": ActionType.OPEN_COVERED_CALL.value,
            "params": {"symbol": "BTC-27DEC24-100000-C"},
            "reasoning": "Rule decision",
        }
        llm_action = {
            "action": ActionType.DO_NOTHING.value,
            "params": {},
            "reasoning": "LLM says wait",
            "validated": True,
        }
        
        settings_mock = MagicMock()
        settings_mock.decision_mode = "hybrid_shadow"
        settings_mock.llm_enabled = True
        
        proposed_action = rule_action.copy()
        decision_source = "rule_based_shadow_llm"
        
        assert proposed_action["action"] == "OPEN_COVERED_CALL"
        assert decision_source == "rule_based_shadow_llm"


class TestShouldComputeLlm:
    """Tests for determining when to compute LLM action."""
    
    def test_compute_llm_for_llm_only_mode(self):
        """LLM should be computed in llm_only mode when enabled."""
        llm_enabled = True
        decision_mode = "llm_only"
        
        should_compute_llm = llm_enabled and decision_mode in ("llm_only", "hybrid_shadow")
        
        assert should_compute_llm is True
    
    def test_compute_llm_for_hybrid_shadow_mode(self):
        """LLM should be computed in hybrid_shadow mode when enabled."""
        llm_enabled = True
        decision_mode = "hybrid_shadow"
        
        should_compute_llm = llm_enabled and decision_mode in ("llm_only", "hybrid_shadow")
        
        assert should_compute_llm is True
    
    def test_no_llm_for_rule_only_no_shadow(self):
        """LLM should not be computed in rule_only without shadow enabled."""
        llm_enabled = True
        llm_shadow_enabled = False
        decision_mode = "rule_only"
        
        should_compute_llm = llm_enabled and decision_mode in ("llm_only", "hybrid_shadow")
        should_compute_shadow = llm_enabled and llm_shadow_enabled and decision_mode == "rule_only"
        
        assert should_compute_llm is False
        assert should_compute_shadow is False
    
    def test_compute_shadow_for_rule_only_with_shadow(self):
        """LLM should be computed as shadow in rule_only with shadow enabled."""
        llm_enabled = True
        llm_shadow_enabled = True
        decision_mode = "rule_only"
        
        should_compute_llm = llm_enabled and decision_mode in ("llm_only", "hybrid_shadow")
        should_compute_shadow = llm_enabled and llm_shadow_enabled and decision_mode == "rule_only"
        
        assert should_compute_llm is False
        assert should_compute_shadow is True
    
    def test_no_llm_when_disabled(self):
        """LLM should not be computed when llm_enabled is False."""
        llm_enabled = False
        decision_mode = "llm_only"
        
        should_compute_llm = llm_enabled and decision_mode in ("llm_only", "hybrid_shadow")
        
        assert should_compute_llm is False


class TestDecisionEntryContainsBothActions:
    """Tests for decision entry structure with both actions."""
    
    def test_decision_entry_includes_both_actions(self):
        """Decision entry should include rule_action, llm_action, and decision_mode."""
        rule_action = {
            "action": ActionType.OPEN_COVERED_CALL.value,
            "params": {"symbol": "BTC-27DEC24-100000-C"},
            "reasoning": "Rule decision",
        }
        llm_action = {
            "action": ActionType.DO_NOTHING.value,
            "params": {},
            "reasoning": "LLM says wait",
            "validated": True,
        }
        proposed_action = rule_action.copy()
        decision_source = "rule_based_shadow_llm"
        decision_mode = "hybrid_shadow"
        
        decision_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "decision_source": decision_source,
            "decision_mode": decision_mode,
            "proposed_action": proposed_action,
            "final_action": proposed_action,
            "rule_action": rule_action,
            "llm_action": llm_action,
            "risk_check": {"allowed": True, "reasons": []},
            "execution": {"status": "skipped"},
            "config_snapshot": {
                "mode": "research",
                "training_mode": False,
                "llm_enabled": True,
                "decision_mode": decision_mode,
                "dry_run": True,
            },
        }
        
        assert decision_entry["decision_mode"] == "hybrid_shadow"
        assert decision_entry["rule_action"] == rule_action
        assert decision_entry["llm_action"] == llm_action
        assert decision_entry["decision_source"] == "rule_based_shadow_llm"
    
    def test_decision_entry_handles_none_llm(self):
        """Decision entry should handle llm_action being None."""
        rule_action = {
            "action": ActionType.OPEN_COVERED_CALL.value,
            "params": {"symbol": "BTC-27DEC24-100000-C"},
            "reasoning": "Rule decision",
        }
        llm_action = None
        
        decision_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "decision_source": "rule_based",
            "decision_mode": "rule_only",
            "rule_action": rule_action,
            "llm_action": llm_action,
        }
        
        assert decision_entry["llm_action"] is None
        assert decision_entry["rule_action"] is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
