"""
CoveredCallStrategy - wraps existing covered call logic into the Strategy interface.
"""
from __future__ import annotations

from typing import Any, List, TYPE_CHECKING

from src.strategies.types import Strategy, StrategyConfig
from src.models import ActionType

if TYPE_CHECKING:
    from src.models import AgentState


class CoveredCallStrategy(Strategy):
    """
    Wraps the current single-bot behaviour (rule-based / LLM / training)
    into a Strategy interface.
    
    This is a thin adapter that delegates to the existing policy modules,
    preserving all existing behaviour.
    """
    
    def propose_actions(self, state: "AgentState") -> List[dict[str, Any]]:
        """
        Propose actions based on the current mode:
        - training: use training_policy.build_training_actions
        - llm: use agent_brain_llm.choose_action_with_llm
        - rule_based: use policy_rule_based.decide_action
        
        Returns a list of action dicts (may be 0, 1, or many for training).
        """
        from src.config import settings
        from src.policy_rule_based import decide_action as rule_decide_action
        from src.agent_brain_llm import choose_action_with_llm
        from src.training_policy import build_actions as build_training_actions
        
        actions: List[dict[str, Any]] = []
        
        if self.config.mode == "training":
            training_actions = build_training_actions(state, settings)
            if training_actions:
                actions.extend(training_actions)
            else:
                fallback = rule_decide_action(state, settings)
                actions.append(fallback)
                
        elif self.config.mode == "llm":
            llm_action = choose_action_with_llm(
                state,
                state.candidate_options,
            )
            actions.append(llm_action)
            
        else:
            rule_action = rule_decide_action(state, settings)
            actions.append(rule_action)
        
        return actions
    
    def get_mode_description(self) -> str:
        """Return a human-readable description of the current mode."""
        mode_names = {
            "training": "Training (multi-profile exploration)",
            "llm": "LLM-based decision making",
            "rule_based": "Rule-based (deterministic)",
        }
        return mode_names.get(self.config.mode, self.config.mode)
