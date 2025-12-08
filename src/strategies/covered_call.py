"""
CoveredCallStrategy - wraps existing covered call logic into the Strategy interface.

This strategy sells covered calls on BTC/ETH:
- Uses rule-based, LLM, or training policies based on configuration
- Integrates with existing policy modules
- Adds strategy_id for multi-strategy attribution in logs
"""
from __future__ import annotations

from typing import Any, List, TYPE_CHECKING

from src.strategies.types import Strategy, StrategyConfig, CandidateAction, StrategyDecision
from src.models import ActionType

if TYPE_CHECKING:
    from src.models import AgentState


class CoveredCallStrategy(Strategy):
    """
    Wraps the current single-bot behaviour (rule-based / LLM / training)
    into a Strategy interface.
    
    This is a thin adapter that delegates to the existing policy modules,
    preserving all existing behaviour while adding strategy_id.
    """
    
    STRATEGY_ID = "covered_call_v1"
    
    @property
    def strategy_id(self) -> str:
        """Unique identifier for this strategy."""
        return self.STRATEGY_ID
    
    def propose_actions(self, state: "AgentState") -> List[dict[str, Any]]:
        """
        Propose actions based on the current mode:
        - training: use training_policy.build_training_actions
        - llm: use agent_brain_llm.choose_action_with_llm
        - rule_based: use policy_rule_based.decide_action
        
        Returns a list of action dicts (may be 0, 1, or many for training).
        Each action dict includes strategy_id for attribution.
        """
        from src.config import settings
        from src.policy_rule_based import decide_action as rule_decide_action
        from src.agent_brain_llm import choose_action_with_llm
        from src.training_policy import build_actions as build_training_actions
        
        actions: List[dict[str, Any]] = []
        
        if self.config.mode == "training":
            training_actions = build_training_actions(state, settings)
            if training_actions:
                for action in training_actions:
                    action["strategy_id"] = self.strategy_id
                actions.extend(training_actions)
            else:
                fallback = rule_decide_action(state, settings)
                fallback["strategy_id"] = self.strategy_id
                actions.append(fallback)
                
        elif self.config.mode == "llm":
            llm_action = choose_action_with_llm(
                state,
                state.candidate_options,
            )
            llm_action["strategy_id"] = self.strategy_id
            actions.append(llm_action)
            
        else:
            rule_action = rule_decide_action(state, settings)
            rule_action["strategy_id"] = self.strategy_id
            actions.append(rule_action)
        
        return actions
    
    def propose_candidate_actions(
        self, state: "AgentState"
    ) -> List[CandidateAction]:
        """
        Propose candidate actions with typed metadata.
        """
        action_dicts = self.propose_actions(state)
        candidates: List[CandidateAction] = []
        
        for ad in action_dicts:
            params = ad.get("params", {})
            diag = ad.get("diagnostics", {})
            
            candidate = CandidateAction(
                action_type=ad.get("action", "DO_NOTHING"),
                symbol=params.get("symbol", params.get("to_symbol", "")),
                underlying=params.get("underlying", ad.get("underlying", "")),
                params=params,
                strike=diag.get("strike"),
                expiry=None,
                dte=diag.get("dte"),
                delta=diag.get("delta"),
                size=params.get("size"),
                score=diag.get("score", 0.0),
                ivrv_score=diag.get("ivrv", 0.0),
                premium_usd=diag.get("premium_usd", 0.0),
                reasoning=ad.get("reasoning", ""),
                is_exploratory=ad.get("is_exploratory", False),
            )
            candidates.append(candidate)
        
        return candidates
    
    def choose_action(
        self,
        state: "AgentState",
        candidates: List[CandidateAction],
    ) -> StrategyDecision:
        """
        Choose the final action from candidates.
        
        For now, just picks the first candidate or returns DO_NOTHING.
        Future: integrate with StrategyPolicy for flexible policy selection.
        """
        if not candidates:
            return StrategyDecision(
                strategy_id=self.strategy_id,
                action=ActionType.DO_NOTHING.value,
                params={},
                reasoning="No candidates available",
                decision_source="rule_based",
                mode=self.config.mode,
            )
        
        best = candidates[0]
        return StrategyDecision(
            strategy_id=self.strategy_id,
            action=best.action_type,
            params=best.params,
            reasoning=best.reasoning,
            decision_source="rule_based" if self.config.mode != "llm" else "llm",
            mode=self.config.mode,
            candidate=best,
            diagnostics={
                "delta": best.delta,
                "dte": best.dte,
                "premium_usd": best.premium_usd,
                "score": best.score,
            },
        )
    
    def get_mode_description(self) -> str:
        """Return a human-readable description of the current mode."""
        mode_names = {
            "training": "Training (multi-profile exploration)",
            "llm": "LLM-based decision making",
            "rule_based": "Rule-based (deterministic)",
        }
        return mode_names.get(self.config.mode, self.config.mode)
