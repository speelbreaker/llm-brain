"""
GregStrategy - Wraps the Greg Mandolini VRP Harvester logic into the main agent loop.

Currently operates in ADVISORY/SIGNAL mode:
1. Runs the Greg Selector (Phase 1) to determine the best strategy for the regime.
2. Reports the selected strategy and reasoning.
3. Does NOT yet execute complex multi-leg structures (Phase 2).
"""
from __future__ import annotations

from typing import List, Dict, Any

from src.strategies.types import (
    Strategy,
    StrategyConfig,
    CandidateAction,
)
from src.bots.gregbot import get_gregbot_evaluations_for_underlying
from src.models import AgentState


class GregStrategy(Strategy):
    """
    Greg Bot Strategy Wrapper.
    
    Integrates the 'GregSelector' logic into the main trading loop.
    """
    
    def propose_actions(self, state: "AgentState") -> List[Dict[str, Any]]:
        """
        Run Greg Selector for each underlying and propose actions.
        
        Current Implementation:
        - Evaluates market conditions.
        - Selects a strategy (e.g. STRATEGY_A_STRADDLE).
        - Returns a DO_NOTHING action with the selection in the reasoning.
        - Future: Will return OPEN actions with specific option contracts.
        """
        actions = []
        
        for underlying in self.config.underlyings:
            # 1. Run Greg Phase 1 Selector
            # This computes sensors and evaluates the decision tree
            try:
                # We use 'test' mode if we are on testnet, else 'live'
                env_mode = "test" if "testnet" in str(state.spot) else "live" 
                # Actually state.spot doesn't have env info. 
                # We can just rely on the default behavior or settings.
                
                evaluation = get_gregbot_evaluations_for_underlying(underlying)
                
                selected_strategy = evaluation.get("selected_strategy", "NO_TRADE")
                reasoning = evaluation.get("decision_reasoning", "No reasoning provided")
                
                # 2. Construct Decision
                # For now, we only log the signal. 
                # Real execution requires a leg-builder (Phase 2).
                
                action_type = "DO_NOTHING"
                
                if selected_strategy != "NO_TRADE":
                    # In the future, this is where we would call a leg builder
                    # e.g. legs = build_straddle_legs(underlying, delta=0.5, dte=30)
                    # and return action="OPEN_MULTI_LEG", params={legs}
                    
                    full_reasoning = f"Greg Selected: {selected_strategy}. Reason: {reasoning}. (Execution pending Phase 2)"
                else:
                    full_reasoning = f"Greg Selected: NO_TRADE. Reason: {reasoning}"

                actions.append({
                    "strategy_id": self.strategy_id,
                    "action": action_type,
                    "underlying": underlying,
                    "params": {
                        "symbol": "", # No specific symbol yet
                        "selected_strategy": selected_strategy
                    },
                    "reasoning": full_reasoning,
                    "diagnostics": {
                        "score": 0.0, # Could use sensor scores here
                        "sensors": evaluation.get("sensors", {})
                    }
                })
                
            except Exception as e:
                print(f"[GregStrategy] Error evaluating {underlying}: {e}")
                actions.append({
                    "strategy_id": self.strategy_id,
                    "action": "DO_NOTHING",
                    "underlying": underlying,
                    "reasoning": f"Greg evaluation failed: {e}"
                })
                
        return actions
