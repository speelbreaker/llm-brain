"""
Core types for the strategy layer.
Defines StrategyConfig, Strategy interface, and action types.

This module provides the shared abstractions that enable:
- Multiple strategies to coexist (Covered Calls, Wheel, etc.)
- Consistent state/action schemas between live agent and backtester
- Typed action representations for logging and execution
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, List, Optional, Any, Dict, TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from src.models import AgentState, CandidateOption

ModeType = Literal["rule_based", "llm", "training"]


@dataclass
class StrategyConfig:
    """Configuration for a trading strategy."""
    name: str
    underlyings: List[str]
    mode: ModeType
    enabled: bool
    training_enabled: bool
    explore_prob: float
    delta_min: float
    delta_max: float
    dte_min: int
    dte_max: int
    ivrv_min: float
    size_fraction: float
    profile_name: Optional[str] = None
    max_calls_per_underlying: int = 1
    training_max_calls_per_expiry: int = 3
    training_strategies: List[str] = field(default_factory=lambda: ["conservative", "moderate", "aggressive"])


@dataclass
class CandidateAction:
    """
    A candidate action proposed by a strategy with scoring metadata.
    
    This represents ONE possible action the strategy could take,
    along with metadata used for ranking and logging.
    """
    action_type: str
    symbol: str
    underlying: str
    params: Dict[str, Any] = field(default_factory=dict)
    
    strike: Optional[float] = None
    expiry: Optional[datetime] = None
    dte: Optional[int] = None
    delta: Optional[float] = None
    size: Optional[float] = None
    
    score: float = 0.0
    ivrv_score: float = 0.0
    premium_usd: float = 0.0
    
    reasoning: str = ""
    is_exploratory: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "action_type": self.action_type,
            "symbol": self.symbol,
            "underlying": self.underlying,
            "params": self.params,
            "strike": self.strike,
            "expiry": self.expiry.isoformat() if self.expiry else None,
            "dte": self.dte,
            "delta": self.delta,
            "size": self.size,
            "score": self.score,
            "ivrv_score": self.ivrv_score,
            "premium_usd": self.premium_usd,
            "reasoning": self.reasoning,
            "is_exploratory": self.is_exploratory,
        }


@dataclass
class StrategyDecision:
    """
    Final decision from a strategy, ready for execution.
    
    This is what the executor/live agent actually processes.
    Includes strategy_id for multi-strategy logging and attribution.
    """
    strategy_id: str
    action: str
    params: Dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""
    
    decision_source: str = "rule_based"
    mode: str = "research"
    policy_version: str = "v1"
    
    candidate: Optional[CandidateAction] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        result = {
            "strategy_id": self.strategy_id,
            "action": self.action,
            "params": self.params,
            "reasoning": self.reasoning,
            "decision_source": self.decision_source,
            "mode": self.mode,
            "policy_version": self.policy_version,
        }
        if self.candidate:
            result["candidate"] = self.candidate.to_dict()
        if self.diagnostics:
            result["diagnostics"] = self.diagnostics
        return result


class StrategyPolicy(Protocol):
    """
    Protocol for policy implementations (rule-based, LLM, etc.).
    
    A policy takes candidates and returns a final decision.
    """
    
    def choose(
        self,
        state: "AgentState",
        candidates: List[CandidateAction],
        strategy_config: StrategyConfig,
    ) -> StrategyDecision:
        """Choose the final action from candidates."""
        ...


class Strategy:
    """
    Base interface for any trading strategy.
    
    A Strategy sees the full AgentState and proposes zero or more actions.
    The risk engine still has final veto.
    
    Subclasses must implement:
    - propose_actions(state) -> list of action dicts
    
    The strategy_id property is used for logging and multi-strategy attribution.
    """
    
    def __init__(self, config: StrategyConfig):
        self.config = config
    
    @property
    def name(self) -> str:
        """Human-readable strategy name."""
        return self.config.name
    
    @property
    def strategy_id(self) -> str:
        """
        Unique identifier for this strategy instance.
        Used in logs and decisions for attribution.
        """
        return f"{self.config.name.lower().replace(' ', '_')}"
    
    def propose_actions(self, state: "AgentState") -> List[Dict[str, Any]]:
        """
        Propose zero or more actions based on the current state.
        
        Args:
            state: Current AgentState with portfolio, candidates, etc.
        
        Returns:
            List of action dicts, each with keys: action, params, reasoning, etc.
            Each dict should include 'strategy_id' for attribution.
        """
        raise NotImplementedError("Subclasses must implement propose_actions")
    
    def propose_candidate_actions(
        self, state: "AgentState"
    ) -> List[CandidateAction]:
        """
        Propose candidate actions with typed metadata.
        
        This is the typed alternative to propose_actions() for new code.
        Default implementation converts from propose_actions().
        
        Args:
            state: Current AgentState
        
        Returns:
            List of CandidateAction with scoring metadata
        """
        action_dicts = self.propose_actions(state)
        candidates: List[CandidateAction] = []
        
        for ad in action_dicts:
            params = ad.get("params", {})
            diag = ad.get("diagnostics", {})
            
            candidate = CandidateAction(
                action_type=ad.get("action", "DO_NOTHING"),
                symbol=params.get("symbol", ""),
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
