"""
Core types for the strategy layer.
Defines StrategyConfig and the base Strategy interface.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, List, Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from src.models import AgentState

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


class Strategy:
    """
    Base interface for any trading strategy.
    A Strategy sees the full AgentState and proposes zero or more actions.
    The risk engine still has final veto.
    """
    
    def __init__(self, config: StrategyConfig):
        self.config = config
    
    @property
    def name(self) -> str:
        return self.config.name
    
    def propose_actions(self, state: "AgentState") -> List[dict[str, Any]]:
        """
        Propose zero or more actions based on the current state.
        
        Args:
            state: Current AgentState with portfolio, candidates, etc.
        
        Returns:
            List of action dicts, each with keys: action, params, reasoning, etc.
        """
        raise NotImplementedError("Subclasses must implement propose_actions")
