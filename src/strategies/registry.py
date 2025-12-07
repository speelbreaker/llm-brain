"""
StrategyRegistry - manages and provides access to registered strategies.
"""
from __future__ import annotations

from typing import List, TYPE_CHECKING

from src.strategies.types import Strategy, StrategyConfig

if TYPE_CHECKING:
    from src.config import Settings


class StrategyRegistry:
    """
    Registry that holds all configured strategies.
    Provides methods to get active strategies for execution.
    """
    
    def __init__(self, strategies: List[Strategy]):
        self._strategies = strategies
    
    def get_active_strategies(self) -> List[Strategy]:
        """Return all enabled strategies."""
        return [s for s in self._strategies if s.config.enabled]
    
    def get_all_strategies(self) -> List[Strategy]:
        """Return all strategies regardless of enabled status."""
        return list(self._strategies)
    
    def get_strategy_by_name(self, name: str) -> Strategy | None:
        """Find a strategy by name."""
        for s in self._strategies:
            if s.name == name:
                return s
        return None
    
    def __len__(self) -> int:
        return len(self._strategies)
    
    def __iter__(self):
        return iter(self._strategies)


def build_default_registry(settings: "Settings") -> StrategyRegistry:
    """
    Build the default set of strategies based on global Settings.
    
    For now, we only create one CoveredCallStrategy that mirrors the
    existing behaviour for BTC and ETH.
    
    Args:
        settings: Application settings
    
    Returns:
        StrategyRegistry with default strategies
    """
    from src.strategies.covered_call import CoveredCallStrategy
    
    if settings.is_training_enabled:
        mode = "training"
    elif settings.llm_enabled:
        mode = "llm"
    else:
        mode = "rule_based"
    
    cfg = StrategyConfig(
        name="CoveredCallLadder",
        underlyings=settings.underlyings,
        mode=mode,
        enabled=True,
        training_enabled=settings.is_training_enabled,
        explore_prob=settings.explore_prob,
        delta_min=settings.effective_delta_min,
        delta_max=settings.effective_delta_max,
        dte_min=settings.effective_dte_min,
        dte_max=settings.effective_dte_max,
        ivrv_min=settings.effective_ivrv_min,
        size_fraction=settings.default_order_size,
        profile_name=settings.training_profile_mode,
        max_calls_per_underlying=settings.max_calls_per_underlying,
        training_max_calls_per_expiry=settings.training_max_calls_per_expiry,
        training_strategies=settings.training_strategies,
    )
    
    strategy = CoveredCallStrategy(cfg)
    return StrategyRegistry([strategy])
