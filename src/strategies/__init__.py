"""
Strategy layer for pluggable trading strategies.

Key abstractions:
- Strategy: Base interface for any trading strategy
- StrategyConfig: Configuration for a strategy
- CandidateAction: Typed action with scoring metadata
- StrategyDecision: Final decision ready for execution
- StrategyRegistry: Manages active strategies
"""
from src.strategies.types import (
    Strategy,
    StrategyConfig,
    ModeType,
    CandidateAction,
    StrategyDecision,
    StrategyPolicy,
)
from src.strategies.registry import StrategyRegistry, build_default_registry
from src.strategies.covered_call import CoveredCallStrategy

__all__ = [
    "Strategy",
    "StrategyConfig",
    "ModeType",
    "CandidateAction",
    "StrategyDecision",
    "StrategyPolicy",
    "StrategyRegistry",
    "build_default_registry",
    "CoveredCallStrategy",
]
