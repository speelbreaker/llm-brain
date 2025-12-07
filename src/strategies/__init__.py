"""
Strategy layer for pluggable trading strategies.
"""
from src.strategies.types import Strategy, StrategyConfig, ModeType
from src.strategies.registry import StrategyRegistry, build_default_registry
from src.strategies.covered_call import CoveredCallStrategy

__all__ = [
    "Strategy",
    "StrategyConfig",
    "ModeType",
    "StrategyRegistry",
    "build_default_registry",
    "CoveredCallStrategy",
]
