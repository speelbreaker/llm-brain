"""Deprecated compatibility shim.

Single source of truth for strategy specs/execution lives in
src/fidelity/canonical_strategies.py.
"""

from __future__ import annotations

from .canonical_strategies import StrategySpec, canonical_strategies


def canonical_strategy_specs() -> list[StrategySpec]:
    return list(canonical_strategies())
