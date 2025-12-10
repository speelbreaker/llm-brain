"""
Runtime IV multiplier override store.

This module provides in-memory storage for IV multiplier overrides applied
from calibration history. Overrides are runtime-only and reset on restart.

Overrides are keyed by (underlying, dte_min, dte_max) for granular control,
with a fallback to (underlying,) for simpler use cases.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple
import threading


OverrideKey = Tuple[str, int, int]

_lock = threading.Lock()
_overrides: Dict[OverrideKey, float] = {}
_simple_overrides: Dict[str, float] = {}


def set_iv_multiplier_override(
    underlying: str,
    value: float,
    dte_min: Optional[int] = None,
    dte_max: Optional[int] = None,
) -> None:
    """
    Set an IV multiplier override.
    
    Args:
        underlying: Asset symbol (BTC, ETH)
        value: The IV multiplier value to use
        dte_min: Optional minimum DTE for granular override
        dte_max: Optional maximum DTE for granular override
    
    If dte_min and dte_max are provided, stores a granular override.
    Otherwise, stores a simple underlying-only override.
    """
    with _lock:
        if dte_min is not None and dte_max is not None:
            key: OverrideKey = (underlying.upper(), dte_min, dte_max)
            _overrides[key] = value
        else:
            _simple_overrides[underlying.upper()] = value


def get_iv_multiplier_override(
    underlying: str,
    dte_min: Optional[int] = None,
    dte_max: Optional[int] = None,
) -> Optional[float]:
    """
    Get the IV multiplier override.
    
    Looks up granular override first, then falls back to simple override.
    
    Returns:
        The override value, or None if no override is set.
    """
    with _lock:
        if dte_min is not None and dte_max is not None:
            key: OverrideKey = (underlying.upper(), dte_min, dte_max)
            if key in _overrides:
                return _overrides[key]
        return _simple_overrides.get(underlying.upper())


def clear_iv_multiplier_override(
    underlying: str,
    dte_min: Optional[int] = None,
    dte_max: Optional[int] = None,
) -> None:
    """
    Clear the IV multiplier override.
    """
    with _lock:
        if dte_min is not None and dte_max is not None:
            key: OverrideKey = (underlying.upper(), dte_min, dte_max)
            _overrides.pop(key, None)
        else:
            _simple_overrides.pop(underlying.upper(), None)


def get_all_overrides() -> Dict[str, Dict]:
    """
    Get all current IV multiplier overrides.
    
    Returns:
        Dictionary with 'granular' and 'simple' override mappings.
    """
    with _lock:
        return {
            "granular": {f"{k[0]}_{k[1]}-{k[2]}d": v for k, v in _overrides.items()},
            "simple": dict(_simple_overrides),
        }


def clear_all_overrides() -> None:
    """
    Clear all IV multiplier overrides.
    """
    with _lock:
        _overrides.clear()
        _simple_overrides.clear()
