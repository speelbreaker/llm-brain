"""
Runtime IV multiplier override store.

This module provides in-memory storage for IV multiplier overrides applied
from calibration history. Overrides are runtime-only and reset on restart.

Overrides are keyed by (underlying, dte_min, dte_max) for granular control,
with a fallback to (underlying,) for simpler use cases.

The "applied" multipliers (source of truth for UI "Current Applied Multipliers")
are tracked with timestamps so the UI can show when calibration was last applied.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import threading


OverrideKey = Tuple[str, int, int]

_lock = threading.Lock()
_overrides: Dict[OverrideKey, float] = {}
_simple_overrides: Dict[str, float] = {}


@dataclass
class AppliedMultiplierState:
    """Tracks the currently applied multiplier with metadata."""
    global_multiplier: float = 1.0
    band_multipliers: Dict[str, float] = field(default_factory=dict)
    last_updated: Optional[datetime] = None
    source: str = "default"
    applied_reason: str = ""


_applied_state: Dict[str, AppliedMultiplierState] = {}


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


def set_applied_multiplier(
    underlying: str,
    global_multiplier: float,
    band_multipliers: Optional[Dict[str, float]] = None,
    source: str = "live",
    applied_reason: str = "",
) -> None:
    """
    Set the applied multiplier for an underlying.
    
    This is the source of truth for the "Current Applied Multipliers" UI panel.
    Called when:
    - A live calibration is applied via policy
    - User clicks "Force-Apply Latest"
    
    NOTE: Auto-calibrate (harvested) should NOT call this.
    """
    with _lock:
        _applied_state[underlying.upper()] = AppliedMultiplierState(
            global_multiplier=global_multiplier,
            band_multipliers=band_multipliers or {},
            last_updated=datetime.now(timezone.utc),
            source=source,
            applied_reason=applied_reason,
        )
        _simple_overrides[underlying.upper()] = global_multiplier


def get_applied_multiplier(underlying: str) -> AppliedMultiplierState:
    """
    Get the applied multiplier state for an underlying.
    
    Returns the state with current values and metadata.
    If no calibration has been applied, returns default (1.0).
    """
    with _lock:
        return _applied_state.get(
            underlying.upper(),
            AppliedMultiplierState(global_multiplier=1.0),
        )


def get_all_applied_multipliers() -> Dict[str, Dict[str, Any]]:
    """
    Get all applied multiplier states for all underlyings.
    
    Returns a dict keyed by underlying with multiplier details.
    """
    with _lock:
        result = {}
        for underlying, state in _applied_state.items():
            result[underlying] = {
                "global_multiplier": state.global_multiplier,
                "band_multipliers": state.band_multipliers,
                "last_updated": state.last_updated.isoformat() if state.last_updated else None,
                "source": state.source,
                "applied_reason": state.applied_reason,
            }
        return result
