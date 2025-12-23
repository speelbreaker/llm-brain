"""
Reconciliation State Module.

Tracks the status of position reconciliation between local tracker and exchange.
Provides cached state for API/UI access.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, List, Dict, Any


class ReconciliationResult(str, Enum):
    """Result of a reconciliation check."""
    CLEAN = "clean"
    DIVERGENT = "divergent"
    ERROR = "error"
    PENDING = "pending"  # Never run yet


@dataclass
class PositionMismatch:
    """Details of a single position mismatch."""
    symbol: str
    mismatch_type: str  # "untracked_on_exchange", "missing_on_exchange", "size_mismatch"
    local_size: Optional[float] = None
    exchange_size: Optional[float] = None
    side: Optional[str] = None
    diff_usd: Optional[float] = None
    
    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "mismatch_type": self.mismatch_type,
            "local_size": self.local_size,
            "exchange_size": self.exchange_size,
            "side": self.side,
            "diff_usd": round(self.diff_usd, 2) if self.diff_usd else None,
        }


@dataclass
class ReconciliationStatus:
    """Current reconciliation status."""
    result: ReconciliationResult
    last_run_time: Optional[datetime] = None
    is_clean: bool = True
    mismatches: List[PositionMismatch] = field(default_factory=list)
    exchange_count: int = 0
    local_count: int = 0
    action_taken: Optional[str] = None  # "halt", "auto_heal", "none"
    error_message: Optional[str] = None
    trading_blocked: bool = False  # True if trading halted due to divergence
    
    def to_dict(self) -> dict:
        return {
            "result": self.result.value,
            "last_run_time": self.last_run_time.isoformat() if self.last_run_time else None,
            "is_clean": self.is_clean,
            "mismatch_count": len(self.mismatches),
            "mismatches": [m.to_dict() for m in self.mismatches],
            "exchange_count": self.exchange_count,
            "local_count": self.local_count,
            "action_taken": self.action_taken,
            "error_message": self.error_message,
            "trading_blocked": self.trading_blocked,
        }


# Module-level cached state
_reconciliation_status: ReconciliationStatus = ReconciliationStatus(
    result=ReconciliationResult.PENDING,
    is_clean=True,
    trading_blocked=False,
)


def get_reconciliation_status() -> ReconciliationStatus:
    """Get current reconciliation status."""
    return _reconciliation_status


def set_reconciliation_status(
    result: ReconciliationResult,
    is_clean: bool,
    mismatches: Optional[List[PositionMismatch]] = None,
    exchange_count: int = 0,
    local_count: int = 0,
    action_taken: Optional[str] = None,
    error_message: Optional[str] = None,
    trading_blocked: bool = False,
) -> ReconciliationStatus:
    """
    Update reconciliation status.
    
    Returns:
        Updated ReconciliationStatus.
    """
    global _reconciliation_status
    
    _reconciliation_status = ReconciliationStatus(
        result=result,
        last_run_time=datetime.now(timezone.utc),
        is_clean=is_clean,
        mismatches=mismatches or [],
        exchange_count=exchange_count,
        local_count=local_count,
        action_taken=action_taken,
        error_message=error_message,
        trading_blocked=trading_blocked,
    )
    
    return _reconciliation_status


def set_trading_blocked(blocked: bool) -> None:
    """Set whether trading is blocked due to reconciliation."""
    global _reconciliation_status
    _reconciliation_status.trading_blocked = blocked


def is_trading_blocked_by_reconciliation() -> bool:
    """Check if trading is blocked due to reconciliation divergence."""
    return _reconciliation_status.trading_blocked


def clear_reconciliation_block() -> None:
    """
    Clear reconciliation block, allowing trading to resume.
    Should be called after manual review or auto-heal.
    """
    global _reconciliation_status
    _reconciliation_status.trading_blocked = False


def build_mismatches_from_diff(diff_dict: Dict[str, Any]) -> List[PositionMismatch]:
    """
    Build PositionMismatch list from reconciliation diff dictionary.
    
    Args:
        diff_dict: Dictionary from reconcile_positions() stats.
    
    Returns:
        List of PositionMismatch objects.
    """
    mismatches: List[PositionMismatch] = []
    
    # Untracked on exchange (exists on exchange but not in local)
    for symbol in diff_dict.get("missing_in_local", []):
        mismatches.append(PositionMismatch(
            symbol=symbol,
            mismatch_type="untracked_on_exchange",
            local_size=0,
            exchange_size=None,  # We don't have the size here
        ))
    
    # Missing on exchange (exists locally but not on exchange)
    for symbol in diff_dict.get("missing_in_exchange", []):
        mismatches.append(PositionMismatch(
            symbol=symbol,
            mismatch_type="missing_on_exchange",
            local_size=None,  # We don't have the size here
            exchange_size=0,
        ))
    
    # Size mismatches
    for mismatch_tuple in diff_dict.get("size_mismatches", []):
        if isinstance(mismatch_tuple, (tuple, list)) and len(mismatch_tuple) >= 3:
            symbol, local_size, exchange_size = mismatch_tuple[:3]
            mismatches.append(PositionMismatch(
                symbol=symbol,
                mismatch_type="size_mismatch",
                local_size=local_size,
                exchange_size=exchange_size,
            ))
    
    return mismatches


def reset_reconciliation_state() -> None:
    """Reset reconciliation state to initial values."""
    global _reconciliation_status
    _reconciliation_status = ReconciliationStatus(
        result=ReconciliationResult.PENDING,
        is_clean=True,
        trading_blocked=False,
    )



