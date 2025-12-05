"""
Risk manager module for enforcing trading limits.
Provides per-trade, per-underlying, and daily trade caps.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, Optional, Sequence, Tuple


@dataclass
class OpenPosition:
    """Represents an open trading position."""
    position_id: str
    underlying: str
    estimated_max_loss_pct: float
    opened_at: datetime


@dataclass
class CandidateTrade:
    """Represents a proposed trade to be validated."""
    underlying: str
    estimated_max_loss_pct: float
    opened_at: datetime
    strategy_id: str


@dataclass
class RiskConfig:
    """Configuration for risk limits."""
    max_risk_per_trade: float = 0.01
    max_risk_per_underlying: float = 0.03
    max_new_trades_per_day: int = 2
    enabled: bool = True


@dataclass
class RiskCheckResult:
    """Result of a risk check."""
    allowed: bool
    reason: Optional[str] = None


class RiskManager:
    """
    Manages trading risk by enforcing position and exposure limits.
    
    Enforces:
    - Maximum risk per individual trade (~1% of equity)
    - Maximum correlated risk per underlying (~3%)
    - Maximum new trades per day (1-2)
    """
    
    def __init__(self, config: RiskConfig):
        self.config = config
        self._new_trades_by_day: Dict[Tuple[str, date], int] = defaultdict(int)

    def reset_daily_counter(self) -> None:
        """Reset the daily trade counter."""
        self._new_trades_by_day.clear()

    def _count_trades_today(self, strategy_id: str, now: datetime) -> int:
        """Count how many trades have been made today for a strategy."""
        key = (strategy_id, now.date())
        return self._new_trades_by_day.get(key, 0)

    def _record_trade_today(self, strategy_id: str, now: datetime) -> None:
        """Record a trade for today."""
        key = (strategy_id, now.date())
        self._new_trades_by_day[key] += 1

    def check_trade(
        self,
        candidate: CandidateTrade,
        open_positions: Sequence[OpenPosition],
    ) -> RiskCheckResult:
        """
        Check if a proposed trade is allowed under risk limits.
        
        Args:
            candidate: The proposed trade
            open_positions: Currently open positions
        
        Returns:
            RiskCheckResult with allowed status and reason if blocked
        """
        if not self.config.enabled:
            return RiskCheckResult(allowed=True, reason=None)

        cfg = self.config
        now = candidate.opened_at

        if candidate.estimated_max_loss_pct > cfg.max_risk_per_trade:
            return RiskCheckResult(
                allowed=False,
                reason=(
                    f"per-trade risk {candidate.estimated_max_loss_pct:.2%} "
                    f"exceeds limit {cfg.max_risk_per_trade:.2%}"
                ),
            )

        total_underlying_risk = sum(
            p.estimated_max_loss_pct
            for p in open_positions
            if p.underlying == candidate.underlying
        ) + candidate.estimated_max_loss_pct

        if total_underlying_risk > cfg.max_risk_per_underlying:
            return RiskCheckResult(
                allowed=False,
                reason=(
                    f"correlated risk on {candidate.underlying} would be "
                    f"{total_underlying_risk:.2%} > limit {cfg.max_risk_per_underlying:.2%}"
                ),
            )

        trades_today = self._count_trades_today(candidate.strategy_id, now)
        if trades_today >= cfg.max_new_trades_per_day:
            return RiskCheckResult(
                allowed=False,
                reason=(
                    f"daily trade cap reached for strategy {candidate.strategy_id} "
                    f"({trades_today}/{cfg.max_new_trades_per_day})"
                ),
            )

        self._record_trade_today(candidate.strategy_id, now)
        return RiskCheckResult(allowed=True, reason=None)

    def would_trade_be_allowed(
        self,
        candidate: CandidateTrade,
        open_positions: Sequence[OpenPosition],
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if a trade would be allowed (without recording it).
        For analysis/logging purposes.
        
        Returns:
            Tuple of (allowed, reason)
        """
        if not self.config.enabled:
            return True, None

        cfg = self.config
        now = candidate.opened_at

        if candidate.estimated_max_loss_pct > cfg.max_risk_per_trade:
            return False, (
                f"per-trade risk {candidate.estimated_max_loss_pct:.2%} "
                f"exceeds limit {cfg.max_risk_per_trade:.2%}"
            )

        total_underlying_risk = sum(
            p.estimated_max_loss_pct
            for p in open_positions
            if p.underlying == candidate.underlying
        ) + candidate.estimated_max_loss_pct

        if total_underlying_risk > cfg.max_risk_per_underlying:
            return False, (
                f"correlated risk on {candidate.underlying} would be "
                f"{total_underlying_risk:.2%} > limit {cfg.max_risk_per_underlying:.2%}"
            )

        trades_today = self._count_trades_today(candidate.strategy_id, now)
        if trades_today >= cfg.max_new_trades_per_day:
            return False, (
                f"daily trade cap reached for strategy {candidate.strategy_id} "
                f"({trades_today}/{cfg.max_new_trades_per_day})"
            )

        return True, None


def maybe_open_trade(
    candidate: CandidateTrade,
    open_positions: Sequence[OpenPosition],
    risk_manager: RiskManager,
    is_training_on_testnet: bool = False,
) -> Tuple[bool, Optional[str]]:
    """
    Centralized function to check if a trade should be opened.
    
    In training mode on testnet: bypasses risk checks for learning.
    In live/paper mode: enforces all risk checks.
    
    Args:
        candidate: The proposed trade
        open_positions: Currently open positions
        risk_manager: RiskManager instance
        is_training_on_testnet: Whether training mode is enabled on testnet
    
    Returns:
        Tuple of (allowed, reason)
    """
    if is_training_on_testnet:
        return True, "training_mode on testnet: risk checks skipped"

    result = risk_manager.check_trade(candidate, open_positions)
    return result.allowed, result.reason
