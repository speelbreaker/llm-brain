"""
Configuration and result models for covered call simulation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Dict, List, Any

from .data_source import Timeframe


@dataclass
class CallSimulationConfig:
    """
    Configuration for a simple covered-call simulation on a single underlying.
    """
    underlying: str
    start: datetime
    end: datetime
    timeframe: Timeframe
    decision_interval_bars: int

    initial_spot_position: float
    contract_size: float
    fee_rate: float
    risk_free_rate: float = 0.0

    target_dte: int = 7
    dte_tolerance: int = 2
    target_delta: float = 0.25
    delta_tolerance: float = 0.05

    hold_to_expiry: bool = True


@dataclass
class SimulatedTrade:
    """
    Result of simulating a single covered call trade.
    """
    instrument_name: str
    underlying: str
    side: Literal["SHORT_CALL"]
    size: float
    open_time: datetime
    close_time: datetime
    open_price: float
    close_price: float
    pnl: float
    pnl_vs_hodl: float
    max_drawdown_pct: float
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "instrument_name": self.instrument_name,
            "underlying": self.underlying,
            "side": self.side,
            "size": self.size,
            "open_time": self.open_time.isoformat(),
            "close_time": self.close_time.isoformat(),
            "open_price": self.open_price,
            "close_price": self.close_price,
            "pnl": self.pnl,
            "pnl_vs_hodl": self.pnl_vs_hodl,
            "max_drawdown_pct": self.max_drawdown_pct,
            "notes": self.notes,
        }


@dataclass
class SimulationResult:
    """
    Aggregate result of running a simulation across multiple decision times.
    """
    trades: List[SimulatedTrade]
    equity_curve: Dict[datetime, float]
    equity_vs_hodl: Dict[datetime, float]
    metrics: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "trades": [t.to_dict() for t in self.trades],
            "equity_curve": {k.isoformat(): v for k, v in self.equity_curve.items()},
            "equity_vs_hodl": {k.isoformat(): v for k, v in self.equity_vs_hodl.items()},
            "metrics": self.metrics,
        }


@dataclass
class TrainingExample:
    """
    A single (state, action, reward) tuple for ML training.
    """
    decision_time: datetime
    underlying: str
    spot: float
    action: str
    reward: float
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for CSV/JSON serialization."""
        return {
            "decision_time": self.decision_time.isoformat(),
            "underlying": self.underlying,
            "spot": self.spot,
            "action": self.action,
            "reward": self.reward,
            **self.extra,
        }
