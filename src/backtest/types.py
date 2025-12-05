"""
Configuration and result models for covered call simulation.

NOTE: When option_margin_type="linear" and option_settlement_ccy="USDC" (defaults),
all prices (spot, option mark_price) and PnL are in USD/USDC.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Dict, List, Any, Optional

from .data_source import Timeframe

ExitStyle = Literal["hold_to_expiry", "tp_and_roll"]


@dataclass
class OptionSnapshot:
    """
    Generic point-in-time snapshot of an option contract.
    Source-agnostic: can be populated from Deribit, CSV, Tardis, etc.
    """
    instrument_name: str
    underlying: str
    kind: Literal["call", "put"]
    strike: float
    expiry: datetime
    delta: Optional[float] = None
    iv: Optional[float] = None
    mark_price: Optional[float] = None
    settlement_ccy: str = "USDC"
    margin_type: Literal["linear", "inverse"] = "linear"


@dataclass
class CallSimulationConfig:
    """
    Configuration for a simple covered-call simulation on a single underlying.
    
    When option_margin_type="linear" and option_settlement_ccy="USDC",
    all amounts (PnL, equity) are in USD/USDC.
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
    
    min_dte: int = 1
    max_dte: int = 21
    delta_min: float = 0.10
    delta_max: float = 0.40

    hold_to_expiry: bool = True
    
    option_margin_type: Literal["linear", "inverse"] = "linear"
    option_settlement_ccy: str = "USDC"
    
    tp_threshold_pct: float = 80.0
    min_dte_to_roll: int = 2
    defend_near_strike_pct: float = 0.98
    max_rolls_per_chain: int = 3
    min_score_to_trade: float = 3.0


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
