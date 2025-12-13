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
PricingMode = Literal["deribit_live", "synthetic_bs"]
SyntheticIVMode = Literal["fixed", "rv_window", "historical_replay"]

# Hybrid synthetic mode enums
SigmaMode = Literal["rv_x_multiplier", "atm_iv_x_multiplier", "mark_iv_x_multiplier"]
ChainMode = Literal["synthetic_grid", "live_chain"]


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
class LiveChainDebugSample:
    """
    Debug sample comparing Deribit mark price vs engine-calculated price.
    Used to verify that Live Chain + Live IV mode correctly preserves exchange marks.
    """
    instrument_name: str
    dte_days: float
    strike: float
    deribit_mark_price: float
    engine_price: float
    abs_diff_pct: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instrument_name": self.instrument_name,
            "dte_days": round(self.dte_days, 2),
            "strike": self.strike,
            "deribit_mark_price": round(self.deribit_mark_price, 6),
            "engine_price": round(self.engine_price, 6),
            "abs_diff_pct": round(self.abs_diff_pct, 4),
        }


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
    
    initial_capital_usd: float = 10000.0
    position_size_underlying: float = 1.0
    
    pricing_mode: PricingMode = "synthetic_bs"
    synthetic_iv_mode: SyntheticIVMode = "fixed"
    synthetic_fixed_iv: float = 0.70
    synthetic_rv_window_days: int = 30
    synthetic_iv_multiplier: float = 1.0
    
    # Hybrid synthetic mode settings
    sigma_mode: SigmaMode = "rv_x_multiplier"
    chain_mode: ChainMode = "synthetic_grid"


RollTrigger = Literal["tp_roll", "defensive_roll", "expiry", "none"]


@dataclass
class EquityPoint:
    """A single point in the equity curve."""
    time: datetime
    equity: float
    hodl_equity: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "time": self.time.isoformat(),
            "equity": self.equity,
            "hodl_equity": self.hodl_equity,
        }


@dataclass
class ChainLeg:
    """A single leg within a multi-roll call chain."""
    index: int
    instrument_name: str
    open_time: datetime
    close_time: datetime
    strike: float
    dte_open: float
    open_price: float
    close_price: float
    pnl: float
    trigger: RollTrigger


@dataclass
class ChainData:
    """Multi-leg chain data attached to a SimulatedTrade."""
    decision_time: datetime
    underlying: str
    total_pnl: float
    max_drawdown_pct: float
    legs: List[ChainLeg] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision_time": self.decision_time.isoformat(),
            "underlying": self.underlying,
            "total_pnl": self.total_pnl,
            "max_drawdown_pct": self.max_drawdown_pct,
            "legs": [
                {
                    "index": leg.index,
                    "instrument_name": leg.instrument_name,
                    "open_time": leg.open_time.isoformat(),
                    "close_time": leg.close_time.isoformat(),
                    "strike": leg.strike,
                    "dte_open": leg.dte_open,
                    "open_price": leg.open_price,
                    "close_price": leg.close_price,
                    "pnl": leg.pnl,
                    "trigger": leg.trigger,
                }
                for leg in self.legs
            ],
        }


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
    chain: Optional[ChainData] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
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
        if self.chain:
            result["chain"] = self.chain.to_dict()
        return result


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
    
    The strategy field indicates which training profile was used:
    - "conservative", "moderate", "aggressive" for profile-based selection
    - "fallback" when no profile matched
    - "ladder_N" for ladder mode positions
    """
    decision_time: datetime
    underlying: str
    spot: float
    action: str
    reward: float
    strategy: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for CSV/JSON serialization."""
        return {
            "decision_time": self.decision_time.isoformat(),
            "underlying": self.underlying,
            "spot": self.spot,
            "action": self.action,
            "reward": self.reward,
            "strategy": self.strategy,
            **self.extra,
        }


@dataclass
class CandidateLevelExample:
    """
    A single candidate-level training example for ML policy learning.
    
    One row per candidate per decision time step.
    For trade steps: exactly one candidate has chosen=1, action="SELL_CALL"
    For no-trade steps: all candidates have chosen=0, action="SKIP"
    
    The strategy field indicates which training profile was used for chosen candidates.
    """
    decision_time: datetime
    underlying: str
    spot: float
    
    instrument: str
    strike: float
    dte: float
    delta: float
    score: float
    iv: Optional[float] = None
    ivrv_ratio: Optional[float] = None
    
    exit_style: str = "hold_to_expiry"
    trade_executed: bool = False
    chosen: bool = False
    action: str = "SKIP"
    strategy: str = ""
    
    reward: float = 0.0
    pnl_vs_hodl: float = 0.0
    max_drawdown_pct: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for CSV/JSON serialization."""
        return {
            "decision_time": self.decision_time.isoformat(),
            "underlying": self.underlying,
            "spot": self.spot,
            "instrument": self.instrument,
            "strike": self.strike,
            "dte": self.dte,
            "delta": self.delta,
            "score": self.score,
            "iv": self.iv,
            "ivrv_ratio": self.ivrv_ratio,
            "exit_style": self.exit_style,
            "trade_executed": int(self.trade_executed),
            "chosen": int(self.chosen),
            "action": self.action,
            "strategy": self.strategy,
            "reward": self.reward,
            "pnl_vs_hodl": self.pnl_vs_hodl,
            "max_drawdown_pct": self.max_drawdown_pct,
        }


@dataclass
class DecisionStepData:
    """
    Per-step data collected during backtest for candidate-level export.
    
    Stores all candidates evaluated at a decision time along with
    which one (if any) was chosen for each exit style.
    """
    decision_time: datetime
    underlying: str
    spot: float
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    chosen_hold_to_expiry: Optional[str] = None
    chosen_tp_and_roll: Optional[str] = None
    trade_result_hold: Optional[Dict[str, float]] = None
    trade_result_tp: Optional[Dict[str, float]] = None
