"""
Pydantic models for the options trading agent.
Defines data structures for instruments, positions, portfolio state, and agent state.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class MarketContext(BaseModel):
    """
    Compact market/chart context for LLM decision-making.
    Includes trend/regime, recent returns, and realized volatility.
    """
    underlying: str = Field(..., description="Underlying asset: BTC or ETH")
    time: datetime = Field(..., description="Timestamp of this snapshot")

    regime: Literal["bull", "sideways", "bear"] = Field(
        default="sideways",
        description="Market regime classification",
    )
    pct_from_50d_ma: float = Field(
        default=0.0,
        description="Price distance from 50-day MA as percentage",
    )
    pct_from_200d_ma: float = Field(
        default=0.0,
        description="Price distance from 200-day MA as percentage",
    )

    return_1d_pct: float = Field(default=0.0, description="1-day return percentage")
    return_7d_pct: float = Field(default=0.0, description="7-day return percentage")
    return_30d_pct: float = Field(default=0.0, description="30-day return percentage")

    realized_vol_7d: float = Field(
        default=0.0,
        description="7-day realized volatility (annualized)",
    )
    realized_vol_30d: float = Field(
        default=0.0,
        description="30-day realized volatility (annualized)",
    )

    support_level: Optional[float] = Field(
        default=None,
        description="Estimated support level (optional)",
    )
    resistance_level: Optional[float] = Field(
        default=None,
        description="Estimated resistance level (optional)",
    )
    distance_to_support_pct: Optional[float] = Field(
        default=None,
        description="Distance to support as percentage (optional)",
    )
    distance_to_resistance_pct: Optional[float] = Field(
        default=None,
        description="Distance to resistance as percentage (optional)",
    )


class OptionType(str, Enum):
    """Option type: call or put."""
    CALL = "call"
    PUT = "put"


class Side(str, Enum):
    """Position or order side."""
    BUY = "buy"
    SELL = "sell"


class ActionType(str, Enum):
    """Possible agent actions."""
    DO_NOTHING = "DO_NOTHING"
    OPEN_COVERED_CALL = "OPEN_COVERED_CALL"
    ROLL_COVERED_CALL = "ROLL_COVERED_CALL"
    CLOSE_COVERED_CALL = "CLOSE_COVERED_CALL"


class OptionInstrument(BaseModel):
    """Represents a tradeable option instrument."""
    symbol: str = Field(..., description="Deribit instrument name, e.g. BTC-27DEC24-100000-C")
    underlying: str = Field(..., description="Underlying asset: BTC or ETH")
    strike: float = Field(..., description="Strike price in USD")
    expiry: datetime = Field(..., description="Expiration datetime")
    option_type: OptionType = Field(..., description="Call or Put")
    
    tick_size: float = Field(default=0.0001, description="Minimum price increment")
    min_trade_amount: float = Field(default=0.1, description="Minimum trade size")
    
    bid: Optional[float] = Field(default=None, description="Current best bid price")
    ask: Optional[float] = Field(default=None, description="Current best ask price")
    mark_price: Optional[float] = Field(default=None, description="Mark price")
    mark_iv: Optional[float] = Field(default=None, description="Mark implied volatility")


class OptionPosition(BaseModel):
    """Represents an open option position."""
    symbol: str = Field(..., description="Instrument name")
    underlying: str = Field(..., description="Underlying asset")
    strike: float = Field(..., description="Strike price")
    expiry: datetime = Field(..., description="Expiration datetime")
    option_type: OptionType = Field(..., description="Call or Put")
    
    side: Side = Field(..., description="Position side (buy = long, sell = short)")
    size: float = Field(..., description="Position size (positive for long, can be negative for short)")
    avg_price: float = Field(..., description="Average entry price")
    
    mark_price: Optional[float] = Field(default=None, description="Current mark price")
    unrealized_pnl: Optional[float] = Field(default=None, description="Unrealized PnL in USD")
    
    expiry_dte: Optional[int] = Field(default=None, description="Days to expiry")
    moneyness: Optional[str] = Field(default=None, description="ITM, ATM, or OTM")
    delta: Optional[float] = Field(default=None, description="Position delta")


class PortfolioState(BaseModel):
    """Represents the current portfolio state."""
    balances: dict[str, float] = Field(
        default_factory=dict,
        description="Balances by currency, e.g. {'BTC': 1.5, 'ETH': 10.0}",
    )
    spot_positions: dict[str, float] = Field(
        default_factory=dict,
        description="Spot holdings by asset for covered call verification, e.g. {'BTC': 0.3, 'ETH': 5.0}",
    )
    equity_usd: float = Field(default=0.0, description="Total equity in USD")
    margin_used_usd: float = Field(default=0.0, description="Used margin in USD")
    margin_available_usd: float = Field(default=0.0, description="Available margin in USD")
    margin_used_pct: float = Field(default=0.0, description="Margin utilization percentage")
    
    net_delta: float = Field(default=0.0, description="Net portfolio delta")
    
    option_positions: list[OptionPosition] = Field(
        default_factory=list,
        description="List of open option positions",
    )


class CandidateOption(BaseModel):
    """A candidate option for potential trading."""
    symbol: str = Field(..., description="Instrument name")
    underlying: str = Field(..., description="Underlying asset")
    strike: float = Field(..., description="Strike price")
    expiry: datetime = Field(..., description="Expiration datetime")
    option_type: OptionType = Field(..., description="Call or Put")
    
    dte: int = Field(..., description="Days to expiry")
    delta: float = Field(..., description="Option delta (approximated)")
    otm_pct: float = Field(..., description="Percentage out of the money")
    
    bid: float = Field(..., description="Best bid price")
    ask: float = Field(..., description="Best ask price")
    mid_price: float = Field(..., description="Mid price")
    premium_usd: float = Field(..., description="Premium in USD")
    
    iv: float = Field(default=0.0, description="Implied volatility")
    rv: float = Field(default=0.0, description="Realized volatility (placeholder)")
    ivrv: float = Field(default=1.0, description="IV/RV ratio")
    
    # Liquidity metrics
    spread_pct: Optional[float] = Field(
        default=None,
        description="Bid/ask spread as percentage of mid price",
    )
    open_interest: Optional[int] = Field(
        default=None,
        description="Open interest (contracts) for this strike",
    )


class VolState(BaseModel):
    """Volatility state snapshot."""
    btc_iv: float = Field(default=0.0, description="BTC implied volatility")
    btc_rv: float = Field(default=0.0, description="BTC realized volatility")
    btc_ivrv: float = Field(default=1.0, description="BTC IV/RV ratio")
    btc_skew: float = Field(default=0.0, description="BTC volatility skew (placeholder)")
    
    eth_iv: float = Field(default=0.0, description="ETH implied volatility")
    eth_rv: float = Field(default=0.0, description="ETH realized volatility")
    eth_ivrv: float = Field(default=1.0, description="ETH IV/RV ratio")
    eth_skew: float = Field(default=0.0, description="ETH volatility skew (placeholder)")


class AgentState(BaseModel):
    """Complete state for agent decision-making."""
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="State timestamp")
    
    underlyings: list[str] = Field(default_factory=list, description="List of underlyings")
    spot: dict[str, float] = Field(
        default_factory=dict,
        description="Spot prices by underlying, e.g. {'BTC': 100000, 'ETH': 3500}",
    )
    
    portfolio: PortfolioState = Field(
        default_factory=PortfolioState,
        description="Current portfolio state",
    )
    
    vol_state: VolState = Field(
        default_factory=VolState,
        description="Volatility state snapshot",
    )
    
    candidate_options: list[CandidateOption] = Field(
        default_factory=list,
        description="Filtered list of candidate options for trading",
    )
    
    market_context: Optional[MarketContext] = Field(
        default=None,
        description="Market/chart context for trend-aware decisions",
    )


class ProposedAction(BaseModel):
    """Represents a proposed action from the policy."""
    action: ActionType = Field(..., description="Action type")
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Action parameters (symbol, size, from_symbol, to_symbol, etc.)",
    )
    reasoning: str = Field(default="", description="Explanation for the action")
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "action": self.action.value,
            "params": self.params,
            "reasoning": self.reasoning,
        }
