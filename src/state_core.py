"""
Shared state-building core for live and backtest agents.

This module provides unified data structures and logic for constructing
AgentState objects from raw market data, regardless of source (live Deribit
API or historical/synthetic backtest data).

The key principle: only the DATA SOURCES differ between live and backtest;
the state construction RULES are identical.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Literal

from src.models import (
    AgentState,
    CandidateOption,
    MarketContext,
    OptionPosition,
    OptionType,
    PortfolioState,
    Side,
    VolState,
)
from src.metrics.volatility import compute_ivrv_ratio


@dataclass
class RawOption:
    """
    Raw option data from any source (live Deribit or synthetic).
    This is the input format for building CandidateOption objects.
    """
    instrument_name: str
    expiry: datetime
    strike: float
    option_type: str  # "call" or "put"
    mark_price: float
    mark_iv: float
    delta: float
    underlying_price: float
    underlying: str = "BTC"
    bid: Optional[float] = None
    ask: Optional[float] = None
    rv: Optional[float] = None


@dataclass
class RawPosition:
    """
    Raw position data from any source.
    """
    instrument_name: str
    underlying: str
    strike: float
    expiry: datetime
    option_type: str  # "call" or "put"
    size: float  # Negative for short positions
    average_price: float
    mark_price: float
    delta: float
    unrealized_pnl_usd: float


@dataclass
class RawPortfolio:
    """
    Raw portfolio data from any source.
    """
    equity_usd: float
    margin_used_pct: float
    balances: Dict[str, float] = field(default_factory=dict)
    positions: List[RawPosition] = field(default_factory=list)
    margin_used_usd: float = 0.0
    margin_available_usd: float = 0.0
    net_delta: float = 0.0


@dataclass
class RawMarketSnapshot:
    """
    Complete market snapshot from any source.
    This is the unified input for build_agent_state_from_raw().
    """
    timestamp: datetime
    underlyings: List[str]
    spot: Dict[str, float]
    portfolio: RawPortfolio
    options: List[RawOption]
    realized_vol: Dict[str, float] = field(default_factory=dict)
    market_context: Optional[MarketContext] = None


def _calculate_dte(expiry: datetime, reference_time: Optional[datetime] = None) -> int:
    """Calculate days to expiry from reference time (or now)."""
    ref = reference_time or datetime.now(timezone.utc)
    if expiry.tzinfo is None:
        expiry = expiry.replace(tzinfo=timezone.utc)
    if ref.tzinfo is None:
        ref = ref.replace(tzinfo=timezone.utc)
    delta = expiry - ref
    return max(0, delta.days)


def _calculate_dte_float(expiry: datetime, reference_time: Optional[datetime] = None) -> float:
    """Calculate days to expiry as float for more precision."""
    ref = reference_time or datetime.now(timezone.utc)
    if expiry.tzinfo is None:
        expiry = expiry.replace(tzinfo=timezone.utc)
    if ref.tzinfo is None:
        ref = ref.replace(tzinfo=timezone.utc)
    delta = expiry - ref
    return max(0.0, delta.total_seconds() / 86400.0)


def _calculate_moneyness(spot: float, strike: float, option_type: str) -> str:
    """Determine if option is ITM, ATM, or OTM."""
    pct_diff = abs(spot - strike) / spot if spot > 0 else 0.0
    
    if pct_diff < 0.02:
        return "ATM"
    
    if option_type.lower() == "call":
        return "OTM" if strike > spot else "ITM"
    else:
        return "OTM" if strike < spot else "ITM"


def _calculate_otm_pct(spot: float, strike: float, option_type: str) -> float:
    """Calculate percentage out of the money."""
    if spot <= 0:
        return 0.0
    if option_type.lower() == "call":
        if strike > spot:
            return (strike - spot) / spot * 100
        return 0.0
    else:
        if strike < spot:
            return (spot - strike) / spot * 100
        return 0.0


def _option_type_to_enum(option_type: str) -> OptionType:
    """Convert string option type to OptionType enum."""
    if option_type.lower() == "call":
        return OptionType.CALL
    return OptionType.PUT


def build_agent_state_from_raw(
    raw: RawMarketSnapshot,
    *,
    delta_min: float = 0.10,
    delta_max: float = 0.40,
    dte_min: int = 1,
    dte_max: int = 21,
    premium_min_usd: float = 0.0,
    max_candidates: int = 50,
    source: Literal["live", "backtest"] = "live",
) -> AgentState:
    """
    Build an AgentState from raw market data.
    
    This is the unified state construction logic used by both live and backtest.
    Only the data sources differ; the filtering and construction rules are identical.
    
    Args:
        raw: RawMarketSnapshot containing all market data
        delta_min: Minimum delta filter for candidates
        delta_max: Maximum delta filter for candidates
        dte_min: Minimum DTE filter
        dte_max: Maximum DTE filter
        premium_min_usd: Minimum premium in USD
        max_candidates: Maximum number of candidates to return
        source: "live" or "backtest" for logging/debugging
    
    Returns:
        Fully constructed AgentState
    """
    reference_time = raw.timestamp
    
    option_positions: List[OptionPosition] = []
    for pos in raw.portfolio.positions:
        dte = _calculate_dte(pos.expiry, reference_time)
        spot = raw.spot.get(pos.underlying, 0.0)
        
        position = OptionPosition(
            symbol=pos.instrument_name,
            underlying=pos.underlying,
            strike=pos.strike,
            expiry=pos.expiry,
            option_type=_option_type_to_enum(pos.option_type),
            side=Side.SELL if pos.size < 0 else Side.BUY,
            size=abs(pos.size),
            avg_price=pos.average_price,
            mark_price=pos.mark_price,
            unrealized_pnl=pos.unrealized_pnl_usd,
            expiry_dte=dte,
            moneyness=_calculate_moneyness(spot, pos.strike, pos.option_type),
            delta=pos.delta,
        )
        option_positions.append(position)
    
    portfolio = PortfolioState(
        balances=raw.portfolio.balances,
        spot_positions=raw.portfolio.balances.copy(),
        equity_usd=raw.portfolio.equity_usd,
        margin_used_usd=raw.portfolio.margin_used_usd,
        margin_available_usd=raw.portfolio.margin_available_usd,
        margin_used_pct=raw.portfolio.margin_used_pct,
        net_delta=raw.portfolio.net_delta,
        option_positions=option_positions,
    )
    
    candidate_options: List[CandidateOption] = []
    
    for opt in raw.options:
        if opt.option_type.lower() != "call":
            continue
        
        spot = opt.underlying_price
        if spot <= 0:
            continue
        
        if opt.strike <= spot:
            continue
        
        dte = _calculate_dte(opt.expiry, reference_time)
        
        if dte < dte_min or dte > dte_max:
            continue
        
        delta = abs(opt.delta)
        if delta < delta_min or delta > delta_max:
            continue
        
        bid = opt.bid if opt.bid is not None else opt.mark_price * 0.95
        ask = opt.ask if opt.ask is not None else opt.mark_price * 1.05
        
        if bid <= 0 or ask <= 0:
            continue
        
        mid_price = (bid + ask) / 2
        premium_usd = mid_price * spot
        
        if premium_usd < premium_min_usd:
            continue
        
        otm_pct = _calculate_otm_pct(spot, opt.strike, opt.option_type)
        
        rv = opt.rv if opt.rv is not None else raw.realized_vol.get(opt.underlying, opt.mark_iv * 0.8)
        ivrv = compute_ivrv_ratio(opt.mark_iv, rv)
        
        candidate = CandidateOption(
            symbol=opt.instrument_name,
            underlying=opt.underlying,
            strike=opt.strike,
            expiry=opt.expiry,
            option_type=_option_type_to_enum(opt.option_type),
            dte=dte,
            delta=delta,
            otm_pct=otm_pct,
            bid=bid,
            ask=ask,
            mid_price=mid_price,
            premium_usd=premium_usd,
            iv=opt.mark_iv,
            rv=rv,
            ivrv=ivrv,
        )
        candidate_options.append(candidate)
    
    candidate_options.sort(key=lambda x: x.premium_usd, reverse=True)
    candidate_options = candidate_options[:max_candidates]
    
    btc_iv = 0.0
    eth_iv = 0.0
    btc_rv = raw.realized_vol.get("BTC", 0.0)
    eth_rv = raw.realized_vol.get("ETH", 0.0)
    
    for c in candidate_options:
        if c.underlying == "BTC" and btc_iv == 0:
            btc_iv = c.iv
            if btc_rv == 0:
                btc_rv = c.rv
        elif c.underlying == "ETH" and eth_iv == 0:
            eth_iv = c.iv
            if eth_rv == 0:
                eth_rv = c.rv
    
    if btc_rv == 0 and btc_iv > 0:
        btc_rv = btc_iv * 0.8
    if eth_rv == 0 and eth_iv > 0:
        eth_rv = eth_iv * 0.8
    
    vol_state = VolState(
        btc_iv=btc_iv,
        btc_rv=btc_rv,
        btc_ivrv=compute_ivrv_ratio(btc_iv, btc_rv),
        btc_skew=0.0,
        eth_iv=eth_iv,
        eth_rv=eth_rv,
        eth_ivrv=compute_ivrv_ratio(eth_iv, eth_rv),
        eth_skew=0.0,
    )
    
    return AgentState(
        timestamp=raw.timestamp,
        underlyings=raw.underlyings,
        spot=raw.spot,
        portfolio=portfolio,
        vol_state=vol_state,
        candidate_options=candidate_options,
        market_context=raw.market_context,
    )
