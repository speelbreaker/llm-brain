"""
State builder module.
Fetches market data and positions to construct the AgentState.

# TODO: duplicate logic with src/backtest/state_builder.py
# Both parse expiry dates and build state snapshots. Consider:
# 1. Shared expiry parsing utility
# 2. Unified state builder interface for live vs historical
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Optional

from src.config import Settings, settings
from src.deribit_client import DeribitClient, DeribitAPIError
from src.market_context import compute_market_context
from src.models import (
    AgentState,
    CandidateOption,
    MarketContext,
    OptionInstrument,
    OptionPosition,
    OptionType,
    PortfolioState,
    Side,
    VolState,
)


def _parse_expiry(instrument_name: str) -> datetime:
    """Parse expiry date from instrument name like BTC-27DEC24-100000-C."""
    parts = instrument_name.split("-")
    if len(parts) < 3:
        raise ValueError(f"Invalid instrument name format: {instrument_name}")
    
    date_str = parts[1]
    day = int(date_str[:2])
    month_str = date_str[2:5].upper()
    year_str = date_str[5:]
    
    months = {
        "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
        "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
    }
    
    month = months.get(month_str, 1)
    year = 2000 + int(year_str) if len(year_str) == 2 else int(year_str)
    
    return datetime(year, month, day, 8, 0, 0, tzinfo=timezone.utc)


def _parse_strike(instrument_name: str) -> float:
    """Parse strike price from instrument name."""
    parts = instrument_name.split("-")
    if len(parts) < 3:
        raise ValueError(f"Invalid instrument name format: {instrument_name}")
    return float(parts[2])


def _parse_option_type(instrument_name: str) -> OptionType:
    """Parse option type from instrument name."""
    if instrument_name.endswith("-C"):
        return OptionType.CALL
    elif instrument_name.endswith("-P"):
        return OptionType.PUT
    raise ValueError(f"Cannot determine option type from: {instrument_name}")


def _calculate_dte(expiry: datetime) -> int:
    """Calculate days to expiry."""
    now = datetime.now(timezone.utc)
    delta = expiry - now
    return max(0, delta.days)


def _approximate_delta(
    spot: float,
    strike: float,
    dte: int,
    option_type: OptionType,
    iv: float = 0.6,
) -> float:
    """
    Approximate option delta using simplified Black-Scholes.
    This is a placeholder - should be replaced with proper Greeks calculation.
    """
    if dte <= 0:
        if option_type == OptionType.CALL:
            return 1.0 if spot > strike else 0.0
        else:
            return -1.0 if spot < strike else 0.0
    
    t = dte / 365.0
    moneyness = math.log(spot / strike)
    vol_sqrt_t = iv * math.sqrt(t)
    
    if vol_sqrt_t == 0:
        d1 = 0
    else:
        d1 = (moneyness + 0.5 * iv * iv * t) / vol_sqrt_t
    
    def norm_cdf(x: float) -> float:
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))
    
    if option_type == OptionType.CALL:
        return norm_cdf(d1)
    else:
        return norm_cdf(d1) - 1


def _calculate_moneyness(spot: float, strike: float, option_type: OptionType) -> str:
    """Determine if option is ITM, ATM, or OTM."""
    pct_diff = abs(spot - strike) / spot
    
    if pct_diff < 0.02:
        return "ATM"
    
    if option_type == OptionType.CALL:
        return "OTM" if strike > spot else "ITM"
    else:
        return "OTM" if strike < spot else "ITM"


def _calculate_otm_pct(spot: float, strike: float, option_type: OptionType) -> float:
    """Calculate percentage out of the money."""
    if option_type == OptionType.CALL:
        if strike > spot:
            return (strike - spot) / spot * 100
        return 0.0
    else:
        if strike < spot:
            return (spot - strike) / spot * 100
        return 0.0


def build_agent_state(
    client: DeribitClient,
    config: Optional[Settings] = None,
) -> AgentState:
    """
    Build the complete agent state by fetching market data and positions.
    
    Args:
        client: Deribit API client
        config: Settings configuration (uses default if None)
    
    Returns:
        AgentState with portfolio, candidates, and market snapshot
    """
    cfg = config or settings
    now = datetime.now(timezone.utc)
    
    spot_prices: dict[str, float] = {}
    for underlying in cfg.underlyings:
        try:
            spot_prices[underlying] = client.get_index_price(underlying)
        except DeribitAPIError as e:
            print(f"Warning: Could not fetch {underlying} spot price: {e}")
            spot_prices[underlying] = 0.0
    
    balances: dict[str, float] = {}
    equity_usd = 0.0
    margin_used_usd = 0.0
    margin_available_usd = 0.0
    
    for underlying in cfg.underlyings:
        try:
            summary = client.get_account_summary(underlying)
            balances[underlying] = summary.get("equity", 0.0)
            
            currency_equity = summary.get("equity", 0.0) * spot_prices.get(underlying, 0.0)
            equity_usd += currency_equity
            
            margin_used_usd += summary.get("initial_margin", 0.0) * spot_prices.get(underlying, 0.0)
            margin_available_usd += summary.get("available_funds", 0.0) * spot_prices.get(underlying, 0.0)
        except DeribitAPIError as e:
            print(f"Warning: Could not fetch {underlying} account summary: {e}")
    
    margin_used_pct = 0.0
    if equity_usd > 0:
        margin_used_pct = (margin_used_usd / equity_usd) * 100
    
    option_positions: list[OptionPosition] = []
    net_delta = 0.0
    
    for underlying in cfg.underlyings:
        try:
            positions = client.get_positions(underlying, kind="option")
            spot = spot_prices.get(underlying, 0.0)
            
            for pos in positions:
                if pos.get("size", 0) == 0:
                    continue
                
                instrument_name = pos["instrument_name"]
                
                try:
                    expiry = _parse_expiry(instrument_name)
                    strike = _parse_strike(instrument_name)
                    opt_type = _parse_option_type(instrument_name)
                except ValueError:
                    continue
                
                dte = _calculate_dte(expiry)
                size = pos.get("size", 0)
                delta = pos.get("delta", 0.0)
                
                position = OptionPosition(
                    symbol=instrument_name,
                    underlying=underlying,
                    strike=strike,
                    expiry=expiry,
                    option_type=opt_type,
                    side=Side.SELL if size < 0 else Side.BUY,
                    size=abs(size),
                    avg_price=pos.get("average_price", 0.0),
                    mark_price=pos.get("mark_price", 0.0),
                    unrealized_pnl=pos.get("floating_profit_loss_usd", 0.0),
                    expiry_dte=dte,
                    moneyness=_calculate_moneyness(spot, strike, opt_type),
                    delta=delta,
                )
                option_positions.append(position)
                
                if delta:
                    net_delta += delta
        except DeribitAPIError as e:
            print(f"Warning: Could not fetch {underlying} positions: {e}")
    
    portfolio = PortfolioState(
        balances=balances,
        spot_positions=balances.copy(),
        equity_usd=equity_usd,
        margin_used_usd=margin_used_usd,
        margin_available_usd=margin_available_usd,
        margin_used_pct=margin_used_pct,
        net_delta=net_delta,
        option_positions=option_positions,
    )
    
    candidate_options: list[CandidateOption] = []
    
    for underlying in cfg.underlyings:
        try:
            instruments = client.get_instruments(underlying, kind="option")
            spot = spot_prices.get(underlying, 0.0)
            
            if spot <= 0:
                continue
            
            for inst in instruments:
                instrument_name = inst["instrument_name"]
                
                if not instrument_name.endswith("-C"):
                    continue
                
                try:
                    expiry = _parse_expiry(instrument_name)
                    strike = _parse_strike(instrument_name)
                    opt_type = _parse_option_type(instrument_name)
                except ValueError:
                    continue
                
                dte = _calculate_dte(expiry)
                
                if dte < cfg.effective_dte_min or dte > cfg.effective_dte_max:
                    continue
                
                if strike <= spot:
                    continue
                
                otm_pct = _calculate_otm_pct(spot, strike, opt_type)
                
                try:
                    ticker = client.get_ticker(instrument_name)
                except DeribitAPIError:
                    continue
                
                bid = ticker.get("best_bid_price", 0.0) or 0.0
                ask = ticker.get("best_ask_price", 0.0) or 0.0
                
                if bid <= 0 or ask <= 0:
                    continue
                
                mid_price = (bid + ask) / 2
                mark_iv = ticker.get("mark_iv", 60.0) or 60.0
                
                premium_usd = mid_price * spot
                
                if premium_usd < cfg.premium_min_usd:
                    continue
                
                delta = _approximate_delta(spot, strike, dte, opt_type, mark_iv / 100)
                
                if delta < cfg.effective_delta_min or delta > cfg.effective_delta_max:
                    continue
                
                rv_placeholder = mark_iv * 0.8
                ivrv = mark_iv / rv_placeholder if rv_placeholder > 0 else 1.0
                
                candidate = CandidateOption(
                    symbol=instrument_name,
                    underlying=underlying,
                    strike=strike,
                    expiry=expiry,
                    option_type=opt_type,
                    dte=dte,
                    delta=delta,
                    otm_pct=otm_pct,
                    bid=bid,
                    ask=ask,
                    mid_price=mid_price,
                    premium_usd=premium_usd,
                    iv=mark_iv,
                    rv=rv_placeholder,
                    ivrv=ivrv,
                )
                candidate_options.append(candidate)
        except DeribitAPIError as e:
            print(f"Warning: Could not fetch {underlying} instruments: {e}")
    
    candidate_options.sort(key=lambda x: x.premium_usd, reverse=True)
    candidate_options = candidate_options[:5]
    
    btc_iv = 0.0
    eth_iv = 0.0
    
    for c in candidate_options:
        if c.underlying == "BTC" and btc_iv == 0:
            btc_iv = c.iv
        elif c.underlying == "ETH" and eth_iv == 0:
            eth_iv = c.iv
    
    vol_state = VolState(
        btc_iv=btc_iv,
        btc_rv=btc_iv * 0.8 if btc_iv > 0 else 0.0,
        btc_ivrv=1.25 if btc_iv > 0 else 1.0,
        btc_skew=0.0,
        eth_iv=eth_iv,
        eth_rv=eth_iv * 0.8 if eth_iv > 0 else 0.0,
        eth_ivrv=1.25 if eth_iv > 0 else 1.0,
        eth_skew=0.0,
    )
    
    market_ctx: Optional[MarketContext] = None
    primary_underlying = "BTC" if "BTC" in cfg.underlyings else (cfg.underlyings[0] if cfg.underlyings else None)
    
    if primary_underlying:
        try:
            market_ctx = compute_market_context(client, primary_underlying, now)
        except Exception as e:
            print(f"Warning: Could not compute market context: {e}")
    
    return AgentState(
        timestamp=now,
        underlyings=cfg.underlyings,
        spot=spot_prices,
        portfolio=portfolio,
        vol_state=vol_state,
        candidate_options=candidate_options,
        market_context=market_ctx,
    )
