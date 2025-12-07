"""
State builder module for LIVE agent.
Fetches market data from Deribit API and constructs AgentState via state_core.

This module:
1. Fetches live data from Deribit (spot, portfolio, options)
2. Converts to RawMarketSnapshot format
3. Delegates to state_core.build_agent_state_from_raw() for unified processing
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from src.config import Settings, settings
from src.deribit_client import DeribitClient, DeribitAPIError
from src.market_context import compute_market_context
from src.utils.expiry import parse_deribit_expiry
from src.models import AgentState, MarketContext, OptionType
from src.state_core import (
    RawOption,
    RawPosition,
    RawPortfolio,
    RawMarketSnapshot,
    build_agent_state_from_raw,
)


def _parse_expiry(instrument_name: str) -> datetime:
    """Parse expiry date from instrument name like BTC-27DEC24-100000-C."""
    result = parse_deribit_expiry(instrument_name)
    if result is None:
        raise ValueError(f"Invalid instrument name format: {instrument_name}")
    return result


def _parse_strike(instrument_name: str) -> float:
    """Parse strike price from instrument name."""
    parts = instrument_name.split("-")
    if len(parts) < 3:
        raise ValueError(f"Invalid instrument name format: {instrument_name}")
    return float(parts[2])


def _parse_option_type(instrument_name: str) -> str:
    """Parse option type from instrument name as string."""
    if instrument_name.endswith("-C"):
        return "call"
    elif instrument_name.endswith("-P"):
        return "put"
    raise ValueError(f"Cannot determine option type from: {instrument_name}")


def _approximate_delta_for_ticker(
    spot: float,
    strike: float,
    dte: int,
    option_type: str,
    iv: float = 0.6,
) -> float:
    """
    Approximate option delta using simplified Black-Scholes.
    Used when Deribit doesn't provide delta in ticker response.
    """
    import math
    
    if dte <= 0:
        if option_type == "call":
            return 1.0 if spot > strike else 0.0
        else:
            return -1.0 if spot < strike else 0.0
    
    t = dte / 365.0
    moneyness = math.log(spot / strike) if spot > 0 and strike > 0 else 0
    vol_sqrt_t = iv * math.sqrt(t)
    
    if vol_sqrt_t == 0:
        d1 = 0
    else:
        d1 = (moneyness + 0.5 * iv * iv * t) / vol_sqrt_t
    
    def norm_cdf(x: float) -> float:
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))
    
    if option_type == "call":
        return norm_cdf(d1)
    else:
        return norm_cdf(d1) - 1


def _fetch_spot_prices(
    client: DeribitClient,
    underlyings: list[str],
) -> dict[str, float]:
    """Fetch spot prices for all underlyings."""
    spot_prices: dict[str, float] = {}
    for underlying in underlyings:
        try:
            spot_prices[underlying] = client.get_index_price(underlying)
        except DeribitAPIError as e:
            print(f"Warning: Could not fetch {underlying} spot price: {e}")
            spot_prices[underlying] = 0.0
    return spot_prices


def _fetch_portfolio(
    client: DeribitClient,
    underlyings: list[str],
    spot_prices: dict[str, float],
) -> RawPortfolio:
    """Fetch portfolio data and positions from Deribit."""
    balances: dict[str, float] = {}
    equity_usd = 0.0
    margin_used_usd = 0.0
    margin_available_usd = 0.0
    
    for underlying in underlyings:
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
    
    positions: list[RawPosition] = []
    net_delta = 0.0
    
    for underlying in underlyings:
        try:
            deribit_positions = client.get_positions(underlying, kind="option")
            spot = spot_prices.get(underlying, 0.0)
            
            for pos in deribit_positions:
                if pos.get("size", 0) == 0:
                    continue
                
                instrument_name = pos["instrument_name"]
                
                try:
                    expiry = _parse_expiry(instrument_name)
                    strike = _parse_strike(instrument_name)
                    opt_type = _parse_option_type(instrument_name)
                except ValueError:
                    continue
                
                size = pos.get("size", 0)
                delta = pos.get("delta", 0.0)
                
                raw_pos = RawPosition(
                    instrument_name=instrument_name,
                    underlying=underlying,
                    strike=strike,
                    expiry=expiry,
                    option_type=opt_type,
                    size=size,
                    average_price=pos.get("average_price", 0.0),
                    mark_price=pos.get("mark_price", 0.0),
                    delta=delta,
                    unrealized_pnl_usd=pos.get("floating_profit_loss_usd", 0.0),
                )
                positions.append(raw_pos)
                
                if delta:
                    net_delta += delta
        except DeribitAPIError as e:
            print(f"Warning: Could not fetch {underlying} positions: {e}")
    
    return RawPortfolio(
        equity_usd=equity_usd,
        margin_used_pct=margin_used_pct,
        balances=balances,
        positions=positions,
        margin_used_usd=margin_used_usd,
        margin_available_usd=margin_available_usd,
        net_delta=net_delta,
    )


def _fetch_options(
    client: DeribitClient,
    underlyings: list[str],
    spot_prices: dict[str, float],
    cfg: Settings,
) -> list[RawOption]:
    """Fetch option chain data from Deribit."""
    now = datetime.now(timezone.utc)
    options: list[RawOption] = []
    
    for underlying in underlyings:
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
                
                dte = max(0, (expiry - now).days)
                
                if dte < cfg.effective_dte_min or dte > cfg.effective_dte_max:
                    continue
                
                if strike <= spot:
                    continue
                
                try:
                    ticker = client.get_ticker(instrument_name)
                except DeribitAPIError:
                    continue
                
                bid = ticker.get("best_bid_price", 0.0) or 0.0
                ask = ticker.get("best_ask_price", 0.0) or 0.0
                
                if bid <= 0 or ask <= 0:
                    continue
                
                mark_price = (bid + ask) / 2
                mark_iv = ticker.get("mark_iv", 60.0) or 60.0
                
                delta = _approximate_delta_for_ticker(spot, strike, dte, opt_type, mark_iv / 100)
                
                rv_placeholder = mark_iv * 0.8
                
                raw_opt = RawOption(
                    instrument_name=instrument_name,
                    expiry=expiry,
                    strike=strike,
                    option_type=opt_type,
                    mark_price=mark_price,
                    mark_iv=mark_iv,
                    delta=delta,
                    underlying_price=spot,
                    underlying=underlying,
                    bid=bid,
                    ask=ask,
                    rv=rv_placeholder,
                )
                options.append(raw_opt)
        except DeribitAPIError as e:
            print(f"Warning: Could not fetch {underlying} instruments: {e}")
    
    return options


def build_agent_state(
    client: DeribitClient,
    config: Optional[Settings] = None,
) -> AgentState:
    """
    Build the complete agent state by fetching market data and positions.
    
    This is the main entry point for live agent state construction.
    It fetches data from Deribit, converts to RawMarketSnapshot,
    and delegates to state_core for unified processing.
    
    Args:
        client: Deribit API client
        config: Settings configuration (uses default if None)
    
    Returns:
        AgentState with portfolio, candidates, and market snapshot
    """
    cfg = config or settings
    now = datetime.now(timezone.utc)
    
    spot_prices = _fetch_spot_prices(client, cfg.underlyings)
    portfolio = _fetch_portfolio(client, cfg.underlyings, spot_prices)
    options = _fetch_options(client, cfg.underlyings, spot_prices, cfg)
    
    market_ctx: Optional[MarketContext] = None
    primary_underlying = "BTC" if "BTC" in cfg.underlyings else (cfg.underlyings[0] if cfg.underlyings else None)
    
    if primary_underlying:
        try:
            market_ctx = compute_market_context(client, primary_underlying, now)
        except Exception as e:
            print(f"Warning: Could not compute market context: {e}")
    
    raw_snapshot = RawMarketSnapshot(
        timestamp=now,
        underlyings=cfg.underlyings,
        spot=spot_prices,
        portfolio=portfolio,
        options=options,
        realized_vol={},
        market_context=market_ctx,
    )
    
    return build_agent_state_from_raw(
        raw_snapshot,
        delta_min=cfg.effective_delta_min,
        delta_max=cfg.effective_delta_max,
        dte_min=cfg.effective_dte_min,
        dte_max=cfg.effective_dte_max,
        premium_min_usd=cfg.premium_min_usd,
        max_candidates=5,
        source="live",
    )
