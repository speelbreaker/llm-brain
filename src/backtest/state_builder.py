"""
Historical state builder for backtests.
Constructs state dicts that mirror live AgentState for scoring and simulation.

This module:
1. Fetches historical/synthetic data from DeribitDataSource
2. Uses shared filtering logic from state_core
3. Returns Dict format for backward compatibility with backtest simulator
4. Also provides build_historical_agent_state() for typed AgentState output

The key principle: both live and backtest builders use the same state_core
logic for constructing AgentState, ensuring no drift between simulation
and live trading.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from dataclasses import replace
from typing import Dict, Any, List, Optional

from .deribit_data_source import DeribitDataSource
from .market_context_backtest import compute_market_context_from_ds, market_context_to_dict
from .types import OptionSnapshot, CallSimulationConfig
from .pricing import bs_call_delta, get_synthetic_iv, get_sigma_for_option, bs_call_price
from src.utils.expiry import parse_deribit_expiry
from src.models import AgentState, MarketContext
from src.state_core import (
    RawOption,
    RawPortfolio,
    RawMarketSnapshot,
    build_agent_state_from_raw,
    _calculate_dte,
    _calculate_dte_float,
    _calculate_moneyness,
    _calculate_otm_pct,
)

logger = logging.getLogger(__name__)


def _generate_synthetic_candidates(
    spot: float,
    t: datetime,
    cfg: CallSimulationConfig,
    sigma: float,
) -> List[OptionSnapshot]:
    """
    Generate synthetic call option candidates for backtest.
    
    Creates options at various strikes around spot with synthetic expiry
    set to target_dte days from the decision time t.
    
    Args:
        spot: Current spot price
        t: Decision time
        cfg: Simulation configuration
        sigma: Implied volatility to use for delta computation
    
    Returns:
        List of synthetic OptionSnapshot candidates filtered by delta range
    """
    candidates: List[OptionSnapshot] = []
    
    synthetic_expiry = t + timedelta(days=cfg.target_dte)
    t_years = cfg.target_dte / 365.0
    
    if cfg.underlying == "BTC":
        if spot >= 50000:
            step = 1000
        elif spot >= 10000:
            step = 500
        else:
            step = 250
    else:
        if spot >= 2000:
            step = 50
        elif spot >= 500:
            step = 25
        else:
            step = 10
    
    base_strike = round(spot / step) * step
    num_strikes_above = 20
    num_strikes_below = 5
    
    for i in range(-num_strikes_below, num_strikes_above + 1):
        strike = base_strike + i * step
        if strike <= 0:
            continue
        
        delta = bs_call_delta(spot, strike, t_years, sigma, cfg.risk_free_rate)
        
        if delta < cfg.delta_min or delta > cfg.delta_max:
            continue
        
        expiry_str = synthetic_expiry.strftime("%d%b%y").upper()
        instrument_name = f"{cfg.underlying}-{expiry_str}-{int(strike)}-C"
        
        candidate = OptionSnapshot(
            instrument_name=instrument_name,
            underlying=cfg.underlying,
            kind="call",
            strike=strike,
            expiry=synthetic_expiry,
            delta=delta,
            iv=sigma,
            mark_price=None,
            settlement_ccy=cfg.option_settlement_ccy,
            margin_type=cfg.option_margin_type,
        )
        candidates.append(candidate)
    
    return candidates


def _generate_live_chain_candidates(
    ds: DeribitDataSource,
    spot: float,
    t: datetime,
    cfg: CallSimulationConfig,
    spot_history: List[tuple],
) -> List[OptionSnapshot]:
    """
    Generate option candidates from live Deribit chain.
    
    Fetches real expiries and strikes from Deribit, filters by DTE/delta/margin,
    and uses the configured sigma_mode to set IV for each option.
    
    Args:
        ds: DeribitDataSource instance
        spot: Current spot price
        t: Decision time
        cfg: Simulation configuration with chain_mode and sigma_mode
        spot_history: List of (datetime, price) tuples for RV calculation
        
    Returns:
        List of OptionSnapshot candidates from live chain, filtered and priced
    """
    all_options: List[OptionSnapshot] = ds.list_option_chain(
        underlying=cfg.underlying,
        as_of=t,
        settlement_ccy=cfg.option_settlement_ccy,
        margin_type=cfg.option_margin_type,
    )
    
    if not all_options:
        logger.warning(f"No live options available for {cfg.underlying} at {t}")
        return []
    
    # Filter by DTE and delta
    filtered = _filter_option_chain(
        all_options=all_options,
        t=t,
        min_dte=cfg.min_dte,
        max_dte=cfg.max_dte,
        delta_min=cfg.delta_min,
        delta_max=cfg.delta_max,
    )
    
    # For mark_iv_x_multiplier mode, use each option's own mark_iv
    # For other modes, we may want to recalculate sigma
    sigma_mode = getattr(cfg, 'sigma_mode', 'rv_x_multiplier')
    
    if sigma_mode == "mark_iv_x_multiplier":
        # Use each option's live mark_iv directly, scaled by multiplier
        result = []
        multiplier = cfg.synthetic_iv_multiplier
        
        for opt in filtered:
            if opt.iv is not None and opt.iv > 0:
                # If multiplier is 1.0 (or very close), preserve Deribit's actual data
                # This ensures "Live Chain + Live IV" mode uses real exchange marks
                if abs(multiplier - 1.0) < 1e-6:
                    # Preserve original Deribit mark values - no recalculation
                    result.append(opt)
                else:
                    # Only recalculate when applying a stress multiplier
                    scaled_iv = opt.iv * multiplier
                    dte = (opt.expiry - t).total_seconds() / (24 * 3600)
                    t_years = max(dte / 365.0, 1e-6)
                    mark_price = bs_call_price(spot, opt.strike, t_years, scaled_iv, cfg.risk_free_rate)
                    delta = bs_call_delta(spot, opt.strike, t_years, scaled_iv, cfg.risk_free_rate)
                    
                    result.append(OptionSnapshot(
                        instrument_name=opt.instrument_name,
                        underlying=opt.underlying,
                        kind=opt.kind,
                        strike=opt.strike,
                        expiry=opt.expiry,
                        delta=delta,
                        iv=scaled_iv,
                        mark_price=mark_price,
                        settlement_ccy=opt.settlement_ccy,
                        margin_type=opt.margin_type,
                    ))
            else:
                # No valid IV - pass through as-is (will fall back to RV in scoring)
                result.append(opt)
        return result
    else:
        # For atm_iv or rv modes, get sigma and recalculate
        sigma = get_sigma_for_option(
            config=cfg,
            spot_history=spot_history,
            as_of=t,
            option_chain=all_options,
            option_mark_iv=None,
            abs_delta=0.25,  # Use target delta for base sigma
            regime_state=None,
        )
        
        result = []
        for opt in filtered:
            abs_delta = abs(opt.delta) if opt.delta else 0.25
            opt_sigma = get_sigma_for_option(
                config=cfg,
                spot_history=spot_history,
                as_of=t,
                option_chain=all_options,
                option_mark_iv=opt.iv,
                abs_delta=abs_delta,
                regime_state=None,
            )
            
            dte = (opt.expiry - t).total_seconds() / (24 * 3600)
            t_years = max(dte / 365.0, 1e-6)
            mark_price = bs_call_price(spot, opt.strike, t_years, opt_sigma, cfg.risk_free_rate)
            delta = bs_call_delta(spot, opt.strike, t_years, opt_sigma, cfg.risk_free_rate)
            
            result.append(OptionSnapshot(
                instrument_name=opt.instrument_name,
                underlying=opt.underlying,
                kind=opt.kind,
                strike=opt.strike,
                expiry=opt.expiry,
                delta=delta,
                iv=opt_sigma,
                mark_price=mark_price,
                settlement_ccy=opt.settlement_ccy,
                margin_type=opt.margin_type,
            ))
        return result


def _filter_option_chain(
    all_options: List[OptionSnapshot],
    t: datetime,
    min_dte: int,
    max_dte: int,
    delta_min: float,
    delta_max: float,
) -> List[OptionSnapshot]:
    """
    Filter option chain using shared logic from state_core.
    
    Args:
        all_options: Full option chain
        t: Reference time for DTE calculation
        min_dte: Minimum DTE filter
        max_dte: Maximum DTE filter
        delta_min: Minimum delta filter
        delta_max: Maximum delta filter
    
    Returns:
        Filtered list of OptionSnapshot
    """
    candidates: List[OptionSnapshot] = []
    
    for opt in all_options:
        if opt.kind != "call":
            continue
        
        expiry = getattr(opt, "expiry", None)
        if expiry is None:
            instrument = getattr(opt, "instrument_name", None) or getattr(opt, "symbol", "")
            expiry = parse_deribit_expiry(str(instrument))
        
        if expiry is None:
            continue
        
        dte = _calculate_dte_float(expiry, t)
        if dte < min_dte or dte > max_dte:
            continue
        
        if opt.delta is None:
            continue
        delta_abs = abs(float(opt.delta))
        if delta_abs < delta_min or delta_abs > delta_max:
            continue
        
        if getattr(opt, "expiry", None) is None:
            opt_with_expiry = replace(opt, expiry=expiry)
            candidates.append(opt_with_expiry)
        else:
            candidates.append(opt)
    
    return candidates


def build_historical_state(
    ds: DeribitDataSource,
    cfg: CallSimulationConfig,
    t: datetime,
) -> Dict[str, Any]:
    """
    Build a historical state dict at time t for simulate_policy.
    
    Returns a dict with:
      {
        "time": t,
        "spot": <float>,
        "market_context": { ... },
        "candidate_options": [OptionSnapshot, ...],
        "portfolio": { ... optional ... }
      }
      
    Args:
        ds: DeribitDataSource instance
        cfg: CallSimulationConfig with target parameters
        t: Decision time for state construction
        
    Returns:
        State dict suitable for scoring and policy evaluation
    """
    underlying = cfg.underlying

    lookback_hours = 24
    spot_lookback = t - timedelta(hours=lookback_hours)
    spot_df = ds.get_spot_ohlc(
        underlying=underlying,
        start=spot_lookback,
        end=t,
        timeframe=cfg.timeframe,
    )
    if spot_df.empty:
        spot = None
    else:
        spot = float(spot_df["close"].iloc[-1])

    mc_obj = compute_market_context_from_ds(ds, underlying=underlying, as_of=t)
    mc_dict = market_context_to_dict(mc_obj)

    candidates: List[OptionSnapshot] = []
    
    # Get spot history for RV calculation (needed for sigma modes)
    rv_lookback = t - timedelta(days=cfg.synthetic_rv_window_days + 7)
    rv_df = ds.get_spot_ohlc(
        underlying=underlying,
        start=rv_lookback,
        end=t,
        timeframe="1d",
    )
    spot_history = []
    if not rv_df.empty:
        for idx, row in rv_df.iterrows():
            spot_history.append((idx, float(row["close"])))
    
    # Get chain_mode from config (defaults to synthetic_grid for backward compatibility)
    chain_mode = getattr(cfg, 'chain_mode', 'synthetic_grid')
    
    if cfg.pricing_mode == "deribit_live":
        # Full live mode - use Deribit chain directly
        all_options: List[OptionSnapshot] = ds.list_option_chain(
            underlying=underlying,
            as_of=t,
            settlement_ccy=cfg.option_settlement_ccy,
            margin_type=cfg.option_margin_type,
        )
        
        candidates = _filter_option_chain(
            all_options=all_options,
            t=t,
            min_dte=cfg.min_dte,
            max_dte=cfg.max_dte,
            delta_min=cfg.delta_min,
            delta_max=cfg.delta_max,
        )
    elif chain_mode == "live_chain":
        # Hybrid mode: live chain with configurable sigma mode
        if spot is not None and spot > 0:
            candidates = _generate_live_chain_candidates(ds, spot, t, cfg, spot_history)
    else:
        # Default: synthetic_grid mode
        if spot is not None and spot > 0:
            # Use new sigma selection logic
            sigma = get_sigma_for_option(
                config=cfg,
                spot_history=spot_history,
                as_of=t,
                option_chain=None,  # No chain for pure synthetic
                option_mark_iv=None,
                abs_delta=cfg.target_delta,
                regime_state=None,
            )
            candidates = _generate_synthetic_candidates(spot, t, cfg, sigma)

    portfolio = {
        "spot_position": cfg.initial_spot_position,
        "equity_usd": None,
    }

    return {
        "time": t,
        "spot": spot,
        "underlying": underlying,
        "market_context": mc_dict,
        "candidate_options": candidates,
        "portfolio": portfolio,
    }


def create_state_builder(
    ds: DeribitDataSource,
    cfg: CallSimulationConfig,
):
    """
    Factory function to create a state_builder callable for simulate_policy.
    
    Usage:
        state_builder = create_state_builder(ds, cfg)
        result = simulator.simulate_policy(
            decision_times=decision_times,
            state_builder=state_builder,
            exit_style="hold_to_expiry",
        )
    """
    def state_builder(t: datetime) -> Dict[str, Any]:
        return build_historical_state(ds, cfg, t)
    
    return state_builder


def _option_snapshot_to_raw_option(
    opt: OptionSnapshot,
    spot: float,
    underlying: str,
    rv: float,
    reference_time: datetime,
) -> RawOption:
    """
    Convert OptionSnapshot to RawOption for state_core.
    
    Args:
        opt: OptionSnapshot from data source
        spot: Current spot price
        underlying: Underlying asset symbol
        rv: Realized volatility
        reference_time: The decision timestamp (NOT datetime.now()) for DTE calculation
    """
    from .pricing import bs_call_price
    
    expiry = opt.expiry
    dte = _calculate_dte_float(expiry, reference_time) if expiry else 0
    
    iv = opt.iv if opt.iv else 0.6
    mark_price = opt.mark_price
    if mark_price is None and spot > 0 and expiry:
        t_years = dte / 365.0 if dte > 0 else 0.001
        mark_price = bs_call_price(spot, opt.strike, t_years, iv, 0.05)
    
    return RawOption(
        instrument_name=opt.instrument_name,
        expiry=expiry or reference_time,
        strike=opt.strike,
        option_type="call" if opt.kind == "call" else "put",
        mark_price=mark_price or 0.0,
        mark_iv=iv * 100,
        delta=opt.delta or 0.0,
        underlying_price=spot,
        underlying=underlying,
        bid=mark_price * 0.95 if mark_price else None,
        ask=mark_price * 1.05 if mark_price else None,
        rv=rv * 100,
    )


def build_historical_agent_state(
    ds: DeribitDataSource,
    cfg: CallSimulationConfig,
    t: datetime,
) -> AgentState:
    """
    Build a historical AgentState at time t using the shared state_core.
    
    This function uses the SAME build_agent_state_from_raw() logic as the
    live agent, ensuring no drift between backtest and live state construction.
    
    Args:
        ds: DeribitDataSource instance
        cfg: CallSimulationConfig with target parameters
        t: Decision time for state construction
        
    Returns:
        AgentState suitable for strategy.propose_actions()
    """
    underlying = cfg.underlying
    
    lookback_hours = 24
    spot_lookback = t - timedelta(hours=lookback_hours)
    spot_df = ds.get_spot_ohlc(
        underlying=underlying,
        start=spot_lookback,
        end=t,
        timeframe=cfg.timeframe,
    )
    spot = float(spot_df["close"].iloc[-1]) if not spot_df.empty else 0.0
    
    mc_obj = compute_market_context_from_ds(ds, underlying=underlying, as_of=t)
    
    rv_lookback = t - timedelta(days=cfg.synthetic_rv_window_days + 7)
    rv_df = ds.get_spot_ohlc(
        underlying=underlying,
        start=rv_lookback,
        end=t,
        timeframe="1d",
    )
    spot_history = []
    if not rv_df.empty:
        for idx, row in rv_df.iterrows():
            spot_history.append((idx, float(row["close"])))
    
    # Use new sigma selection logic
    sigma = get_sigma_for_option(
        config=cfg,
        spot_history=spot_history,
        as_of=t,
        option_chain=None,
        option_mark_iv=None,
        abs_delta=cfg.target_delta,
        regime_state=None,
    )
    rv = sigma * 0.8
    
    # Get chain_mode from config
    chain_mode = getattr(cfg, 'chain_mode', 'synthetic_grid')
    
    options: List[OptionSnapshot] = []
    if cfg.pricing_mode == "deribit_live":
        # Full live mode
        all_options = ds.list_option_chain(
            underlying=underlying,
            as_of=t,
            settlement_ccy=cfg.option_settlement_ccy,
            margin_type=cfg.option_margin_type,
        )
        options = _filter_option_chain(
            all_options=all_options,
            t=t,
            min_dte=cfg.min_dte,
            max_dte=cfg.max_dte,
            delta_min=cfg.delta_min,
            delta_max=cfg.delta_max,
        )
    elif chain_mode == "live_chain":
        # Hybrid mode: live chain with configurable sigma mode
        if spot > 0:
            options = _generate_live_chain_candidates(ds, spot, t, cfg, spot_history)
    else:
        # Default: synthetic_grid mode
        if spot > 0:
            options = _generate_synthetic_candidates(spot, t, cfg, sigma)
    
    raw_options = [
        _option_snapshot_to_raw_option(opt, spot, underlying, rv, t)
        for opt in options
    ]
    
    raw_portfolio = RawPortfolio(
        equity_usd=100000.0,
        margin_used_pct=0.0,
        balances={underlying: cfg.initial_spot_position},
        positions=[],
        margin_used_usd=0.0,
        margin_available_usd=100000.0,
        net_delta=0.0,
    )
    
    raw_snapshot = RawMarketSnapshot(
        timestamp=t,
        underlyings=[underlying],
        spot={underlying: spot},
        portfolio=raw_portfolio,
        options=raw_options,
        realized_vol={underlying: rv * 100},
        market_context=mc_obj,
    )
    
    return build_agent_state_from_raw(
        raw_snapshot,
        delta_min=cfg.delta_min,
        delta_max=cfg.delta_max,
        dte_min=cfg.min_dte,
        dte_max=cfg.max_dte,
        premium_min_usd=0.0,
        max_candidates=50,
        source="backtest",
    )


def create_agent_state_builder(
    ds: DeribitDataSource,
    cfg: CallSimulationConfig,
):
    """
    Factory function to create a typed AgentState builder for strategies.
    
    This returns the SAME AgentState type as the live agent uses,
    enabling direct use of Strategy.propose_actions().
    
    Usage:
        state_builder = create_agent_state_builder(ds, cfg)
        agent_state = state_builder(t)
        actions = strategy.propose_actions(agent_state)
    """
    def state_builder(t: datetime) -> AgentState:
        return build_historical_agent_state(ds, cfg, t)
    
    return state_builder
