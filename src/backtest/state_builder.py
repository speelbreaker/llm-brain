"""
Historical state builder for backtests.
Constructs state dicts that mirror live AgentState for scoring and simulation.

# TODO: duplicate of src/state_builder.py? Both build agent state snapshots.
# Consider unifying with a shared interface: build_state(source, time, mode="live"|"historical")
"""
from __future__ import annotations

import re
import logging
from datetime import datetime, timezone, timedelta
from dataclasses import replace
from typing import Dict, Any, List, Optional

from .deribit_data_source import DeribitDataSource
from .market_context_backtest import compute_market_context_from_ds, market_context_to_dict
from .types import OptionSnapshot, CallSimulationConfig
from .pricing import bs_call_delta, get_synthetic_iv
from src.utils.expiry import parse_deribit_expiry

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
    
    if cfg.pricing_mode == "synthetic_bs":
        if spot is not None and spot > 0:
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
            
            sigma = get_synthetic_iv(cfg, spot_history, t)
            candidates = _generate_synthetic_candidates(spot, t, cfg, sigma)
    else:
        all_options: List[OptionSnapshot] = ds.list_option_chain(
            underlying=underlying,
            as_of=t,
            settlement_ccy=cfg.option_settlement_ccy,
            margin_type=cfg.option_margin_type,
        )
        
        min_dte = cfg.min_dte
        max_dte = cfg.max_dte
        delta_min = cfg.delta_min
        delta_max = cfg.delta_max

        for opt in all_options:
            if opt.kind != "call":
                continue

            expiry = getattr(opt, "expiry", None)
            if expiry is None:
                instrument = getattr(opt, "instrument_name", None) or getattr(opt, "symbol", "")
                expiry = parse_deribit_expiry(str(instrument))

            if expiry is None:
                continue

            dte = (expiry - t).total_seconds() / 86400.0
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
