"""
Historical state builder for backtests.
Constructs state dicts that mirror live AgentState for scoring and simulation.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import asdict

from .deribit_data_source import DeribitDataSource
from .market_context_backtest import compute_market_context_from_ds, market_context_to_dict
from .types import OptionSnapshot, CallSimulationConfig


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

    all_options: List[OptionSnapshot] = ds.list_option_chain(
        underlying=underlying,
        as_of=t,
        settlement_ccy=cfg.option_settlement_ccy,
        margin_type=cfg.option_margin_type,
    )
    
    candidates: List[OptionSnapshot] = []
    min_dte = cfg.min_dte
    max_dte = cfg.max_dte
    delta_min = cfg.delta_min
    delta_max = cfg.delta_max

    for opt in all_options:
        if opt.kind != "call":
            continue

        expiry = opt.expiry
        dte = (expiry - t).total_seconds() / 86400.0
        if dte < min_dte or dte > max_dte:
            continue

        if opt.delta is None:
            continue
        delta_abs = abs(float(opt.delta))
        if delta_abs < delta_min or delta_abs > delta_max:
            continue

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
