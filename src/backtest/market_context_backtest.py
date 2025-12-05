"""
Market context computation for backtests using DeribitDataSource.
Mirrors the live compute_market_context but uses historical data.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional, Union, Dict, Any
from dataclasses import asdict

import numpy as np
import pandas as pd

from src.models import MarketContext
from .deribit_data_source import DeribitDataSource


def compute_market_context_from_ds(
    ds: DeribitDataSource,
    underlying: str,
    as_of: datetime,
    lookback_days: int = 60,
) -> Optional[MarketContext]:
    """
    Build a compact market context summary for backtests using DeribitDataSource.
    Uses daily candles over the last ~60 days.
    
    Args:
        ds: DeribitDataSource instance
        underlying: "BTC" or "ETH"
        as_of: Timestamp for which to compute context
        lookback_days: Number of days to look back for data
        
    Returns:
        MarketContext with regime, returns, vol, MA distances, or None if insufficient data
    """
    start = as_of - timedelta(days=lookback_days)

    df = ds.get_spot_ohlc(
        underlying=underlying,
        start=start,
        end=as_of,
        timeframe="1d",
    )
    if df.empty:
        return None

    close = df["close"].sort_index()
    if len(close) < 30:
        return None

    def pct_return(days: int) -> float:
        if len(close) < days + 1:
            return 0.0
        c_now = close.iloc[-1]
        c_prev = close.iloc[-(days + 1)]
        return float((c_now / c_prev - 1.0) * 100.0)

    return_1d_pct = pct_return(1)
    return_7d_pct = pct_return(7)
    return_30d_pct = pct_return(30)

    log_ret = np.log(close / close.shift(1)).dropna()

    def realized_vol(days: int) -> float:
        if len(log_ret) < days:
            return 0.0
        window = log_ret.iloc[-days:]
        if window.empty:
            return 0.0
        daily_vol = float(window.std())
        return float(daily_vol * np.sqrt(365.0))

    realized_vol_7d = realized_vol(7)
    realized_vol_30d = realized_vol(30)

    ma_50_series = close.rolling(window=50, min_periods=1).mean()
    ma_50 = float(ma_50_series.iloc[-1]) if len(ma_50_series) > 0 else float(close.iloc[-1])
    
    if len(close) >= 200:
        ma_200_series = close.rolling(window=200, min_periods=1).mean()
        ma_200 = float(ma_200_series.iloc[-1]) if len(ma_200_series) > 0 else ma_50
    else:
        ma_200 = ma_50

    last = float(close.iloc[-1])
    pct_from_50d_ma = (last / ma_50 - 1.0) * 100.0 if ma_50 > 0 else 0.0
    pct_from_200d_ma = (last / ma_200 - 1.0) * 100.0 if ma_200 > 0 else 0.0

    if pct_from_200d_ma > 5.0 and return_30d_pct > 10.0:
        regime = "bull"
    elif pct_from_200d_ma < -5.0 and return_30d_pct < -10.0:
        regime = "bear"
    else:
        regime = "sideways"

    return MarketContext(
        underlying=underlying,
        time=as_of,
        regime=regime,
        pct_from_50d_ma=pct_from_50d_ma,
        pct_from_200d_ma=pct_from_200d_ma,
        return_1d_pct=return_1d_pct,
        return_7d_pct=return_7d_pct,
        return_30d_pct=return_30d_pct,
        realized_vol_7d=realized_vol_7d,
        realized_vol_30d=realized_vol_30d,
        support_level=None,
        resistance_level=None,
        distance_to_support_pct=None,
        distance_to_resistance_pct=None,
    )


def market_context_to_dict(mc: Union[MarketContext, Dict[str, Any], None]) -> Dict[str, Any]:
    """
    Convert MarketContext to dict, handling both dataclass and dict inputs.
    Returns empty dict if None.
    """
    if mc is None:
        return {}
    if isinstance(mc, dict):
        return mc
    return asdict(mc) if hasattr(mc, '__dataclass_fields__') else mc.model_dump()
