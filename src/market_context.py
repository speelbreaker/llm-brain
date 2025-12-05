"""
Market context computation for chart-aware LLM decisions.
Computes trend/regime, recent returns, and realized volatility.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Literal, Optional

import numpy as np
import pandas as pd

from src.models import MarketContext
from src.deribit_client import DeribitClient, DeribitAPIError


def compute_market_context(
    client: DeribitClient,
    underlying: str,
    as_of: datetime,
) -> Optional[MarketContext]:
    """
    Build a compact 'chart-aware' summary for the LLM.
    Uses daily candles for the last ~60 days.
    
    Args:
        client: Deribit API client
        underlying: Asset symbol (BTC or ETH)
        as_of: Reference timestamp for the snapshot
    
    Returns:
        MarketContext with trend/regime, returns, and volatility metrics,
        or None if insufficient data
    """
    start = as_of - timedelta(days=60)
    
    try:
        res = client.get_tradingview_chart_data(
            instrument_name=f"{underlying.lower()}_usd",
            start=start,
            end=as_of,
            resolution="1D",
        )
    except DeribitAPIError:
        try:
            res = client.get_tradingview_chart_data(
                instrument_name=f"{underlying}-PERPETUAL",
                start=start,
                end=as_of,
                resolution="1D",
            )
        except DeribitAPIError:
            return None
    
    if not res or not res.get("ticks"):
        return None
    
    timestamps = [
        datetime.fromtimestamp(ts / 1000, tz=timezone.utc) for ts in res["ticks"]
    ]
    df = pd.DataFrame(
        {
            "close": res["close"],
        },
        index=pd.DatetimeIndex(timestamps, name="timestamp"),
    ).sort_index()
    
    if len(df) < 30:
        return None
    
    close = df["close"]
    
    def pct_return(days: int) -> float:
        if len(close) < days + 1:
            return 0.0
        c_now = float(close.iloc[-1])
        c_prev = float(close.iloc[-(days + 1)])
        if c_prev == 0:
            return 0.0
        return float((c_now / c_prev - 1.0) * 100.0)
    
    return_1d_pct = pct_return(1)
    return_7d_pct = pct_return(7)
    return_30d_pct = pct_return(30)
    
    log_ret = np.log(close / close.shift(1)).dropna()
    
    def realized_vol(days: int) -> float:
        if len(log_ret) < days:
            window = log_ret
        else:
            window = log_ret.iloc[-days:]
        if len(window) == 0:
            return 0.0
        daily_vol = float(window.std())
        return float(daily_vol * np.sqrt(365.0) * 100.0)
    
    realized_vol_7d = realized_vol(7)
    realized_vol_30d = realized_vol(30)
    
    ma_50_series = close.rolling(window=min(50, len(close))).mean()
    ma_50 = float(ma_50_series.iloc[-1]) if len(ma_50_series) > 0 else 0.0
    
    if len(close) >= 200:
        ma_200_series = close.rolling(window=200).mean()
    else:
        ma_200_series = close.rolling(window=len(close)).mean()
    ma_200 = float(ma_200_series.iloc[-1]) if len(ma_200_series) > 0 else ma_50
    
    last = float(close.iloc[-1])
    
    if ma_50 > 0:
        pct_from_50d_ma = float((last / ma_50 - 1.0) * 100.0)
    else:
        pct_from_50d_ma = 0.0
    
    if ma_200 > 0:
        pct_from_200d_ma = float((last / ma_200 - 1.0) * 100.0)
    else:
        pct_from_200d_ma = 0.0
    
    if pct_from_200d_ma > 5.0 and return_30d_pct > 10.0:
        regime: Literal["bull", "sideways", "bear"] = "bull"
    elif pct_from_200d_ma < -5.0 and return_30d_pct < -10.0:
        regime = "bear"
    else:
        regime = "sideways"
    
    return MarketContext(
        underlying=underlying,
        time=as_of,
        regime=regime,
        pct_from_50d_ma=round(pct_from_50d_ma, 2),
        pct_from_200d_ma=round(pct_from_200d_ma, 2),
        return_1d_pct=round(return_1d_pct, 2),
        return_7d_pct=round(return_7d_pct, 2),
        return_30d_pct=round(return_30d_pct, 2),
        realized_vol_7d=round(realized_vol_7d, 2),
        realized_vol_30d=round(realized_vol_30d, 2),
        support_level=None,
        resistance_level=None,
        distance_to_support_pct=None,
        distance_to_resistance_pct=None,
    )
