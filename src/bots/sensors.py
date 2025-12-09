"""
Sensor computation module for bots.

Computes technical indicators (ADX, RSI, MA200) and volatility metrics
from OHLC data fetched via Deribit API.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from functools import lru_cache
from math import sqrt, log
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.deribit_client import DeribitClient


@lru_cache(maxsize=2)
def _fetch_ohlc_cached(underlying: str, cache_key: str) -> pd.DataFrame:
    """
    Fetch daily OHLC data for the past 250 days (enough for 200-day MA).
    Cached by underlying and cache_key (date-based).
    """
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=250)
    
    with DeribitClient() as client:
        index_name = f"{underlying.lower()}_usd"
        try:
            res = client.get_tradingview_chart_data(
                instrument_name=index_name,
                start=start,
                end=end,
                resolution="1D",
            )
        except Exception:
            perp_name = f"{underlying}-PERPETUAL"
            res = client.get_tradingview_chart_data(
                instrument_name=perp_name,
                start=start,
                end=end,
                resolution="1D",
            )
    
    if not res.get("ticks"):
        return pd.DataFrame()
    
    timestamps = [
        datetime.fromtimestamp(ts / 1000, tz=timezone.utc) for ts in res["ticks"]
    ]
    df = pd.DataFrame(
        {
            "open": res["open"],
            "high": res["high"],
            "low": res["low"],
            "close": res["close"],
            "volume": res.get("volume", [0] * len(res["open"])),
        },
        index=pd.DatetimeIndex(timestamps, name="timestamp"),
    )
    return df


def get_ohlc_data(underlying: str) -> pd.DataFrame:
    """Get OHLC data with hourly cache invalidation."""
    now = datetime.now(timezone.utc)
    cache_key = f"{now.year}-{now.month}-{now.day}-{now.hour}"
    return _fetch_ohlc_cached(underlying, cache_key)


def compute_rsi(closes: pd.Series, period: int = 14) -> Optional[float]:
    """
    Compute RSI using Wilder's smoothing method.
    Returns value between 0-100, or None if insufficient data.
    """
    if len(closes) < period + 1:
        return None
    
    deltas = closes.diff()
    gains = deltas.where(deltas > 0, 0.0)
    losses = (-deltas).where(deltas < 0, 0.0)
    
    avg_gain = gains.iloc[1:period+1].mean()
    avg_loss = losses.iloc[1:period+1].mean()
    
    for i in range(period + 1, len(closes)):
        avg_gain = (avg_gain * (period - 1) + gains.iloc[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses.iloc[i]) / period
    
    if avg_loss == 0:
        return 100.0
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return float(rsi)


def compute_adx(df: pd.DataFrame, period: int = 14) -> Optional[float]:
    """
    Compute ADX (Average Directional Index).
    Returns value between 0-100, or None if insufficient data.
    """
    if len(df) < period * 2:
        return None
    
    high = df["high"]
    low = df["low"]
    close = df["close"]
    
    plus_dm = high.diff()
    minus_dm = low.diff().abs() * -1
    
    plus_dm = plus_dm.where((plus_dm > minus_dm.abs()) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.abs().where((minus_dm.abs() > plus_dm) & (minus_dm.abs() > 0), 0.0)
    
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.rolling(window=period).mean()
    
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
    
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    adx = dx.rolling(window=period).mean()
    
    last_adx = adx.iloc[-1]
    if pd.isna(last_adx):
        return None
    return float(last_adx)


def compute_ma200(closes: pd.Series) -> Optional[float]:
    """Compute 200-day simple moving average."""
    if len(closes) < 200:
        return None
    return float(closes.iloc[-200:].mean())


def compute_price_vs_ma200(spot: float, ma200: Optional[float]) -> Optional[float]:
    """Compute % distance from 200-day MA."""
    if ma200 is None or ma200 <= 0:
        return None
    return ((spot - ma200) / ma200) * 100


def compute_realized_volatility(closes: pd.Series, window: int = 30) -> Optional[float]:
    """
    Compute annualized realized volatility from daily closes.
    Returns as percentage (e.g., 45.0 for 45%).
    """
    if len(closes) < window + 1:
        return None
    
    recent = closes.iloc[-(window + 1):]
    log_returns = (recent / recent.shift(1)).apply(lambda x: log(x) if x > 0 else 0).dropna()
    
    if len(log_returns) < 2:
        return None
    
    daily_vol = log_returns.std()
    annualized = daily_vol * sqrt(365) * 100
    return float(annualized)


class SensorBundle:
    """Container for computed sensor values for an underlying."""
    
    def __init__(self, underlying: str):
        self.underlying = underlying
        self.vrp_30d: Optional[float] = None
        self.chop_factor_7d: Optional[float] = None
        self.iv_rank_6m: Optional[float] = None
        self.term_structure_spread: Optional[float] = None
        self.skew_25d: Optional[float] = None
        self.adx_14d: Optional[float] = None
        self.rsi_14d: Optional[float] = None
        self.price_vs_ma200: Optional[float] = None
        self.spot: Optional[float] = None
        self.iv_30d: Optional[float] = None
        self.rv_30d: Optional[float] = None
        self.rv_7d: Optional[float] = None
        self.ma200: Optional[float] = None
        self._missing: List[str] = []
    
    def to_dict(self) -> Dict[str, Optional[float]]:
        """Return sensor values as dictionary."""
        return {
            "vrp_30d": self.vrp_30d,
            "chop_factor_7d": self.chop_factor_7d,
            "iv_rank_6m": self.iv_rank_6m,
            "term_structure_spread": self.term_structure_spread,
            "skew_25d": self.skew_25d,
            "adx_14d": self.adx_14d,
            "rsi_14d": self.rsi_14d,
            "price_vs_ma200": self.price_vs_ma200,
        }
    
    @property
    def missing_sensors(self) -> List[str]:
        """List of sensor names that are missing data."""
        return self._missing


def compute_sensors_for_underlying(
    underlying: str,
    iv_30d: Optional[float] = None,
    iv_7d: Optional[float] = None,
    skew: Optional[float] = None,
) -> SensorBundle:
    """
    Compute all Greg sensors for a given underlying.
    
    Args:
        underlying: "BTC" or "ETH"
        iv_30d: Current 30-day implied volatility (from options chain or vol_state)
        iv_7d: Current 7-day implied volatility (for term structure)
        skew: Current 25-delta skew
    
    Returns:
        SensorBundle with computed values
    """
    bundle = SensorBundle(underlying)
    
    try:
        df = get_ohlc_data(underlying)
    except Exception:
        bundle._missing = [
            "vrp_30d", "chop_factor_7d", "iv_rank_6m", "term_structure_spread",
            "skew_25d", "adx_14d", "rsi_14d", "price_vs_ma200"
        ]
        return bundle
    
    if df.empty or len(df) < 30:
        bundle._missing = [
            "vrp_30d", "chop_factor_7d", "iv_rank_6m", "term_structure_spread",
            "skew_25d", "adx_14d", "rsi_14d", "price_vs_ma200"
        ]
        return bundle
    
    closes = df["close"]
    bundle.spot = float(closes.iloc[-1])
    
    bundle.rv_30d = compute_realized_volatility(closes, window=30)
    bundle.rv_7d = compute_realized_volatility(closes, window=7)
    
    bundle.iv_30d = iv_30d
    
    if iv_30d is not None and bundle.rv_30d is not None:
        bundle.vrp_30d = iv_30d - bundle.rv_30d
    else:
        bundle._missing.append("vrp_30d")
    
    if bundle.rv_7d is not None and iv_30d is not None and iv_30d > 0:
        bundle.chop_factor_7d = bundle.rv_7d / iv_30d
    else:
        bundle._missing.append("chop_factor_7d")
    
    if iv_7d is not None and iv_30d is not None:
        bundle.term_structure_spread = iv_7d - iv_30d
    else:
        bundle._missing.append("term_structure_spread")
    
    if skew is not None and skew != 0:
        bundle.skew_25d = skew
    else:
        bundle._missing.append("skew_25d")
    
    bundle._missing.append("iv_rank_6m")
    
    bundle.adx_14d = compute_adx(df, period=14)
    if bundle.adx_14d is None:
        bundle._missing.append("adx_14d")
    
    bundle.rsi_14d = compute_rsi(closes, period=14)
    if bundle.rsi_14d is None:
        bundle._missing.append("rsi_14d")
    
    bundle.ma200 = compute_ma200(closes)
    bundle.price_vs_ma200 = compute_price_vs_ma200(bundle.spot, bundle.ma200)
    if bundle.price_vs_ma200 is None:
        bundle._missing.append("price_vs_ma200")
    
    return bundle
