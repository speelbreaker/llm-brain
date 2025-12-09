"""
Sensor computation module for bots.

Computes technical indicators (ADX, RSI, MA200) and volatility metrics
from OHLC data and options chain data fetched via Deribit API.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from functools import lru_cache
from math import sqrt, log
from typing import Any, Dict, List, Optional, Tuple

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


@lru_cache(maxsize=4)
def _fetch_options_chain_cached(underlying: str, cache_key: str) -> List[Dict[str, Any]]:
    """
    Fetch active options instruments for an underlying.
    Cached with 10-minute invalidation.
    """
    with DeribitClient() as client:
        instruments = client.get_instruments(currency=underlying, kind="option")
    return instruments


def get_options_chain(underlying: str) -> List[Dict[str, Any]]:
    """Get options chain with 10-minute cache invalidation."""
    now = datetime.now(timezone.utc)
    cache_key = f"{now.year}-{now.month}-{now.day}-{now.hour}-{now.minute // 10}"
    return _fetch_options_chain_cached(underlying, cache_key)


def _parse_option_expiry(instrument_name: str) -> Optional[datetime]:
    """Parse expiry date from Deribit option instrument name."""
    parts = instrument_name.split("-")
    if len(parts) < 4:
        return None
    try:
        date_str = parts[1]
        return datetime.strptime(date_str, "%d%b%y").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _compute_dte(expiry: datetime) -> int:
    """Compute days to expiry."""
    now = datetime.now(timezone.utc)
    return max(0, (expiry - now).days)


def compute_term_structure_and_atm_iv(
    underlying: str,
    spot: float,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Compute term structure spread and ATM IVs for 7d and 30d tenors.
    
    Returns:
        (term_structure_spread, iv_7d, iv_30d)
    """
    try:
        instruments = get_options_chain(underlying)
    except Exception:
        return None, None, None
    
    if not instruments:
        return None, None, None
    
    now = datetime.now(timezone.utc)
    
    expiry_groups: Dict[str, List[Dict]] = {}
    for inst in instruments:
        if inst.get("option_type") != "call":
            continue
        expiry = _parse_option_expiry(inst["instrument_name"])
        if expiry is None:
            continue
        dte = _compute_dte(expiry)
        if dte < 1:
            continue
        
        exp_key = expiry.strftime("%Y-%m-%d")
        if exp_key not in expiry_groups:
            expiry_groups[exp_key] = []
        expiry_groups[exp_key].append({
            "instrument_name": inst["instrument_name"],
            "strike": inst["strike"],
            "dte": dte,
            "expiry": expiry,
        })
    
    if not expiry_groups:
        return None, None, None
    
    def find_closest_expiry(target_dte: int, min_dte: int, max_dte: int) -> Optional[str]:
        """Find expiry closest to target DTE within range."""
        valid = []
        for exp_key, opts in expiry_groups.items():
            if opts:
                dte = opts[0]["dte"]
                if min_dte <= dte <= max_dte:
                    valid.append((exp_key, dte, abs(dte - target_dte)))
        if not valid:
            return None
        valid.sort(key=lambda x: x[2])
        return valid[0][0]
    
    # Expanded ranges to handle limited testnet expiries
    # 7d: look for anything in 2-20 DTE range
    # 30d: look for anything in 14-60 DTE range (broader to catch monthly expiries)
    exp_7d = find_closest_expiry(7, 2, 20)
    exp_30d = find_closest_expiry(30, 14, 60)
    
    iv_7d = None
    iv_30d = None
    
    def get_atm_iv(exp_key: str) -> Optional[float]:
        """Get ATM IV by finding option closest to spot and fetching mark_iv."""
        if exp_key not in expiry_groups:
            return None
        
        options = expiry_groups[exp_key]
        options_sorted = sorted(options, key=lambda o: abs(o["strike"] - spot))
        
        if not options_sorted:
            return None
        
        atm_option = options_sorted[0]
        
        try:
            with DeribitClient() as client:
                ticker = client.get_ticker(atm_option["instrument_name"])
                mark_iv = ticker.get("mark_iv")
                if mark_iv is not None and mark_iv > 0:
                    return float(mark_iv)
        except Exception:
            pass
        
        return None
    
    if exp_7d:
        iv_7d = get_atm_iv(exp_7d)
    
    if exp_30d:
        iv_30d = get_atm_iv(exp_30d)
    
    term_spread = None
    if iv_7d is not None and iv_30d is not None:
        term_spread = iv_7d - iv_30d
    
    return term_spread, iv_7d, iv_30d


def compute_skew_25d(underlying: str, spot: float) -> Optional[float]:
    """
    Compute 25-delta skew: IV_25d_put - IV_25d_call.
    
    Uses options with delta closest to +/- 0.25 from a ~30d expiry.
    """
    try:
        instruments = get_options_chain(underlying)
    except Exception:
        return None
    
    if not instruments:
        return None
    
    # Expanded ranges to handle limited testnet expiries
    target_dte = 30
    min_dte = 14
    max_dte = 60
    
    valid_options = []
    for inst in instruments:
        expiry = _parse_option_expiry(inst["instrument_name"])
        if expiry is None:
            continue
        dte = _compute_dte(expiry)
        if min_dte <= dte <= max_dte:
            valid_options.append({
                "instrument_name": inst["instrument_name"],
                "strike": inst["strike"],
                "option_type": inst.get("option_type"),
                "dte": dte,
            })
    
    if not valid_options:
        return None
    
    best_dte = min(valid_options, key=lambda o: abs(o["dte"] - target_dte))["dte"]
    options_at_expiry = [o for o in valid_options if o["dte"] == best_dte]
    
    calls = [o for o in options_at_expiry if o["option_type"] == "call"]
    puts = [o for o in options_at_expiry if o["option_type"] == "put"]
    
    if not calls or not puts:
        return None
    
    def find_25d_option(options: List[Dict], is_call: bool) -> Optional[Dict]:
        """Find option closest to 25-delta."""
        if is_call:
            target_strike = spot * 1.05
        else:
            target_strike = spot * 0.95
        
        sorted_opts = sorted(options, key=lambda o: abs(o["strike"] - target_strike))
        return sorted_opts[0] if sorted_opts else None
    
    call_25d = find_25d_option(calls, is_call=True)
    put_25d = find_25d_option(puts, is_call=False)
    
    if not call_25d or not put_25d:
        return None
    
    try:
        with DeribitClient() as client:
            call_ticker = client.get_ticker(call_25d["instrument_name"])
            put_ticker = client.get_ticker(put_25d["instrument_name"])
            
            call_iv = call_ticker.get("mark_iv")
            put_iv = put_ticker.get("mark_iv")
            
            if call_iv is not None and put_iv is not None and call_iv > 0 and put_iv > 0:
                return float(put_iv - call_iv)
    except Exception:
        pass
    
    return None


def compute_iv_rank_lite(iv_30d: Optional[float], underlying: str) -> Optional[float]:
    """
    Compute a lite IV rank based on historical IV range.
    
    Uses a heuristic based on typical crypto IV ranges:
    - BTC: typical range 40-100%
    - ETH: typical range 50-120%
    
    Returns value in [0, 1] range.
    """
    if iv_30d is None or iv_30d <= 0:
        return None
    
    if underlying.upper() == "BTC":
        iv_low = 35.0
        iv_high = 100.0
    else:
        iv_low = 45.0
        iv_high = 120.0
    
    if iv_30d <= iv_low:
        return 0.0
    if iv_30d >= iv_high:
        return 1.0
    
    rank = (iv_30d - iv_low) / (iv_high - iv_low)
    return float(rank)


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
        self.iv_7d: Optional[float] = None
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
    
    def to_debug_dict(self) -> Dict[str, Any]:
        """Return sensor values with debug inputs for each sensor."""
        underlying = self.underlying.upper()
        iv_low = 35.0 if underlying == "BTC" else 45.0
        iv_high = 100.0 if underlying == "BTC" else 120.0
        
        return {
            "vrp_30d": {
                "value": self.vrp_30d,
                "inputs": {
                    "atm_iv_30d": self.iv_30d,
                    "rv_30d": self.rv_30d,
                },
            },
            "chop_factor_7d": {
                "value": self.chop_factor_7d,
                "inputs": {
                    "rv_7d": self.rv_7d,
                    "iv_30d": self.iv_30d,
                },
            },
            "iv_rank_6m": {
                "value": self.iv_rank_6m,
                "inputs": {
                    "current_iv": self.iv_30d,
                    "iv_min_6m": iv_low,
                    "iv_max_6m": iv_high,
                    "note": "Using heuristic range, not historical data",
                },
            },
            "term_structure_spread": {
                "value": self.term_structure_spread,
                "inputs": {
                    "iv_7d": self.iv_7d,
                    "iv_30d": self.iv_30d,
                },
            },
            "skew_25d": {
                "value": self.skew_25d,
                "inputs": {
                    "note": "25-delta put IV - 25-delta call IV (from ~30d expiry)",
                },
            },
            "adx_14d": {
                "value": self.adx_14d,
                "inputs": {
                    "period": 14,
                },
            },
            "rsi_14d": {
                "value": self.rsi_14d,
                "inputs": {
                    "period": 14,
                },
            },
            "price_vs_ma200": {
                "value": self.price_vs_ma200,
                "inputs": {
                    "spot": self.spot,
                    "ma200": self.ma200,
                },
            },
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
    
    term_spread_live, iv_7d_live, iv_30d_live = compute_term_structure_and_atm_iv(
        underlying, bundle.spot
    )
    
    if iv_30d is None and iv_30d_live is not None:
        iv_30d = iv_30d_live
    if iv_7d is None and iv_7d_live is not None:
        iv_7d = iv_7d_live
    
    bundle.iv_30d = iv_30d
    bundle.iv_7d = iv_7d
    
    if iv_30d is not None and bundle.rv_30d is not None:
        bundle.vrp_30d = iv_30d - bundle.rv_30d
    else:
        bundle._missing.append("vrp_30d")
    
    if bundle.rv_7d is not None and iv_30d is not None and iv_30d > 0:
        bundle.chop_factor_7d = bundle.rv_7d / iv_30d
    else:
        bundle._missing.append("chop_factor_7d")
    
    if term_spread_live is not None:
        bundle.term_structure_spread = term_spread_live
    elif iv_7d is not None and iv_30d is not None:
        bundle.term_structure_spread = iv_7d - iv_30d
    else:
        bundle._missing.append("term_structure_spread")
    
    if skew is not None and skew != 0:
        bundle.skew_25d = skew
    else:
        skew_live = compute_skew_25d(underlying, bundle.spot)
        if skew_live is not None:
            bundle.skew_25d = skew_live
        else:
            bundle._missing.append("skew_25d")
    
    bundle.iv_rank_6m = compute_iv_rank_lite(iv_30d, underlying)
    if bundle.iv_rank_6m is None:
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
