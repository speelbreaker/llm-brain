"""
Calibration module to compare synthetic Black-Scholes prices with live Deribit option prices.

This module provides:
1. Black-Scholes call pricing
2. Deribit public API helpers
3. Calibration logic to compute pricing errors

Uses RV-based IV model matching the synthetic backtester, NOT Deribit's mark_iv.
Uses only public Deribit endpoints - no authentication required.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import httpx

from src.backtest.pricing import (
    compute_realized_volatility,
    compute_synthetic_iv_with_skew,
    bs_call_delta,
)
from src.config import settings


DERIBIT_API = "https://www.deribit.com/api/v2"


def _norm_cdf(x: float) -> float:
    """Standard normal CDF using math.erf."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def black_scholes_call_price(
    spot: float,
    strike: float,
    t_years: float,
    sigma: float,
    r: float = 0.0,
) -> float:
    """
    Standard Black-Scholes European call price.
    
    Args:
        spot: Current underlying price
        strike: Option strike price
        t_years: Time to expiration in years
        sigma: Implied volatility (annualized)
        r: Risk-free rate (continuous compounding)
    
    Returns:
        Call option price
    """
    if t_years <= 0 or sigma <= 0 or spot <= 0 or strike <= 0:
        return max(0.0, spot - strike)

    sqrt_t = math.sqrt(t_years)
    d1 = (math.log(spot / strike) + (r + 0.5 * sigma ** 2) * t_years) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t

    nd1 = _norm_cdf(d1)
    nd2 = _norm_cdf(d2)

    return spot * nd1 - strike * math.exp(-r * t_years) * nd2


def black_scholes_put_price(
    spot: float,
    strike: float,
    t_years: float,
    sigma: float,
    r: float = 0.0,
) -> float:
    """
    Standard Black-Scholes European put price.
    
    Args:
        spot: Current underlying price
        strike: Option strike price
        t_years: Time to expiration in years
        sigma: Implied volatility (annualized)
        r: Risk-free rate (continuous compounding)
    
    Returns:
        Put option price
    """
    if t_years <= 0 or sigma <= 0 or spot <= 0 or strike <= 0:
        return max(0.0, strike - spot)

    sqrt_t = math.sqrt(t_years)
    d1 = (math.log(spot / strike) + (r + 0.5 * sigma ** 2) * t_years) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t

    nd1 = _norm_cdf(-d1)
    nd2 = _norm_cdf(-d2)

    return strike * math.exp(-r * t_years) * nd2 - spot * nd1


def deribit_get(path: str, params: Dict[str, Any]) -> Any:
    """Make a GET request to Deribit public API."""
    url = f"{DERIBIT_API}/{path}"
    with httpx.Client(timeout=15.0) as client:
        resp = client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        if "result" not in data:
            raise RuntimeError(f"Unexpected Deribit response: {data}")
        return data["result"]


def get_index_price(underlying: str) -> float:
    """Get the current index price for underlying (BTC or ETH) in USD."""
    index_name = f"{underlying.lower()}_usd"
    result = deribit_get("public/get_index_price", {"index_name": index_name})
    return float(result["index_price"])


def get_spot_history_for_rv(
    underlying: str,
    as_of: datetime,
    window_days: int = 7,
) -> List[Tuple[datetime, float]]:
    """
    Fetch daily index close prices for the given underlying from Deribit
    over roughly `window_days` days, to use for realized volatility.

    Uses public/get_tradingview_chart_data with instrument_name like "btc_usd".
    """
    start = as_of - timedelta(days=window_days + 10)

    params = {
        "instrument_name": f"{underlying.lower()}_usd",
        "start_timestamp": int(start.timestamp() * 1000),
        "end_timestamp": int(as_of.timestamp() * 1000),
        "resolution": "1D",
    }
    try:
        res = deribit_get("public/get_tradingview_chart_data", params)
    except Exception:
        params["instrument_name"] = f"{underlying}-PERPETUAL"
        res = deribit_get("public/get_tradingview_chart_data", params)

    ticks = res.get("ticks") or res.get("timestamp") or []
    closes = res.get("close") or []
    if not ticks or not closes:
        return []

    points: List[Tuple[datetime, float]] = []
    for ts, close in zip(ticks, closes):
        t = datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc)
        points.append((t, float(close)))

    points.sort(key=lambda x: x[0])
    return points


@dataclass
class OptionQuote:
    """Point-in-time quote for a Deribit option."""
    instrument_name: str
    kind: str
    strike: float
    expiration: datetime
    mark_price: float
    mark_iv: Optional[float]
    delta: Optional[float]
    dte_days: float
    settlement_currency: str = "BTC"
    vega: Optional[float] = None


def get_call_chain(
    underlying: str,
    min_dte: float,
    max_dte: float,
) -> List[OptionQuote]:
    """
    Fetch all CALL options for the given underlying and filter by DTE.
    
    Uses public Deribit endpoints:
    - public/get_instruments (currency=underlying, kind=option, expired=false)
    - public/ticker (instrument_name=<option>)
    """
    result = deribit_get(
        "public/get_instruments",
        {
            "currency": underlying,
            "kind": "option",
            "expired": "false",
        },
    )
    instruments = result

    now = datetime.now(timezone.utc)
    quotes: List[OptionQuote] = []

    for inst in instruments:
        if inst.get("option_type") != "call":
            continue

        instrument_name = inst["instrument_name"]
        strike = float(inst["strike"])
        expiration_ts_ms = inst["expiration_timestamp"]
        expiration = datetime.fromtimestamp(expiration_ts_ms / 1000.0, tz=timezone.utc)

        dte_days = (expiration - now).total_seconds() / 86400.0
        if dte_days < min_dte or dte_days > max_dte:
            continue

        settlement_currency = inst.get("settlement_currency", underlying)
        
        try:
            ticker = deribit_get(
                "public/ticker",
                {"instrument_name": instrument_name},
            )
            mark_price = float(ticker.get("mark_price", 0.0) or 0.0)
            mark_iv = ticker.get("mark_iv", None)
            greeks = ticker.get("greeks") or {}
            delta = greeks.get("delta", None)
            vega = greeks.get("vega", None)
            
            if mark_price <= 0.0:
                continue

            quotes.append(
                OptionQuote(
                    instrument_name=instrument_name,
                    kind="call",
                    strike=strike,
                    expiration=expiration,
                    mark_price=mark_price,
                    mark_iv=float(mark_iv) if mark_iv is not None else None,
                    delta=float(delta) if delta is not None else None,
                    dte_days=dte_days,
                    settlement_currency=settlement_currency,
                    vega=float(vega) if vega is not None else None,
                )
            )
        except Exception:
            continue

    return quotes


def get_option_chain(
    underlying: str,
    min_dte: float,
    max_dte: float,
    option_types: List[str] = ["C"],
) -> List[OptionQuote]:
    """
    Fetch options for the given underlying, filtering by DTE and option type.
    
    Args:
        underlying: BTC or ETH
        min_dte: Minimum days to expiry
        max_dte: Maximum days to expiry
        option_types: List of option types to include: 'C' for calls, 'P' for puts
                     Default ['C'] for backward compatibility
    
    Returns:
        List of OptionQuote objects
    """
    type_map = {"C": "call", "P": "put"}
    allowed_types = set()
    for ot in option_types:
        ot_upper = ot.upper()
        if ot_upper in type_map:
            allowed_types.add(type_map[ot_upper])
        elif ot_upper in ("CALL", "PUT"):
            allowed_types.add(ot_upper.lower())
    
    if not allowed_types:
        allowed_types = {"call"}
    
    result = deribit_get(
        "public/get_instruments",
        {
            "currency": underlying,
            "kind": "option",
            "expired": "false",
        },
    )
    instruments = result

    now = datetime.now(timezone.utc)
    quotes: List[OptionQuote] = []

    for inst in instruments:
        opt_type = inst.get("option_type")
        if opt_type not in allowed_types:
            continue

        instrument_name = inst["instrument_name"]
        strike = float(inst["strike"])
        expiration_ts_ms = inst["expiration_timestamp"]
        expiration = datetime.fromtimestamp(expiration_ts_ms / 1000.0, tz=timezone.utc)

        dte_days = (expiration - now).total_seconds() / 86400.0
        if dte_days < min_dte or dte_days > max_dte:
            continue

        settlement_currency = inst.get("settlement_currency", underlying)
        
        try:
            ticker = deribit_get(
                "public/ticker",
                {"instrument_name": instrument_name},
            )
            mark_price = float(ticker.get("mark_price", 0.0) or 0.0)
            mark_iv = ticker.get("mark_iv", None)
            greeks = ticker.get("greeks") or {}
            delta = greeks.get("delta", None)
            vega = greeks.get("vega", None)
            
            if mark_price <= 0.0:
                continue

            quotes.append(
                OptionQuote(
                    instrument_name=instrument_name,
                    kind=opt_type,
                    strike=strike,
                    expiration=expiration,
                    mark_price=mark_price,
                    mark_iv=float(mark_iv) if mark_iv is not None else None,
                    delta=float(delta) if delta is not None else None,
                    dte_days=dte_days,
                    settlement_currency=settlement_currency,
                    vega=float(vega) if vega is not None else None,
                )
            )
        except Exception:
            continue

    return quotes


def synthetic_iv_from_rv(
    base_iv: float,
    iv_multiplier: float,
) -> float:
    """
    Synthetic IV model for calibration using realized volatility.

    This matches the synthetic backtester's pricing model:
        sigma_synth = realized_vol(window_days) * synthetic_iv_multiplier

    Args:
        base_iv: An annualized realized volatility (e.g. 0.70 for 70%)
        iv_multiplier: Scales the base IV, same as CallSimulationConfig.synthetic_iv_multiplier

    Returns:
        sigma to plug into Black-Scholes
    """
    return max(1e-6, base_iv * iv_multiplier)


@dataclass
class CalibrationRow:
    """Single option comparison row."""
    instrument: str
    dte: float
    strike: float
    mark_price: float
    syn_price: float
    diff: float
    diff_pct: float
    mark_iv: Optional[float]
    syn_iv: float


@dataclass
class CalibrationResult:
    """Complete calibration result with summary metrics and detail rows."""
    underlying: str
    spot: float
    min_dte: float
    max_dte: float
    iv_multiplier: float
    default_iv: float
    count: int
    mae_pct: float
    bias_pct: float
    timestamp: datetime
    rows: List[CalibrationRow]
    rv_annualized: Optional[float] = None
    atm_iv: Optional[float] = None
    recommended_iv_multiplier: Optional[float] = None


def run_calibration(
    underlying: str,
    min_dte: float,
    max_dte: float,
    iv_multiplier: float = 1.0,
    default_iv: float = 0.6,
    r: float = 0.0,
    max_samples: int = 80,
    rv_window_days: int = 7,
) -> CalibrationResult:
    """
    Compare synthetic BS prices vs Deribit mark prices for CALLs.
    
    Uses RV-based IV model with skew matching the synthetic backtester:
        sigma_synth = realized_vol(window_days) * iv_multiplier * skew_factor(delta)
    
    1. Fetch current spot and recent spot history from Deribit
    2. Compute realized volatility from spot history
    3. Fetch call chain via get_call_chain()
    4. Sub-sample to at most max_samples quotes
    5. For each quote:
       - Compute t_years = dte_days / 365.0
       - Compute delta for this strike
       - Compute sigma using RV * multiplier * skew_factor(delta)
       - Compute synthetic_price via Black-Scholes
       - diff = synthetic_price - mark_price
       - diff_pct = diff / mark_price * 100
    6. Compute:
       - mae_pct = mean(|diff_pct|)
       - bias_pct = mean(diff_pct)
    7. Return CalibrationResult with rows and aggregates
    """
    now = datetime.now(timezone.utc)
    spot = get_index_price(underlying)
    
    spot_history = get_spot_history_for_rv(underlying, as_of=now, window_days=rv_window_days)
    
    if spot_history:
        rv_annualized = compute_realized_volatility(
            prices=spot_history,
            as_of=now,
            window_days=rv_window_days,
        )
    else:
        rv_annualized = default_iv
    
    base_iv = rv_annualized
    
    quotes = get_call_chain(underlying, min_dte=min_dte, max_dte=max_dte)
    quotes = sorted(quotes, key=lambda q: (q.dte_days, q.strike))
    
    if not quotes:
        return CalibrationResult(
            underlying=underlying,
            spot=spot,
            min_dte=min_dte,
            max_dte=max_dte,
            iv_multiplier=iv_multiplier,
            default_iv=default_iv,
            count=0,
            mae_pct=0.0,
            bias_pct=0.0,
            timestamp=now,
            rows=[],
            rv_annualized=rv_annualized,
        )

    atm_iv: Optional[float] = None
    best_delta_diff: Optional[float] = None
    for q in quotes:
        if q.mark_iv is None or q.mark_iv <= 0:
            continue
        if q.delta is None:
            continue
        delta_val = float(q.delta)
        diff = abs(delta_val - 0.5)
        if best_delta_diff is None or diff < best_delta_diff:
            best_delta_diff = diff
            atm_iv = float(q.mark_iv) / 100.0

    recommended_iv_multiplier: Optional[float] = None
    if rv_annualized and rv_annualized > 0.0 and atm_iv and atm_iv > 0.0:
        recommended_iv_multiplier = atm_iv / rv_annualized

    if len(quotes) > max_samples:
        step = max(1, len(quotes) // max_samples)
        quotes = quotes[::step]

    rows: List[CalibrationRow] = []
    errors_pct: List[float] = []

    for q in quotes:
        t_years = max(0.0001, q.dte_days / 365.0)
        
        base_iv_for_delta = max(1e-6, rv_annualized * iv_multiplier)
        abs_delta = abs(bs_call_delta(
            spot=spot,
            strike=q.strike,
            t_years=t_years,
            sigma=base_iv_for_delta,
            r=r,
        ))
        
        sigma = compute_synthetic_iv_with_skew(
            underlying=underlying,
            option_type="call",
            abs_delta=abs_delta,
            rv_annualized=rv_annualized,
            iv_multiplier=iv_multiplier,
            skew_enabled=settings.synthetic_skew_enabled,
            skew_min_dte=settings.synthetic_skew_min_dte,
            skew_max_dte=settings.synthetic_skew_max_dte,
        )

        synthetic_price_usd = black_scholes_call_price(
            spot=spot,
            strike=q.strike,
            t_years=t_years,
            sigma=sigma,
            r=r,
        )
        
        is_inverse = q.settlement_currency.upper() in ("BTC", "ETH")
        if is_inverse:
            synthetic_price = synthetic_price_usd / spot
        else:
            synthetic_price = synthetic_price_usd

        diff = synthetic_price - q.mark_price
        if q.mark_price > 0:
            diff_pct = (diff / q.mark_price) * 100.0
        else:
            diff_pct = 0.0

        errors_pct.append(diff_pct)

        rows.append(
            CalibrationRow(
                instrument=q.instrument_name,
                dte=q.dte_days,
                strike=q.strike,
                mark_price=q.mark_price,
                syn_price=synthetic_price,
                diff=diff,
                diff_pct=diff_pct,
                mark_iv=q.mark_iv,
                syn_iv=sigma,
            )
        )

    mae_pct = sum(abs(x) for x in errors_pct) / len(errors_pct) if errors_pct else 0.0
    bias_pct = sum(errors_pct) / len(errors_pct) if errors_pct else 0.0

    return CalibrationResult(
        underlying=underlying,
        spot=spot,
        min_dte=min_dte,
        max_dte=max_dte,
        iv_multiplier=iv_multiplier,
        default_iv=default_iv,
        count=len(rows),
        mae_pct=mae_pct,
        bias_pct=bias_pct,
        timestamp=now,
        rows=rows,
        rv_annualized=rv_annualized,
        atm_iv=atm_iv,
        recommended_iv_multiplier=recommended_iv_multiplier,
    )
