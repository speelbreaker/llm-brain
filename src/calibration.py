"""
Calibration module to compare synthetic Black-Scholes prices with live Deribit option prices.

This module provides:
1. Black-Scholes call pricing
2. Deribit public API helpers
3. Calibration logic to compute pricing errors

Uses only public Deribit endpoints - no authentication required.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx


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
            delta = ticker.get("delta", None)
            
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
                )
            )
        except Exception:
            continue

    return quotes


def synthetic_iv(
    quote: OptionQuote,
    default_iv: float,
    iv_multiplier: float,
) -> float:
    """
    Compute synthetic IV for calibration.
    
    If quote.mark_iv is present and > 0:
        base_iv = quote.mark_iv / 100 (Deribit returns percentage)
    else:
        base_iv = default_iv
    
    Returns max(1e-6, base_iv * iv_multiplier).
    """
    if quote.mark_iv is not None and quote.mark_iv > 0:
        base_iv = quote.mark_iv / 100.0
    else:
        base_iv = default_iv

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


def run_calibration(
    underlying: str,
    min_dte: float,
    max_dte: float,
    iv_multiplier: float = 1.0,
    default_iv: float = 0.6,
    r: float = 0.0,
    max_samples: int = 80,
) -> CalibrationResult:
    """
    Compare synthetic BS prices vs Deribit mark prices for CALLs.
    
    1. Fetch current spot from Deribit
    2. Fetch call chain via get_call_chain()
    3. Sub-sample to at most max_samples quotes
    4. For each quote:
       - Compute t_years = dte_days / 365.0
       - Compute sigma = synthetic_iv(...)
       - Compute synthetic_price via Black-Scholes
       - diff = synthetic_price - mark_price
       - diff_pct = diff / mark_price * 100
    5. Compute:
       - mae_pct = mean(|diff_pct|)
       - bias_pct = mean(diff_pct)
    6. Return CalibrationResult with rows and aggregates
    """
    now = datetime.now(timezone.utc)
    spot = get_index_price(underlying)
    
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
        )

    if len(quotes) > max_samples:
        step = max(1, len(quotes) // max_samples)
        quotes = quotes[::step]

    rows: List[CalibrationRow] = []
    errors_pct: List[float] = []

    for q in quotes:
        t_years = max(0.0001, q.dte_days / 365.0)
        sigma = synthetic_iv(q, default_iv=default_iv, iv_multiplier=iv_multiplier)

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
    )
