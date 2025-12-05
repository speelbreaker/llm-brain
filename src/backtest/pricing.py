"""
Black-Scholes pricing helpers for synthetic option pricing mode.

Provides self-consistent option pricing for historical backtests without
requiring actual historical option data from exchanges.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from math import log, sqrt, exp, erf
from typing import List, Optional, Tuple

from .types import CallSimulationConfig


def _norm_cdf(x: float) -> float:
    """Standard normal cumulative distribution function."""
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def bs_call_price(
    spot: float,
    strike: float,
    t_years: float,
    sigma: float,
    r: float = 0.0
) -> float:
    """
    Black-Scholes European call option price.
    
    Args:
        spot: Underlying price
        strike: Option strike price
        t_years: Time to expiry in years
        sigma: Annualized volatility (e.g., 0.70 for 70%)
        r: Risk-free rate (default 0)
    
    Returns:
        Call option price in same units as spot
    """
    if spot <= 0 or strike <= 0 or t_years <= 0 or sigma <= 0:
        return 0.0
    
    d1 = (log(spot / strike) + (r + 0.5 * sigma * sigma) * t_years) / (sigma * sqrt(t_years))
    d2 = d1 - sigma * sqrt(t_years)
    
    return spot * _norm_cdf(d1) - strike * exp(-r * t_years) * _norm_cdf(d2)


def bs_call_delta(
    spot: float,
    strike: float,
    t_years: float,
    sigma: float,
    r: float = 0.0
) -> float:
    """
    Black-Scholes call delta.
    
    Args:
        spot: Underlying price
        strike: Option strike price
        t_years: Time to expiry in years
        sigma: Annualized volatility
        r: Risk-free rate (default 0)
    
    Returns:
        Call delta between 0 and 1
    """
    if spot <= 0 or strike <= 0 or t_years <= 0 or sigma <= 0:
        return 0.0
    
    d1 = (log(spot / strike) + (r + 0.5 * sigma * sigma) * t_years) / (sigma * sqrt(t_years))
    return _norm_cdf(d1)


def compute_realized_volatility(
    prices: List[Tuple[datetime, float]],
    as_of: datetime,
    window_days: int = 30
) -> float:
    """
    Compute annualized realized volatility from historical prices.
    
    Args:
        prices: List of (datetime, price) tuples, sorted by datetime
        as_of: The reference datetime
        window_days: Number of days to look back
    
    Returns:
        Annualized realized volatility (e.g., 0.70 for 70%)
    """
    if len(prices) < 2:
        return 0.70
    
    cutoff = as_of - timedelta(days=window_days)
    window_prices = [(t, p) for t, p in prices if cutoff <= t <= as_of]
    
    if len(window_prices) < 2:
        return 0.70
    
    window_prices.sort(key=lambda x: x[0])
    
    log_returns = []
    for i in range(1, len(window_prices)):
        p1 = window_prices[i - 1][1]
        p2 = window_prices[i][1]
        if p1 > 0 and p2 > 0:
            log_returns.append(log(p2 / p1))
    
    if len(log_returns) < 2:
        return 0.70
    
    mean_return = sum(log_returns) / len(log_returns)
    variance = sum((r - mean_return) ** 2 for r in log_returns) / (len(log_returns) - 1)
    daily_vol = sqrt(variance)
    
    annualized_vol = daily_vol * sqrt(365)
    
    return max(0.10, min(annualized_vol, 2.0))


def get_synthetic_iv(
    config: CallSimulationConfig,
    spot_history: List[Tuple[datetime, float]],
    as_of: datetime
) -> float:
    """
    Get the implied volatility to use for synthetic pricing.
    
    Args:
        config: Simulation configuration
        spot_history: List of (datetime, price) tuples
        as_of: Current decision time
    
    Returns:
        Implied volatility to use for Black-Scholes pricing
    """
    if config.synthetic_iv_mode == "fixed":
        return config.synthetic_fixed_iv
    
    rv = compute_realized_volatility(
        spot_history,
        as_of,
        config.synthetic_rv_window_days
    )
    return rv * config.synthetic_iv_multiplier


def price_option_synthetic(
    spot: float,
    strike: float,
    expiry: datetime,
    as_of: datetime,
    sigma: float,
    r: float = 0.0
) -> Tuple[float, float]:
    """
    Price a call option using synthetic Black-Scholes model.
    
    Args:
        spot: Current underlying price
        strike: Option strike price
        expiry: Option expiry datetime
        as_of: Current datetime
        sigma: Implied volatility
        r: Risk-free rate
    
    Returns:
        Tuple of (price, delta)
    """
    t_years = max((expiry - as_of).total_seconds() / (365.0 * 24 * 3600), 1e-6)
    
    price = bs_call_price(spot, strike, t_years, sigma, r)
    delta = bs_call_delta(spot, strike, t_years, sigma, r)
    
    return price, delta
