"""
Black-Scholes pricing helpers for synthetic option pricing mode.

Provides self-consistent option pricing for historical backtests without
requiring actual historical option data from exchanges.

Supports regime-aware IV dynamics using Greg-sensor cluster regimes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from math import log, sqrt, exp, erf
from typing import List, Optional, Tuple, Dict, TYPE_CHECKING

import numpy as np

from .types import CallSimulationConfig, OptionSnapshot

if TYPE_CHECKING:
    from src.synthetic.regimes import RegimeParams, RegimeModel


@dataclass
class RegimeState:
    """
    Tracks the current state of the regime-aware IV dynamics.
    
    Used to maintain continuity between simulation steps.
    """
    regime_id: int = 0
    regime: Optional["RegimeParams"] = None
    iv_atm: float = 50.0
    skew_state: float = 0.0
    rv_30d: float = 50.0
    rng: Optional[np.random.Generator] = field(default=None, repr=False)
    
    def __post_init__(self):
        if self.rng is None:
            self.rng = np.random.default_rng()


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


_volatility_history_cache: Dict[str, "pd.DataFrame"] = {}


def get_volatility_history_date_range(
    underlying: str,
) -> Optional[Tuple[datetime, datetime]]:
    """
    Get the available date range for historical volatility data.
    
    Args:
        underlying: Currency symbol (BTC, ETH)
        
    Returns:
        Tuple of (start_date, end_date) or None if no data available
    """
    df = _load_volatility_history_for_replay(underlying)
    
    if df is None or df.empty:
        return None
    
    return (df.index.min().to_pydatetime(), df.index.max().to_pydatetime())


def _load_volatility_history_for_replay(underlying: str) -> Optional["pd.DataFrame"]:
    """
    Load historical volatility data for replay mode with caching.
    
    Returns:
        DataFrame with timestamp (as index), iv_30d, rv_30d, vrp_30d
    """
    import pandas as pd
    from pathlib import Path
    
    cache_key = underlying.upper()
    
    if cache_key in _volatility_history_cache:
        return _volatility_history_cache[cache_key]
    
    filepath = Path("data/volatility_history") / f"{underlying.lower()}_volatility_history.parquet"
    
    if not filepath.exists():
        return None
    
    try:
        df = pd.read_parquet(filepath)
        df = df.set_index("timestamp").sort_index()
        _volatility_history_cache[cache_key] = df
        return df
    except Exception:
        return None


def get_historical_iv_at_time(
    underlying: str,
    as_of: datetime,
) -> Optional[float]:
    """
    Get historical IV at a specific time using nearest neighbor lookup.
    
    Args:
        underlying: Currency symbol (BTC, ETH)
        as_of: The datetime to get IV for
        
    Returns:
        IV as decimal (e.g., 0.50 for 50%), or None if not available
        Returns None if as_of is outside the available data range
    """
    import pandas as pd
    
    df = _load_volatility_history_for_replay(underlying)
    
    if df is None or df.empty:
        return None
    
    as_of_tz = as_of
    if as_of.tzinfo is None:
        from datetime import timezone
        as_of_tz = as_of.replace(tzinfo=timezone.utc)
    
    if as_of_tz < df.index.min() or as_of_tz > df.index.max():
        return None
    
    idx = df.index.get_indexer([as_of_tz], method="nearest")[0]
    iv_pct = df["iv_30d"].iloc[idx]
    
    return float(iv_pct) / 100.0


def get_historical_rv_at_time(
    underlying: str,
    as_of: datetime,
) -> Optional[float]:
    """
    Get historical RV at a specific time.
    
    Args:
        underlying: Currency symbol (BTC, ETH)
        as_of: The datetime to get RV for
        
    Returns:
        RV as decimal (e.g., 0.50 for 50%), or None if not available
        Returns None if as_of is outside the available data range
    """
    import pandas as pd
    
    df = _load_volatility_history_for_replay(underlying)
    
    if df is None or df.empty:
        return None
    
    as_of_tz = as_of
    if as_of.tzinfo is None:
        from datetime import timezone
        as_of_tz = as_of.replace(tzinfo=timezone.utc)
    
    if as_of_tz < df.index.min() or as_of_tz > df.index.max():
        return None
    
    idx = df.index.get_indexer([as_of_tz], method="nearest")[0]
    rv_pct = df["rv_30d"].iloc[idx]
    
    return float(rv_pct) / 100.0 if pd.notna(rv_pct) else None


def get_synthetic_iv(
    config: CallSimulationConfig,
    spot_history: List[Tuple[datetime, float]],
    as_of: datetime,
    regime_state: Optional["RegimeState"] = None,
) -> float:
    """
    Get the implied volatility to use for synthetic pricing.
    
    Args:
        config: Simulation configuration
        spot_history: List of (datetime, price) tuples
        as_of: Current decision time
        regime_state: Optional regime state for AR(1) dynamics
    
    Returns:
        Implied volatility to use for Black-Scholes pricing
    """
    if config.synthetic_iv_mode == "fixed":
        return config.synthetic_fixed_iv
    
    if config.synthetic_iv_mode == "historical_replay":
        historical_iv = get_historical_iv_at_time(config.underlying, as_of)
        if historical_iv is not None:
            return historical_iv * config.synthetic_iv_multiplier
    
    rv = compute_realized_volatility(
        spot_history,
        as_of,
        config.synthetic_rv_window_days
    )
    
    if regime_state is not None and regime_state.regime is not None:
        return regime_state.iv_atm / 100.0
    
    return rv * config.synthetic_iv_multiplier


def get_atm_iv_from_chain(
    option_chain: List[OptionSnapshot],
    spot: float,
    target_dte: int,
    dte_tolerance: int = 2,
    as_of: Optional[datetime] = None,
) -> Optional[float]:
    """
    Extract ATM IV from an option chain for a specific DTE band.
    
    Finds the near-ATM call (delta closest to 0.5) within the DTE range
    and returns its IV.
    
    Args:
        option_chain: List of OptionSnapshot objects with iv and delta
        spot: Current spot price
        target_dte: Target days to expiry
        dte_tolerance: Tolerance around target DTE
        as_of: Reference timestamp for DTE calculation (uses now if None)
        
    Returns:
        ATM IV as decimal (e.g., 0.50 for 50%), or None if not found
    """
    from datetime import timezone
    
    if not option_chain:
        return None
    
    # Filter to calls in the DTE range
    candidates = []
    # Use provided timestamp or fall back to now (for live trading)
    reference_time = as_of if as_of is not None else datetime.now(timezone.utc)
    if reference_time.tzinfo is None:
        reference_time = reference_time.replace(tzinfo=timezone.utc)
    
    for opt in option_chain:
        if opt.kind != "call":
            continue
        if opt.iv is None or opt.delta is None:
            continue
        
        dte = (opt.expiry - reference_time).total_seconds() / (24 * 3600)
        if abs(dte - target_dte) <= dte_tolerance:
            candidates.append((opt, abs(opt.delta - 0.5)))
    
    if not candidates:
        # Fall back to any call with valid IV
        for opt in option_chain:
            if opt.kind == "call" and opt.iv is not None and opt.delta is not None:
                candidates.append((opt, abs(opt.delta - 0.5)))
    
    if not candidates:
        return None
    
    # Find closest to ATM (delta = 0.5)
    candidates.sort(key=lambda x: x[1])
    best_opt = candidates[0][0]
    
    return best_opt.iv


def get_sigma_for_option(
    config: CallSimulationConfig,
    spot_history: List[Tuple[datetime, float]],
    as_of: datetime,
    option_chain: Optional[List[OptionSnapshot]] = None,
    option_mark_iv: Optional[float] = None,
    abs_delta: Optional[float] = None,
    regime_state: Optional[RegimeState] = None,
) -> float:
    """
    Get the sigma (IV) to use for option pricing based on sigma_mode.
    
    This is the main entry point for hybrid synthetic mode sigma selection.
    
    Args:
        config: Simulation configuration with sigma_mode
        spot_history: List of (datetime, price) tuples for RV calculation
        as_of: Current decision time
        option_chain: Optional option chain for atm_iv_x_multiplier mode
        option_mark_iv: Optional mark IV for mark_iv_x_multiplier mode
        abs_delta: Optional absolute delta for skew adjustment
        regime_state: Optional regime state for AR(1) dynamics
        
    Returns:
        Sigma (IV) as decimal for Black-Scholes pricing
    """
    from src.synthetic_skew import get_skew_factor
    
    sigma_mode = getattr(config, 'sigma_mode', 'rv_x_multiplier')
    
    if sigma_mode == "mark_iv_x_multiplier":
        # Use the option's live mark_iv directly, bypass RV completely
        if option_mark_iv is not None and option_mark_iv > 0:
            base_iv = option_mark_iv
            # Optionally scale by multiplier for stress scenarios
            scaled_iv = base_iv * config.synthetic_iv_multiplier
            return max(0.01, min(scaled_iv, 5.0))
        else:
            # Fall back to rv_x_multiplier if mark_iv not available
            sigma_mode = "rv_x_multiplier"
    
    if sigma_mode == "atm_iv_x_multiplier":
        # Pull ATM IV from chain and use as base, apply skew
        atm_iv = None
        if option_chain:
            atm_iv = get_atm_iv_from_chain(
                option_chain,
                spot_history[-1][1] if spot_history else 0,
                config.target_dte,
                config.dte_tolerance,
                as_of=as_of,
            )
        
        if atm_iv is not None:
            base_iv = atm_iv * config.synthetic_iv_multiplier
            
            # Apply skew if delta is provided
            if abs_delta is not None:
                skew_factor = get_skew_factor(
                    underlying=config.underlying,
                    option_type="call",
                    abs_delta=abs_delta,
                    skew_enabled=True,
                    min_dte=float(config.min_dte),
                    max_dte=float(config.max_dte),
                )
                base_iv = base_iv * skew_factor
            
            return max(0.01, min(base_iv, 5.0))
        else:
            # Fall back to rv_x_multiplier if ATM IV not available
            sigma_mode = "rv_x_multiplier"
    
    # Default: rv_x_multiplier - use existing behavior
    rv = compute_realized_volatility(
        spot_history,
        as_of,
        config.synthetic_rv_window_days
    )
    
    # Check for regime state
    if regime_state is not None and regime_state.regime is not None:
        base_iv = regime_state.iv_atm / 100.0
    else:
        base_iv = rv * config.synthetic_iv_multiplier
    
    # Apply skew if delta is provided
    if abs_delta is not None:
        skew_factor = get_skew_factor(
            underlying=config.underlying,
            option_type="call",
            abs_delta=abs_delta,
            skew_enabled=True,
            min_dte=float(config.min_dte),
            max_dte=float(config.max_dte),
        )
        base_iv = base_iv * skew_factor
    
    return max(0.01, min(base_iv, 5.0))


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


def compute_synthetic_iv_with_skew(
    underlying: str,
    option_type: str,
    abs_delta: float,
    rv_annualized: float,
    iv_multiplier: float = 1.0,
    skew_enabled: bool = True,
    skew_min_dte: float = 3.0,
    skew_max_dte: float = 14.0,
) -> float:
    """
    Compute synthetic annualized IV for the synthetic universe with skew.

    Formula:
        base_iv = rv_annualized * iv_multiplier
        skew_factor = get_skew_factor(underlying, option_type, abs_delta, ...)
        sigma = base_iv * skew_factor

    Args:
        underlying: "BTC" or "ETH"
        option_type: "call" or "put"
        abs_delta: Absolute delta of the option (0 to 1)
        rv_annualized: Realized volatility (annualized)
        iv_multiplier: Multiplier for the base IV
        skew_enabled: Whether to apply skew
        skew_min_dte: Min DTE for skew estimation
        skew_max_dte: Max DTE for skew estimation

    Returns:
        sigma (annualized volatility) to plug into Black-Scholes
    """
    from src.synthetic_skew import get_skew_factor

    base_iv = max(1e-6, rv_annualized * iv_multiplier)
    
    skew_factor = get_skew_factor(
        underlying=underlying,
        option_type=option_type,
        abs_delta=abs_delta,
        skew_enabled=skew_enabled,
        min_dte=skew_min_dte,
        max_dte=skew_max_dte,
    )
    
    return max(1e-6, base_iv * skew_factor)


def create_regime_state(
    underlying: str,
    initial_rv: float = 50.0,
    seed: Optional[int] = None,
) -> RegimeState:
    """
    Create a new RegimeState with loaded regime model.
    
    Args:
        underlying: Asset symbol (BTC, ETH)
        initial_rv: Initial realized volatility (percentage)
        seed: Random seed for reproducibility
        
    Returns:
        Initialized RegimeState
    """
    from src.synthetic.regimes import (
        load_regime_model,
        get_default_regimes,
        get_default_transition_matrix,
    )
    
    rng = np.random.default_rng(seed)
    
    model = load_regime_model(underlying)
    
    if model is not None:
        regimes = model.regimes
        start_regime = rng.choice(list(regimes.keys()))
        regime = regimes[start_regime]
    else:
        regimes = get_default_regimes()
        start_regime = 0
        regime = regimes[start_regime]
    
    initial_iv = initial_rv * (1 + regime.mu_vrp_30d)
    
    return RegimeState(
        regime_id=start_regime,
        regime=regime,
        iv_atm=initial_iv,
        skew_state=0.0,
        rv_30d=initial_rv,
        rng=rng,
    )


def step_regime_state(
    state: RegimeState,
    rv_30d: float,
    underlying: str,
) -> RegimeState:
    """
    Advance the regime state by one time step.
    
    Uses AR(1) IV dynamics and optionally samples regime transitions.
    
    Args:
        state: Current regime state
        rv_30d: Current realized volatility (percentage)
        underlying: Asset symbol for loading transition matrix
        
    Returns:
        Updated RegimeState
    """
    from src.synthetic.regimes import (
        load_regime_model,
        get_default_regimes,
        get_default_transition_matrix,
        evolve_iv_and_skew,
        sample_regime_path,
    )
    
    model = load_regime_model(underlying)
    
    if model is not None:
        regimes = model.regimes
        transition_matrix = model.transition_matrix
    else:
        regimes = get_default_regimes()
        transition_matrix = get_default_transition_matrix(len(regimes))
    
    if state.rng is None:
        state.rng = np.random.default_rng()
    
    n_regimes = len(regimes)
    probs = transition_matrix[state.regime_id]
    probs = probs / probs.sum()
    new_regime_id = state.rng.choice(n_regimes, p=probs)
    new_regime = regimes.get(new_regime_id, state.regime)
    
    if new_regime is None:
        new_regime = regimes.get(0)
    
    iv_atm, skew_state = evolve_iv_and_skew(
        iv_atm_prev=state.iv_atm,
        rv_30d_t=rv_30d,
        regime=new_regime,
        rng=state.rng,
    )
    
    return RegimeState(
        regime_id=new_regime_id,
        regime=new_regime,
        iv_atm=iv_atm,
        skew_state=skew_state,
        rv_30d=rv_30d,
        rng=state.rng,
    )


def get_regime_aware_iv_for_delta(
    state: RegimeState,
    delta: float,
) -> float:
    """
    Get IV for a specific delta using regime skew template.
    
    Args:
        state: Current regime state
        delta: Option delta (0 to 1 for calls)
        
    Returns:
        IV as decimal (e.g., 0.50 for 50%)
    """
    from src.synthetic.regimes import iv_for_delta
    
    if state.regime is None:
        return state.iv_atm / 100.0
    
    iv_pct = iv_for_delta(
        iv_atm_t=state.iv_atm,
        regime=state.regime,
        skew_state_t=state.skew_state,
        delta=delta,
    )
    
    return iv_pct / 100.0
