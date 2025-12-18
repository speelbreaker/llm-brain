"""
Unified Synthetic Volatility Engine.

Single canonical source for all synthetic IV calculations across:
- Backtests (historical simulations)
- Selector scans
- Hybrid pricing modes
- Live trading

Order of operations: base_iv → multiplier → skew

This engine consolidates:
- Vol surface / calibration config pathway (multipliers, DTE bands, skew anchors)
- Synthetic skew engine pathway (source controls, time-awareness)
- Regime dynamics for IV/skew evolution (optional wrapper)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Literal, Optional, Tuple, TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from src.backtest.types import CallSimulationConfig


SkewSource = Literal["none", "harvested", "live"]
SurfaceSource = Literal["calibration", "config", "none"]


class SyntheticVolConfig(BaseModel):
    """Configuration for a single synthetic vol calculation."""
    sigma_mode: Literal["rv_x_multiplier", "atm_iv_x_multiplier", "mark_iv_x_multiplier"] = "rv_x_multiplier"
    surface_source: SurfaceSource = "calibration"
    skew_source: SkewSource = "none"
    rv_window_days: int = Field(default=7, ge=1, le=365)
    
    class Config:
        extra = "forbid"


@dataclass
class SyntheticVolResult:
    """Result from synthetic vol calculation with audit trail."""
    iv: float
    atm_iv: float
    skew_ratio: float
    multiplier: float
    sigma_mode: str
    skew_source: str
    surface_source: str
    dte_days: float


class SyntheticVolEngine:
    """
    Unified engine for synthetic volatility calculations.
    
    Single source of truth for answering:
    "What IV do we use for this option at time t?"
    
    Usage:
        engine = SyntheticVolEngine()
        iv = engine.get_iv(
            underlying="BTC",
            as_of=datetime.now(timezone.utc),
            dte_days=7.0,
            abs_delta=0.25,
            option_type="call",
            spot_history=spot_history,
            config=SyntheticVolConfig(skew_source="none"),
        )
    """
    
    def __init__(self):
        pass
    
    def get_iv(
        self,
        underlying: str,
        as_of: datetime,
        dte_days: float,
        abs_delta: float,
        option_type: str = "call",
        spot_history: Optional[List[Tuple[datetime, float]]] = None,
        option_chain: Optional[list] = None,
        option_mark_iv: Optional[float] = None,
        config: Optional[SyntheticVolConfig] = None,
        sim_config: Optional["CallSimulationConfig"] = None,
    ) -> float:
        """
        Get synthetic IV for an option.
        
        Order of operations:
        1. Compute base ATM IV (RV/ATM/mark based on sigma_mode)
        2. Apply calibration multiplier (global + DTE band)
        3. Apply skew adjustment
        
        Args:
            underlying: Asset symbol (BTC, ETH)
            as_of: Reference time for calculation
            dte_days: Days to expiry
            abs_delta: Absolute delta (0-1)
            option_type: "call" or "put"
            spot_history: Price history for RV calculation
            option_chain: Option chain for ATM IV extraction
            option_mark_iv: Mark IV for mark_iv mode
            config: Synthetic vol configuration
            sim_config: Optional simulation config for legacy compat
            
        Returns:
            Synthetic IV as decimal (e.g., 0.70 for 70%)
            
        Raises:
            ValueError: If skew_source="live" for historical as_of
        """
        if config is None:
            config = SyntheticVolConfig()
        
        self._validate_no_lookahead(as_of, config.skew_source)
        
        atm_iv = self.get_atm_iv(
            underlying=underlying,
            as_of=as_of,
            dte_days=dte_days,
            spot_history=spot_history,
            option_chain=option_chain,
            option_mark_iv=option_mark_iv,
            config=config,
            sim_config=sim_config,
        )
        
        multiplier = self._get_multiplier(dte_days, config.surface_source)
        
        base_iv = atm_iv * multiplier
        
        skew_ratio = self.get_skew_anchor_ratio(
            underlying=underlying,
            as_of=as_of,
            dte_days=dte_days,
            abs_delta=abs_delta,
            option_type=option_type,
            config=config,
            sim_config=sim_config,
        )
        
        iv = base_iv * skew_ratio
        
        return max(0.01, min(iv, 5.0))
    
    def get_atm_iv(
        self,
        underlying: str,
        as_of: datetime,
        dte_days: float,
        spot_history: Optional[List[Tuple[datetime, float]]] = None,
        option_chain: Optional[list] = None,
        option_mark_iv: Optional[float] = None,
        config: Optional[SyntheticVolConfig] = None,
        sim_config: Optional["CallSimulationConfig"] = None,
    ) -> float:
        """
        Get base ATM IV before multiplier/skew.
        
        Priority based on sigma_mode:
        1. mark_iv_x_multiplier: Use option_mark_iv if provided
        2. atm_iv_x_multiplier: Extract from option chain
        3. rv_x_multiplier: Compute from spot history (fallback)
        
        Returns:
            ATM IV as decimal (before multiplier)
        """
        from src.backtest.pricing import compute_realized_volatility, get_atm_iv_from_chain
        
        if config is None:
            config = SyntheticVolConfig()
        
        sigma_mode = config.sigma_mode
        rv_window = config.rv_window_days
        if sim_config:
            rv_window = getattr(sim_config, 'synthetic_rv_window_days', rv_window)
        
        if sigma_mode == "mark_iv_x_multiplier" and option_mark_iv is not None and option_mark_iv > 0:
            return option_mark_iv
        
        if sigma_mode == "atm_iv_x_multiplier" and option_chain:
            target_dte = dte_days
            dte_tolerance = 2
            if sim_config:
                target_dte = getattr(sim_config, 'target_dte', dte_days)
                dte_tolerance = getattr(sim_config, 'dte_tolerance', 2)
            
            spot = spot_history[-1][1] if spot_history else 0
            atm_iv = get_atm_iv_from_chain(
                option_chain,
                spot,
                int(target_dte),
                dte_tolerance,
                as_of=as_of,
            )
            if atm_iv is not None:
                return atm_iv
        
        if spot_history:
            rv = compute_realized_volatility(spot_history, as_of, rv_window)
            return rv
        
        return 0.50
    
    def get_skew_anchor_ratio(
        self,
        underlying: str,
        as_of: datetime,
        dte_days: float,
        abs_delta: float,
        option_type: str = "call",
        config: Optional[SyntheticVolConfig] = None,
        sim_config: Optional["CallSimulationConfig"] = None,
    ) -> float:
        """
        Get skew anchor ratio for delta adjustment.
        
        Priority:
        1. Calibrated skew anchors (if surface_source=calibration and DTE in range)
        2. Synthetic skew engine (based on skew_source: none/harvested/live)
        
        Args:
            underlying: Asset symbol
            as_of: Reference time
            dte_days: Days to expiry
            abs_delta: Absolute delta (0-1)
            option_type: "call" or "put"
            config: Synthetic vol config
            sim_config: Optional simulation config
            
        Returns:
            Skew ratio multiplier (typically 0.8 - 1.2)
        """
        from src.synthetic.vol_surface import get_vol_surface_config
        from src.synthetic_skew import get_skew_factor
        
        if config is None:
            config = SyntheticVolConfig()
        
        if config.surface_source in ("calibration", "config"):
            vs = get_vol_surface_config()
            if vs.skew is not None and vs.skew.enabled:
                if vs.skew.min_dte <= dte_days <= vs.skew.max_dte:
                    return vs.get_skew_anchor_ratio(abs_delta)
        
        if config.skew_source == "none":
            return 1.0
        
        min_dte = 3.0
        max_dte = 14.0
        if sim_config:
            min_dte = float(getattr(sim_config, 'min_dte', 3))
            max_dte = float(getattr(sim_config, 'max_dte', 14))
        
        return get_skew_factor(
            underlying=underlying,
            option_type=option_type,
            abs_delta=abs_delta,
            skew_enabled=True,
            min_dte=min_dte,
            max_dte=max_dte,
            as_of=as_of,
            source=config.skew_source,
        )
    
    def get_iv_with_details(
        self,
        underlying: str,
        as_of: datetime,
        dte_days: float,
        abs_delta: float,
        option_type: str = "call",
        spot_history: Optional[List[Tuple[datetime, float]]] = None,
        option_chain: Optional[list] = None,
        option_mark_iv: Optional[float] = None,
        config: Optional[SyntheticVolConfig] = None,
        sim_config: Optional["CallSimulationConfig"] = None,
    ) -> SyntheticVolResult:
        """
        Get IV with full audit trail for debugging/display.
        
        Returns a SyntheticVolResult with all intermediate values.
        """
        if config is None:
            config = SyntheticVolConfig()
        
        self._validate_no_lookahead(as_of, config.skew_source)
        
        atm_iv = self.get_atm_iv(
            underlying=underlying,
            as_of=as_of,
            dte_days=dte_days,
            spot_history=spot_history,
            option_chain=option_chain,
            option_mark_iv=option_mark_iv,
            config=config,
            sim_config=sim_config,
        )
        
        multiplier = self._get_multiplier(dte_days, config.surface_source)
        
        skew_ratio = self.get_skew_anchor_ratio(
            underlying=underlying,
            as_of=as_of,
            dte_days=dte_days,
            abs_delta=abs_delta,
            option_type=option_type,
            config=config,
            sim_config=sim_config,
        )
        
        iv = max(0.01, min(atm_iv * multiplier * skew_ratio, 5.0))
        
        return SyntheticVolResult(
            iv=iv,
            atm_iv=atm_iv,
            skew_ratio=skew_ratio,
            multiplier=multiplier,
            sigma_mode=config.sigma_mode,
            skew_source=config.skew_source,
            surface_source=config.surface_source,
            dte_days=dte_days,
        )
    
    def _get_multiplier(self, dte_days: float, surface_source: SurfaceSource) -> float:
        """Get IV multiplier from calibration or default."""
        if surface_source in ("calibration", "config"):
            from src.synthetic.vol_surface import get_vol_surface_config
            vs = get_vol_surface_config()
            return vs.get_iv_multiplier_for_dte(dte_days)
        return 1.0
    
    def _validate_no_lookahead(self, as_of: datetime, skew_source: SkewSource) -> None:
        """
        Prevent look-ahead bias by blocking live skew for historical dates.
        
        Raises ValueError if as_of is in the past and skew_source is "live".
        """
        if skew_source != "live":
            return
        
        now = datetime.now(timezone.utc)
        if as_of.tzinfo is None:
            as_of = as_of.replace(tzinfo=timezone.utc)
        
        hours_ago = (now - as_of).total_seconds() / 3600
        if hours_ago > 1:
            raise ValueError(
                f"Look-ahead bias detected: skew_source='live' not allowed for "
                f"historical as_of={as_of.isoformat()} (more than 1 hour ago). "
                f"Use skew_source='harvested' or 'none' for historical backtests."
            )


_engine_instance: Optional[SyntheticVolEngine] = None


def get_synthetic_vol_engine() -> SyntheticVolEngine:
    """Get singleton engine instance."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = SyntheticVolEngine()
    return _engine_instance


def get_iv(
    underlying: str,
    as_of: datetime,
    dte_days: float,
    abs_delta: float,
    option_type: str = "call",
    spot_history: Optional[List[Tuple[datetime, float]]] = None,
    option_chain: Optional[list] = None,
    option_mark_iv: Optional[float] = None,
    skew_source: SkewSource = "none",
    surface_source: SurfaceSource = "calibration",
    sigma_mode: str = "rv_x_multiplier",
) -> float:
    """
    Convenience function for getting IV through the unified engine.
    
    This is the canonical way to get synthetic IV across the codebase.
    """
    engine = get_synthetic_vol_engine()
    config = SyntheticVolConfig(
        sigma_mode=sigma_mode,
        surface_source=surface_source,
        skew_source=skew_source,
    )
    return engine.get_iv(
        underlying=underlying,
        as_of=as_of,
        dte_days=dte_days,
        abs_delta=abs_delta,
        option_type=option_type,
        spot_history=spot_history,
        option_chain=option_chain,
        option_mark_iv=option_mark_iv,
        config=config,
    )
