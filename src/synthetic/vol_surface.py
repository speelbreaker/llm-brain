"""
Vol surface configuration for synthetic universe.

Provides DTE-band-specific IV multipliers and skew configuration
that can be populated from calibration results.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class DteBand(BaseModel):
    """A single DTE band with its own IV multiplier."""
    name: str = Field(..., description="Band name (e.g., 'weekly', 'monthly')")
    min_dte: float = Field(..., ge=0, description="Minimum DTE for this band")
    max_dte: float = Field(..., description="Maximum DTE for this band")
    iv_multiplier: float = Field(default=1.0, description="IV multiplier for this band")


class SkewTemplate(BaseModel):
    """Skew configuration with anchor ratios."""
    enabled: bool = Field(default=True, description="Whether skew is enabled")
    min_dte: float = Field(default=3.0, description="Min DTE for skew")
    max_dte: float = Field(default=14.0, description="Max DTE for skew")
    anchor_ratios: Dict[str, float] = Field(
        default_factory=lambda: {"0.15": 1.1, "0.25": 1.05, "0.35": 1.02},
        description="Anchor ratios by delta"
    )
    mode: Literal["put_heavy", "call_heavy", "neutral"] = Field(
        default="put_heavy",
        description="Skew mode affecting anchor ratio interpretation"
    )
    scale: float = Field(default=1.0, description="Scale factor for anchor ratios")


class VolSurfaceConfig(BaseModel):
    """
    Complete vol surface configuration for synthetic pricing.
    
    Can be populated from calibration results or manually configured.
    """
    iv_mode: Literal["fixed", "rv_window"] = Field(default="rv_window")
    rv_window_days: int = Field(default=7, ge=1, le=365)
    iv_multiplier: float = Field(default=1.0, ge=0.1, le=3.0)
    
    dte_bands: Optional[List[DteBand]] = Field(default=None, description="DTE-specific IV multipliers")
    skew: Optional[SkewTemplate] = Field(default=None, description="Skew configuration")
    
    regime_override: bool = Field(default=True, description="Allow regime to override IV multiplier")
    vrp_offset_enabled: bool = Field(default=False, description="Add VRP as vol-point offset")
    
    def get_iv_multiplier_for_dte(self, dte: float) -> float:
        """
        Get the appropriate IV multiplier for a given DTE.
        
        Priority:
        1. If DTE falls within a dte_band, use that band's multiplier
        2. Otherwise, use the global iv_multiplier
        """
        if self.dte_bands:
            for band in self.dte_bands:
                if band.min_dte <= dte <= band.max_dte:
                    return band.iv_multiplier
        return self.iv_multiplier
    
    def get_skew_anchor_ratio(self, abs_delta: float) -> float:
        """
        Get skew anchor ratio for a given absolute delta.
        
        Uses a smooth parametric curve fit (quadratic in log-space) through
        anchor points for >=3 anchors. Falls back to linear interpolation
        for <3 anchors.
        
        The curve fit ensures smooth, monotone-ish behavior without kinks.
        
        Returns:
            Skew ratio multiplier (clamped to [0.5, 2.0] for safety)
        """
        if not self.skew or not self.skew.enabled:
            return 1.0
        
        anchors = self.skew.anchor_ratios
        if not anchors:
            return 1.0
        
        sorted_deltas = sorted([float(d) for d in anchors.keys()])
        ratios = [anchors.get(f"{d:.2f}", 1.0) for d in sorted_deltas]
        
        if len(sorted_deltas) >= 3:
            ratio = self._compute_quadratic_log_ratio(abs_delta, sorted_deltas, ratios)
        else:
            ratio = self._linear_interpolate_ratio(abs_delta, sorted_deltas, ratios)
        
        if self.skew.mode == "neutral":
            ratio = 1.0 + (ratio - 1.0) * 0.5
        elif self.skew.mode == "call_heavy":
            ratio = 2.0 - ratio
        
        ratio *= self.skew.scale
        
        return max(0.5, min(ratio, 2.0))
    
    def _linear_interpolate_ratio(
        self, 
        abs_delta: float, 
        sorted_deltas: List[float], 
        ratios: List[float]
    ) -> float:
        """Linear interpolation fallback for <3 anchors."""
        if abs_delta <= sorted_deltas[0]:
            return ratios[0]
        elif abs_delta >= sorted_deltas[-1]:
            return ratios[-1]
        
        for i in range(len(sorted_deltas) - 1):
            if sorted_deltas[i] <= abs_delta <= sorted_deltas[i + 1]:
                lower_delta = sorted_deltas[i]
                upper_delta = sorted_deltas[i + 1]
                lower_ratio = ratios[i]
                upper_ratio = ratios[i + 1]
                
                if upper_delta == lower_delta:
                    return lower_ratio
                
                t = (abs_delta - lower_delta) / (upper_delta - lower_delta)
                return lower_ratio + t * (upper_ratio - lower_ratio)
        
        return 1.0
    
    def _compute_quadratic_log_ratio(
        self,
        abs_delta: float,
        sorted_deltas: List[float],
        ratios: List[float],
    ) -> float:
        """
        Fit a quadratic curve in log-space through anchor points.
        
        Uses least squares to fit: log(ratio) = a + b*x + c*x^2
        where x = abs_delta - 0.25 (centered around typical OTM delta).
        
        Then ratio = exp(log_ratio).
        """
        import numpy as np
        
        center = 0.25
        x = np.array([d - center for d in sorted_deltas])
        y = np.array([np.log(max(r, 0.01)) for r in ratios])
        
        X = np.column_stack([np.ones_like(x), x, x**2])
        coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        a, b, c = coeffs
        
        x_query = abs_delta - center
        log_ratio = a + b * x_query + c * x_query**2
        
        log_ratio = max(-1.0, min(log_ratio, 1.0))
        
        return np.exp(log_ratio)
    
    @classmethod
    def from_calibration(cls, recommended_vol_surface: Dict[str, Any]) -> "VolSurfaceConfig":
        """Create VolSurfaceConfig from calibration recommended_vol_surface dict."""
        dte_bands = None
        if "dte_bands" in recommended_vol_surface and recommended_vol_surface["dte_bands"]:
            dte_bands = [DteBand(**b) for b in recommended_vol_surface["dte_bands"]]
        
        skew = None
        if "skew" in recommended_vol_surface and recommended_vol_surface["skew"]:
            skew = SkewTemplate(**recommended_vol_surface["skew"])
        
        return cls(
            iv_mode=recommended_vol_surface.get("iv_mode", "rv_window"),
            rv_window_days=recommended_vol_surface.get("rv_window_days", 7),
            iv_multiplier=recommended_vol_surface.get("iv_multiplier", 1.0),
            dte_bands=dte_bands,
            skew=skew,
        )


_runtime_vol_surface: Optional[VolSurfaceConfig] = None


def _get_default_dte_bands() -> List[DteBand]:
    """
    Get default DTE bands for term-structure realism.
    
    Rationale:
    - Weekly options (3-10 DTE): Higher gamma, faster decay, use 1.0x multiplier
    - Monthly options (20-40 DTE): Lower gamma, smoother IV, use 1.10x multiplier
      (monthly IV tends to be 5-15% higher than weekly in crypto due to event risk)
    - Outside bands: Fall back to global iv_multiplier
    
    These defaults provide realistic term structure out-of-the-box without
    requiring calibration. Calibration results override these defaults.
    """
    return [
        DteBand(name="weekly", min_dte=3.0, max_dte=10.0, iv_multiplier=1.00),
        DteBand(name="monthly", min_dte=20.0, max_dte=40.0, iv_multiplier=1.10),
    ]


def get_vol_surface_config() -> VolSurfaceConfig:
    """
    Get the current runtime vol surface configuration.
    
    If no runtime config is set, returns a default config with sensible
    DTE bands for term-structure realism. Calibration/runtime overrides
    replace these defaults entirely.
    """
    global _runtime_vol_surface
    if _runtime_vol_surface is None:
        _runtime_vol_surface = VolSurfaceConfig(
            dte_bands=_get_default_dte_bands()
        )
    return _runtime_vol_surface


def set_vol_surface_config(config: VolSurfaceConfig) -> None:
    """Set the runtime vol surface configuration."""
    global _runtime_vol_surface
    _runtime_vol_surface = config


def reset_vol_surface_config() -> None:
    """
    Reset the runtime vol surface config to None.
    
    Next call to get_vol_surface_config() will return defaults.
    Useful for testing.
    """
    global _runtime_vol_surface
    _runtime_vol_surface = None


def update_vol_surface_from_calibration(recommended: Dict[str, Any]) -> VolSurfaceConfig:
    """
    Update the runtime vol surface config from calibration results.
    
    Returns the new config.
    """
    config = VolSurfaceConfig.from_calibration(recommended)
    set_vol_surface_config(config)
    return config
