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
        
        Interpolates between defined anchors.
        """
        if not self.skew or not self.skew.enabled:
            return 1.0
        
        anchors = self.skew.anchor_ratios
        if not anchors:
            return 1.0
        
        sorted_deltas = sorted([float(d) for d in anchors.keys()])
        
        if abs_delta <= sorted_deltas[0]:
            ratio = anchors.get(f"{sorted_deltas[0]:.2f}", 1.0)
        elif abs_delta >= sorted_deltas[-1]:
            ratio = anchors.get(f"{sorted_deltas[-1]:.2f}", 1.0)
        else:
            lower_delta = sorted_deltas[0]
            upper_delta = sorted_deltas[-1]
            for i, d in enumerate(sorted_deltas[:-1]):
                if d <= abs_delta <= sorted_deltas[i + 1]:
                    lower_delta = d
                    upper_delta = sorted_deltas[i + 1]
                    break
            
            lower_ratio = anchors.get(f"{lower_delta:.2f}", 1.0)
            upper_ratio = anchors.get(f"{upper_delta:.2f}", 1.0)
            
            if upper_delta == lower_delta:
                ratio = lower_ratio
            else:
                t = (abs_delta - lower_delta) / (upper_delta - lower_delta)
                ratio = lower_ratio + t * (upper_ratio - lower_ratio)
        
        if self.skew.mode == "neutral":
            ratio = 1.0 + (ratio - 1.0) * 0.5
        elif self.skew.mode == "call_heavy":
            ratio = 2.0 - ratio
        
        ratio *= self.skew.scale
        
        return ratio
    
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


def get_vol_surface_config() -> VolSurfaceConfig:
    """Get the current runtime vol surface configuration."""
    global _runtime_vol_surface
    if _runtime_vol_surface is None:
        _runtime_vol_surface = VolSurfaceConfig()
    return _runtime_vol_surface


def set_vol_surface_config(config: VolSurfaceConfig) -> None:
    """Set the runtime vol surface configuration."""
    global _runtime_vol_surface
    _runtime_vol_surface = config


def update_vol_surface_from_calibration(recommended: Dict[str, Any]) -> VolSurfaceConfig:
    """
    Update the runtime vol surface config from calibration results.
    
    Returns the new config.
    """
    config = VolSurfaceConfig.from_calibration(recommended)
    set_vol_surface_config(config)
    return config
