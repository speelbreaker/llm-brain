"""
Extended calibration configuration models.

This module provides Pydantic models for the enhanced calibration system with:
- Multi-DTE bands
- DTE×delta buckets
- Liquidity filters
- Skew fitting
- Harvest data source support
- Verbose output options

All fields are backward compatible - old configs without new fields still work.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field


class HarvestConfig(BaseModel):
    """Configuration for using harvested Parquet data."""
    data_root: str = Field(default="data/live_deribit", description="Root directory for harvester data")
    underlying: Optional[str] = Field(default=None, description="Override underlying (defaults to top-level)")
    start_time: Optional[datetime] = Field(default=None, description="Inclusive start time filter")
    end_time: Optional[datetime] = Field(default=None, description="Inclusive end time filter")
    snapshot_step: int = Field(default=1, ge=1, description="Use every N-th snapshot to thin data")
    max_snapshots: Optional[int] = Field(default=None, description="Max number of snapshots to process")


class BandConfig(BaseModel):
    """Configuration for a DTE band."""
    name: str = Field(..., description="Band name (e.g. 'weekly', 'monthly')")
    min_dte: float = Field(..., ge=0, description="Minimum DTE for this band")
    max_dte: float = Field(..., description="Maximum DTE for this band")
    max_samples: Optional[int] = Field(default=None, description="Override max_samples for this band")


class CalibrationFilters(BaseModel):
    """Liquidity filters for calibration."""
    min_mark_price: Optional[float] = Field(default=None, description="Minimum mark price to include")
    min_open_interest: Optional[float] = Field(default=None, description="Minimum open interest to include")
    min_vega: Optional[float] = Field(default=None, description="Minimum vega to include")


class SkewConfig(BaseModel):
    """Skew configuration within calibration."""
    enabled: bool = Field(default=True, description="Whether skew is enabled")
    min_dte: float = Field(default=3.0, description="Minimum DTE for skew estimation")
    max_dte: float = Field(default=14.0, description="Maximum DTE for skew estimation")


class CalibrationConfig(BaseModel):
    """
    Complete calibration configuration.
    
    Backward compatible - all new fields have sensible defaults.
    Old configs with just underlying/min_dte/max_dte still work.
    """
    underlying: str = Field(..., description="Underlying asset (BTC or ETH)")
    min_dte: float = Field(default=3.0, ge=0, description="Minimum DTE")
    max_dte: float = Field(default=30.0, description="Maximum DTE")
    iv_multiplier: float = Field(default=1.0, ge=0.1, le=3.0, description="IV multiplier for synthetic pricing")
    default_iv: float = Field(default=0.6, ge=0.01, le=3.0, description="Default IV when RV unavailable")
    rv_window_days: int = Field(default=7, ge=1, le=365, description="Window for realized volatility computation")
    risk_free_rate: float = Field(default=0.0, description="Risk-free rate for Black-Scholes")
    max_samples: int = Field(default=80, ge=1, description="Maximum samples to process")
    skew: Optional[SkewConfig] = Field(default_factory=SkewConfig, description="Skew configuration")
    
    source: Literal["live", "harvested"] = Field(default="live", description="Data source: live API or harvested Parquet")
    harvest: Optional[HarvestConfig] = Field(default=None, description="Harvest configuration (when source='harvested')")
    
    bands: Optional[List[BandConfig]] = Field(default=None, description="Optional DTE bands for multi-band calibration")
    bucket_by_dte: Optional[List[Tuple[float, float]]] = Field(default=None, description="DTE buckets for metrics [(min, max), ...]")
    bucket_by_abs_delta: Optional[List[Tuple[float, float]]] = Field(default=None, description="Absolute delta buckets [(min, max), ...]")
    
    filters: Optional[CalibrationFilters] = Field(default=None, description="Liquidity filters")
    
    fit_skew: bool = Field(default=False, description="Whether to fit skew anchor ratios")
    return_rows: bool = Field(default=True, description="Whether to return individual option rows")
    emit_recommended_vol_surface: bool = Field(default=True, description="Whether to emit recommended vol_surface snippet")
    
    class Config:
        extra = "allow"


class DteBandResult(BaseModel):
    """Result for a single DTE band."""
    name: str
    min_dte: float
    max_dte: float
    count: int
    mae_pct: float
    bias_pct: float
    recommended_iv_multiplier: Optional[float] = None
    avg_mark_iv: Optional[float] = None
    avg_synth_iv: Optional[float] = None


class BucketResult(BaseModel):
    """Result for a DTE/delta bucket."""
    name: str
    count: int
    mae_pct: float
    bias_pct: float
    avg_mark_iv: Optional[float] = None
    avg_synth_iv: Optional[float] = None
    recommended_iv_multiplier: Optional[float] = None


class ResidualsSummary(BaseModel):
    """Summary of residuals distribution."""
    p50_pct_error: float
    p90_pct_error: float
    pct_gt_10pct_error: float
    by_delta_bucket: Optional[Dict[str, Dict[str, float]]] = None
    by_dte_bucket: Optional[Dict[str, Dict[str, float]]] = None


class GlobalMetrics(BaseModel):
    """Extended global calibration metrics."""
    mae_pct: float = Field(..., description="Mean absolute error percentage")
    bias_pct: float = Field(..., description="Bias (mean signed error) percentage")
    mae_vol_points: Optional[float] = Field(default=None, description="MAE in absolute vol points")
    vega_weighted_mae_pct: Optional[float] = Field(default=None, description="Vega-weighted MAE percentage")


class SkewFitResult(BaseModel):
    """Result of skew fitting."""
    anchor_ratios: Dict[str, float] = Field(..., description="Fitted anchor ratios by delta string")
    min_dte: float
    max_dte: float


class SkewMisfit(BaseModel):
    """Misfit between current and recommended skew."""
    max_abs_diff: float
    anchor_diffs: Dict[str, float]


class RecommendedVolSurface(BaseModel):
    """Recommended vol surface configuration."""
    iv_mode: str = Field(default="rv_window")
    rv_window_days: int
    iv_multiplier: float
    dte_bands: Optional[List[Dict[str, Any]]] = None
    skew: Optional[Dict[str, Any]] = None


class VolSurfaceDiff(BaseModel):
    """Difference between recommended and current vol surface."""
    iv_multiplier_delta: float
    anchor_ratios_delta: Optional[Dict[str, float]] = None


class SnapshotSensors(BaseModel):
    """Market sensors computed at calibration time."""
    vrp_30d: Optional[float] = None
    vrp_7d: Optional[float] = None
    chop_factor: Optional[float] = None
    adx_14: Optional[float] = None
    iv_rank_30d: Optional[float] = None
    skew_25d: Optional[float] = None
    term_slope: Optional[float] = None


class LiquidityFilterResult(BaseModel):
    """Result of liquidity filtering."""
    min_mark_price: Optional[float] = None
    min_open_interest: Optional[float] = None
    min_vega: Optional[float] = None
    dropped_count: int = 0


class ExtendedCalibrationResult(BaseModel):
    """
    Extended calibration result with all new metrics.
    
    Backward compatible with original CalibrationResult fields.
    """
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
    rv_annualized: Optional[float] = None
    atm_iv: Optional[float] = None
    recommended_iv_multiplier: Optional[float] = None
    
    global_metrics: Optional[GlobalMetrics] = None
    residuals_summary: Optional[ResidualsSummary] = None
    buckets: Optional[List[BucketResult]] = None
    bands: Optional[List[DteBandResult]] = None
    liquidity_filters: Optional[LiquidityFilterResult] = None
    
    recommended_skew: Optional[SkewFitResult] = None
    skew_misfit: Optional[SkewMisfit] = None
    
    recommended_vol_surface: Optional[RecommendedVolSurface] = None
    vol_surface_diff: Optional[VolSurfaceDiff] = None
    
    snapshot_sensors: Optional[SnapshotSensors] = None
    
    rows: Optional[List[Dict[str, Any]]] = None
    
    class Config:
        extra = "allow"


class HistoricalCalibrationResult(ExtendedCalibrationResult):
    """
    Result from historical calibration using harvested data.
    
    Extends ExtendedCalibrationResult with aggregate fields over many snapshots.
    """
    snapshot_count: int = Field(default=0, description="Number of snapshots processed")
    time_range_start: Optional[datetime] = None
    time_range_end: Optional[datetime] = None
    rv_median: Optional[float] = None
    sensor_means: Optional[Dict[str, float]] = None
    regime_distribution: Optional[Dict[str, float]] = None
