"""
Calibration update policy with smoothing, thresholds, and history management.

This module provides:
- CalibrationUpdatePolicy: Configuration for when to apply calibration updates
- History storage for calibration runs (file-based JSON)
- Smoothing functions (EWMA) for multiplier updates
- Decision logic for whether to apply an update

Directory: data/calibration_runs/
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field


CALIBRATION_RUNS_DIR = Path("data/calibration_runs")


@dataclass
class CalibrationUpdatePolicy:
    """Policy configuration for calibration updates."""
    
    min_delta_global: float = 0.03
    min_delta_band: float = 0.03
    min_sample_size: int = 50
    min_vega_sum: float = 100.0
    smoothing_window_days: int = 14
    ewma_alpha: float = 0.3
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "min_delta_global": self.min_delta_global,
            "min_delta_band": self.min_delta_band,
            "min_sample_size": self.min_sample_size,
            "min_vega_sum": self.min_vega_sum,
            "smoothing_window_days": self.smoothing_window_days,
            "ewma_alpha": self.ewma_alpha,
        }


class BandMultiplier(BaseModel):
    """IV multiplier for a specific DTE band."""
    name: str
    min_dte: float
    max_dte: float
    iv_multiplier: float


class CalibrationRunRecord(BaseModel):
    """A single calibration run stored in history."""
    
    timestamp: datetime
    underlying: str
    source: str
    
    config_hash: Optional[str] = None
    
    global_metrics: Dict[str, Any] = Field(default_factory=dict)
    bands: Optional[List[Dict[str, Any]]] = None
    
    recommended_iv_multiplier: float
    recommended_band_multipliers: Optional[List[BandMultiplier]] = None
    
    sample_size: int
    vega_sum: float
    
    smoothed_global_multiplier: Optional[float] = None
    smoothed_band_multipliers: Optional[List[BandMultiplier]] = None
    
    applied: bool = False
    applied_reason: str = ""
    
    class Config:
        extra = "allow"


class CurrentAppliedMultipliers(BaseModel):
    """Current applied IV multipliers (loaded from vol surface config)."""
    global_multiplier: float = 1.0
    band_multipliers: Optional[List[BandMultiplier]] = None
    last_updated: Optional[datetime] = None


def _ensure_runs_dir() -> Path:
    """Ensure the calibration runs directory exists."""
    CALIBRATION_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    return CALIBRATION_RUNS_DIR


def _run_filename(timestamp: datetime, underlying: str, source: str) -> str:
    """Generate filename for a calibration run."""
    ts_str = timestamp.strftime("%Y%m%d_%H%M%S")
    return f"{ts_str}_{underlying}_{source}.json"


def record_calibration_result(
    underlying: str,
    source: str,
    recommended_iv_multiplier: float,
    recommended_band_multipliers: Optional[List[BandMultiplier]],
    sample_size: int,
    vega_sum: float,
    global_metrics: Optional[Dict[str, Any]] = None,
    bands: Optional[List[Dict[str, Any]]] = None,
    config_hash: Optional[str] = None,
    smoothed_global: Optional[float] = None,
    smoothed_bands: Optional[List[BandMultiplier]] = None,
    applied: bool = False,
    applied_reason: str = "",
) -> CalibrationRunRecord:
    """
    Save a calibration run to history.
    
    Returns the created record.
    """
    _ensure_runs_dir()
    
    now = datetime.now(timezone.utc)
    
    record = CalibrationRunRecord(
        timestamp=now,
        underlying=underlying,
        source=source,
        config_hash=config_hash,
        global_metrics=global_metrics or {},
        bands=bands,
        recommended_iv_multiplier=recommended_iv_multiplier,
        recommended_band_multipliers=recommended_band_multipliers,
        sample_size=sample_size,
        vega_sum=vega_sum,
        smoothed_global_multiplier=smoothed_global,
        smoothed_band_multipliers=smoothed_bands,
        applied=applied,
        applied_reason=applied_reason,
    )
    
    filename = _run_filename(now, underlying, source)
    filepath = CALIBRATION_RUNS_DIR / filename
    
    with open(filepath, "w") as f:
        json.dump(record.model_dump(mode="json"), f, indent=2, default=str)
    
    return record


def load_recent_calibration_history(
    underlying: str,
    limit: int = 50,
    source_filter: Optional[str] = None,
) -> List[CalibrationRunRecord]:
    """
    Load recent calibration runs for an underlying.
    
    Args:
        underlying: Filter by underlying (BTC or ETH)
        limit: Maximum number of runs to return
        source_filter: Optional filter by source ("live" or "harvested")
        
    Returns:
        List of CalibrationRunRecord sorted by timestamp descending
    """
    _ensure_runs_dir()
    
    records: List[Tuple[datetime, CalibrationRunRecord]] = []
    
    for filepath in CALIBRATION_RUNS_DIR.glob("*.json"):
        try:
            with open(filepath) as f:
                data = json.load(f)
            
            if data.get("underlying") != underlying:
                continue
            
            if source_filter and data.get("source") != source_filter:
                continue
            
            if "timestamp" in data and isinstance(data["timestamp"], str):
                data["timestamp"] = datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00"))
            
            record = CalibrationRunRecord(**data)
            records.append((record.timestamp, record))
            
        except Exception:
            continue
    
    records.sort(key=lambda x: x[0], reverse=True)
    
    return [r for _, r in records[:limit]]


def get_smoothed_multipliers(
    history: List[CalibrationRunRecord],
    current_recommended: float,
    current_recommended_bands: Optional[List[BandMultiplier]],
    policy: CalibrationUpdatePolicy,
) -> Tuple[float, Optional[List[BandMultiplier]]]:
    """
    Compute smoothed multipliers using EWMA.
    
    Args:
        history: Recent calibration runs (newest first)
        current_recommended: Current run's recommended global multiplier
        current_recommended_bands: Current run's recommended band multipliers
        policy: Update policy with smoothing parameters
        
    Returns:
        Tuple of (smoothed_global, smoothed_bands)
    """
    window_days = policy.smoothing_window_days
    alpha = policy.ewma_alpha
    
    from datetime import timedelta
    cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)
    
    relevant_history = []
    for rec in history:
        rec_ts = rec.timestamp
        if rec_ts is None:
            continue
        if rec_ts.tzinfo is None:
            rec_ts = rec_ts.replace(tzinfo=timezone.utc)
        if rec_ts >= cutoff:
            relevant_history.append(rec)
    
    global_values = [current_recommended]
    for rec in relevant_history:
        global_values.append(rec.recommended_iv_multiplier)
    
    smoothed_global = _ewma(global_values, alpha)
    
    smoothed_bands = None
    if current_recommended_bands:
        smoothed_bands = []
        for band in current_recommended_bands:
            band_values = [band.iv_multiplier]
            for rec in relevant_history:
                if rec.recommended_band_multipliers:
                    for hist_band in rec.recommended_band_multipliers:
                        if hist_band.name == band.name:
                            band_values.append(hist_band.iv_multiplier)
                            break
            
            smoothed_mult = _ewma(band_values, alpha)
            smoothed_bands.append(BandMultiplier(
                name=band.name,
                min_dte=band.min_dte,
                max_dte=band.max_dte,
                iv_multiplier=round(smoothed_mult, 4),
            ))
    
    return round(smoothed_global, 4), smoothed_bands


def _ewma(values: List[float], alpha: float) -> float:
    """Compute EWMA with most recent value first."""
    if not values:
        return 1.0
    if len(values) == 1:
        return values[0]
    
    result = values[0]
    for i in range(1, len(values)):
        weight = (1 - alpha) ** i
        result = alpha * values[i] + (1 - alpha) * result
    
    return result


@dataclass
class UpdateDecision:
    """Result of should_apply_update check."""
    should_apply: bool
    reason: str
    details: Dict[str, Any] = field(default_factory=dict)


def should_apply_update(
    current_applied: CurrentAppliedMultipliers,
    smoothed_global: float,
    smoothed_bands: Optional[List[BandMultiplier]],
    policy: CalibrationUpdatePolicy,
    sample_size: int,
    vega_sum: float,
) -> UpdateDecision:
    """
    Decide whether to apply a calibration update.
    
    Checks:
    1. Sample size >= min_sample_size
    2. Vega sum >= min_vega_sum
    3. Global delta >= min_delta_global OR any band delta >= min_delta_band
    
    Returns:
        UpdateDecision with should_apply flag and reason
    """
    if sample_size < policy.min_sample_size:
        return UpdateDecision(
            should_apply=False,
            reason=f"Sample size too small ({sample_size} < {policy.min_sample_size})",
            details={"sample_size": sample_size, "min_required": policy.min_sample_size},
        )
    
    if vega_sum < policy.min_vega_sum:
        return UpdateDecision(
            should_apply=False,
            reason=f"Vega sum too low ({vega_sum:.1f} < {policy.min_vega_sum})",
            details={"vega_sum": vega_sum, "min_required": policy.min_vega_sum},
        )
    
    global_delta = abs(smoothed_global - current_applied.global_multiplier)
    global_passes = global_delta >= policy.min_delta_global
    
    band_passes = False
    band_deltas: Dict[str, float] = {}
    
    if smoothed_bands and current_applied.band_multipliers:
        current_bands_by_name = {b.name: b for b in current_applied.band_multipliers}
        for band in smoothed_bands:
            current_band = current_bands_by_name.get(band.name)
            if current_band:
                delta = abs(band.iv_multiplier - current_band.iv_multiplier)
                band_deltas[band.name] = delta
                if delta >= policy.min_delta_band:
                    band_passes = True
    
    if not global_passes and not band_passes:
        return UpdateDecision(
            should_apply=False,
            reason=f"Change too small (global Δ={global_delta:.4f} < {policy.min_delta_global})",
            details={
                "global_delta": global_delta,
                "min_global_delta": policy.min_delta_global,
                "band_deltas": band_deltas,
            },
        )
    
    reasons = []
    if global_passes:
        reasons.append(f"global Δ={global_delta:.4f}")
    if band_passes:
        passing_bands = [n for n, d in band_deltas.items() if d >= policy.min_delta_band]
        reasons.append(f"bands changed: {', '.join(passing_bands)}")
    reasons.append(f"sample={sample_size}")
    reasons.append(f"vega={vega_sum:.0f}")
    
    return UpdateDecision(
        should_apply=True,
        reason="; ".join(reasons),
        details={
            "global_delta": global_delta,
            "band_deltas": band_deltas,
            "sample_size": sample_size,
            "vega_sum": vega_sum,
        },
    )


def update_run_record(record: CalibrationRunRecord, applied: bool, reason: str) -> None:
    """Update a run record's applied status in the file."""
    _ensure_runs_dir()
    
    filename = _run_filename(record.timestamp, record.underlying, record.source)
    filepath = CALIBRATION_RUNS_DIR / filename
    
    if filepath.exists():
        with open(filepath) as f:
            data = json.load(f)
        
        data["applied"] = applied
        data["applied_reason"] = reason
        
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)


def get_default_policy() -> CalibrationUpdatePolicy:
    """Get the default calibration update policy."""
    return CalibrationUpdatePolicy()


_runtime_policy: Optional[CalibrationUpdatePolicy] = None


def get_policy() -> CalibrationUpdatePolicy:
    """Get the current runtime policy."""
    global _runtime_policy
    if _runtime_policy is None:
        _runtime_policy = get_default_policy()
    return _runtime_policy


def set_policy(policy: CalibrationUpdatePolicy) -> None:
    """Set the runtime policy."""
    global _runtime_policy
    _runtime_policy = policy


def get_current_applied_multipliers(underlying: str = "BTC") -> CurrentAppliedMultipliers:
    """
    Get the currently applied multipliers from the calibration store.
    
    This is the source of truth for the "Current Applied Multipliers" UI panel.
    The store is updated when:
    - A live calibration is applied via policy
    - User clicks "Force-Apply Latest"
    
    Falls back to vol surface config if no explicit applied state exists.
    """
    from src.calibration_store import get_applied_multiplier
    from src.synthetic.vol_surface import get_vol_surface_config
    
    applied_state = get_applied_multiplier(underlying)
    
    if applied_state.last_updated is not None:
        band_multipliers = None
        if applied_state.band_multipliers:
            band_multipliers = [
                BandMultiplier(
                    name=name,
                    min_dte=0,
                    max_dte=365,
                    iv_multiplier=mult,
                )
                for name, mult in applied_state.band_multipliers.items()
            ]
        
        return CurrentAppliedMultipliers(
            global_multiplier=applied_state.global_multiplier,
            band_multipliers=band_multipliers,
            last_updated=applied_state.last_updated,
        )
    
    config = get_vol_surface_config()
    
    band_multipliers = None
    if config.dte_bands:
        band_multipliers = [
            BandMultiplier(
                name=b.name,
                min_dte=b.min_dte,
                max_dte=b.max_dte,
                iv_multiplier=b.iv_multiplier,
            )
            for b in config.dte_bands
        ]
    
    return CurrentAppliedMultipliers(
        global_multiplier=config.iv_multiplier,
        band_multipliers=band_multipliers,
        last_updated=None,
    )


def run_calibration_with_policy(
    underlying: str,
    source: Literal["live", "harvested"] = "live",
    force: bool = False,
    policy: Optional[CalibrationUpdatePolicy] = None,
    **config_kwargs,
) -> Tuple[CalibrationRunRecord, UpdateDecision]:
    """
    Run calibration and apply policy to decide whether to update vol surface.
    
    Args:
        underlying: BTC or ETH
        source: "live" or "harvested"
        force: If True, apply regardless of thresholds
        policy: Update policy (uses default if None)
        **config_kwargs: Additional CalibrationConfig kwargs
        
    Returns:
        Tuple of (record, decision)
    """
    from src.calibration_config import CalibrationConfig, HarvestConfig
    from src.calibration_extended import run_calibration_extended
    from src.synthetic.vol_surface import (
        get_vol_surface_config, 
        set_vol_surface_config,
        VolSurfaceConfig,
        DteBand,
    )
    
    policy = policy or get_policy()
    
    harvest = None
    if source == "harvested":
        harvest = HarvestConfig(
            data_root=config_kwargs.get("data_root", "data/live_deribit"),
            underlying=underlying,
        )
    
    config = CalibrationConfig(
        underlying=underlying,
        source=source,
        harvest=harvest,
        emit_recommended_vol_surface=True,
        return_rows=False,
        **{k: v for k, v in config_kwargs.items() if k not in ["data_root"]},
    )
    
    result = run_calibration_extended(config)
    
    recommended_global = result.recommended_iv_multiplier or 1.0
    recommended_bands: Optional[List[BandMultiplier]] = None
    if result.bands:
        recommended_bands = [
            BandMultiplier(
                name=b.name,
                min_dte=b.min_dte,
                max_dte=b.max_dte,
                iv_multiplier=b.recommended_iv_multiplier or 1.0,
            )
            for b in result.bands
            if b.recommended_iv_multiplier is not None
        ]
    
    sample_size = result.count
    vega_sum = 0.0
    if result.global_metrics and hasattr(result.global_metrics, 'vega_weighted_mae_pct'):
        vega_sum = sample_size * 10.0
    else:
        vega_sum = sample_size * 10.0
    
    history = load_recent_calibration_history(underlying, limit=50)
    
    smoothed_global, smoothed_bands = get_smoothed_multipliers(
        history, recommended_global, recommended_bands, policy
    )
    
    current_applied = get_current_applied_multipliers()
    
    if force:
        decision = UpdateDecision(
            should_apply=True,
            reason="Forced by user",
            details={"forced": True},
        )
    else:
        decision = should_apply_update(
            current_applied, smoothed_global, smoothed_bands, 
            policy, sample_size, vega_sum
        )
    
    if decision.should_apply:
        from src.calibration_store import set_applied_multiplier
        from src.synthetic.vol_surface import SkewTemplate
        
        current_config = get_vol_surface_config()
        
        dte_bands = None
        band_multipliers_dict: Dict[str, float] = {}
        if smoothed_bands:
            dte_bands = [
                DteBand(
                    name=b.name,
                    min_dte=b.min_dte,
                    max_dte=b.max_dte,
                    iv_multiplier=b.iv_multiplier,
                )
                for b in smoothed_bands
            ]
            band_multipliers_dict = {b.name: b.iv_multiplier for b in smoothed_bands}
        
        skew_config = current_config.skew
        skew_anchor_ratios: Optional[Dict[str, float]] = None
        if result.recommended_skew and result.recommended_skew.anchor_ratios:
            skew_anchor_ratios = result.recommended_skew.anchor_ratios
            skew_config = SkewTemplate(
                enabled=True,
                min_dte=result.recommended_skew.min_dte,
                max_dte=result.recommended_skew.max_dte,
                anchor_ratios=skew_anchor_ratios,
                mode=current_config.skew.mode if current_config.skew else "put_heavy",
                scale=current_config.skew.scale if current_config.skew else 1.0,
            )
        
        new_config = VolSurfaceConfig(
            iv_mode=current_config.iv_mode,
            rv_window_days=current_config.rv_window_days,
            iv_multiplier=smoothed_global,
            dte_bands=dte_bands or current_config.dte_bands,
            skew=skew_config,
            regime_override=current_config.regime_override,
            vrp_offset_enabled=current_config.vrp_offset_enabled,
        )
        set_vol_surface_config(new_config)
        
        set_applied_multiplier(
            underlying=underlying,
            global_multiplier=smoothed_global,
            band_multipliers=band_multipliers_dict if band_multipliers_dict else None,
            skew_anchor_ratios=skew_anchor_ratios,
            source=source,
            applied_reason=decision.reason,
        )
    
    global_metrics_dict = {}
    if result.global_metrics:
        global_metrics_dict = result.global_metrics.model_dump()
    
    bands_dict = None
    if result.bands:
        bands_dict = [b.model_dump() for b in result.bands]
    
    record = record_calibration_result(
        underlying=underlying,
        source=source,
        recommended_iv_multiplier=recommended_global,
        recommended_band_multipliers=recommended_bands,
        sample_size=sample_size,
        vega_sum=vega_sum,
        global_metrics=global_metrics_dict,
        bands=bands_dict,
        smoothed_global=smoothed_global,
        smoothed_bands=smoothed_bands,
        applied=decision.should_apply,
        applied_reason=decision.reason,
    )
    
    return record, decision
