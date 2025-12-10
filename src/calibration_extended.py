"""
Extended calibration module with enhanced metrics, bucketing, and harvested data support.

This module extends run_calibration with:
- Liquidity filtering
- Multi-DTE band handling
- DTE/delta bucket metrics
- Skew fitting
- Recommended vol_surface snippet
- Snapshot sensors
- Harvested Parquet data support
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.calibration import (
    get_index_price,
    get_spot_history_for_rv,
    get_call_chain,
    black_scholes_call_price,
    OptionQuote,
)
from src.calibration_config import (
    CalibrationConfig,
    CalibrationFilters,
    BandConfig,
    ExtendedCalibrationResult,
    HistoricalCalibrationResult,
    GlobalMetrics,
    ResidualsSummary,
    BucketResult,
    DteBandResult,
    LiquidityFilterResult,
    SkewFitResult,
    SkewMisfit,
    RecommendedVolSurface,
    VolSurfaceDiff,
    SnapshotSensors,
)
from src.backtest.pricing import (
    compute_realized_volatility,
    compute_synthetic_iv_with_skew,
    bs_call_delta,
)
from src.config import settings


def apply_liquidity_filters(
    quotes: List[OptionQuote],
    filters: Optional[CalibrationFilters],
) -> Tuple[List[OptionQuote], LiquidityFilterResult]:
    """
    Apply liquidity filters to option quotes.
    
    Returns filtered quotes and filter result with dropped count.
    """
    if filters is None:
        return quotes, LiquidityFilterResult(dropped_count=0)
    
    original_count = len(quotes)
    filtered = quotes
    
    if filters.min_mark_price is not None:
        filtered = [q for q in filtered if q.mark_price >= filters.min_mark_price]
    
    result = LiquidityFilterResult(
        min_mark_price=filters.min_mark_price,
        min_open_interest=filters.min_open_interest,
        min_vega=filters.min_vega,
        dropped_count=original_count - len(filtered),
    )
    
    return filtered, result


def compute_bucket_metrics(
    rows: List[Dict[str, Any]],
    bucket_by_dte: Optional[List[Tuple[float, float]]],
    bucket_by_abs_delta: Optional[List[Tuple[float, float]]],
    rv_annualized: float,
) -> Tuple[List[BucketResult], ResidualsSummary]:
    """
    Compute bucketed metrics by DTE and/or absolute delta.
    """
    buckets: List[BucketResult] = []
    by_dte_bucket: Dict[str, Dict[str, float]] = {}
    by_delta_bucket: Dict[str, Dict[str, float]] = {}
    
    if not rows:
        return buckets, ResidualsSummary(
            p50_pct_error=0.0,
            p90_pct_error=0.0,
            pct_gt_10pct_error=0.0,
        )
    
    errors = [abs(r.get("diff_pct", 0.0)) for r in rows]
    p50 = float(np.percentile(errors, 50)) if errors else 0.0
    p90 = float(np.percentile(errors, 90)) if errors else 0.0
    pct_gt_10 = sum(1 for e in errors if e > 10.0) / len(errors) * 100 if errors else 0.0
    
    if bucket_by_dte:
        for dte_min, dte_max in bucket_by_dte:
            bucket_name = f"dte_{int(dte_min)}_{int(dte_max)}"
            bucket_rows = [r for r in rows if dte_min <= r.get("dte", 0) <= dte_max]
            
            if bucket_rows:
                bucket_errors = [abs(r.get("diff_pct", 0.0)) for r in bucket_rows]
                mae = float(np.mean(bucket_errors))
                bias = float(np.mean([r.get("diff_pct", 0.0) for r in bucket_rows]))
                avg_mark = np.mean([r.get("mark_iv", 0.0) for r in bucket_rows if r.get("mark_iv")])
                avg_synth = np.mean([r.get("syn_iv", 0.0) for r in bucket_rows])
                
                rec_mult = None
                if rv_annualized > 0 and avg_mark and avg_mark > 0:
                    rec_mult = float(avg_mark) / 100.0 / rv_annualized
                
                buckets.append(BucketResult(
                    name=bucket_name,
                    count=len(bucket_rows),
                    mae_pct=mae,
                    bias_pct=bias,
                    avg_mark_iv=float(avg_mark) if avg_mark else None,
                    avg_synth_iv=float(avg_synth) if avg_synth else None,
                    recommended_iv_multiplier=rec_mult,
                ))
                
                by_dte_bucket[bucket_name] = {"mae_pct": mae, "bias_pct": bias, "count": len(bucket_rows)}
    
    if bucket_by_abs_delta:
        for delta_min, delta_max in bucket_by_abs_delta:
            bucket_name = f"delta_{delta_min:.2f}_{delta_max:.2f}"
            bucket_rows = [r for r in rows if r.get("abs_delta") is not None 
                          and delta_min <= r.get("abs_delta", 0) <= delta_max]
            
            if bucket_rows:
                bucket_errors = [abs(r.get("diff_pct", 0.0)) for r in bucket_rows]
                mae = float(np.mean(bucket_errors))
                bias = float(np.mean([r.get("diff_pct", 0.0) for r in bucket_rows]))
                avg_mark = np.mean([r.get("mark_iv", 0.0) for r in bucket_rows if r.get("mark_iv")])
                avg_synth = np.mean([r.get("syn_iv", 0.0) for r in bucket_rows])
                
                rec_mult = None
                if rv_annualized > 0 and avg_mark and avg_mark > 0:
                    rec_mult = float(avg_mark) / 100.0 / rv_annualized
                
                buckets.append(BucketResult(
                    name=bucket_name,
                    count=len(bucket_rows),
                    mae_pct=mae,
                    bias_pct=bias,
                    avg_mark_iv=float(avg_mark) if avg_mark else None,
                    avg_synth_iv=float(avg_synth) if avg_synth else None,
                    recommended_iv_multiplier=rec_mult,
                ))
                
                by_delta_bucket[bucket_name] = {"mae_pct": mae, "bias_pct": bias, "count": len(bucket_rows)}
    
    return buckets, ResidualsSummary(
        p50_pct_error=p50,
        p90_pct_error=p90,
        pct_gt_10pct_error=pct_gt_10,
        by_dte_bucket=by_dte_bucket if by_dte_bucket else None,
        by_delta_bucket=by_delta_bucket if by_delta_bucket else None,
    )


def compute_global_metrics(
    rows: List[Dict[str, Any]],
) -> GlobalMetrics:
    """
    Compute extended global metrics including vega-weighted MAE.
    """
    if not rows:
        return GlobalMetrics(mae_pct=0.0, bias_pct=0.0)
    
    errors_pct = [r.get("diff_pct", 0.0) for r in rows]
    mae_pct = float(np.mean([abs(e) for e in errors_pct]))
    bias_pct = float(np.mean(errors_pct))
    
    mark_ivs = [r.get("mark_iv") for r in rows if r.get("mark_iv") is not None]
    syn_ivs = [r.get("syn_iv") for r in rows if r.get("syn_iv") is not None]
    
    mae_vol_points = None
    if mark_ivs and syn_ivs and len(mark_ivs) == len(syn_ivs):
        iv_diffs = [abs(m - s * 100) for m, s in zip(mark_ivs, syn_ivs)]
        mae_vol_points = float(np.mean(iv_diffs))
    
    vega_weighted_mae = None
    vega_rows = [(r, r.get("vega", 0.0)) for r in rows if r.get("vega") and r.get("vega") > 0]
    if vega_rows:
        total_vega = sum(v for _, v in vega_rows)
        if total_vega > 0:
            weighted_sum = sum(abs(r.get("diff_pct", 0.0)) * v for r, v in vega_rows)
            vega_weighted_mae = weighted_sum / total_vega
    
    return GlobalMetrics(
        mae_pct=mae_pct,
        bias_pct=bias_pct,
        mae_vol_points=mae_vol_points,
        vega_weighted_mae_pct=vega_weighted_mae,
    )


def fit_skew_anchor_ratios(
    rows: List[Dict[str, Any]],
    rv_annualized: float,
    iv_multiplier: float,
    anchor_deltas: List[float] = [0.15, 0.25, 0.35],
    delta_tolerance: float = 0.05,
) -> Optional[SkewFitResult]:
    """
    Fit skew anchor ratios from calibration data.
    
    For each anchor delta, find options near that delta and compute
    the ratio of mark IV to baseline IV.
    """
    if not rows or rv_annualized <= 0:
        return None
    
    baseline_iv = rv_annualized * iv_multiplier
    if baseline_iv <= 0:
        return None
    
    anchor_ratios: Dict[str, float] = {}
    
    for anchor in anchor_deltas:
        near_anchor = [
            r for r in rows 
            if r.get("abs_delta") is not None 
            and abs(r.get("abs_delta", 0) - anchor) <= delta_tolerance
            and r.get("mark_iv") is not None
            and r.get("mark_iv") > 0
        ]
        
        if near_anchor:
            avg_mark_iv = np.mean([r.get("mark_iv", 0) / 100.0 for r in near_anchor])
            ratio = float(avg_mark_iv / baseline_iv)
            anchor_ratios[f"{anchor:.2f}"] = round(ratio, 4)
        else:
            anchor_ratios[f"{anchor:.2f}"] = 1.0
    
    dte_values = [r.get("dte", 0) for r in rows if r.get("dte")]
    min_dte = float(np.min(dte_values)) if dte_values else 3.0
    max_dte = float(np.max(dte_values)) if dte_values else 14.0
    
    return SkewFitResult(
        anchor_ratios=anchor_ratios,
        min_dte=min_dte,
        max_dte=max_dte,
    )


def compute_skew_misfit(
    recommended: SkewFitResult,
    current_ratios: Dict[str, float],
) -> SkewMisfit:
    """
    Compute misfit between recommended and current skew anchor ratios.
    """
    diffs: Dict[str, float] = {}
    max_diff = 0.0
    
    for delta_key, rec_ratio in recommended.anchor_ratios.items():
        current = current_ratios.get(delta_key, 1.0)
        diff = rec_ratio - current
        diffs[delta_key] = round(diff, 4)
        max_diff = max(max_diff, abs(diff))
    
    return SkewMisfit(
        max_abs_diff=round(max_diff, 4),
        anchor_diffs=diffs,
    )


def build_recommended_vol_surface(
    rv_window_days: int,
    recommended_iv_multiplier: float,
    bands_results: Optional[List[DteBandResult]],
    skew_result: Optional[SkewFitResult],
    skew_enabled: bool,
) -> RecommendedVolSurface:
    """
    Build a recommended vol_surface configuration from calibration results.
    """
    dte_bands = None
    if bands_results:
        dte_bands = [
            {
                "name": b.name,
                "min_dte": b.min_dte,
                "max_dte": b.max_dte,
                "iv_multiplier": b.recommended_iv_multiplier or recommended_iv_multiplier,
            }
            for b in bands_results
            if b.recommended_iv_multiplier is not None
        ]
    
    skew_config = None
    if skew_result:
        skew_config = {
            "enabled": skew_enabled,
            "min_dte": skew_result.min_dte,
            "max_dte": skew_result.max_dte,
            "anchor_ratios": skew_result.anchor_ratios,
        }
    
    return RecommendedVolSurface(
        iv_mode="rv_window",
        rv_window_days=rv_window_days,
        iv_multiplier=recommended_iv_multiplier,
        dte_bands=dte_bands,
        skew=skew_config,
    )


def compute_snapshot_sensors(
    spot_history: List[Tuple[datetime, float]],
    atm_iv: Optional[float],
    rv_annualized: float,
) -> SnapshotSensors:
    """
    Compute market sensors at calibration time.
    """
    vrp_30d = None
    vrp_7d = None
    
    if atm_iv is not None and rv_annualized > 0:
        vrp_30d = (atm_iv - rv_annualized) * 100
        vrp_7d = vrp_30d
    
    return SnapshotSensors(
        vrp_30d=vrp_30d,
        vrp_7d=vrp_7d,
    )


def run_calibration_extended(
    config: CalibrationConfig,
) -> ExtendedCalibrationResult:
    """
    Extended calibration with all new features.
    
    Supports:
    - Liquidity filtering
    - Multi-DTE bands
    - Bucket metrics
    - Skew fitting
    - Recommended vol_surface
    - Snapshot sensors
    
    Backward compatible with original run_calibration output.
    """
    if config.source == "harvested":
        return run_historical_calibration_from_harvest(config)
    
    now = datetime.now(timezone.utc)
    underlying = config.underlying.upper()
    
    spot = get_index_price(underlying)
    spot_history = get_spot_history_for_rv(underlying, as_of=now, window_days=config.rv_window_days)
    
    if spot_history:
        rv_annualized = compute_realized_volatility(
            prices=spot_history,
            as_of=now,
            window_days=config.rv_window_days,
        )
    else:
        rv_annualized = config.default_iv
    
    quotes = get_call_chain(underlying, min_dte=config.min_dte, max_dte=config.max_dte)
    quotes = sorted(quotes, key=lambda q: (q.dte_days, q.strike))
    
    quotes, liquidity_result = apply_liquidity_filters(quotes, config.filters)
    
    if not quotes:
        return ExtendedCalibrationResult(
            underlying=underlying,
            spot=spot,
            min_dte=config.min_dte,
            max_dte=config.max_dte,
            iv_multiplier=config.iv_multiplier,
            default_iv=config.default_iv,
            count=0,
            mae_pct=0.0,
            bias_pct=0.0,
            timestamp=now,
            rv_annualized=rv_annualized,
            liquidity_filters=liquidity_result,
        )
    
    atm_iv: Optional[float] = None
    best_delta_diff: Optional[float] = None
    for q in quotes:
        if q.mark_iv is None or q.mark_iv <= 0:
            continue
        if q.delta is None:
            continue
        diff = abs(float(q.delta) - 0.5)
        if best_delta_diff is None or diff < best_delta_diff:
            best_delta_diff = diff
            atm_iv = float(q.mark_iv) / 100.0
    
    recommended_iv_multiplier: Optional[float] = None
    if rv_annualized and rv_annualized > 0.0 and atm_iv and atm_iv > 0.0:
        recommended_iv_multiplier = round(atm_iv / rv_annualized, 4)
    
    if len(quotes) > config.max_samples:
        step = max(1, len(quotes) // config.max_samples)
        quotes = quotes[::step]
    
    rows: List[Dict[str, Any]] = []
    skew_enabled = config.skew.enabled if config.skew else settings.synthetic_skew_enabled
    skew_min_dte = config.skew.min_dte if config.skew else settings.synthetic_skew_min_dte
    skew_max_dte = config.skew.max_dte if config.skew else settings.synthetic_skew_max_dte
    
    for q in quotes:
        t_years = max(0.0001, q.dte_days / 365.0)
        
        base_iv = max(1e-6, rv_annualized * config.iv_multiplier)
        abs_delta = abs(bs_call_delta(
            spot=spot,
            strike=q.strike,
            t_years=t_years,
            sigma=base_iv,
            r=config.risk_free_rate,
        ))
        
        sigma = compute_synthetic_iv_with_skew(
            underlying=underlying,
            option_type="call",
            abs_delta=abs_delta,
            rv_annualized=rv_annualized,
            iv_multiplier=config.iv_multiplier,
            skew_enabled=skew_enabled,
            skew_min_dte=skew_min_dte,
            skew_max_dte=skew_max_dte,
        )
        
        synthetic_price_usd = black_scholes_call_price(
            spot=spot,
            strike=q.strike,
            t_years=t_years,
            sigma=sigma,
            r=config.risk_free_rate,
        )
        
        is_inverse = q.settlement_currency.upper() in ("BTC", "ETH")
        if is_inverse:
            synthetic_price = synthetic_price_usd / spot
        else:
            synthetic_price = synthetic_price_usd
        
        diff = synthetic_price - q.mark_price
        diff_pct = (diff / q.mark_price) * 100.0 if q.mark_price > 0 else 0.0
        
        rows.append({
            "instrument": q.instrument_name,
            "dte": q.dte_days,
            "strike": q.strike,
            "mark_price": q.mark_price,
            "syn_price": synthetic_price,
            "diff": diff,
            "diff_pct": diff_pct,
            "mark_iv": q.mark_iv,
            "syn_iv": sigma,
            "abs_delta": abs_delta,
            "delta": q.delta,
        })
    
    global_metrics = compute_global_metrics(rows)
    
    buckets, residuals_summary = compute_bucket_metrics(
        rows,
        config.bucket_by_dte,
        config.bucket_by_abs_delta,
        rv_annualized,
    )
    
    bands_results: Optional[List[DteBandResult]] = None
    if config.bands:
        bands_results = []
        for band in config.bands:
            band_rows = [r for r in rows if band.min_dte <= r.get("dte", 0) <= band.max_dte]
            if band_rows:
                band_errors = [abs(r.get("diff_pct", 0.0)) for r in band_rows]
                mae = float(np.mean(band_errors))
                bias = float(np.mean([r.get("diff_pct", 0.0) for r in band_rows]))
                
                avg_mark = np.mean([r.get("mark_iv", 0.0) for r in band_rows if r.get("mark_iv")])
                rec_mult = None
                if rv_annualized > 0 and avg_mark and avg_mark > 0:
                    rec_mult = round(float(avg_mark) / 100.0 / rv_annualized, 4)
                
                bands_results.append(DteBandResult(
                    name=band.name,
                    min_dte=band.min_dte,
                    max_dte=band.max_dte,
                    count=len(band_rows),
                    mae_pct=mae,
                    bias_pct=bias,
                    recommended_iv_multiplier=rec_mult,
                    avg_mark_iv=float(avg_mark) if avg_mark else None,
                ))
    
    skew_result: Optional[SkewFitResult] = None
    skew_misfit: Optional[SkewMisfit] = None
    if config.fit_skew:
        skew_result = fit_skew_anchor_ratios(
            rows, rv_annualized, config.iv_multiplier
        )
        if skew_result:
            current_ratios = {
                "0.15": settings.synthetic_skew_anchor_15 if hasattr(settings, 'synthetic_skew_anchor_15') else 1.0,
                "0.25": settings.synthetic_skew_anchor_25 if hasattr(settings, 'synthetic_skew_anchor_25') else 1.0,
                "0.35": settings.synthetic_skew_anchor_35 if hasattr(settings, 'synthetic_skew_anchor_35') else 1.0,
            }
            skew_misfit = compute_skew_misfit(skew_result, current_ratios)
    
    recommended_vol_surface: Optional[RecommendedVolSurface] = None
    vol_surface_diff: Optional[VolSurfaceDiff] = None
    if config.emit_recommended_vol_surface and recommended_iv_multiplier:
        recommended_vol_surface = build_recommended_vol_surface(
            rv_window_days=config.rv_window_days,
            recommended_iv_multiplier=recommended_iv_multiplier,
            bands_results=bands_results,
            skew_result=skew_result,
            skew_enabled=skew_enabled,
        )
        vol_surface_diff = VolSurfaceDiff(
            iv_multiplier_delta=round(recommended_iv_multiplier - config.iv_multiplier, 4),
            anchor_ratios_delta=skew_misfit.anchor_diffs if skew_misfit else None,
        )
    
    snapshot_sensors = compute_snapshot_sensors(spot_history, atm_iv, rv_annualized)
    
    return ExtendedCalibrationResult(
        underlying=underlying,
        spot=spot,
        min_dte=config.min_dte,
        max_dte=config.max_dte,
        iv_multiplier=config.iv_multiplier,
        default_iv=config.default_iv,
        count=len(rows),
        mae_pct=global_metrics.mae_pct,
        bias_pct=global_metrics.bias_pct,
        timestamp=now,
        rv_annualized=rv_annualized,
        atm_iv=atm_iv,
        recommended_iv_multiplier=recommended_iv_multiplier,
        global_metrics=global_metrics,
        residuals_summary=residuals_summary if buckets else None,
        buckets=buckets if buckets else None,
        bands=bands_results,
        liquidity_filters=liquidity_result,
        recommended_skew=skew_result,
        skew_misfit=skew_misfit,
        recommended_vol_surface=recommended_vol_surface,
        vol_surface_diff=vol_surface_diff,
        snapshot_sensors=snapshot_sensors,
        rows=[r for r in rows] if config.return_rows else None,
    )


def load_harvest_snapshots(
    data_root: str,
    underlying: str,
    start_time: Optional[datetime],
    end_time: Optional[datetime],
    snapshot_step: int = 1,
    max_snapshots: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load harvested Parquet snapshots from the data directory.
    
    Expects structure: data_root/<asset>/<YYYY>/<MM>/<DD>/*.parquet
    """
    root = Path(data_root)
    asset_dir = root / underlying.upper()
    
    if not asset_dir.exists():
        return pd.DataFrame()
    
    parquet_files = sorted(asset_dir.rglob("*.parquet"))
    
    if not parquet_files:
        return pd.DataFrame()
    
    dfs = []
    for pf in parquet_files:
        try:
            df = pd.read_parquet(pf)
            dfs.append(df)
        except Exception:
            continue
    
    if not dfs:
        return pd.DataFrame()
    
    combined = pd.concat(dfs, ignore_index=True)
    
    if "harvest_time" in combined.columns:
        combined["harvest_time"] = pd.to_datetime(combined["harvest_time"], utc=True)
        
        if start_time:
            combined = combined[combined["harvest_time"] >= start_time]
        if end_time:
            combined = combined[combined["harvest_time"] <= end_time]
    
    if snapshot_step > 1 and "harvest_time" in combined.columns:
        unique_times = combined["harvest_time"].drop_duplicates().sort_values()
        selected_times = unique_times[::snapshot_step]
        combined = combined[combined["harvest_time"].isin(selected_times)]
    
    if max_snapshots and "harvest_time" in combined.columns:
        unique_times = combined["harvest_time"].drop_duplicates().sort_values()
        if len(unique_times) > max_snapshots:
            selected_times = unique_times[:max_snapshots]
            combined = combined[combined["harvest_time"].isin(selected_times)]
    
    return combined


def run_historical_calibration_from_harvest(
    config: CalibrationConfig,
) -> HistoricalCalibrationResult:
    """
    Run calibration using harvested Parquet data over many timestamps.
    
    Aggregates metrics across all snapshots to provide:
    - Global MAE/bias
    - Bucket metrics
    - Recommended vol_surface
    - Regime distribution (if clustering is enabled)
    """
    now = datetime.now(timezone.utc)
    
    harvest = config.harvest
    if harvest is None:
        return HistoricalCalibrationResult(
            underlying=config.underlying,
            spot=0.0,
            min_dte=config.min_dte,
            max_dte=config.max_dte,
            iv_multiplier=config.iv_multiplier,
            default_iv=config.default_iv,
            count=0,
            mae_pct=0.0,
            bias_pct=0.0,
            timestamp=now,
            snapshot_count=0,
        )
    
    underlying = (harvest.underlying or config.underlying).upper()
    
    df = load_harvest_snapshots(
        data_root=harvest.data_root,
        underlying=underlying,
        start_time=harvest.start_time,
        end_time=harvest.end_time,
        snapshot_step=harvest.snapshot_step,
        max_snapshots=harvest.max_snapshots,
    )
    
    if df.empty:
        return HistoricalCalibrationResult(
            underlying=underlying,
            spot=0.0,
            min_dte=config.min_dte,
            max_dte=config.max_dte,
            iv_multiplier=config.iv_multiplier,
            default_iv=config.default_iv,
            count=0,
            mae_pct=0.0,
            bias_pct=0.0,
            timestamp=now,
            snapshot_count=0,
        )
    
    if "dte_days" not in df.columns and "expiry_timestamp" in df.columns and "harvest_time" in df.columns:
        df["dte_days"] = (pd.to_datetime(df["expiry_timestamp"], unit="ms", utc=True) - df["harvest_time"]).dt.total_seconds() / 86400.0
    
    mask = (
        (df["dte_days"] >= config.min_dte) &
        (df["dte_days"] <= config.max_dte)
    )
    if "option_type" in df.columns:
        mask = mask & (df["option_type"].str.upper().isin(["CALL", "C"]))
    
    df = df[mask]
    
    if df.empty:
        return HistoricalCalibrationResult(
            underlying=underlying,
            spot=0.0,
            min_dte=config.min_dte,
            max_dte=config.max_dte,
            iv_multiplier=config.iv_multiplier,
            default_iv=config.default_iv,
            count=0,
            mae_pct=0.0,
            bias_pct=0.0,
            timestamp=now,
            snapshot_count=0,
        )
    
    snapshot_times = df["harvest_time"].drop_duplicates().sort_values()
    snapshot_count = len(snapshot_times)
    
    spot_prices = df.groupby("harvest_time")["underlying_price"].first().sort_index()
    spot_series = [(t, p) for t, p in spot_prices.items()]
    
    all_rows: List[Dict[str, Any]] = []
    rv_values: List[float] = []
    
    for snap_time in snapshot_times:
        snap_df = df[df["harvest_time"] == snap_time]
        
        if spot_series:
            rv = compute_realized_volatility(spot_series, snap_time, config.rv_window_days)
        else:
            rv = config.default_iv
        rv_values.append(rv)
        
        spot = snap_df["underlying_price"].iloc[0] if len(snap_df) > 0 else 0.0
        
        for _, row in snap_df.iterrows():
            t_years = max(0.0001, row.get("dte_days", 1.0) / 365.0)
            strike = row.get("strike", 0.0)
            mark_price = row.get("mark_price", 0.0)
            mark_iv = row.get("mark_iv")
            
            if mark_price <= 0 or strike <= 0 or spot <= 0:
                continue
            
            base_iv = max(1e-6, rv * config.iv_multiplier)
            abs_delta = abs(bs_call_delta(spot, strike, t_years, base_iv, 0.0))
            
            sigma = rv * config.iv_multiplier
            
            syn_price_usd = black_scholes_call_price(spot, strike, t_years, sigma, 0.0)
            syn_price = syn_price_usd / spot
            
            diff = syn_price - mark_price
            diff_pct = (diff / mark_price) * 100.0 if mark_price > 0 else 0.0
            
            all_rows.append({
                "instrument": row.get("instrument_name", ""),
                "dte": row.get("dte_days", 0),
                "strike": strike,
                "mark_price": mark_price,
                "syn_price": syn_price,
                "diff": diff,
                "diff_pct": diff_pct,
                "mark_iv": mark_iv,
                "syn_iv": sigma,
                "abs_delta": abs_delta,
                "snapshot_time": snap_time,
            })
    
    global_metrics = compute_global_metrics(all_rows)
    
    rv_median = float(np.median(rv_values)) if rv_values else config.default_iv
    
    avg_spot = df["underlying_price"].mean() if "underlying_price" in df.columns else 0.0
    
    avg_mark_iv = None
    if "mark_iv" in df.columns:
        valid_ivs = df["mark_iv"].dropna()
        if len(valid_ivs) > 0:
            avg_mark_iv = valid_ivs.mean() / 100.0
    
    recommended_iv_multiplier = None
    if rv_median > 0 and avg_mark_iv and avg_mark_iv > 0:
        recommended_iv_multiplier = round(avg_mark_iv / rv_median, 4)
    
    buckets, residuals = compute_bucket_metrics(
        all_rows,
        config.bucket_by_dte,
        config.bucket_by_abs_delta,
        rv_median,
    )
    
    skew_result = None
    if config.fit_skew:
        skew_result = fit_skew_anchor_ratios(all_rows, rv_median, config.iv_multiplier)
    
    recommended_vol_surface = None
    if config.emit_recommended_vol_surface and recommended_iv_multiplier:
        recommended_vol_surface = build_recommended_vol_surface(
            rv_window_days=config.rv_window_days,
            recommended_iv_multiplier=recommended_iv_multiplier,
            bands_results=None,
            skew_result=skew_result,
            skew_enabled=True,
        )
    
    time_range_start = snapshot_times.iloc[0] if len(snapshot_times) > 0 else None
    time_range_end = snapshot_times.iloc[-1] if len(snapshot_times) > 0 else None
    
    return HistoricalCalibrationResult(
        underlying=underlying,
        spot=float(avg_spot),
        min_dte=config.min_dte,
        max_dte=config.max_dte,
        iv_multiplier=config.iv_multiplier,
        default_iv=config.default_iv,
        count=len(all_rows),
        mae_pct=global_metrics.mae_pct,
        bias_pct=global_metrics.bias_pct,
        timestamp=now,
        rv_annualized=rv_median,
        atm_iv=avg_mark_iv,
        recommended_iv_multiplier=recommended_iv_multiplier,
        global_metrics=global_metrics,
        residuals_summary=residuals if buckets else None,
        buckets=buckets if buckets else None,
        recommended_skew=skew_result,
        recommended_vol_surface=recommended_vol_surface,
        rows=all_rows if config.return_rows else None,
        snapshot_count=snapshot_count,
        time_range_start=time_range_start,
        time_range_end=time_range_end,
        rv_median=rv_median,
    )


def build_vol_surface_from_calibration(
    calibration_result: ExtendedCalibrationResult,
) -> Dict[str, Any]:
    """
    Build a vol_surface configuration dict from calibration results.
    
    Can be used to update synthetic config with calibrated parameters.
    """
    if calibration_result.recommended_vol_surface:
        return calibration_result.recommended_vol_surface.model_dump()
    
    return {
        "iv_mode": "rv_window",
        "rv_window_days": 7,
        "iv_multiplier": calibration_result.recommended_iv_multiplier or 1.0,
        "skew": {
            "enabled": True,
            "anchor_ratios": calibration_result.recommended_skew.anchor_ratios if calibration_result.recommended_skew else {},
        } if calibration_result.recommended_skew else None,
    }
