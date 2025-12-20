"""
Position and calibration routes for the Options Trading Agent.
Includes open/closed positions, calibration endpoints, and reconciliation.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

import httpx

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.status_store import status_store
from src.config import settings
from src.position_tracker import position_tracker
from src.calibration_extended import run_calibration_extended, CalibrationConfig

router = APIRouter()


class ForceApplyCalibrationRequest(BaseModel):
    underlying: str = "BTC"
    source: str = "live"
    min_dte: float = 3.0
    max_dte: float = 30.0


class ReconciliationConfigUpdate(BaseModel):
    """Request model for updating position reconciliation configuration."""
    position_reconcile_action: Optional[Literal["halt", "auto_heal"]] = None
    position_reconcile_on_startup: Optional[bool] = None
    position_reconcile_on_each_loop: Optional[bool] = None
    position_reconcile_tolerance_usd: Optional[float] = Field(default=None, ge=0.0)


@router.get("/api/positions/open")
def get_open_positions() -> JSONResponse:
    """
    Return open positions for the UI with live mark prices and PnL from Deribit.
    - Merges bot-managed chains from PositionTracker with live Deribit data.
    - Falls back to live Deribit positions if no bot-managed chains exist.
    """
    status = status_store.get() or {}
    state = status.get("state") or {}
    portfolio = state.get("portfolio") or {}
    live_positions = portfolio.get("positions") or []
    spot_prices = state.get("spot") or {}
    
    live_by_symbol: Dict[str, Dict[str, Any]] = {}
    for p in live_positions:
        symbol = p.get("symbol")
        if symbol:
            live_by_symbol[symbol] = p
    
    payload = position_tracker.get_open_positions_payload()
    bot_positions = payload.get("positions") or []
    
    if bot_positions:
        enriched_positions: List[Dict[str, Any]] = []
        for pos in bot_positions:
            enriched = dict(pos)
            symbol = enriched.get("symbol")
            underlying = enriched.get("underlying", "BTC")
            spot = float(spot_prices.get(underlying, 0.0))
            live_data = live_by_symbol.get(symbol, {})
            
            if live_data:
                live_mark = float(live_data.get("mark_price") or 0.0)
                live_pnl = float(live_data.get("unrealized_pnl") or 0.0)
                entry_price_btc = float(enriched.get("entry_price") or 0.0)
                qty = abs(float(enriched.get("quantity") or 1.0))
                
                if live_mark > 0:
                    enriched["mark_price"] = live_mark
                    enriched["unrealized_pnl"] = live_pnl
                    if entry_price_btc > 0 and qty > 0 and spot > 0:
                        notional_usd = entry_price_btc * qty * spot
                        enriched["unrealized_pnl_pct"] = (live_pnl / notional_usd) * 100.0 if notional_usd > 0 else 0.0
            
            enriched_positions.append(enriched)
        
        total_pnl = sum(float(p.get("unrealized_pnl", 0.0)) for p in enriched_positions)
        total_notional_usd = 0.0
        for p in enriched_positions:
            underlying = p.get("underlying", "BTC")
            spot = float(spot_prices.get(underlying, 0.0))
            entry = abs(float(p.get("entry_price", 0.0)))
            qty = abs(float(p.get("quantity", 0.0)))
            total_notional_usd += entry * qty * spot
        
        totals = {
            "positions_count": len(enriched_positions),
            "unrealized_pnl": total_pnl,
            "unrealized_pnl_pct": (total_pnl / total_notional_usd * 100.0) if total_notional_usd > 0 else 0.0,
        }
        
        return JSONResponse(content={"positions": enriched_positions, "totals": totals})
    
    positions: List[Dict[str, Any]] = []
    for p in live_positions:
        try:
            side = p.get("side", "sell")
            option_type = p.get("option_type", "CALL")
            pnl = float(p.get("unrealized_pnl") or 0.0)
            entry = float(p.get("avg_price", 0.0))
            size = abs(float(p.get("size", 0.0)))
            underlying = p.get("underlying", "BTC")
            spot = float(spot_prices.get(underlying, 0.0))
            notional_usd = entry * size * spot if entry > 0 and size > 0 and spot > 0 else 1.0
            pnl_pct = (pnl / notional_usd * 100.0) if notional_usd > 0 else 0.0
            
            positions.append({
                "position_id": f"live-{p.get('symbol')}",
                "underlying": underlying,
                "symbol": p.get("symbol"),
                "option_type": option_type,
                "strategy_type": "LIVE_POSITION",
                "side": "SHORT" if side == "sell" else "LONG",
                "quantity": size,
                "entry_price": entry,
                "mark_price": float(p.get("mark_price") or 0.0),
                "unrealized_pnl": pnl,
                "unrealized_pnl_pct": pnl_pct,
                "entry_time": None,
                "expiry": None,
                "dte": float(p.get("expiry_dte") or 0.0),
                "num_rolls": 0,
                "mode": "LIVE",
                "entry_mode": "NATURAL",
                "exit_style": "unknown",
            })
        except Exception:
            continue

    total_pnl = sum(pos["unrealized_pnl"] for pos in positions) if positions else 0.0
    total_notional_usd = 0.0
    for pos in positions:
        underlying = pos.get("underlying", "BTC")
        spot = float(spot_prices.get(underlying, 0.0))
        total_notional_usd += pos["entry_price"] * pos["quantity"] * spot

    totals = {
        "positions_count": len(positions),
        "unrealized_pnl": total_pnl,
        "unrealized_pnl_pct": (total_pnl / total_notional_usd * 100.0) if total_notional_usd > 0 else 0.0,
    }

    return JSONResponse(content={"positions": positions, "totals": totals})


@router.get("/api/positions/closed")
def get_closed_positions() -> JSONResponse:
    """Return closed bot-managed chains with realized PnL."""
    payload = position_tracker.get_closed_positions_payload()
    return JSONResponse(content=payload)


@router.get("/api/calibration")
def get_calibration(
    underlying: str = "BTC",
    min_dte: float = 3.0,
    max_dte: float = 10.0,
    iv_multiplier: float = 1.0,
    default_iv: float = 0.6,
) -> JSONResponse:
    """
    Run a quick synthetic-vs-Deribit calibration for near-dated calls.
    Returns JSON with summary metrics and up to ~80 sample rows.
    Also returns term structure bands (weekly/monthly/quarterly) from broader DTE range.
    """
    if underlying not in ("BTC", "ETH"):
        return JSONResponse(
            status_code=400,
            content={"error": "underlying must be BTC or ETH"},
        )
    
    try:
        config = CalibrationConfig(
            underlying=underlying,
            min_dte=min_dte,
            max_dte=max_dte,
            iv_multiplier=iv_multiplier,
            default_iv=default_iv,
            option_types=["C"],
            return_rows=True,
            fit_skew=True,
        )
        result = run_calibration_extended(config)
        
        term_structure_bands = None
        try:
            broad_config = CalibrationConfig(
                underlying=underlying,
                min_dte=1.0,
                max_dte=120.0,
                iv_multiplier=iv_multiplier,
                default_iv=default_iv,
                option_types=["C"],
                return_rows=False,
            )
            broad_result = run_calibration_extended(broad_config)
            if broad_result.bands:
                term_structure_bands = [
                    {
                        "band_name": b.name,
                        "dte_range": f"{b.min_dte}-{b.max_dte}",
                        "option_type": b.option_type,
                        "count": b.count,
                        "mae_pct": b.mae_pct,
                        "bias_pct": b.bias_pct,
                        "recommended_iv_multiplier": b.recommended_iv_multiplier,
                        "vega_weighted_mae_pct": b.vega_weighted_mae_pct,
                    }
                    for b in broad_result.bands
                ]
        except Exception:
            pass

        bands_data = None
        if result.bands:
            bands_data = [
                {
                    "band_name": b.name,
                    "dte_range": f"{b.min_dte}-{b.max_dte}",
                    "option_type": b.option_type,
                    "count": b.count,
                    "mae_pct": b.mae_pct,
                    "bias_pct": b.bias_pct,
                    "recommended_iv_multiplier": b.recommended_iv_multiplier,
                }
                for b in result.bands
            ]
        
        by_option_type_data = None
        if result.by_option_type:
            by_option_type_data = {
                ot: {
                    "count": m.count,
                    "mae_pct": m.mae_pct,
                    "bias_pct": m.bias_pct,
                    "mae_vol_points": m.mae_vol_points,
                    "vega_weighted_mae_pct": m.vega_weighted_mae_pct,
                }
                for ot, m in result.by_option_type.items()
            }

        skew_fit_data = None
        if result.recommended_skew:
            from src.calibration_store import get_current_skew_ratios
            current_ratios = get_current_skew_ratios(underlying)
            current_skew = {
                "anchor_ratios": current_ratios,
                "min_dte": result.recommended_skew.min_dte,
                "max_dte": result.recommended_skew.max_dte,
            }
            skew_fit_data = {
                "recommended_skew": {
                    "anchor_ratios": result.recommended_skew.anchor_ratios,
                    "min_dte": result.recommended_skew.min_dte,
                    "max_dte": result.recommended_skew.max_dte,
                },
                "current_skew": current_skew,
                "skew_misfit": {
                    "anchor_diffs": result.skew_misfit.anchor_diffs if result.skew_misfit else {},
                    "max_abs_diff": result.skew_misfit.max_abs_diff if result.skew_misfit else 0.0,
                } if result.skew_misfit else None,
            }

        payload = {
            "underlying": result.underlying,
            "spot": result.spot,
            "min_dte": result.min_dte,
            "max_dte": result.max_dte,
            "iv_multiplier": result.iv_multiplier,
            "default_iv": result.default_iv,
            "rv_annualized": result.rv_annualized,
            "rv_source": result.rv_source,
            "atm_iv": result.atm_iv,
            "atm_source": result.atm_source,
            "recommended_iv_multiplier": result.recommended_iv_multiplier,
            "count": result.count,
            "mae_pct": result.mae_pct,
            "bias_pct": result.bias_pct,
            "timestamp": result.timestamp.isoformat(),
            "option_types_used": result.option_types_used,
            "bands": bands_data,
            "term_structure_bands": term_structure_bands,
            "by_option_type": by_option_type_data,
            "skew_fit": skew_fit_data,
            "rows": result.rows if result.rows else [],
        }
        return JSONResponse(content=payload)
    except ValueError as e:
        return JSONResponse(
            status_code=400,
            content={"error": "validation_error", "message": str(e), "error_type": "validation"},
        )
    except httpx.TimeoutException as e:
        return JSONResponse(
            status_code=504,
            content={"error": "deribit_timeout", "message": "Deribit API timeout, please retry", "error_type": "timeout"},
        )
    except httpx.HTTPError as e:
        return JSONResponse(
            status_code=502,
            content={"error": "deribit_error", "message": f"Deribit API error: {str(e)}", "error_type": "api_error"},
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "calibration_failed", "message": str(e), "error_type": "internal"},
        )


@router.get("/api/calibration/history")
def get_calibration_history(
    underlying: str = "BTC",
    limit: int = 20,
) -> JSONResponse:
    """
    Get recent calibration history entries from the database.
    """
    if underlying not in ("BTC", "ETH"):
        return JSONResponse(
            status_code=400,
            content={"error": "underlying must be BTC or ETH"},
        )
    
    try:
        from src.db.models_calibration import list_recent_calibrations
        entries = list_recent_calibrations(underlying=underlying, limit=limit)
        
        return JSONResponse(content={
            "underlying": underlying,
            "entries": [
                {
                    "id": e.id,
                    "created_at": e.created_at.isoformat() if e.created_at else None,
                    "dte_min": e.dte_min,
                    "dte_max": e.dte_max,
                    "lookback_days": e.lookback_days,
                    "multiplier": e.multiplier,
                    "mae_pct": e.mae_pct,
                    "vega_weighted_mae_pct": e.vega_weighted_mae_pct,
                    "bias_pct": e.bias_pct,
                    "num_samples": e.num_samples,
                    "source": e.source,
                    "status": e.status,
                    "reason": e.reason,
                }
                for e in entries
            ],
        })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "failed_to_fetch_history", "message": str(e)},
        )


@router.post("/api/calibration/use_latest")
def use_latest_calibration(request: dict) -> JSONResponse:
    """
    Apply the latest calibration multiplier from history as a runtime override.
    
    Also updates the "Current Applied Multipliers" panel via set_applied_multiplier.
    
    Body: {"underlying": "BTC", "dte_min": 3, "dte_max": 10}
    """
    underlying = request.get("underlying", "BTC")
    dte_min = request.get("dte_min", 3)
    dte_max = request.get("dte_max", 10)
    
    if underlying not in ("BTC", "ETH"):
        return JSONResponse(
            status_code=400,
            content={"error": "underlying must be BTC or ETH"},
        )
    
    try:
        from src.db.models_calibration import get_latest_calibration
        from src.calibration_store import set_iv_multiplier_override, set_applied_multiplier
        
        entry = get_latest_calibration(
            underlying=underlying,
            dte_min=dte_min,
            dte_max=dte_max,
            skip_failed=True,
        )
        
        if entry is None:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "no_calibration_found",
                    "message": f"No valid calibration found for {underlying} in {dte_min}-{dte_max} DTE range. All calibrations may have failed guardrails. Run a new calibration.",
                },
            )
        
        set_iv_multiplier_override(underlying, entry.multiplier, dte_min, dte_max)
        
        set_applied_multiplier(
            underlying=underlying,
            global_multiplier=entry.multiplier,
            band_multipliers=None,
            source=entry.source or "harvested",
            applied_reason=f"User force-applied from {dte_min}-{dte_max} DTE band",
        )
        
        return JSONResponse(content={
            "status": "ok",
            "underlying": underlying,
            "dte_min": dte_min,
            "dte_max": dte_max,
            "multiplier": entry.multiplier,
            "mae_pct": entry.mae_pct,
            "num_samples": entry.num_samples,
            "created_at": entry.created_at.isoformat() if entry.created_at else None,
        })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "failed_to_apply_calibration", "message": str(e)},
        )


@router.post("/api/calibration/apply_direct")
def apply_calibration_direct(request: dict) -> JSONResponse:
    """
    Apply a calibration multiplier directly from the frontend (from "Run Calibration" result).
    
    Body: {"underlying": "BTC", "dte_min": 3, "dte_max": 10, "multiplier": 1.106, "mae_pct": 19.75, "num_samples": 36}
    """
    from src.db.models_calibration import MIN_REASONABLE_MULT, MAX_REASONABLE_MULT
    from src.calibration_store import set_iv_multiplier_override
    
    underlying = request.get("underlying", "BTC")
    dte_min = request.get("dte_min", 3)
    dte_max = request.get("dte_max", 10)
    multiplier = request.get("multiplier")
    
    if underlying not in ("BTC", "ETH"):
        return JSONResponse(
            status_code=400,
            content={"error": "underlying must be BTC or ETH"},
        )
    
    if multiplier is None:
        return JSONResponse(
            status_code=400,
            content={"error": "multiplier is required"},
        )
    
    if multiplier < MIN_REASONABLE_MULT or multiplier > MAX_REASONABLE_MULT:
        return JSONResponse(
            status_code=400,
            content={
                "error": "calibration_out_of_bounds",
                "message": f"Multiplier {multiplier:.4f} is outside guardrail bounds ({MIN_REASONABLE_MULT}-{MAX_REASONABLE_MULT}).",
            },
        )
    
    try:
        set_iv_multiplier_override(underlying, multiplier, dte_min, dte_max)
        
        return JSONResponse(content={
            "status": "ok",
            "underlying": underlying,
            "dte_min": dte_min,
            "dte_max": dte_max,
            "multiplier": multiplier,
        })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "failed_to_apply_calibration", "message": str(e)},
        )


@router.get("/api/calibration/overrides")
def get_calibration_overrides() -> JSONResponse:
    """
    Get current IV multiplier runtime overrides.
    """
    try:
        from src.calibration_store import get_all_overrides
        overrides = get_all_overrides()
        return JSONResponse(content={"overrides": overrides})
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "failed_to_get_overrides", "message": str(e)},
        )


@router.get("/api/calibration/policy")
def get_calibration_policy() -> JSONResponse:
    """
    Get the current calibration update policy configuration.
    This explains the thresholds used for deciding when to apply calibration updates.
    """
    try:
        from src.calibration_update_policy import get_policy
        policy = get_policy()
        return JSONResponse(content={
            "min_delta_global": policy.min_delta_global,
            "min_delta_band": policy.min_delta_band,
            "min_sample_size": policy.min_sample_size,
            "min_vega_sum": policy.min_vega_sum,
            "smoothing_window_days": policy.smoothing_window_days,
            "ewma_alpha": policy.ewma_alpha,
            "explanation": (
                f"The system smooths calibration results over the last {policy.smoothing_window_days} days "
                f"and only updates IV multipliers when: (1) The change is larger than {policy.min_delta_global} "
                f"(e.g., 0.03), and (2) There are at least {policy.min_sample_size} samples with sufficient vega "
                f"({policy.min_vega_sum}+). This prevents overreacting to noisy days and keeps the synthetic universe stable."
            ),
        })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "failed_to_get_policy", "message": str(e)},
        )


@router.get("/api/calibration/current_multipliers")
def get_current_multipliers(underlying: str = "BTC") -> JSONResponse:
    """
    Get the currently applied IV multipliers.
    
    This reads from the calibration store, which is updated when:
    - A live calibration is applied via policy
    - User clicks "Force-Apply Latest"
    """
    try:
        from src.calibration_update_policy import get_current_applied_multipliers
        
        current = get_current_applied_multipliers(underlying)
        
        last_applied = current.last_updated.isoformat() if current.last_updated else None
        
        bands_list = None
        if current.band_multipliers:
            bands_list = [
                {
                    "name": b.name,
                    "min_dte": b.min_dte,
                    "max_dte": b.max_dte,
                    "iv_multiplier": b.iv_multiplier,
                }
                for b in current.band_multipliers
            ]
        
        return JSONResponse(content={
            "underlying": underlying,
            "global_multiplier": current.global_multiplier,
            "band_multipliers": bands_list,
            "last_updated": last_applied,
        })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "failed_to_get_multipliers", "message": str(e)},
        )


@router.get("/api/calibration/runs")
def get_calibration_runs(
    underlying: str = "BTC",
    limit: int = 20,
) -> JSONResponse:
    """
    Get recent calibration runs from the file-based history store.
    Returns full run details including smoothed values and apply decisions.
    """
    try:
        from src.calibration_update_policy import load_recent_calibration_history
        
        runs = load_recent_calibration_history(underlying, limit=limit)
        
        return JSONResponse(content={
            "underlying": underlying,
            "runs": [
                {
                    "timestamp": r.timestamp.isoformat() if r.timestamp else None,
                    "source": r.source,
                    "recommended_iv_multiplier": r.recommended_iv_multiplier,
                    "smoothed_global_multiplier": r.smoothed_global_multiplier,
                    "sample_size": r.sample_size,
                    "vega_sum": r.vega_sum,
                    "applied": r.applied,
                    "applied_reason": r.applied_reason,
                    "bands": [
                        {"name": b.name, "iv_multiplier": b.iv_multiplier}
                        for b in (r.recommended_band_multipliers or [])
                    ] if r.recommended_band_multipliers else None,
                    "smoothed_bands": [
                        {"name": b.name, "iv_multiplier": b.iv_multiplier}
                        for b in (r.smoothed_band_multipliers or [])
                    ] if r.smoothed_band_multipliers else None,
                }
                for r in runs
            ],
        })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "failed_to_get_runs", "message": str(e)},
        )


@router.get("/api/calibration/fidelity/latest")
def get_fidelity_latest(
    underlying: str = "BTC",
) -> JSONResponse:
    """Return the latest Synthetic Fidelity report (if any)."""
    if underlying not in ("BTC", "ETH"):
        return JSONResponse(status_code=400, content={"error": "underlying must be BTC or ETH"})

    try:
        from src.fidelity.fidelity_store import load_latest_report

        report = load_latest_report(underlying)
        return JSONResponse(
            content={
                "ok": True,
                "underlying": underlying,
                "report": report,
            }
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})


@router.get("/api/calibration/fidelity/history")
def get_fidelity_history(
    underlying: str = "BTC",
    limit: int = 30,
) -> JSONResponse:
    """Return recent Synthetic Fidelity reports from the file-based history store."""
    if underlying not in ("BTC", "ETH"):
        return JSONResponse(status_code=400, content={"error": "underlying must be BTC or ETH"})

    try:
        from src.fidelity.fidelity_store import list_recent_reports

        runs = list_recent_reports(underlying, limit=limit)
        return JSONResponse(
            content={
                "ok": True,
                "underlying": underlying,
                "runs": runs,
            }
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})


@router.get("/calibration/fidelity/latest")
def get_fidelity_latest_mvp() -> JSONResponse:
    """MVP endpoint: return the latest fidelity report (run_id-scoped store)."""
    try:
        from src.fidelity.fidelity_store import load_latest_index, load_report_by_id

        latest = load_latest_index()
        if not latest or not latest.get("run_id"):
            return JSONResponse(content={"ok": True, "run_id": None, "report": None})

        run_id = str(latest.get("run_id"))
        report = load_report_by_id(run_id)
        return JSONResponse(content={"ok": True, "run_id": run_id, "report": report, "latest": latest})
    except Exception as e:
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})


@router.get("/calibration/fidelity/history")
def get_fidelity_history_mvp(limit: int = 30) -> JSONResponse:
    """MVP endpoint: list recent fidelity runs (run_id-scoped store)."""
    try:
        from src.fidelity.fidelity_store import list_history_runs

        runs = list_history_runs(limit=limit)
        return JSONResponse(content={"ok": True, "runs": runs})
    except Exception as e:
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})


@router.get("/calibration/fidelity/report/{run_id}")
def get_fidelity_report_mvp(run_id: str) -> JSONResponse:
    """MVP endpoint: fetch a specific run's report by run_id."""
    try:
        from src.fidelity.fidelity_store import load_report_by_id

        report = load_report_by_id(run_id)
        if report is None:
            return JSONResponse(status_code=404, content={"ok": False, "error": "not_found"})
        return JSONResponse(content={"ok": True, "run_id": run_id, "report": report})
    except Exception as e:
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})


@router.get("/calibration/fidelity/spec")
def get_fidelity_spec_mvp() -> JSONResponse:
    """MVP endpoint: return the suite spec (strategies + scoring components)."""
    try:
        from src.fidelity.spec import fidelity_spec

        return JSONResponse(content={"ok": True, "spec": fidelity_spec()})
    except Exception as e:
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})


@router.post("/api/calibration/force_apply")
def force_apply_calibration(request: ForceApplyCalibrationRequest) -> JSONResponse:
    """
    Force-apply the latest calibration to the vol surface config.
    This runs calibration with force=True, bypassing thresholds.
    """
    try:
        from src.calibration_update_policy import run_calibration_with_policy
        from typing import Literal
        
        source: Literal["live", "harvested"] = "live" if request.source == "live" else "harvested"
        
        record, decision = run_calibration_with_policy(
            underlying=request.underlying,
            source=source,
            force=True,
            min_dte=request.min_dte,
            max_dte=request.max_dte,
        )
        
        return JSONResponse(content={
            "status": "ok",
            "underlying": request.underlying,
            "source": request.source,
            "recommended_iv_multiplier": record.recommended_iv_multiplier,
            "smoothed_iv_multiplier": record.smoothed_global_multiplier,
            "applied": record.applied,
            "applied_reason": record.applied_reason,
            "sample_size": record.sample_size,
            "timestamp": record.timestamp.isoformat() if record.timestamp else None,
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": "failed_to_force_apply", "message": str(e)},
        )


@router.post("/api/calibration/run_with_policy")
def run_calibration_with_policy_endpoint(request: ForceApplyCalibrationRequest) -> JSONResponse:
    """
    Run calibration with the update policy (normal mode, respects thresholds).
    """
    try:
        from src.calibration_update_policy import run_calibration_with_policy
        from typing import Literal
        
        source: Literal["live", "harvested"] = "live" if request.source == "live" else "harvested"
        
        record, decision = run_calibration_with_policy(
            underlying=request.underlying,
            source=source,
            force=False,
            min_dte=request.min_dte,
            max_dte=request.max_dte,
        )
        
        return JSONResponse(content={
            "status": "ok",
            "underlying": request.underlying,
            "source": request.source,
            "recommended_iv_multiplier": record.recommended_iv_multiplier,
            "smoothed_iv_multiplier": record.smoothed_global_multiplier,
            "applied": record.applied,
            "applied_reason": record.applied_reason,
            "sample_size": record.sample_size,
            "timestamp": record.timestamp.isoformat() if record.timestamp else None,
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": "failed_to_run_calibration", "message": str(e)},
        )


@router.post("/api/calibration/apply_skew")
def apply_skew_ratios(request: dict) -> JSONResponse:
    """
    Apply recommended skew anchor ratios directly.
    
    Body: {"underlying": "BTC", "anchor_ratios": {"0.15": 0.96, "0.25": 0.94, "0.35": 0.92}}
    """
    from src.calibration_store import set_skew_anchor_ratios, get_applied_multiplier, set_applied_multiplier
    
    underlying = request.get("underlying", "BTC")
    anchor_ratios = request.get("anchor_ratios", {})
    
    if underlying not in ("BTC", "ETH"):
        return JSONResponse(
            status_code=400,
            content={"error": "underlying must be BTC or ETH"},
        )
    
    if not anchor_ratios or not isinstance(anchor_ratios, dict):
        return JSONResponse(
            status_code=400,
            content={"error": "anchor_ratios is required and must be a dict"},
        )
    
    try:
        set_skew_anchor_ratios(underlying, anchor_ratios)
        
        current_state = get_applied_multiplier(underlying)
        set_applied_multiplier(
            underlying=underlying,
            global_multiplier=current_state.global_multiplier,
            band_multipliers=current_state.band_multipliers if current_state.band_multipliers else None,
            skew_anchor_ratios=anchor_ratios,
            source=current_state.source,
            applied_reason="Skew ratios applied directly",
        )
        
        return JSONResponse(content={
            "status": "ok",
            "underlying": underlying,
            "anchor_ratios": anchor_ratios,
        })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "failed_to_apply_skew", "message": str(e)},
        )


@router.get("/api/calibration/auto_status")
def get_auto_calibration_status() -> JSONResponse:
    """
    Get the status of the most recent auto-calibrations for BTC and ETH.
    Returns the latest calibration_history entries for each underlying.
    """
    try:
        from src.db.models_calibration import list_recent_calibrations
        
        underlyings = ["BTC", "ETH"]
        results = {}
        
        for underlying in underlyings:
            entries = list_recent_calibrations(underlying=underlying, limit=1)
            if entries:
                e = entries[0]
                results[underlying] = {
                    "id": e.id,
                    "created_at": e.created_at.isoformat() if e.created_at else None,
                    "dte_range": f"{e.dte_min}-{e.dte_max}",
                    "lookback_days": e.lookback_days,
                    "multiplier": e.multiplier,
                    "mae_pct": e.mae_pct,
                    "vega_weighted_mae_pct": e.vega_weighted_mae_pct,
                    "num_samples": e.num_samples,
                    "source": e.source,
                    "status": e.status,
                    "reason": e.reason,
                }
            else:
                results[underlying] = None
        
        any_recent = any(r is not None for r in results.values())
        overall_status = "ok"
        if any_recent:
            statuses = [r.get("status", "ok") for r in results.values() if r]
            if all(s == "failed" for s in statuses):
                overall_status = "failed"
            elif any(s == "failed" for s in statuses):
                overall_status = "degraded"
            elif any(s == "degraded" for s in statuses):
                overall_status = "degraded"
        else:
            overall_status = "no_data"
        
        return JSONResponse(content={
            "overall_status": overall_status,
            "underlyings": results,
        })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "failed_to_fetch_auto_status", "message": str(e)},
        )


@router.get("/api/data_status/intraday")
def intraday_data_status() -> JSONResponse:
    """
    Return status of the Deribit intraday data scraping / storage.
    Read-only; does not trigger scraping.
    """
    from src.data_status import get_intraday_data_status
    
    try:
        status = get_intraday_data_status(settings)
        return JSONResponse(
            content={
                "ok": status.ok,
                "source": status.source,
                "backend": status.backend,
                "rows_total": status.rows_total,
                "days_covered": status.days_covered,
                "first_timestamp": status.first_timestamp.isoformat() if status.first_timestamp else None,
                "last_timestamp": status.last_timestamp.isoformat() if status.last_timestamp else None,
                "approx_size_mb": status.approx_size_mb,
                "target_interval_sec": status.target_interval_sec,
                "is_running": status.is_running,
                "error": status.error,
            }
        )
    except Exception as e:
        return JSONResponse(content={"ok": False, "error": str(e)}, status_code=500)


@router.post("/api/reconcile_positions")
def reconcile_positions_endpoint() -> JSONResponse:
    """Run position reconciliation once and return results."""
    try:
        from src.deribit_client import DeribitClient
        from src.reconciliation import run_reconciliation_once
        
        with DeribitClient() as client:
            spot_prices = {}
            for underlying in settings.underlyings:
                try:
                    spot_prices[underlying] = client.get_index_price(underlying)
                except Exception:
                    pass
            
            diff = run_reconciliation_once(
                deribit_client=client,
                position_tracker=position_tracker,
                settings=settings,
                spot_prices=spot_prices,
            )
            
            summary = {
                "deribit_positions": diff.exchange_count,
                "tracked_positions": diff.local_count,
                "missing_on_deribit": [
                    p.get("symbol", "unknown") for p in diff.missing_on_exchange
                ],
                "missing_in_tracker": [
                    p.get("instrument_name", p.get("symbol", "unknown"))
                    for p in diff.untracked_on_exchange
                ],
                "mismatched_size": [
                    {
                        "symbol": m.instrument_name,
                        "tracker": m.size_tracker,
                        "exchange": m.size_exchange,
                    }
                    for m in diff.size_mismatches
                ],
            }
            
            details = []
            if diff.is_clean:
                details.append("All positions match between Deribit and tracker.")
            else:
                if diff.missing_on_exchange:
                    details.append(f"{len(diff.missing_on_exchange)} position(s) missing on Deribit")
                if diff.untracked_on_exchange:
                    details.append(f"{len(diff.untracked_on_exchange)} position(s) untracked locally")
                if diff.size_mismatches:
                    details.append(f"{len(diff.size_mismatches)} size mismatch(es)")
            
            return JSONResponse(content={
                "ok": True,
                "is_clean": diff.is_clean,
                "summary": summary,
                "details": details,
            })
    except Exception as e:
        return JSONResponse(content={"ok": False, "error": str(e)})


@router.get("/api/reconciliation_config")
def get_reconciliation_config() -> JSONResponse:
    """Get current position reconciliation configuration."""
    try:
        return JSONResponse(content={
            "ok": True,
            "position_reconcile_action": settings.position_reconcile_action,
            "position_reconcile_on_startup": settings.position_reconcile_on_startup,
            "position_reconcile_on_each_loop": settings.position_reconcile_on_each_loop,
            "position_reconcile_tolerance_usd": settings.position_reconcile_tolerance_usd,
        })
    except Exception as e:
        return JSONResponse(content={"ok": False, "error": str(e)})


@router.post("/api/reconciliation_config")
def update_reconciliation_config(req: ReconciliationConfigUpdate) -> JSONResponse:
    """Update position reconciliation config at runtime (in-memory only)."""
    try:
        if req.position_reconcile_action is not None:
            settings.position_reconcile_action = req.position_reconcile_action
        
        if req.position_reconcile_on_startup is not None:
            settings.position_reconcile_on_startup = req.position_reconcile_on_startup
        
        if req.position_reconcile_on_each_loop is not None:
            settings.position_reconcile_on_each_loop = req.position_reconcile_on_each_loop
        
        if req.position_reconcile_tolerance_usd is not None:
            settings.position_reconcile_tolerance_usd = req.position_reconcile_tolerance_usd
        
        return get_reconciliation_config()
    except Exception as e:
        return JSONResponse(content={"ok": False, "error": str(e)})
