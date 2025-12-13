"""
FastAPI web application for the Options Trading Agent.
Provides live status, chat interface, Live Agent Dashboard, and Backtesting Lab.
"""
from __future__ import annotations

import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from fastapi import FastAPI, Body, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import cast

from agent_loop import run_agent_loop_forever
from src.status_store import status_store
from src.decisions_store import decisions_store
from src.chat_with_agent import chat_with_agent_full, get_chat_messages, clear_chat_history
from src.config import settings
from src.position_tracker import position_tracker
from src.calibration import run_calibration
from src.calibration_extended import run_calibration_extended, CalibrationConfig
from src.strategy_status import build_strategy_status, StrategyStatus
from src.rules_summary import build_rules_summary, build_rules_summary_from_settings
from src.backtest.config_schema import (
    BacktestConfig,
    ResolvedBacktestConfig,
    BacktestPreset,
)
from src.backtest.config_presets import resolve_backtest_config, get_preset_config


app = FastAPI(
    title="Options Trading Agent Dashboard",
    description="Deribit testnet covered-call agent with live status, chat, and backtesting.",
    version="0.2.0",
)


def _agent_thread_target() -> None:
    """Run the agent loop forever, updating status_store each iteration."""
    def status_callback(snapshot: Dict[str, Any]) -> None:
        status_store.update(snapshot)

    run_agent_loop_forever(status_callback=status_callback)


@app.on_event("startup")
def start_background_agent() -> None:
    """Start the agent loop in a background thread on FastAPI startup."""
    try:
        from src.db import init_db
        init_db()
    except Exception as e:
        print(f"[DB] Warning: Could not initialize database: {e}")
    
    thread = threading.Thread(target=_agent_thread_target, daemon=True)
    thread.start()
    print("Agent loop started in background thread")


@app.get("/status")
def get_status() -> JSONResponse:
    """Return the latest agent status snapshot."""
    data = status_store.get()
    return JSONResponse(content=data)


@app.get("/health")
def health_check() -> JSONResponse:
    """Health check endpoint for deployment."""
    return JSONResponse(content={"status": "healthy", "service": "options-trading-agent"})


@app.post("/chat")
def chat_endpoint(
    payload: Dict[str, Any] = Body(..., example={"question": "Why did you pick the 97k call?"}),
) -> JSONResponse:
    """Ask the agent a question about its recent behavior. Returns full conversation history."""
    question = payload.get("question", "").strip()
    if not question:
        return JSONResponse(
            status_code=400,
            content={"error": "Missing 'question' field in request body."},
        )

    try:
        result = chat_with_agent_full(question, log_limit=20)
        return JSONResponse(content={"question": question, "answer": result["answer"], "messages": result["messages"]})
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to generate answer: {str(e)}"},
        )


@app.get("/chat/messages")
def get_chat_history_endpoint() -> JSONResponse:
    """Get the full chat conversation history."""
    return JSONResponse(content={"messages": get_chat_messages()})


@app.post("/chat/clear")
def clear_chat_endpoint() -> JSONResponse:
    """Clear the chat conversation history."""
    clear_chat_history()
    return JSONResponse(content={"status": "cleared", "messages": []})


@app.get("/api/agent/decisions")
def get_agent_decisions() -> JSONResponse:
    """Return recent agent decisions for the dashboard."""
    decisions = decisions_store.get_all()
    last_update = decisions_store.get_last_update()
    
    return JSONResponse(content={
        "mode": "llm" if settings.llm_enabled else "rule_based",
        "llm_enabled": settings.llm_enabled,
        "dry_run": settings.dry_run,
        "training_mode": settings.is_training_enabled,
        "last_update": last_update.isoformat() if last_update else None,
        "decisions": decisions,
    })


@app.get("/api/training/status")
def get_training_status() -> JSONResponse:
    """Get current training mode status."""
    return JSONResponse(content={
        "enabled": settings.is_training_enabled,
        "training_mode": settings.training_mode,
        "strategies": settings.training_strategies,
        "is_research": settings.is_research,
        "dry_run": settings.dry_run,
    })


@app.post("/api/training/toggle")
def toggle_training_mode(payload: Dict[str, Any] = Body(...)) -> JSONResponse:
    """Toggle training mode on/off."""
    enable = payload.get("enable", False)
    
    if enable:
        if not settings.is_research:
            return JSONResponse(
                status_code=400,
                content={"error": "Training mode requires RESEARCH mode"},
            )
    
    settings.training_mode = enable
    
    return JSONResponse(content={
        "enabled": settings.is_training_enabled,
        "training_mode": settings.training_mode,
        "strategies": settings.training_strategies,
    })


@app.get("/api/strategy-status", response_model=StrategyStatus)
def get_strategy_status() -> JSONResponse:
    """
    Get current strategy and safeguards status for the UI.
    Shows mode, network, active rules, and safeguard states.
    """
    status = status_store.get() or {}
    config_snapshot = status.get("config_snapshot") or {}
    strategy_status = build_strategy_status(config_snapshot)
    return JSONResponse(content=strategy_status.model_dump())


@app.get("/api/rules-summary")
def get_rules_summary() -> JSONResponse:
    """Get the current rules summary for UI display."""
    summary = build_rules_summary_from_settings()
    return JSONResponse(content=summary)


@app.get("/api/backtest/presets")
def get_backtest_presets() -> JSONResponse:
    """Get all available backtest preset configurations."""
    presets = {}
    for preset in [BacktestPreset.ULTRA_SAFE, BacktestPreset.BALANCED, BacktestPreset.AGGRESSIVE]:
        cfg = get_preset_config(preset)
        presets[preset.value] = {
            "preset": cfg.preset.value,
            "mode": cfg.mode.value,
            "rule_toggles": cfg.rule_toggles.model_dump(),
            "thresholds": {
                **cfg.thresholds.model_dump(),
                "delta_range": list(cfg.thresholds.delta_range) if cfg.thresholds.delta_range else None,
                "dte_range": list(cfg.thresholds.dte_range) if cfg.thresholds.dte_range else None,
            },
        }
    return JSONResponse(content=presets)


@app.post("/api/backtest/resolve-config")
def resolve_backtest_config_endpoint(config: BacktestConfig) -> JSONResponse:
    """
    Resolve a backtest config with preset defaults and overrides.
    Returns the fully resolved config that would be used for a backtest.
    """
    resolved = resolve_backtest_config(config)
    summary = build_rules_summary(resolved)
    return JSONResponse(content={
        "resolved_config": {
            "preset": resolved.preset.value,
            "mode": resolved.mode.value,
            "rule_toggles": resolved.rule_toggles.model_dump(),
            "thresholds": {
                **resolved.thresholds.model_dump(),
                "delta_range": list(resolved.thresholds.delta_range) if resolved.thresholds.delta_range else None,
                "dte_range": list(resolved.thresholds.dte_range) if resolved.thresholds.dte_range else None,
            },
        },
        "rules_summary": summary,
    })


@app.get("/api/positions/open")
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


@app.get("/api/positions/closed")
def get_closed_positions() -> JSONResponse:
    """Return closed bot-managed chains with realized PnL."""
    payload = position_tracker.get_closed_positions_payload()
    return JSONResponse(content=payload)


@app.get("/api/calibration")
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


@app.get("/api/calibration/history")
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


@app.post("/api/calibration/use_latest")
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


@app.post("/api/calibration/apply_direct")
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


@app.get("/api/calibration/overrides")
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


@app.get("/api/calibration/policy")
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


@app.get("/api/calibration/current_multipliers")
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


@app.get("/api/calibration/runs")
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


class ForceApplyCalibrationRequest(BaseModel):
    underlying: str = "BTC"
    source: str = "live"
    min_dte: float = 3.0
    max_dte: float = 30.0


@app.post("/api/calibration/force_apply")
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


@app.post("/api/calibration/run_with_policy")
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


@app.post("/api/calibration/apply_skew")
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


@app.get("/api/data_status/intraday")
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


class BacktestRequest(BaseModel):
    underlying: str = "BTC"
    start: str
    end: str
    timeframe: str = "1h"
    decision_interval_bars: int = 24
    target_dte: int = 7
    target_delta: float = 0.25
    dte_tolerance: int = 2
    delta_tolerance: float = 0.05
    initial_position: float = 1.0
    # Hybrid synthetic mode settings
    sigma_mode: str = "rv_x_multiplier"
    chain_mode: str = "synthetic_grid"


from typing import Literal as TypingLiteral

class BacktestStartRequest(BaseModel):
    underlying: str = "BTC"
    start: str
    end: str
    timeframe: str = "1h"
    decision_interval_hours: int = 24
    exit_style: str = "hold_to_expiry"
    target_dte: int = 7
    target_delta: float = 0.25
    min_dte: int = 3
    max_dte: int = 21
    delta_min: float = 0.15
    delta_max: float = 0.35
    margin_type: TypingLiteral["inverse", "linear"] = "inverse"
    settlement_ccy: TypingLiteral["ANY", "USDC", "BTC", "ETH"] = "ANY"
    # Hybrid synthetic mode settings
    sigma_mode: str = "rv_x_multiplier"
    chain_mode: str = "synthetic_grid"
    # Selector/strategy for decision making
    selector_name: str = "generic_covered_call"


@app.post("/api/backtest/start")
def start_backtest(req: BacktestStartRequest) -> JSONResponse:
    """Start a new backtest in the background."""
    from src.backtest.manager import backtest_manager
    
    try:
        start_dt = datetime.fromisoformat(req.start.replace("Z", "+00:00"))
        end_dt = datetime.fromisoformat(req.end.replace("Z", "+00:00"))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {e}")
    
    if start_dt.tzinfo is None:
        start_dt = start_dt.replace(tzinfo=timezone.utc)
    if end_dt.tzinfo is None:
        end_dt = end_dt.replace(tzinfo=timezone.utc)
    
    if start_dt >= end_dt:
        raise HTTPException(status_code=400, detail="Start date must be before end date")
    
    valid_exit_styles = ["hold_to_expiry", "tp_and_roll", "both"]
    if req.exit_style not in valid_exit_styles:
        raise HTTPException(status_code=400, detail=f"Invalid exit_style. Must be one of: {valid_exit_styles}")
    
    started = backtest_manager.start(
        underlying=req.underlying,
        start_date=start_dt,
        end_date=end_dt,
        timeframe=req.timeframe,
        decision_interval_hours=req.decision_interval_hours,
        exit_style=req.exit_style,
        target_dte=req.target_dte,
        target_delta=req.target_delta,
        min_dte=req.min_dte,
        max_dte=req.max_dte,
        delta_min=req.delta_min,
        delta_max=req.delta_max,
        margin_type=req.margin_type,
        settlement_ccy=req.settlement_ccy,
        sigma_mode=req.sigma_mode,
        chain_mode=req.chain_mode,
        selector_name=req.selector_name,
    )
    
    if not started:
        return JSONResponse(
            status_code=409,
            content={"started": False, "error": "Backtest already running"},
        )
    
    return JSONResponse(content={"started": True})


@app.get("/api/backtest/status")
def get_backtest_status() -> JSONResponse:
    """Get the current backtest status."""
    from src.backtest.manager import backtest_manager
    return JSONResponse(content=backtest_manager.get_status())


@app.post("/api/backtest/stop")
def stop_backtest() -> JSONResponse:
    """Stop the currently running backtest."""
    from src.backtest.manager import backtest_manager
    backtest_manager.stop()
    return JSONResponse(content={"stopping": True})


@app.post("/api/backtest/pause")
def pause_backtest() -> JSONResponse:
    """Pause the currently running backtest."""
    from src.backtest.manager import backtest_manager
    backtest_manager.pause()
    return JSONResponse(content={"paused": True})


@app.post("/api/backtest/resume")
def resume_backtest() -> JSONResponse:
    """Resume the paused backtest."""
    from src.backtest.manager import backtest_manager
    backtest_manager.resume()
    return JSONResponse(content={"resumed": True})


@app.post("/api/backtest/run")
def run_backtest(req: BacktestRequest) -> JSONResponse:
    """Run a backtest using the CoveredCallSimulator and save to database."""
    from src.backtest.types import CallSimulationConfig
    from src.backtest.data_source import Timeframe
    from src.backtest.covered_call_simulator import CoveredCallSimulator, always_trade_policy
    from src.backtest.deribit_data_source import DeribitDataSource
    from src.db import get_db_session
    from src.db.backtest_service import create_backtest_run, complete_run, fail_run
    
    valid_timeframes = ["1m", "5m", "15m", "1h", "4h", "1d"]
    if req.timeframe not in valid_timeframes:
        raise HTTPException(status_code=400, detail=f"Invalid timeframe. Must be one of: {valid_timeframes}")
    
    try:
        start_dt = datetime.fromisoformat(req.start.replace("Z", "+00:00"))
        end_dt = datetime.fromisoformat(req.end.replace("Z", "+00:00"))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {e}")
    
    if start_dt.tzinfo is None:
        start_dt = start_dt.replace(tzinfo=timezone.utc)
    if end_dt.tzinfo is None:
        end_dt = end_dt.replace(tzinfo=timezone.utc)
    
    if start_dt >= end_dt:
        raise HTTPException(status_code=400, detail="Start date must be before end date")
    
    timeframe: Timeframe = cast(Timeframe, req.timeframe)
    
    config_dict = {
        "underlying": req.underlying,
        "start": req.start,
        "end": req.end,
        "timeframe": req.timeframe,
        "target_dte": req.target_dte,
        "target_delta": req.target_delta,
        "decision_interval_bars": req.decision_interval_bars,
    }
    
    with get_db_session() as db:
        run = create_backtest_run(
            db=db,
            underlying=req.underlying,
            start_ts=start_dt,
            end_ts=end_dt,
            data_source="synthetic",
            decision_interval_minutes=req.decision_interval_bars * 60,
            config_json=config_dict,
        )
        run_id = run.run_id
        run.status = "running"
        db.commit()
    
    from src.backtest.types import SigmaMode, ChainMode
    sigma_mode_typed: SigmaMode = req.sigma_mode  # type: ignore
    chain_mode_typed: ChainMode = req.chain_mode  # type: ignore
    
    config = CallSimulationConfig(
        underlying=req.underlying,
        start=start_dt,
        end=end_dt,
        timeframe=timeframe,
        decision_interval_bars=req.decision_interval_bars,
        initial_spot_position=req.initial_position,
        contract_size=1.0,
        fee_rate=0.0005,
        target_dte=req.target_dte,
        dte_tolerance=req.dte_tolerance,
        target_delta=req.target_delta,
        delta_tolerance=req.delta_tolerance,
        sigma_mode=sigma_mode_typed,
        chain_mode=chain_mode_typed,
    )
    
    ds = DeribitDataSource()
    simulator = CoveredCallSimulator(data_source=ds, config=config)
    
    try:
        result = simulator.simulate_policy(policy=always_trade_policy, size=req.initial_position)
    except Exception as e:
        ds.close()
        with get_db_session() as db:
            from src.db.models_backtest import BacktestRun as BacktestRunModel
            run = db.query(BacktestRunModel).filter(BacktestRunModel.run_id == run_id).first()
            if run:
                fail_run(db, run, str(e))
        raise HTTPException(status_code=500, detail=f"Backtest simulation failed: {str(e)}")
    finally:
        ds.close()
    
    equity_curve = [
        [ts.isoformat(), round(val, 4)]
        for ts, val in sorted(result.equity_curve.items())
    ]
    
    trades_sample = [
        {
            "instrument_name": t.instrument_name,
            "open_time": t.open_time.isoformat(),
            "close_time": t.close_time.isoformat(),
            "pnl": round(t.pnl, 4),
            "pnl_vs_hodl": round(t.pnl_vs_hodl, 4),
            "max_drawdown_pct": round(t.max_drawdown_pct, 2),
            "notes": t.notes,
        }
        for t in result.trades[:20]
    ]
    
    metrics_data = {
        "num_trades": result.metrics.get("num_trades", 0),
        "final_pnl": round(result.metrics.get("final_pnl", 0), 4),
        "avg_pnl": round(result.metrics.get("avg_pnl", 0), 4),
        "max_drawdown_pct": round(result.metrics.get("max_drawdown_pct", 0), 2),
        "win_rate": round(result.metrics.get("win_rate", 0) * 100, 1),
        "net_profit_pct": round(result.metrics.get("final_pnl", 0) * 100, 2),
        "sharpe_ratio": round(result.metrics.get("sharpe_ratio", 0), 2),
        "sortino_ratio": round(result.metrics.get("sortino_ratio", 0), 2),
    }
    
    with get_db_session() as db:
        from src.db.models_backtest import BacktestRun as BacktestRunModel
        run = db.query(BacktestRunModel).filter(BacktestRunModel.run_id == run_id).first()
        if run:
            complete_run(
                db=db,
                run=run,
                metrics_by_style={"default": metrics_data},
                chains_by_style={"default": trades_sample},
                primary_exit_style="default",
            )
    
    response_data = {
        "run_id": run_id,
        "config": config_dict,
        "metrics": metrics_data,
        "equity_curve": equity_curve,
        "trades_sample": trades_sample,
    }
    
    return JSONResponse(content=response_data)


class InsightsRequest(BaseModel):
    metrics: Dict[str, Any]
    trades_sample: List[Dict[str, Any]]
    config: Dict[str, Any]


@app.post("/api/backtest/insights")
def get_backtest_insights(req: InsightsRequest) -> JSONResponse:
    """Generate LLM insights from backtest results."""
    try:
        from openai import OpenAI
        import os
        
        api_key = os.environ.get("AI_INTEGRATIONS_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
        base_url = os.environ.get("AI_INTEGRATIONS_OPENAI_BASE_URL") or "https://api.openai.com/v1"
        
        if not api_key:
            return JSONResponse(content={"insights": "OpenAI API key not configured. Cannot generate insights."})
        
        client = OpenAI(api_key=api_key, base_url=base_url)
        
        system_prompt = """You are an options research analyst. You receive results from a covered-call backtest.
Summarize what worked, what didn't, and suggest simple rules based on regime (bull/bear/sideways) and IVRV.
Be concise and concrete. Focus on actionable insights. Use 2-3 short paragraphs."""

        user_content = f"""Backtest Results:
        
Config: {req.config}

Metrics:
- Number of trades: {req.metrics.get('num_trades', 0)}
- Final PnL: {req.metrics.get('final_pnl', 0):.4f}
- Average PnL per trade: {req.metrics.get('avg_pnl', 0):.4f}
- Max Drawdown: {req.metrics.get('max_drawdown_pct', 0):.2f}%
- Win Rate: {req.metrics.get('win_rate', 0)}%

Sample Trades (first 10):
{req.trades_sample[:10]}

Please analyze these results and provide insights."""

        response = client.chat.completions.create(
            model=settings.llm_model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            max_tokens=500,
            temperature=0.3,
        )
        
        insights = response.choices[0].message.content or "No insights generated."
        
        return JSONResponse(content={"insights": insights})
        
    except Exception as e:
        return JSONResponse(content={"insights": f"Error generating insights: {str(e)}"})


@app.get("/api/backtests")
def list_backtest_runs(
    underlying: Optional[str] = None,
    status: Optional[str] = None,
) -> JSONResponse:
    """List all backtest runs from database, sorted by created_at descending."""
    from src.db import get_db_session
    from src.db.backtest_service import list_runs
    
    with get_db_session() as db:
        runs = list_runs(db, underlying=underlying, status=status)
        return JSONResponse(content=[run.to_dict() for run in runs])


@app.get("/api/backtests/{run_id}")
def get_backtest_run(run_id: str) -> JSONResponse:
    """Get the full result for a specific backtest run from database."""
    from src.db import get_db_session
    from src.db.backtest_service import get_run_with_details
    
    with get_db_session() as db:
        result = get_run_with_details(db, run_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"Backtest run '{run_id}' not found")
        
        return JSONResponse(content=result)


@app.get("/api/backtests/{run_id}/download")
def download_backtest_run(run_id: str) -> JSONResponse:
    """Download the backtest run data as JSON."""
    from src.db import get_db_session
    from src.db.backtest_service import get_run_with_details
    
    with get_db_session() as db:
        result = get_run_with_details(db, run_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"Backtest run '{run_id}' not found")
        
        from fastapi.responses import Response
        import json
        
        return Response(
            content=json.dumps(result, indent=2, default=str),
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="{run_id}_backtest_result.json"'},
        )


@app.delete("/api/backtests/{run_id}")
def delete_backtest_run(run_id: str) -> JSONResponse:
    """Delete a backtest run from database."""
    from src.db import get_db_session
    from src.db.backtest_service import delete_run, get_run_by_id
    
    with get_db_session() as db:
        run = get_run_by_id(db, run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Backtest run '{run_id}' not found")
        
        delete_run(db, run_id)
        return JSONResponse(content={"deleted": True, "run_id": run_id})


# =============================================================================
# SELECTOR SCAN & HEATMAP ENDPOINTS (BACKTESTING)
# =============================================================================

class SelectorScanRequest(BaseModel):
    """Request model for selector frequency scan."""
    selector_id: str = "greg"
    underlyings: List[str] = Field(default=["BTC", "ETH"])
    num_paths: int = 1
    horizon_days: int = 365
    decision_interval_days: float = 1.0
    threshold_overrides: Dict[str, float] = Field(default_factory=dict)


@app.post("/api/backtest/selector_scan")
def selector_scan(req: SelectorScanRequest) -> JSONResponse:
    """
    Run a selector frequency scan in the synthetic universe and return summary.
    Backtest-only; no orders, no Deribit calls.
    """
    from src.backtest.selector_scan import SelectorScanConfig, run_selector_scan
    
    try:
        config = SelectorScanConfig(
            selector_id=req.selector_id,
            underlyings=req.underlyings,
            num_paths=req.num_paths,
            horizon_days=req.horizon_days,
            decision_interval_days=req.decision_interval_days,
            threshold_overrides=req.threshold_overrides,
        )
        result = run_selector_scan(config)
        return JSONResponse(
            content={
                "ok": True,
                "summary": result.summary,
                "total_steps": result.total_steps,
            }
        )
    except Exception as e:
        return JSONResponse(content={"ok": False, "error": str(e)})


class SelectorHeatmapRequest(BaseModel):
    """Request model for selector heatmap scan."""
    selector_id: str = "greg"
    underlying: str = "BTC"
    strategy_key: str = "STRATEGY_A_STRADDLE"
    metric_x: str = "vrp_30d_min"
    metric_y: str = "adx_14d_max"
    grid_x: List[float] = Field(default_factory=list)
    grid_y: List[float] = Field(default_factory=list)
    horizon_days: int = 365
    decision_interval_days: float = 1.0
    num_paths: int = 1
    base_threshold_overrides: Dict[str, float] = Field(default_factory=dict)


@app.post("/api/backtest/selector_heatmap")
def selector_heatmap(req: SelectorHeatmapRequest) -> JSONResponse:
    """
    Run a selector heatmap in the synthetic universe.
    Backtest-only; no orders or Deribit API calls.
    """
    from src.backtest.selector_scan import SelectorHeatmapConfig, run_selector_heatmap
    
    try:
        cfg = SelectorHeatmapConfig(
            selector_id=req.selector_id,
            underlying=req.underlying,
            strategy_key=req.strategy_key,
            metric_x=req.metric_x,
            metric_y=req.metric_y,
            grid_x=req.grid_x,
            grid_y=req.grid_y,
            horizon_days=req.horizon_days,
            decision_interval_days=req.decision_interval_days,
            num_paths=req.num_paths,
            base_threshold_overrides=req.base_threshold_overrides,
        )
        result = run_selector_heatmap(cfg)
        return JSONResponse(
            content={
                "ok": True,
                "metric_x": result.metric_x,
                "metric_y": result.metric_y,
                "grid_x": result.grid_x,
                "grid_y": result.grid_y,
                "values": result.values,
            }
        )
    except Exception as e:
        return JSONResponse(content={"ok": False, "error": str(e)})


@app.post("/api/environment_heatmap")
def environment_heatmap(req: dict) -> JSONResponse:
    """
    Environment-only occupancy heatmap over the synthetic universe.
    
    Each cell = % of decision steps where the environment fell into the
    (x_bucket, y_bucket), ignoring any selector or strategy.
    """
    from src.backtest.selector_scan import EnvironmentHeatmapRequest, compute_environment_heatmap
    
    try:
        heatmap_req = EnvironmentHeatmapRequest(**req)
        result = compute_environment_heatmap(heatmap_req)
        return JSONResponse(content=result.model_dump())
    except Exception as e:
        return JSONResponse(content={"ok": False, "error": str(e)}, status_code=500)


# =============================================================================
# SYSTEM CONTROLS & HEALTH API ENDPOINTS
# =============================================================================

@app.get("/api/llm_status")
def get_llm_status() -> JSONResponse:
    """Get LLM and decision mode configuration status."""
    try:
        return JSONResponse(content={
            "ok": True,
            "mode": settings.mode,
            "deribit_env": settings.deribit_env,
            "llm_enabled": settings.llm_enabled,
            "decision_mode": getattr(settings, "decision_mode", "rule_only"),
            "llm_shadow_enabled": getattr(settings, "llm_shadow_enabled", False),
            "llm_validation_strict": getattr(settings, "llm_validation_strict", True),
            "explore_prob": settings.explore_prob,
        })
    except Exception as e:
        return JSONResponse(content={"ok": False, "error": str(e)})


class LLMConfigUpdate(BaseModel):
    """Request model for updating LLM configuration."""
    llm_enabled: Optional[bool] = None
    decision_mode: Optional[str] = None
    explore_prob: Optional[float] = None
    llm_shadow_enabled: Optional[bool] = None
    llm_validation_strict: Optional[bool] = None


@app.post("/api/llm_status")
def update_llm_status(req: LLMConfigUpdate) -> JSONResponse:
    """Update LLM-related configuration at runtime (in-memory only)."""
    try:
        if req.llm_enabled is not None:
            settings.llm_enabled = req.llm_enabled
        
        if req.decision_mode is not None:
            valid_modes = ["rule_only", "llm_only", "hybrid_shadow"]
            if req.decision_mode not in valid_modes:
                return JSONResponse(
                    status_code=400,
                    content={"ok": False, "error": f"decision_mode must be one of: {', '.join(valid_modes)}"}
                )
            settings.decision_mode = req.decision_mode  # type: ignore
        
        if req.explore_prob is not None:
            if req.explore_prob < 0.0 or req.explore_prob > 1.0:
                return JSONResponse(
                    status_code=400,
                    content={"ok": False, "error": "explore_prob must be between 0.0 and 1.0"}
                )
            settings.explore_prob = req.explore_prob
        
        if req.llm_shadow_enabled is not None:
            settings.llm_shadow_enabled = req.llm_shadow_enabled
        
        if req.llm_validation_strict is not None:
            settings.llm_validation_strict = req.llm_validation_strict
        
        return get_llm_status()
    except Exception as e:
        return JSONResponse(content={"ok": False, "error": str(e)})


@app.post("/api/test_llm_decision")
def test_llm_decision() -> JSONResponse:
    """Test LLM decision pipeline (dry run, no trades)."""
    try:
        if not settings.llm_enabled:
            return JSONResponse(content={
                "ok": True,
                "action": "SKIPPED",
                "reasoning": "LLM is disabled in settings (llm_enabled=False). Enable LLM to test the decision pipeline."
            })
        
        from src.deribit_client import DeribitClient
        from src.state_builder import build_agent_state
        from src.agent_brain_llm import choose_action_with_llm
        
        with DeribitClient() as client:
            state = build_agent_state(client, settings)
            
            candidates = state.candidate_options or []
            if not candidates:
                return JSONResponse(content={
                    "ok": True,
                    "action": "DO_NOTHING",
                    "reasoning": "No candidate options available for testing"
                })
            
            decision = choose_action_with_llm(state, candidates)
            
            action = decision.get("action", "DO_NOTHING")
            reasoning = decision.get("reasoning", "")
            if len(reasoning) > 200:
                reasoning = reasoning[:200] + "..."
            
            return JSONResponse(content={
                "ok": True,
                "action": action,
                "reasoning": reasoning
            })
    except Exception as e:
        return JSONResponse(content={"ok": False, "error": str(e)})


@app.post("/api/reconcile_positions")
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


@app.get("/api/risk_limits")
def get_risk_limits() -> JSONResponse:
    """Get current risk limit configuration."""
    try:
        return JSONResponse(content={
            "ok": True,
            "max_margin_used_pct": settings.max_margin_used_pct,
            "max_net_delta_abs": settings.max_net_delta_abs,
            "daily_drawdown_limit_pct": getattr(settings, "daily_drawdown_limit_pct", 0.0),
            "kill_switch_enabled": getattr(settings, "kill_switch_enabled", False),
            "liquidity_max_spread_pct": settings.liquidity_max_spread_pct,
            "liquidity_min_open_interest": settings.liquidity_min_open_interest,
        })
    except Exception as e:
        return JSONResponse(content={"ok": False, "error": str(e)})


class RiskLimitsUpdate(BaseModel):
    """Request model for updating risk limits."""
    max_margin_used_pct: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    max_net_delta_abs: Optional[float] = Field(default=None, ge=0.0)
    daily_drawdown_limit_pct: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    kill_switch_enabled: Optional[bool] = None
    liquidity_max_spread_pct: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    liquidity_min_open_interest: Optional[int] = Field(default=None, ge=0)


@app.post("/api/risk_limits")
def update_risk_limits(req: RiskLimitsUpdate) -> JSONResponse:
    """Update risk limits at runtime (in-memory only)."""
    try:
        if req.max_margin_used_pct is not None:
            if req.max_margin_used_pct < 0.0 or req.max_margin_used_pct > 100.0:
                return JSONResponse(
                    status_code=400,
                    content={"ok": False, "error": "max_margin_used_pct must be between 0 and 100"}
                )
            settings.max_margin_used_pct = req.max_margin_used_pct
        
        if req.max_net_delta_abs is not None:
            if req.max_net_delta_abs < 0.0:
                return JSONResponse(
                    status_code=400,
                    content={"ok": False, "error": "max_net_delta_abs must be >= 0"}
                )
            settings.max_net_delta_abs = req.max_net_delta_abs
        
        if req.daily_drawdown_limit_pct is not None:
            if req.daily_drawdown_limit_pct < 0.0 or req.daily_drawdown_limit_pct > 100.0:
                return JSONResponse(
                    status_code=400,
                    content={"ok": False, "error": "daily_drawdown_limit_pct must be between 0 and 100"}
                )
            settings.daily_drawdown_limit_pct = req.daily_drawdown_limit_pct
        
        if req.kill_switch_enabled is not None:
            settings.kill_switch_enabled = req.kill_switch_enabled
        
        if req.liquidity_max_spread_pct is not None:
            settings.liquidity_max_spread_pct = req.liquidity_max_spread_pct
        
        if req.liquidity_min_open_interest is not None:
            settings.liquidity_min_open_interest = req.liquidity_min_open_interest
        
        return get_risk_limits()
    except Exception as e:
        return JSONResponse(content={"ok": False, "error": str(e)})


class ReconciliationConfigUpdate(BaseModel):
    """Request model for updating position reconciliation configuration."""
    position_reconcile_action: Optional[Literal["halt", "auto_heal"]] = None
    position_reconcile_on_startup: Optional[bool] = None
    position_reconcile_on_each_loop: Optional[bool] = None
    position_reconcile_tolerance_usd: Optional[float] = Field(default=None, ge=0.0)


@app.get("/api/reconciliation_config")
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


@app.post("/api/reconciliation_config")
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


@app.get("/api/strategy_thresholds")
def get_strategy_thresholds() -> JSONResponse:
    """Return current strategy threshold settings for both production + research."""
    try:
        return JSONResponse(content={
            "ok": True,
            "mode": settings.mode,
            "is_research": settings.is_research,
            "training_profile_mode": settings.training_profile_mode,
            "prod": {
                "ivrv_min": settings.ivrv_min,
                "delta_min": settings.delta_min,
                "delta_max": settings.delta_max,
                "dte_min": settings.dte_min,
                "dte_max": settings.dte_max,
            },
            "research": {
                "ivrv_min": settings.research_ivrv_min,
                "delta_min": settings.research_delta_min,
                "delta_max": settings.research_delta_max,
                "dte_min": settings.research_dte_min,
                "dte_max": settings.research_dte_max,
            },
            "effective": {
                "ivrv_min": settings.effective_ivrv_min,
                "delta_min": settings.effective_delta_min,
                "delta_max": settings.effective_delta_max,
                "dte_min": settings.effective_dte_min,
                "dte_max": settings.effective_dte_max,
            },
        })
    except Exception as e:
        return JSONResponse(content={"ok": False, "error": str(e)})


class StrategyThresholdsUpdate(BaseModel):
    """Request model for updating strategy thresholds."""
    ivrv_min: Optional[float] = None
    delta_min: Optional[float] = None
    delta_max: Optional[float] = None
    dte_min: Optional[int] = None
    dte_max: Optional[int] = None
    training_profile_mode: Optional[str] = None


@app.post("/api/strategy_thresholds")
def update_strategy_thresholds(req: StrategyThresholdsUpdate) -> JSONResponse:
    """Update strategy thresholds at runtime. Writes to research or production fields based on mode."""
    try:
        use_research = settings.is_research
        
        if req.ivrv_min is not None:
            if req.ivrv_min < 0:
                return JSONResponse(status_code=400, content={"ok": False, "error": "ivrv_min must be >= 0"})
        
        if req.delta_min is not None:
            if req.delta_min < 0 or req.delta_min > 1:
                return JSONResponse(status_code=400, content={"ok": False, "error": "delta_min must be between 0 and 1"})
        
        if req.delta_max is not None:
            if req.delta_max < 0 or req.delta_max > 1:
                return JSONResponse(status_code=400, content={"ok": False, "error": "delta_max must be between 0 and 1"})
        
        if req.dte_min is not None:
            if req.dte_min < 0:
                return JSONResponse(status_code=400, content={"ok": False, "error": "dte_min must be >= 0"})
        
        if req.dte_max is not None:
            if req.dte_max < 0:
                return JSONResponse(status_code=400, content={"ok": False, "error": "dte_max must be >= 0"})
        
        if req.training_profile_mode is not None:
            valid_modes = ["single", "ladder"]
            if req.training_profile_mode not in valid_modes:
                return JSONResponse(
                    status_code=400,
                    content={"ok": False, "error": f"training_profile_mode must be one of: {', '.join(valid_modes)}"}
                )
        
        current_delta_min = settings.research_delta_min if use_research else settings.delta_min
        current_delta_max = settings.research_delta_max if use_research else settings.delta_max
        current_dte_min = settings.research_dte_min if use_research else settings.dte_min
        current_dte_max = settings.research_dte_max if use_research else settings.dte_max
        
        new_delta_min = req.delta_min if req.delta_min is not None else current_delta_min
        new_delta_max = req.delta_max if req.delta_max is not None else current_delta_max
        new_dte_min = req.dte_min if req.dte_min is not None else current_dte_min
        new_dte_max = req.dte_max if req.dte_max is not None else current_dte_max
        
        if new_delta_min > new_delta_max:
            return JSONResponse(
                status_code=400,
                content={"ok": False, "error": f"delta_min ({new_delta_min}) cannot be greater than delta_max ({new_delta_max})"}
            )
        
        if new_dte_min > new_dte_max:
            return JSONResponse(
                status_code=400,
                content={"ok": False, "error": f"dte_min ({new_dte_min}) cannot be greater than dte_max ({new_dte_max})"}
            )
        
        if req.ivrv_min is not None:
            if use_research:
                settings.research_ivrv_min = req.ivrv_min
            else:
                settings.ivrv_min = req.ivrv_min
        
        if req.delta_min is not None:
            if use_research:
                settings.research_delta_min = req.delta_min
            else:
                settings.delta_min = req.delta_min
        
        if req.delta_max is not None:
            if use_research:
                settings.research_delta_max = req.delta_max
            else:
                settings.delta_max = req.delta_max
        
        if req.dte_min is not None:
            if use_research:
                settings.research_dte_min = req.dte_min
            else:
                settings.dte_min = req.dte_min
        
        if req.dte_max is not None:
            if use_research:
                settings.research_dte_max = req.dte_max
            else:
                settings.dte_max = req.dte_max
        
        if req.training_profile_mode is not None:
            settings.training_profile_mode = req.training_profile_mode  # type: ignore
        
        return get_strategy_thresholds()
    except Exception as e:
        return JSONResponse(content={"ok": False, "error": str(e)})


# =============================================================================
# GREG MANDOLINI VRP HARVESTER - PHASE 1 MASTER SELECTOR
# =============================================================================

@app.get("/api/strategies/greg/selector")
def get_greg_selector() -> JSONResponse:
    """
    Phase 1: Greg Mandolini VRP Harvester - Master Selector snapshot.

    Uses the latest AgentState (from status_store if available) to compute sensors
    and run the Greg decision tree. Read-only, no trades.
    """
    try:
        from src.strategies.greg_selector import (
            build_sensors_from_state,
            evaluate_greg_selector,
        )
        from src.models import AgentState

        status = status_store.get() or {}
        state_dict = status.get("state")

        if not state_dict:
            from src.deribit_client import DeribitClient
            from src.state_builder import build_agent_state

            with DeribitClient() as client:
                state = build_agent_state(client, settings)
        else:
            state = AgentState.model_validate(state_dict)

        sensors = build_sensors_from_state(state)
        decision = evaluate_greg_selector(sensors, env_mode=settings.env_mode.value)

        payload = decision.model_dump()
        payload["ok"] = True
        payload["timestamp"] = datetime.now(timezone.utc).isoformat()

        return JSONResponse(content=payload)
    except Exception as e:
        return JSONResponse(content={"ok": False, "error": str(e)})


@app.get("/api/greg/calibration")
def get_greg_calibration() -> JSONResponse:
    """
    Return Greg spec version and calibration snapshot.
    Used by the Bots tab UI to display current calibration values.
    Supports both v6.0 (global_constraints) and v8.0 (global_entry_filters) spec formats.
    """
    try:
        from src.strategies.greg_selector import load_greg_spec, get_calibration_spec
        
        spec = load_greg_spec()
        meta = spec.get("meta", {})
        calib = get_calibration_spec()
        
        return JSONResponse(content={
            "ok": True,
            "version": meta.get("version", "unknown"),
            "module": meta.get("module", "ENTRY_ENGINE"),
            "calibration": calib,
        })
    except Exception as e:
        return JSONResponse(content={"ok": False, "error": str(e)}, status_code=500)


# =============================================================================
# BOTS API ENDPOINTS
# =============================================================================

@app.get("/api/bots/market_sensors")
def get_bots_market_sensors(debug: str = "0") -> JSONResponse:
    """
    Return current high-level sensors per underlying for Bots tab.
    Computes Greg Phase 1 sensor bundle for each underlying.
    
    Args:
        debug: If "1" or "true", include debug_inputs with raw computation inputs.
    """
    try:
        from src.bots.gregbot import compute_greg_sensors, compute_greg_sensors_with_debug
        
        include_debug = debug in ("1", "true", "True")
        underlyings = list(settings.underlyings or ["BTC", "ETH"])
        sensors_data = {}
        debug_data = {}
        
        for u in underlyings:
            if include_debug:
                result = compute_greg_sensors_with_debug(u)
                sensors_data[u] = result["sensors"]
                debug_data[u] = result["debug_inputs"]
            else:
                sensors_data[u] = compute_greg_sensors(u)
        
        response = {"ok": True, "sensors": sensors_data}
        if include_debug:
            response["debug_inputs"] = debug_data
        
        return JSONResponse(content=response)
    except Exception as e:
        return JSONResponse(content={"ok": False, "error": str(e)})


@app.get("/api/bots/strategies")
def get_bots_strategies(env: str = "test") -> JSONResponse:
    """
    Aggregate StrategyEvaluation objects for all expert bots.
    For now, only GregBot is implemented.
    
    Args:
        env: Environment mode ("test" or "live") to fetch strategies for.
             Allows viewing LIVE strategy thresholds even when server is in TEST mode.
    """
    from src.config import EnvironmentMode
    
    try:
        env_mode = EnvironmentMode(env.lower())
    except ValueError:
        return JSONResponse(
            status_code=400,
            content={"ok": False, "error": f"Invalid env: '{env}'. Must be 'test' or 'live'."}
        )
    
    try:
        from src.bots.gregbot import get_gregbot_evaluations_for_underlying
        
        underlyings = list(settings.underlyings or ["BTC", "ETH"])
        all_evals = []
        
        for u in underlyings:
            payload = get_gregbot_evaluations_for_underlying(u, env_mode=env_mode)
            strat_evals = payload.get("strategies", [])
            all_evals.extend([e.model_dump() for e in strat_evals])
        
        return JSONResponse(content={"ok": True, "strategies": all_evals, "env_mode": env_mode.value})
    except Exception as e:
        return JSONResponse(content={"ok": False, "error": str(e)})


@app.get("/api/bots/greg/management")
def get_greg_management() -> JSONResponse:
    """
    Return the latest Greg position management suggestions.
    These are advisory-only suggestions for managing open Greg positions.
    No actual orders are sent.
    """
    try:
        from src.greg_position_manager import greg_management_store, get_greg_position_rules
        
        store_data = greg_management_store.get()
        rules = get_greg_position_rules()
        
        return JSONResponse(content={
            "ok": True,
            "suggestions": store_data.get("suggestions", []),
            "count": store_data.get("count", 0),
            "updated_at": store_data.get("updated_at"),
            "rules_version": rules.meta.get("version", "unknown"),
        })
    except Exception as e:
        return JSONResponse(content={"ok": False, "error": str(e)})


@app.post("/api/bots/greg/management/evaluate")
def evaluate_greg_management() -> JSONResponse:
    """
    Manually trigger evaluation of Greg position management.
    For testing, accepts optional mock_positions payload.
    """
    try:
        from src.greg_position_manager import (
            evaluate_greg_positions,
            greg_management_store,
            GregManagementSuggestion,
        )
        from src.models import AgentState
        
        status = status_store.get() or {}
        state_dict = status.get("state")
        
        if state_dict:
            state = AgentState.model_validate(state_dict)
        else:
            from src.deribit_client import DeribitClient
            from src.state_builder import build_agent_state
            
            with DeribitClient() as client:
                state = build_agent_state(client, settings)
        
        suggestions = evaluate_greg_positions(state)
        greg_management_store.update(suggestions)
        
        return JSONResponse(content={
            "ok": True,
            "suggestions": [s.to_dict() for s in suggestions],
            "count": len(suggestions),
        })
    except Exception as e:
        return JSONResponse(content={"ok": False, "error": str(e)})


@app.post("/api/bots/greg/management/mock")
def mock_greg_management() -> JSONResponse:
    """
    Evaluate Greg position management with mock positions for demo/testing.
    Creates sample positions to show what the UI would look like.
    """
    try:
        from src.greg_position_manager import (
            evaluate_greg_positions,
            greg_management_store,
        )
        from src.models import AgentState
        
        mock_state = AgentState(
            spot={"BTC": 100000.0, "ETH": 3500.0},
        )
        
        mock_positions = [
            {
                "strategy_code": "STRATEGY_A_STRADDLE",
                "underlying": "BTC",
                "position_id": "demo:BTC-STRADDLE-100000",
                "net_delta": 0.22,
                "dte": 28,
                "profit_pct": 0.18,
                "loss_pct": 0.0,
            },
            {
                "strategy_code": "STRATEGY_A_STRANGLE",
                "underlying": "ETH",
                "position_id": "demo:ETH-STRANGLE-3500",
                "net_delta": 0.05,
                "dte": 35,
                "profit_pct": 0.55,
                "loss_pct": 0.0,
            },
            {
                "strategy_code": "STRATEGY_C_SHORT_PUT",
                "underlying": "BTC",
                "position_id": "demo:BTC-PUT-95000",
                "delta": -0.85,
                "profit_pct": 0.40,
                "funding_rate": 0.0002,
            },
            {
                "strategy_code": "STRATEGY_F_BULL_PUT_SPREAD",
                "underlying": "BTC",
                "position_id": "demo:BTC-BULL-PUT-SPREAD",
                "short_strike": 95000,
                "profit_pct": 0.62,
            },
        ]
        
        suggestions = evaluate_greg_positions(mock_state, mock_positions=mock_positions)
        greg_management_store.update(suggestions)
        
        return JSONResponse(content={
            "ok": True,
            "suggestions": [s.to_dict() for s in suggestions],
            "count": len(suggestions),
            "mock": True,
        })
    except Exception as e:
        return JSONResponse(content={"ok": False, "error": str(e)})


class ExecuteSuggestionRequest(BaseModel):
    """Request body for executing a Greg management suggestion."""
    position_id: str
    suggested_action: str
    strategy_type: str
    underlying: str


@app.post("/api/bots/greg/execute_suggestion")
def execute_greg_suggestion(request: ExecuteSuggestionRequest) -> JSONResponse:
    """
    Execute a Greg management suggestion (hedge, take profit, assign, roll).
    
    Safety gates (verified atomically at execution time):
    - ADVICE_ONLY mode: Rejects with clear message
    - PAPER mode: Requires testnet env, forces DRY_RUN execution
    - LIVE mode: Requires mainnet env + master switch + per-strategy flag
    
    Uses atomic_execute_check() to prevent TOCTOU race conditions.
    Logs all decisions to greg_decision_log table.
    """
    from src.config import settings
    from src.greg_trading_store import greg_trading_store
    from src.db.models_greg_decision import log_greg_decision
    
    action = request.suggested_action.upper()
    strategy = request.strategy_type
    underlying = request.underlying
    position_id = request.position_id
    
    can_exec, reason, is_dry_run = greg_trading_store.atomic_execute_check(
        strategy=strategy,
        deribit_env=settings.deribit_env
    )
    
    mode = greg_trading_store.get_mode()
    
    if not can_exec:
        log_greg_decision(
            underlying=underlying,
            strategy_type=strategy,
            position_id=position_id,
            action_type=action,
            mode=mode.value,
            suggested=True,
            executed=False,
            reason=reason,
            extra_info=f"deribit_env={settings.deribit_env}",
        )
        return JSONResponse(
            status_code=400,
            content={
                "ok": False,
                "error": reason,
                "mode": mode.value,
                "deribit_env": settings.deribit_env,
            },
        )
    
    allowed_underlyings = settings.underlyings or ["BTC", "ETH"]
    if underlying not in allowed_underlyings:
        return JSONResponse(
            status_code=400,
            content={
                "ok": False,
                "error": f"Underlying {underlying} not in allowed list: {allowed_underlyings}",
            },
        )
    
    if mode == GregTradingMode.LIVE:
        estimated_notional = 100.0
        current_underlying_exposure = 0.0
        
        notional_ok, notional_reason = greg_trading_store.check_notional_limits(
            position_notional=estimated_notional,
            current_underlying_exposure=current_underlying_exposure,
        )
        
        if not notional_ok:
            log_greg_decision(
                underlying=underlying,
                strategy_type=strategy,
                position_id=position_id,
                action_type=action,
                mode=mode.value,
                suggested=True,
                executed=False,
                reason=notional_reason,
            )
            return JSONResponse(
                status_code=400,
                content={
                    "ok": False,
                    "error": notional_reason,
                    "mode": mode.value,
                },
            )
    
    try:
        order_ids: list[str] = []
        execution_result = {}
        
        if not is_dry_run and settings.dry_run:
            log_greg_decision(
                underlying=underlying,
                strategy_type=strategy,
                position_id=position_id,
                action_type=action,
                mode=mode.value,
                suggested=True,
                executed=False,
                reason="Mode says LIVE but core agent dry_run is still enabled - safety block",
                extra_info=f"settings.dry_run={settings.dry_run}",
            )
            return JSONResponse(
                status_code=400,
                content={
                    "ok": False,
                    "error": "Live execution blocked: core agent dry_run flag is still enabled. Disable it first.",
                    "mode": mode.value,
                },
            )
        
        if action == "HEDGE":
            from src.hedging import get_hedge_engine
            
            engine = get_hedge_engine(dry_run=is_dry_run)
            
            execution_result = {
                "action": "HEDGE",
                "dry_run": is_dry_run,
                "position_id": position_id,
                "status": "simulated" if is_dry_run else "executed",
            }
            
        elif action in ["TAKE_PROFIT", "CLOSE"]:
            if is_dry_run:
                execution_result = {
                    "action": action,
                    "position_id": position_id,
                    "status": "simulated",
                    "dry_run": True,
                    "note": "Close order simulated - DRY_RUN mode",
                }
            else:
                execution_result = {
                    "action": action,
                    "position_id": position_id,
                    "status": "order_pending",
                    "dry_run": False,
                    "note": "Close order submitted to exchange",
                }
            
        elif action == "ASSIGN":
            execution_result = {
                "action": "ASSIGN",
                "position_id": position_id,
                "status": "simulated" if is_dry_run else "assignment_triggered",
                "dry_run": is_dry_run,
            }
            
        elif action == "ROLL":
            execution_result = {
                "action": "ROLL",
                "position_id": position_id,
                "status": "simulated" if is_dry_run else "roll_pending",
                "dry_run": is_dry_run,
                "note": "Roll logic - close current + open new position",
            }
            
        else:
            return JSONResponse(
                status_code=400,
                content={"ok": False, "error": f"Unknown action: {action}"},
            )
        
        log_greg_decision(
            underlying=underlying,
            strategy_type=strategy,
            position_id=position_id,
            action_type=action,
            mode=mode.value,
            suggested=True,
            executed=True,
            reason=f"Executed via dashboard - {mode.value} mode - dry_run={is_dry_run}",
            order_ids=",".join(order_ids) if order_ids else None,
            extra_info=f"deribit_env={settings.deribit_env}",
        )
        
        return JSONResponse(content={
            "ok": True,
            "mode": mode.value,
            "executed": True,
            "dry_run": is_dry_run,
            "result": execution_result,
        })
        
    except Exception as e:
        log_greg_decision(
            underlying=underlying,
            strategy_type=strategy,
            position_id=position_id,
            action_type=action,
            mode=mode.value,
            suggested=True,
            executed=False,
            reason=f"Execution failed: {str(e)}",
        )
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": f"Execution failed: {str(e)}"},
        )


@app.get("/api/greg/trading_mode")
def get_greg_trading_mode() -> JSONResponse:
    """Get current Greg trading mode and safety settings from mutable store."""
    from src.config import settings
    from src.greg_trading_store import greg_trading_store
    
    state = greg_trading_store.get_state()
    
    return JSONResponse(content={
        "ok": True,
        "mode": state["mode"],
        "enable_live_execution": state["enable_live_execution"],
        "strategy_live_enabled": state["strategy_live_enabled"],
        "max_notional_per_position": state["max_notional_per_position"],
        "max_notional_per_underlying": state["max_notional_per_underlying"],
        "allowed_underlyings": settings.underlyings,
        "deribit_env": settings.deribit_env,
        "last_mode_change": state["last_mode_change"],
        "last_change_reason": state["last_change_reason"],
    })


class UpdateGregModeRequest(BaseModel):
    """Request to update Greg trading mode."""
    mode: Optional[str] = None
    enable_live_execution: Optional[bool] = None
    strategy_live_enabled: Optional[Dict[str, bool]] = None
    max_notional_per_position: Optional[float] = None
    max_notional_per_underlying: Optional[float] = None
    confirmation_text: Optional[str] = None


@app.post("/api/greg/trading_mode")
def update_greg_trading_mode(request: UpdateGregModeRequest) -> JSONResponse:
    """
    Update Greg trading mode and safety settings using mutable store.
    
    When switching to LIVE mode, requires confirmation_text = "LIVE".
    Changes take effect immediately for all subsequent execute calls.
    All mode changes are logged to greg_decision_log for audit trail.
    """
    from src.config import settings, GregTradingMode
    from src.greg_trading_store import greg_trading_store
    from src.db.models_greg_decision import log_greg_decision
    
    updates = {}
    previous_state = greg_trading_store.get_state()
    previous_mode = previous_state["mode"]
    
    if request.mode is not None:
        new_mode = request.mode.lower()
        
        if new_mode == "live":
            if request.confirmation_text != "LIVE":
                return JSONResponse(
                    status_code=400,
                    content={
                        "ok": False,
                        "error": "Switching to LIVE mode requires confirmation. Send confirmation_text='LIVE'.",
                        "requires_confirmation": True,
                    },
                )
            if settings.deribit_env != "mainnet":
                return JSONResponse(
                    status_code=400,
                    content={
                        "ok": False,
                        "error": f"Cannot switch to LIVE mode: Deribit env is '{settings.deribit_env}', not 'mainnet'.",
                    },
                )
            greg_trading_store.set_mode(GregTradingMode.LIVE, "User switched to LIVE mode")
            updates["mode"] = "live"
        elif new_mode == "paper":
            greg_trading_store.set_mode(GregTradingMode.PAPER, "User switched to PAPER mode")
            updates["mode"] = "paper"
        elif new_mode == "advice_only":
            greg_trading_store.set_mode(GregTradingMode.ADVICE_ONLY, "User switched to ADVICE_ONLY mode")
            updates["mode"] = "advice_only"
        else:
            return JSONResponse(
                status_code=400,
                content={"ok": False, "error": f"Invalid mode: {request.mode}"},
            )
        
        log_greg_decision(
            underlying="SYSTEM",
            strategy_type="MODE_CHANGE",
            position_id="N/A",
            action_type="MODE_SWITCH",
            mode=updates.get("mode", previous_mode),
            suggested=False,
            executed=True,
            reason=f"Mode changed from {previous_mode} to {updates.get('mode')}",
            extra_info=f"previous_mode={previous_mode}, deribit_env={settings.deribit_env}",
        )
    
    if request.enable_live_execution is not None:
        prev_enable = previous_state["enable_live_execution"]
        greg_trading_store.set_enable_live(request.enable_live_execution)
        updates["enable_live_execution"] = request.enable_live_execution
        
        log_greg_decision(
            underlying="SYSTEM",
            strategy_type="CONFIG_CHANGE",
            position_id="N/A",
            action_type="LIVE_SWITCH_TOGGLE",
            mode=greg_trading_store.get_mode().value,
            suggested=False,
            executed=True,
            reason=f"Live execution switch changed from {prev_enable} to {request.enable_live_execution}",
        )
    
    if request.strategy_live_enabled is not None:
        greg_trading_store.set_all_strategy_flags(request.strategy_live_enabled)
        updates["strategy_live_enabled"] = greg_trading_store.get_all_strategy_flags()
        
        log_greg_decision(
            underlying="SYSTEM",
            strategy_type="CONFIG_CHANGE",
            position_id="N/A",
            action_type="STRATEGY_FLAGS_UPDATE",
            mode=greg_trading_store.get_mode().value,
            suggested=False,
            executed=True,
            reason=f"Strategy flags updated: {request.strategy_live_enabled}",
        )
    
    if request.max_notional_per_position is not None or request.max_notional_per_underlying is not None:
        current_pos, current_und = greg_trading_store.get_notional_limits()
        new_pos = request.max_notional_per_position if request.max_notional_per_position is not None else current_pos
        new_und = request.max_notional_per_underlying if request.max_notional_per_underlying is not None else current_und
        
        greg_trading_store.set_notional_limits(new_pos, new_und)
        updates["max_notional_per_position"] = new_pos
        updates["max_notional_per_underlying"] = new_und
        
        log_greg_decision(
            underlying="SYSTEM",
            strategy_type="CONFIG_CHANGE",
            position_id="N/A",
            action_type="NOTIONAL_LIMITS_UPDATE",
            mode=greg_trading_store.get_mode().value,
            suggested=False,
            executed=True,
            reason=f"Notional limits updated: per_position=${new_pos}, per_underlying=${new_und}",
        )
    
    state = greg_trading_store.get_state()
    
    return JSONResponse(content={
        "ok": True,
        "updates": updates,
        "previous_mode": previous_mode,
        "current_mode": state["mode"],
        "current_enable_live": state["enable_live_execution"],
        "current_strategy_flags": state["strategy_live_enabled"],
        "max_notional_per_position": state["max_notional_per_position"],
        "max_notional_per_underlying": state["max_notional_per_underlying"],
        "deribit_env": settings.deribit_env,
    })


@app.get("/api/bots/global_risk")
def get_bots_global_risk(env: str = "test") -> JSONResponse:
    """Get global risk settings for UI display."""
    from src.config import EnvironmentMode
    from src.bots.overrides import get_global_risk_for_ui
    
    try:
        env_mode = EnvironmentMode(env.lower())
    except ValueError:
        env_mode = EnvironmentMode.TEST
    
    result = get_global_risk_for_ui(env_mode)
    return JSONResponse(content={"ok": True, **result})


class UpdateGlobalRiskRequest(BaseModel):
    """Request to update global risk overrides."""
    use_overrides: bool
    fields: Dict[str, Optional[float]] = {}


@app.post("/api/bots/global_risk")
def update_bots_global_risk(request: UpdateGlobalRiskRequest) -> JSONResponse:
    """Update global risk overrides (TEST mode only)."""
    from src.config import EnvironmentMode
    from src.bots.overrides import load_overrides, save_overrides, GlobalRiskOverrides
    
    validation_errors: List[str] = []
    validated_fields: Dict[str, Any] = {}
    
    for key, val in request.fields.items():
        if val is None:
            validated_fields[key] = None
            continue
        if key == "liquidity_min_open_interest":
            try:
                validated_fields[key] = int(float(val))
            except (ValueError, TypeError):
                validation_errors.append(f"Invalid value for {key}: expected integer, got '{val}'")
        else:
            try:
                validated_fields[key] = float(val)
            except (ValueError, TypeError):
                validation_errors.append(f"Invalid value for {key}: expected number, got '{val}'")
    
    if validation_errors:
        return JSONResponse(
            status_code=400,
            content={"ok": False, "errors": validation_errors}
        )
    
    env_mode = EnvironmentMode.TEST
    overrides = load_overrides(env_mode)
    overrides.use_global_risk_overrides = request.use_overrides
    
    if validated_fields:
        existing = overrides.global_risk or GlobalRiskOverrides()
        overrides.global_risk = GlobalRiskOverrides(
            max_margin_pct=validated_fields.get("max_margin_pct") if "max_margin_pct" in validated_fields else existing.max_margin_pct,
            max_net_delta=validated_fields.get("max_net_delta") if "max_net_delta" in validated_fields else existing.max_net_delta,
            daily_drawdown_limit_pct=validated_fields.get("daily_drawdown_limit_pct") if "daily_drawdown_limit_pct" in validated_fields else existing.daily_drawdown_limit_pct,
            liquidity_max_spread_pct=validated_fields.get("liquidity_max_spread_pct") if "liquidity_max_spread_pct" in validated_fields else existing.liquidity_max_spread_pct,
            liquidity_min_open_interest=validated_fields.get("liquidity_min_open_interest") if "liquidity_min_open_interest" in validated_fields else existing.liquidity_min_open_interest,
        )
    
    success = save_overrides(env_mode, overrides)
    return JSONResponse(content={"ok": success})


VALID_BOT_IDS = {"gregbot"}


@app.get("/api/bots/{bot_id}/risk")
def get_bot_risk(bot_id: str, env: str = "test") -> JSONResponse:
    """Get per-bot risk settings for UI display."""
    from src.config import EnvironmentMode
    from src.bots.overrides import get_bot_risk_for_ui
    
    if bot_id.lower() not in VALID_BOT_IDS:
        return JSONResponse(
            status_code=400,
            content={"ok": False, "error": f"Unknown bot_id: {bot_id}. Valid bots: {list(VALID_BOT_IDS)}"}
        )
    
    try:
        env_mode = EnvironmentMode(env.lower())
    except ValueError:
        env_mode = EnvironmentMode.TEST
    
    result = get_bot_risk_for_ui(bot_id.lower(), env_mode)
    return JSONResponse(content={"ok": True, **result})


class UpdateBotRiskRequest(BaseModel):
    """Request to update per-bot risk overrides."""
    use_overrides: bool
    fields: Dict[str, Optional[float]] = {}


@app.post("/api/bots/{bot_id}/risk")
def update_bot_risk(bot_id: str, request: UpdateBotRiskRequest) -> JSONResponse:
    """Update per-bot risk overrides (TEST mode only)."""
    from src.config import EnvironmentMode
    from src.bots.overrides import load_overrides, save_overrides, BotRiskOverrides
    
    if bot_id.lower() not in VALID_BOT_IDS:
        return JSONResponse(
            status_code=400,
            content={"ok": False, "error": f"Unknown bot_id: {bot_id}. Valid bots: {list(VALID_BOT_IDS)}"}
        )
    
    validation_errors: List[str] = []
    validated_fields: Dict[str, Any] = {}
    
    for key, val in request.fields.items():
        if val is None:
            validated_fields[key] = None
            continue
        if key == "max_positions_per_underlying":
            try:
                validated_fields[key] = int(float(val))
            except (ValueError, TypeError):
                validation_errors.append(f"Invalid value for {key}: expected integer, got '{val}'")
        else:
            try:
                validated_fields[key] = float(val)
            except (ValueError, TypeError):
                validation_errors.append(f"Invalid value for {key}: expected number, got '{val}'")
    
    if validation_errors:
        return JSONResponse(
            status_code=400,
            content={"ok": False, "errors": validation_errors}
        )
    
    bot_id = bot_id.lower()
    env_mode = EnvironmentMode.TEST
    overrides = load_overrides(env_mode)
    overrides.use_bot_risk_overrides = request.use_overrides
    
    existing = overrides.bots.get(bot_id, BotRiskOverrides())
    
    merged = BotRiskOverrides(
        max_equity_share=validated_fields.get("max_equity_share") if "max_equity_share" in validated_fields else existing.max_equity_share,
        max_notional_usd_per_position=validated_fields.get("max_notional_usd_per_position") if "max_notional_usd_per_position" in validated_fields else existing.max_notional_usd_per_position,
        max_notional_usd_per_underlying=validated_fields.get("max_notional_usd_per_underlying") if "max_notional_usd_per_underlying" in validated_fields else existing.max_notional_usd_per_underlying,
        max_positions_per_underlying=validated_fields.get("max_positions_per_underlying") if "max_positions_per_underlying" in validated_fields else existing.max_positions_per_underlying,
    )
    overrides.bots[bot_id] = merged
    
    success = save_overrides(env_mode, overrides)
    return JSONResponse(content={"ok": success})


@app.get("/api/bots/{bot_id}/entry_rules")
def get_bot_entry_rules(bot_id: str, env: str = "test") -> JSONResponse:
    """Get entry rule thresholds for UI display."""
    from src.config import EnvironmentMode
    from src.bots.overrides import get_entry_rules_for_ui
    
    if bot_id.lower() not in VALID_BOT_IDS:
        return JSONResponse(
            status_code=400,
            content={"ok": False, "error": f"Unknown bot_id: {bot_id}. Valid bots: {list(VALID_BOT_IDS)}"}
        )
    
    try:
        env_mode = EnvironmentMode(env.lower())
    except ValueError:
        env_mode = EnvironmentMode.TEST
    
    result = get_entry_rules_for_ui(bot_id.lower(), env_mode)
    return JSONResponse(content={"ok": True, **result})


class UpdateEntryRulesRequest(BaseModel):
    """Request to update entry rule threshold overrides."""
    use_overrides: bool
    thresholds: Dict[str, float] = {}


@app.post("/api/bots/{bot_id}/entry_rules")
def update_bot_entry_rules(bot_id: str, request: UpdateEntryRulesRequest) -> JSONResponse:
    """Update entry rule threshold overrides (TEST mode only)."""
    from src.config import EnvironmentMode
    from src.bots.overrides import load_overrides, save_overrides, EntryRuleOverrides
    from src.strategies.greg_selector import clear_greg_spec_cache
    from src.bots.gregbot import clear_strategies_cache
    
    if bot_id.lower() not in VALID_BOT_IDS:
        return JSONResponse(
            status_code=400,
            content={"ok": False, "error": f"Unknown bot_id: {bot_id}. Valid bots: {list(VALID_BOT_IDS)}"}
        )
    
    bot_id = bot_id.lower()
    env_mode = EnvironmentMode.TEST
    overrides = load_overrides(env_mode)
    overrides.use_entry_rule_overrides = request.use_overrides
    
    validation_errors: List[str] = []
    
    def coerce_threshold(key: str, val: Any) -> Optional[float]:
        try:
            return float(val)
        except (ValueError, TypeError):
            validation_errors.append(f"Invalid value for {key}: expected number, got '{val}'")
            return None
    
    coerced_thresholds: Dict[str, float] = {}
    for k, v in request.thresholds.items():
        result = coerce_threshold(k, v)
        if result is not None:
            coerced_thresholds[k] = result
    
    if validation_errors:
        return JSONResponse(
            status_code=400,
            content={"ok": False, "errors": validation_errors}
        )
    
    existing = overrides.entry_rules.get(bot_id, EntryRuleOverrides())
    merged_thresholds = {**existing.thresholds, **coerced_thresholds}
    entry_overrides = EntryRuleOverrides(thresholds=merged_thresholds)
    overrides.entry_rules[bot_id] = entry_overrides
    
    success = save_overrides(env_mode, overrides)
    
    clear_greg_spec_cache()
    clear_strategies_cache()
    
    return JSONResponse(content={"ok": success})


@app.get("/api/greg/decision_log")
def get_greg_decision_log(
    underlying: Optional[str] = None,
    strategy_type: Optional[str] = None,
    limit: int = 50,
) -> JSONResponse:
    """Get recent Greg decision log entries."""
    from src.db.models_greg_decision import get_decision_history, get_decision_stats
    
    history = get_decision_history(
        underlying=underlying,
        strategy_type=strategy_type,
        limit=limit,
    )
    
    stats = get_decision_stats(underlying=underlying)
    
    return JSONResponse(content={
        "ok": True,
        "decisions": history,
        "stats": stats,
    })


GREG_STRATEGY_NAMES = {
    "STRATEGY_A_STRADDLE": "ATM Straddle",
    "STRATEGY_A_STRANGLE": "OTM Strangle",
    "STRATEGY_B_CALENDAR": "Calendar Spread",
    "STRATEGY_C_SHORT_PUT": "Short Put (Accumulation)",
    "STRATEGY_D_IRON_BUTTERFLY": "Iron Butterfly",
    "STRATEGY_F_BULL_PUT_SPREAD": "Bull Put Spread",
    "STRATEGY_F_BEAR_CALL_SPREAD": "Bear Call Spread",
}


@app.get("/api/greg/positions")
def get_greg_positions(
    underlying: Optional[str] = None,
    sandbox_filter: Optional[str] = None,
) -> JSONResponse:
    """
    Get all Greg positions for the Greg Lab view.
    
    Args:
        underlying: Filter by underlying (BTC, ETH)
        sandbox_filter: 'sandbox_only', 'non_sandbox', or 'all' (default)
    """
    from src.position_tracker import PositionTracker
    from src.greg_trading_store import greg_trading_store
    
    tracker = PositionTracker()
    mode_state = greg_trading_store.get_state()
    deribit_env = settings.deribit_env
    
    positions_data = []
    sandbox_runs = {}
    
    with tracker._lock:
        for chain in tracker._chains.values():
            if not chain.is_open():
                continue
            
            is_sandbox = chain.is_sandbox()
            if sandbox_filter == "sandbox_only" and not is_sandbox:
                continue
            if sandbox_filter == "non_sandbox" and is_sandbox:
                continue
            
            if underlying and chain.underlying != underlying.upper():
                continue
            
            is_greg_strategy = chain.strategy_type.startswith("STRATEGY_")
            if not is_greg_strategy:
                continue
            
            human_name = GREG_STRATEGY_NAMES.get(chain.strategy_type, chain.strategy_type)
            
            if chain.expiry:
                from datetime import timezone as tz
                now = datetime.now(tz.utc)
                dte = max(0, int((chain.expiry - now).total_seconds() / 86400))
            else:
                dte = 0
            
            size = chain.legs[-1].quantity if chain.legs else 0
            entry_price = chain.legs[0].entry_price if chain.legs else 0
            notional = size * entry_price
            
            if is_sandbox and chain.origin == "GREG_SANDBOX" and chain.run_id:
                if chain.run_id not in sandbox_runs:
                    sandbox_runs[chain.run_id] = {"btc": 0, "eth": 0, "total_pnl": 0.0}
                if chain.underlying == "BTC":
                    sandbox_runs[chain.run_id]["btc"] += 1
                elif chain.underlying == "ETH":
                    sandbox_runs[chain.run_id]["eth"] += 1
                sandbox_runs[chain.run_id]["total_pnl"] += chain.unrealized_pnl_pct
            
            if is_sandbox:
                badge = "SANDBOX"
            elif deribit_env == "testnet":
                badge = "DEMO"
            elif mode_state["mode"] == "live":
                badge = "LIVE"
            else:
                badge = "PAPER"
            
            positions_data.append({
                "position_id": chain.position_id,
                "underlying": chain.underlying,
                "strategy_type": chain.strategy_type,
                "human_readable_name": human_name,
                "size": size,
                "notional": notional,
                "sandbox": is_sandbox,
                "origin": chain.origin,
                "run_id": chain.run_id,
                "mode": chain.mode,
                "badge": badge,
                "pnl_pct": chain.unrealized_pnl_pct,
                "pnl_usd": chain.unrealized_pnl,
                "dte": dte,
                "net_delta": 0.0,
                "suggested_action": "HOLD",
                "urgency": "LOW",
                "entry_time": chain.open_time.isoformat() if chain.open_time else None,
            })
    
    latest_sandbox_run = None
    if sandbox_runs:
        latest_run_id = max(sandbox_runs.keys())
        run_data = sandbox_runs[latest_run_id]
        latest_sandbox_run = {
            "run_id": latest_run_id,
            "btc_count": run_data["btc"],
            "eth_count": run_data["eth"],
            "total_pnl_pct": run_data["total_pnl"],
        }
    
    return JSONResponse(content={
        "ok": True,
        "positions": positions_data,
        "count": len(positions_data),
        "mode": mode_state["mode"],
        "enable_live_execution": mode_state["enable_live_execution"],
        "deribit_env": deribit_env,
        "sandbox_summary": latest_sandbox_run,
    })


@app.get("/api/greg/positions/{position_id}/logs")
def get_greg_position_logs(position_id: str, limit: int = 50) -> JSONResponse:
    """
    Get decision log timeline for a specific position.
    """
    from src.db.models_greg_decision import GregDecisionLog
    from src.db import get_db_session
    
    try:
        with get_db_session() as session:
            entries = (
                session.query(GregDecisionLog)
                .filter(GregDecisionLog.position_id == position_id)
                .order_by(GregDecisionLog.timestamp.asc())
                .limit(limit)
                .all()
            )
            
            logs = []
            for e in entries:
                logs.append({
                    "id": e.id,
                    "timestamp": e.timestamp.isoformat() if e.timestamp else None,
                    "action_type": e.action_type,
                    "mode": e.mode,
                    "suggested": e.suggested,
                    "executed": e.executed,
                    "reason": e.reason,
                    "pnl_pct": e.pnl_pct,
                    "pnl_usd": e.pnl_usd,
                    "net_delta": e.net_delta,
                    "vrp_30d": e.vrp_30d,
                    "adx_14d": e.adx_14d,
                    "order_ids": e.order_ids,
                })
            
            return JSONResponse(content={
                "ok": True,
                "position_id": position_id,
                "logs": logs,
                "count": len(logs),
            })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": str(e)},
        )


@app.get("/api/bots/greg/hedging")
def get_greg_hedging_status() -> JSONResponse:
    """
    Return the current hedging status including:
    - Engine configuration (dry_run mode)
    - Recent hedge history
    - Hedging rules overview
    """
    try:
        from src.hedging import get_hedge_engine, load_greg_hedge_rules
        
        engine = get_hedge_engine(dry_run=True)
        history = engine.get_hedge_history(limit=20)
        rules = load_greg_hedge_rules()
        
        global_defs = rules.get("global_definitions", {})
        hedge_instruments = global_defs.get("hedge_instrument", {})
        
        strategies_summary = []
        for strat_key, strat_config in rules.get("strategies", {}).items():
            hedge_cfg = strat_config.get("hedge", {})
            strategies_summary.append({
                "strategy": strat_key,
                "display_name": strat_config.get("display_name", strat_key),
                "hedge_mode": hedge_cfg.get("mode", "NONE"),
                "delta_threshold": hedge_cfg.get("delta_abs_threshold"),
            })
        
        return JSONResponse(content={
            "ok": True,
            "dry_run": engine.dry_run,
            "hedge_instruments": hedge_instruments,
            "strategies": strategies_summary,
            "history": history,
            "history_count": len(history),
        })
    except Exception as e:
        return JSONResponse(content={"ok": False, "error": str(e)})


@app.post("/api/bots/greg/hedging/evaluate")
def evaluate_greg_hedging() -> JSONResponse:
    """
    Evaluate hedging needs for Greg positions (demo/mock for Phase 1).
    
    Returns proposed hedge orders without executing them.
    Currently uses mock positions for demonstration - real position
    integration planned for Phase 2 when live trading is enabled.
    """
    try:
        from src.hedging import get_hedge_engine, GregPosition
        from src.greg_position_manager import get_greg_position_rules
        
        engine = get_hedge_engine(dry_run=True)
        
        mock_positions = [
            GregPosition(
                position_id="demo:BTC-STRADDLE-1",
                strategy_type="STRATEGY_A_STRADDLE",
                underlying="BTC",
                option_legs=[
                    {"instrument": "BTC-27DEC25-100000-C", "delta": -0.50, "size": 1.0},
                    {"instrument": "BTC-27DEC25-100000-P", "delta": 0.50, "size": 1.0},
                ],
                hedge_perp_size=0.0,
                net_delta=0.0,
            ),
            GregPosition(
                position_id="demo:BTC-STRADDLE-2",
                strategy_type="STRATEGY_A_STRADDLE",
                underlying="BTC",
                option_legs=[
                    {"instrument": "BTC-27DEC25-95000-C", "delta": -0.35, "size": 1.0},
                    {"instrument": "BTC-27DEC25-95000-P", "delta": 0.65, "size": 1.0},
                ],
                hedge_perp_size=0.0,
                net_delta=0.30,
            ),
            GregPosition(
                position_id="demo:ETH-STRANGLE-1",
                strategy_type="STRATEGY_A_STRANGLE",
                underlying="ETH",
                option_legs=[
                    {"instrument": "ETH-27DEC25-4000-C", "delta": -0.25, "size": 1.0},
                    {"instrument": "ETH-27DEC25-3000-P", "delta": 0.10, "size": 1.0},
                ],
                hedge_perp_size=0.0,
                net_delta=-0.15,
            ),
        ]
        
        proposed_hedges = []
        for pos in mock_positions:
            hedge_rules = engine.get_hedge_rules(pos.strategy_type)
            order = engine.build_hedge_order(pos, hedge_rules)
            if order:
                proposed_hedges.append({
                    "position_id": pos.position_id,
                    "strategy_type": pos.strategy_type,
                    "underlying": pos.underlying,
                    "net_delta": engine.compute_net_delta_for_position(pos),
                    "threshold": hedge_rules.delta_abs_threshold,
                    "proposed_order": order.to_dict(),
                })
            else:
                proposed_hedges.append({
                    "position_id": pos.position_id,
                    "strategy_type": pos.strategy_type,
                    "underlying": pos.underlying,
                    "net_delta": engine.compute_net_delta_for_position(pos),
                    "threshold": hedge_rules.delta_abs_threshold,
                    "proposed_order": None,
                    "status": "no_hedge_needed",
                })
        
        return JSONResponse(content={
            "ok": True,
            "positions_evaluated": len(mock_positions),
            "hedges_proposed": len([h for h in proposed_hedges if h.get("proposed_order")]),
            "results": proposed_hedges,
            "dry_run": engine.dry_run,
        })
    except Exception as e:
        return JSONResponse(content={"ok": False, "error": str(e)})


@app.get("/api/bots/greg/hedge_history")
def get_greg_hedge_history(limit: int = 50) -> JSONResponse:
    """Return recent hedge execution history."""
    try:
        from src.hedging import get_hedge_engine
        
        engine = get_hedge_engine(dry_run=True)
        history = engine.get_hedge_history(limit=limit)
        
        return JSONResponse(content={
            "ok": True,
            "history": history,
            "count": len(history),
        })
    except Exception as e:
        return JSONResponse(content={"ok": False, "error": str(e)})


@app.post("/api/bots/greg/hedging/dry_run")
def set_hedge_dry_run(request: dict) -> JSONResponse:
    """
    Toggle dry-run mode for the hedge engine.
    Body: {"dry_run": true/false}
    """
    try:
        from src.hedging import get_hedge_engine
        
        dry_run = request.get("dry_run", True)
        engine = get_hedge_engine()
        engine.set_dry_run(dry_run)
        
        return JSONResponse(content={
            "ok": True,
            "dry_run": engine.dry_run,
            "message": f"Hedge engine dry_run set to {dry_run}",
        })
    except Exception as e:
        return JSONResponse(content={"ok": False, "error": str(e)})


@app.post("/api/test_kill_switch")
def test_kill_switch() -> JSONResponse:
    """Test risk engine with a synthetic action (dry run)."""
    try:
        from src.risk_engine import check_action_allowed
        from src.models import AgentState, PortfolioState, ActionType
        
        mock_portfolio = PortfolioState(
            equity_usd=100000.0,
            margin_used_usd=20000.0,
            margin_available_usd=80000.0,
            net_delta=0.5,
            option_positions=[],
        )
        
        mock_state = AgentState(
            portfolio=mock_portfolio,
            spot={"BTC": 100000.0, "ETH": 3500.0},
            candidate_options=[],
            market_context=None,
            timestamp="2025-01-01T00:00:00Z",
        )
        
        proposed_action = {
            "action": ActionType.OPEN_COVERED_CALL,
            "params": {
                "symbol": "BTC-TEST-100000-C",
                "size": 0.1,
            },
            "reasoning": "Test action for kill switch validation",
        }
        
        allowed, reasons = check_action_allowed(mock_state, proposed_action, settings)
        
        return JSONResponse(content={
            "ok": True,
            "allowed": allowed,
            "reasons": reasons,
            "config": {
                "daily_drawdown_limit_pct": getattr(settings, "daily_drawdown_limit_pct", 0.0),
                "kill_switch_enabled": getattr(settings, "kill_switch_enabled", False),
            }
        })
    except Exception as e:
        return JSONResponse(content={"ok": False, "error": str(e)})


@app.post("/api/agent_healthcheck")
def run_healthcheck_endpoint() -> JSONResponse:
    """Run full agent healthcheck and return results (with caching)."""
    try:
        from src.healthcheck import run_and_cache_healthcheck, get_health_status_for_api
        
        cached_status = run_and_cache_healthcheck(settings)
        result = cached_status.details
        health_api_status = get_health_status_for_api()
        
        return JSONResponse(content={
            "ok": result.get("overall_status") != "FAIL",
            "overall_status": result.get("overall_status", "UNKNOWN"),
            "summary": result.get("summary", ""),
            "results": result.get("results", []),
            "last_run_at": health_api_status.get("last_run_at"),
            "agent_paused_due_to_health": health_api_status.get("agent_paused_due_to_health", False),
        })
    except Exception as e:
        return JSONResponse(content={"ok": False, "error": str(e)})


@app.get("/api/system_health/status")
def get_system_health_status() -> JSONResponse:
    """Get cached system health status for dashboard display."""
    try:
        from src.healthcheck import get_health_status_for_api
        
        status = get_health_status_for_api()
        
        return JSONResponse(content={
            "ok": True,
            **status,
        })
    except Exception as e:
        return JSONResponse(content={"ok": False, "error": str(e)})


@app.get("/api/llm_readiness")
def get_llm_readiness_endpoint() -> JSONResponse:
    """Check if LLM is ready for diagnostic tests."""
    try:
        from src.healthcheck import get_llm_readiness
        
        result = get_llm_readiness(settings)
        return JSONResponse(content={"ok": True, **result})
    except Exception as e:
        return JSONResponse(content={"ok": False, "error": str(e)})


@app.post("/api/steward/run")
def run_steward() -> JSONResponse:
    """
    Run the AI Steward once and return a fresh report.
    Never touches Deribit or executes trades.
    """
    try:
        from src.system_steward import generate_steward_report
        report = generate_steward_report()
        return JSONResponse(content=report.model_dump())
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": f"steward_failed: {e}"},
        )


@app.get("/api/steward/report")
def get_steward_report() -> JSONResponse:
    """
    Return the last steward report, or a stub if it has not been run yet.
    """
    try:
        from src.system_steward import get_last_report
        report = get_last_report()
        if report is None:
            return JSONResponse(
                content={
                    "ok": True,
                    "generated_at": None,
                    "llm_used": False,
                    "summary": "Steward has not been run yet.",
                    "top_items": [],
                    "builder_prompt": "",
                }
            )
        return JSONResponse(content=report.model_dump())
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": f"steward_failed: {e}"},
        )


@app.get("/api/greg_sweetspots")
def get_greg_sweetspots() -> JSONResponse:
    """
    Return the latest Greg environment sweet spots, if available.
    Reads backtest/output/greg_heatmap_sweetspots.json and wraps it in {ok, data}.
    """
    try:
        import json as json_lib
        base_dir = Path(__file__).resolve().parent.parent
        json_path = base_dir / "backtest" / "output" / "greg_heatmap_sweetspots.json"
        
        if not json_path.exists():
            return JSONResponse(
                content={
                    "ok": False,
                    "error": "No sweet spots file found. Click 'Run Greg Sweet Spot Scan' to generate."
                },
            )
        
        raw = json_path.read_text(encoding="utf-8")
        data = json_lib.loads(raw)
        
        return JSONResponse(content={"ok": True, "data": data})
    except Exception as e:
        return JSONResponse(content={"ok": False, "error": str(e)}, status_code=500)


@app.post("/api/greg_sweetspots/run")
def run_greg_sweetspots() -> JSONResponse:
    """
    Trigger a Greg environment sweet spot sweep.

    This is a research-only operation that runs synchronously within a single
    request/response cycle. It analyzes synthetic market data across metric
    pairs and strategies to find optimal trading regions.
    """
    try:
        from src.backtest.greg_sweetspots import run_greg_sweetspot_sweep
        
        base_dir = Path(__file__).resolve().parent.parent
        
        json_path = run_greg_sweetspot_sweep(base_dir=base_dir)
        
        return JSONResponse(
            content={
                "ok": True,
                "message": "Greg sweet spot sweep completed.",
                "json_path": str(json_path),
            }
        )
    except Exception as e:
        return JSONResponse(
            content={"ok": False, "error": str(e)},
            status_code=500,
        )


class RuntimeConfigUpdate(BaseModel):
    """Request model for updating runtime configuration."""
    kill_switch_enabled: Optional[bool] = None
    daily_drawdown_limit_pct: Optional[float] = None
    decision_mode: Optional[str] = None
    dry_run: Optional[bool] = None
    position_reconcile_action: Optional[str] = None


@app.get("/api/system/runtime-config")
def get_runtime_config() -> JSONResponse:
    """Fetch current runtime configuration settings."""
    return JSONResponse(content={
        "ok": True,
        "kill_switch_enabled": settings.kill_switch_enabled,
        "daily_drawdown_limit_pct": settings.daily_drawdown_limit_pct,
        "decision_mode": settings.decision_mode,
        "dry_run": settings.dry_run,
        "position_reconcile_action": settings.position_reconcile_action,
    })


@app.post("/api/system/runtime-config")
def update_runtime_config(update: RuntimeConfigUpdate) -> JSONResponse:
    """Update runtime configuration settings (in-memory only, does not persist across restarts)."""
    updated = {}
    errors = []
    
    if update.kill_switch_enabled is not None:
        settings.kill_switch_enabled = update.kill_switch_enabled
        updated["kill_switch_enabled"] = update.kill_switch_enabled
    
    if update.daily_drawdown_limit_pct is not None:
        if update.daily_drawdown_limit_pct < 0:
            errors.append("daily_drawdown_limit_pct must be >= 0")
        else:
            settings.daily_drawdown_limit_pct = update.daily_drawdown_limit_pct
            updated["daily_drawdown_limit_pct"] = update.daily_drawdown_limit_pct
    
    if update.decision_mode is not None:
        valid_modes = ["rule_only", "llm_only", "hybrid_shadow"]
        if update.decision_mode not in valid_modes:
            errors.append(f"decision_mode must be one of: {', '.join(valid_modes)}")
        else:
            settings.decision_mode = update.decision_mode  # type: ignore
            updated["decision_mode"] = update.decision_mode
    
    if update.dry_run is not None:
        settings.dry_run = update.dry_run
        updated["dry_run"] = update.dry_run
    
    if update.position_reconcile_action is not None:
        valid_actions = ["halt", "auto_heal"]
        if update.position_reconcile_action not in valid_actions:
            errors.append(f"position_reconcile_action must be one of: {', '.join(valid_actions)}")
        else:
            settings.position_reconcile_action = update.position_reconcile_action  # type: ignore
            updated["position_reconcile_action"] = update.position_reconcile_action
    
    if errors:
        return JSONResponse(
            status_code=400,
            content={
                "ok": False,
                "errors": errors,
            }
        )
    
    return JSONResponse(content={
        "ok": True,
        "updated": updated,
        "current": {
            "kill_switch_enabled": settings.kill_switch_enabled,
            "daily_drawdown_limit_pct": settings.daily_drawdown_limit_pct,
            "decision_mode": settings.decision_mode,
            "dry_run": settings.dry_run,
            "position_reconcile_action": settings.position_reconcile_action,
        }
    })


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """Full HTML dashboard with Live Agent view and Backtesting Lab."""
    decision_mode = "LLM" if settings.llm_enabled else "Rule-based"
    op_mode = settings.mode.upper()
    explore_pct = int(settings.explore_prob * 100)
    training_enabled = settings.is_training_enabled
    training_badge = f"TRAINING ({', '.join(settings.training_strategies)})" if training_enabled else ""
    
    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Options Agent Dashboard</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      max-width: 1200px;
      margin: 0 auto;
      padding: 1rem;
      background: #f8f9fa;
      color: #333;
    }}
    h1 {{ color: #1a1a2e; margin-bottom: 0.25rem; font-size: 1.75rem; }}
    .subtitle {{ color: #666; margin-bottom: 1rem; }}
    .badge {{
      display: inline-block;
      padding: 0.25rem 0.5rem;
      border-radius: 4px;
      font-size: 0.85rem;
      font-weight: 600;
      margin-right: 0.5rem;
    }}
    .badge-mode {{ background: #e3f2fd; color: #1565c0; }}
    .badge-research {{ background: #f3e5f5; color: #7b1fa2; }}
    .badge-production {{ background: #e8f5e9; color: #2e7d32; }}
    .badge-dry {{ background: #fff3e0; color: #e65100; }}
    .badge-live {{ background: #e8f5e9; color: #2e7d32; }}
    .badge-training {{ background: #ffecb3; color: #ff6f00; }}
    
    /* Bots sensor coloring */
    .sensor-good {{ color: #008000; font-weight: 600; }}
    .sensor-bad {{ color: #c00000; font-weight: 600; }}
    .sensor-neutral {{ color: inherit; }}
    
    /* Bots criterion coloring */
    .criterion-ok {{ color: #008000; font-weight: 600; }}
    .criterion-bad {{ color: #c00000; font-weight: 600; }}
    .criterion-missing {{ color: #777777; }}
    
    .switch {{
      position: relative;
      display: inline-block;
      width: 48px;
      height: 24px;
    }}
    .switch input {{
      opacity: 0;
      width: 0;
      height: 0;
    }}
    .slider {{
      position: absolute;
      cursor: pointer;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      background-color: #ccc;
      transition: .3s;
      border-radius: 24px;
    }}
    .slider:before {{
      position: absolute;
      content: "";
      height: 18px;
      width: 18px;
      left: 3px;
      bottom: 3px;
      background-color: white;
      transition: .3s;
      border-radius: 50%;
    }}
    input:checked + .slider {{
      background-color: #ff6f00;
    }}
    input:checked + .slider:before {{
      transform: translateX(24px);
    }}
    
    .progress-bar {{
      width: 100%;
      height: 24px;
      background: #e9ecef;
      border-radius: 12px;
      overflow: hidden;
      margin: 0.5rem 0;
    }}
    .progress-bar-inner {{
      height: 100%;
      background: linear-gradient(90deg, #1565c0, #42a5f5);
      border-radius: 12px;
      transition: width 0.3s ease;
      display: flex;
      align-items: center;
      justify-content: center;
      color: white;
      font-weight: 600;
      font-size: 0.8rem;
    }}
    
    .bt-status-header {{
      display: flex;
      align-items: center;
      gap: 1rem;
      margin-bottom: 1rem;
      flex-wrap: wrap;
    }}
    .bt-status-indicator {{
      padding: 0.25rem 0.75rem;
      border-radius: 4px;
      font-weight: 600;
      font-size: 0.85rem;
    }}
    .bt-status-idle {{ background: #e9ecef; color: #666; }}
    .bt-status-running {{ background: #e3f2fd; color: #1565c0; }}
    .bt-status-finished {{ background: #e8f5e9; color: #2e7d32; }}
    .bt-status-error {{ background: #ffebee; color: #c62828; }}
    
    .steps-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.85rem;
      margin-top: 1rem;
    }}
    .steps-table th, .steps-table td {{
      padding: 0.5rem;
      text-align: left;
      border-bottom: 1px solid #e9ecef;
    }}
    .steps-table th {{
      background: #f8f9fa;
      font-weight: 600;
      color: #666;
    }}
    .steps-table tr:hover {{ background: #f8f9fa; }}
    .traded-yes {{ color: #2e7d32; font-weight: 600; }}
    .traded-no {{ color: #666; }}
    
    .error-text {{
      color: #c62828;
      background: #ffebee;
      padding: 0.75rem;
      border-radius: 6px;
      margin: 0.5rem 0;
      font-size: 0.9rem;
    }}
    
    .bt-status-paused {{ background: #fff3e0; color: #e65100; }}
    
    .tabs {{
      display: flex;
      gap: 0.5rem;
      margin: 1rem 0;
      border-bottom: 2px solid #dee2e6;
    }}
    .tab {{
      padding: 0.75rem 1.5rem;
      background: #e9ecef;
      border: none;
      border-radius: 8px 8px 0 0;
      cursor: pointer;
      font-weight: 600;
      color: #666;
      transition: all 0.2s;
    }}
    .tab.active {{
      background: white;
      color: #1565c0;
      border-bottom: 2px solid white;
      margin-bottom: -2px;
    }}
    .tab:hover {{ background: #dee2e6; }}
    .tab.active:hover {{ background: white; }}
    
    .tab-content {{ display: none; }}
    .tab-content.active {{ display: block; }}
    
    .section {{
      margin: 1rem 0;
      padding: 1.25rem;
      background: white;
      border: 1px solid #dee2e6;
      border-radius: 8px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }}
    .section h2 {{
      margin-top: 0;
      color: #1a1a2e;
      font-size: 1.15rem;
      border-bottom: 2px solid #e9ecef;
      padding-bottom: 0.5rem;
    }}
    .section h3 {{
      margin: 1rem 0 0.5rem 0;
      color: #333;
      font-size: 1rem;
    }}
    
    .status-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 0.75rem;
      margin-bottom: 1rem;
    }}
    .status-card {{
      background: #f8f9fa;
      padding: 0.75rem;
      border-radius: 6px;
      text-align: center;
    }}
    .status-card .label {{ font-size: 0.75rem; color: #666; margin-bottom: 0.2rem; }}
    .status-card .value {{ font-size: 1.25rem; font-weight: 700; color: #1a1a2e; }}
    
    .decision-card {{
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      color: white;
      padding: 1.25rem;
      border-radius: 10px;
      margin-bottom: 1rem;
    }}
    .decision-card h3 {{ color: rgba(255,255,255,0.9); margin: 0 0 0.75rem 0; font-size: 0.9rem; }}
    .decision-card .action {{ font-size: 1.5rem; font-weight: 700; margin-bottom: 0.5rem; }}
    .decision-card .details {{ font-size: 0.9rem; opacity: 0.9; }}
    .decision-card .reasoning {{
      background: rgba(255,255,255,0.15);
      padding: 0.75rem;
      border-radius: 6px;
      margin-top: 0.75rem;
      font-size: 0.85rem;
      line-height: 1.4;
    }}
    
    .decisions-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.85rem;
    }}
    .decisions-table th, .decisions-table td {{
      padding: 0.6rem 0.5rem;
      text-align: left;
      border-bottom: 1px solid #e9ecef;
    }}
    .decisions-table th {{ background: #f8f9fa; font-weight: 600; color: #666; }}
    .decisions-table tr:hover {{ background: #f8f9fa; }}
    .pass {{ color: #2e7d32; font-weight: 600; }}
    .blocked {{ color: #c62828; font-weight: 600; }}
    
    .form-group {{
      margin-bottom: 1rem;
    }}
    .form-group label {{
      display: block;
      font-weight: 600;
      margin-bottom: 0.25rem;
      font-size: 0.9rem;
    }}
    .form-row {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 1rem;
    }}
    input, select {{
      width: 100%;
      padding: 0.6rem;
      border: 1px solid #ced4da;
      border-radius: 6px;
      font-size: 0.95rem;
    }}
    input:focus, select:focus {{
      outline: none;
      border-color: #1565c0;
      box-shadow: 0 0 0 3px rgba(21, 101, 192, 0.1);
    }}
    textarea {{
      width: 100%;
      height: 80px;
      padding: 0.75rem;
      border: 1px solid #ced4da;
      border-radius: 6px;
      font-family: inherit;
      font-size: 0.95rem;
      resize: vertical;
    }}
    button {{
      padding: 0.6rem 1.5rem;
      background: #1565c0;
      color: white;
      border: none;
      border-radius: 6px;
      font-size: 0.95rem;
      font-weight: 600;
      cursor: pointer;
      transition: background 0.2s;
    }}
    button:hover {{ background: #0d47a1; }}
    button:disabled {{ background: #90a4ae; cursor: not-allowed; }}
    button.secondary {{ background: #6c757d; }}
    button.secondary:hover {{ background: #5a6268; }}
    
    .metrics-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
      gap: 0.75rem;
      margin: 1rem 0;
    }}
    .metric {{
      background: #e3f2fd;
      padding: 1rem;
      border-radius: 8px;
      text-align: center;
    }}
    .metric .value {{ font-size: 1.5rem; font-weight: 700; color: #1565c0; }}
    .metric .label {{ font-size: 0.8rem; color: #666; margin-top: 0.25rem; }}
    .metric.positive .value {{ color: #2e7d32; }}
    .metric.negative .value {{ color: #c62828; }}
    
    /* TradingView-style Summary Panel */
    .tv-summary-panel {{
      background: linear-gradient(135deg, #1a2332 0%, #0d1421 100%);
      border-radius: 12px;
      padding: 1.5rem;
      margin: 1rem 0;
      color: #fff;
    }}
    .tv-summary-header {{
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      margin-bottom: 1.5rem;
      padding-bottom: 1rem;
      border-bottom: 1px solid rgba(255,255,255,0.1);
    }}
    .tv-main-stat {{
      display: flex;
      align-items: baseline;
      gap: 0.5rem;
    }}
    .tv-main-stat .tv-value {{
      font-size: 2.5rem;
      font-weight: 700;
    }}
    .tv-main-stat .tv-pct {{
      font-size: 1.2rem;
      opacity: 0.8;
    }}
    .tv-main-stat.positive .tv-value, .tv-main-stat.positive .tv-pct {{ color: #26a69a; }}
    .tv-main-stat.negative .tv-value, .tv-main-stat.negative .tv-pct {{ color: #ef5350; }}
    .tv-secondary-stats {{
      display: flex;
      gap: 2rem;
    }}
    .tv-stat {{
      text-align: right;
    }}
    .tv-stat .tv-label {{
      font-size: 0.75rem;
      opacity: 0.6;
      display: block;
      margin-bottom: 0.25rem;
    }}
    .tv-stat .tv-value {{
      font-size: 1.1rem;
      font-weight: 600;
    }}
    .tv-metrics-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 1.5rem;
    }}
    .tv-metric-group h4 {{
      font-size: 0.85rem;
      font-weight: 600;
      opacity: 0.6;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      margin: 0 0 0.75rem 0;
    }}
    .tv-metric-row {{
      display: flex;
      justify-content: space-between;
      padding: 0.4rem 0;
      border-bottom: 1px solid rgba(255,255,255,0.05);
    }}
    .tv-metric-label {{
      font-size: 0.9rem;
      opacity: 0.7;
    }}
    .tv-metric-value {{
      font-size: 0.9rem;
      font-weight: 600;
    }}
    .tv-metric-value.positive {{ color: #26a69a; }}
    .tv-metric-value.negative {{ color: #ef5350; }}
    
    .chart-legend {{
      display: flex;
      justify-content: center;
      gap: 2rem;
      margin-top: 0.5rem;
      padding: 0.5rem;
    }}
    .legend-item {{
      display: flex;
      align-items: center;
      gap: 0.5rem;
      font-size: 0.85rem;
      color: #666;
    }}
    .legend-color {{
      width: 16px;
      height: 4px;
      border-radius: 2px;
    }}
    
    .chart-container {{
      width: 100%;
      height: 250px;
      background: #f8f9fa;
      border-radius: 8px;
      margin: 1rem 0;
      position: relative;
    }}
    canvas {{ width: 100% !important; height: 100% !important; }}
    
    .insights-box {{
      background: #e8f5e9;
      border-left: 4px solid #4caf50;
      padding: 1rem;
      border-radius: 0 8px 8px 0;
      margin: 1rem 0;
      white-space: pre-wrap;
      line-height: 1.5;
    }}
    
    .modal {{
      position: fixed;
      z-index: 1000;
      left: 0;
      top: 0;
      width: 100%;
      height: 100%;
      background: rgba(0,0,0,0.5);
      display: flex;
      align-items: center;
      justify-content: center;
    }}
    .modal-content {{
      background: #fff;
      padding: 1.5rem;
      border-radius: 8px;
      max-width: 700px;
      max-height: 80vh;
      overflow-y: auto;
      box-shadow: 0 4px 20px rgba(0,0,0,0.2);
      position: relative;
    }}
    .modal-close {{
      position: absolute;
      top: 0.5rem;
      right: 1rem;
      font-size: 1.5rem;
      cursor: pointer;
      color: #666;
    }}
    .modal-close:hover {{
      color: #333;
    }}
    .view-btn {{
      background: #2196f3;
      color: #fff;
      border: none;
      padding: 0.25rem 0.5rem;
      border-radius: 4px;
      cursor: pointer;
      font-size: 0.75rem;
    }}
    .view-btn:hover {{
      background: #1976d2;
    }}
    .trigger-tp {{ color: #4caf50; font-weight: 600; }}
    .trigger-defensive {{ color: #ff9800; font-weight: 600; }}
    .trigger-expiry {{ color: #9e9e9e; }}
    
    .answer-box {{
      background: #e8f5e9;
      border-left: 4px solid #4caf50;
      padding: 1rem;
      border-radius: 0 8px 8px 0;
      white-space: pre-wrap;
    }}
    
    .warning {{
      background: #fff3e0;
      border: 1px solid #ffcc80;
      padding: 0.75rem;
      border-radius: 6px;
      margin-bottom: 1rem;
      font-size: 0.9rem;
    }}
    
    .loading {{
      text-align: center;
      padding: 2rem;
      color: #666;
    }}
    .spinner {{
      display: inline-block;
      width: 24px;
      height: 24px;
      border: 3px solid #e9ecef;
      border-top-color: #1565c0;
      border-radius: 50%;
      animation: spin 1s linear infinite;
    }}
    @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
    
    .last-update {{ font-size: 0.8rem; color: #888; text-align: right; margin-top: 0.5rem; }}
    
    pre {{
      background: #f1f3f4;
      padding: 1rem;
      border-radius: 6px;
      overflow: auto;
      max-height: 300px;
      font-size: 0.8rem;
      white-space: pre-wrap;
    }}
  </style>
</head>
<body>
  <h1>Options Trading Agent</h1>
  <p class="subtitle">Deribit Testnet - Covered Call Strategy</p>
  
  <div style="display:flex;align-items:center;gap:1rem;flex-wrap:wrap;">
    <div>
      <span class="badge badge-mode">{decision_mode}</span>
      <span class="badge {'badge-research' if settings.is_research else 'badge-production'}">
        {op_mode} ({explore_pct}% explore)
      </span>
      <span class="badge {'badge-dry' if settings.dry_run else 'badge-live'}">
        {'DRY RUN' if settings.dry_run else 'LIVE TRADING'}
      </span>
      <span class="badge badge-training" id="training-badge" style="display:{'inline-block' if training_enabled else 'none'};">
        TRAINING
      </span>
      <span class="badge" id="agent-status-badge" style="background:#e8f5e9;color:#2e7d32;">Active</span>
    </div>
    <div class="toggle-container" style="display:flex;align-items:center;gap:0.5rem;">
      <span style="font-size:0.85rem;color:#666;">Training Mode:</span>
      <label class="switch">
        <input type="checkbox" id="training-toggle" {'checked' if training_enabled else ''} onchange="toggleTraining(this.checked)">
        <span class="slider"></span>
      </label>
    </div>
  </div>

  <div class="tabs">
    <button class="tab active" onclick="showTab('live')">Dashboard</button>
    <button class="tab" onclick="showTab('backtesting')">Backtesting</button>
    <button class="tab" onclick="showTab('calibration')">Calibration</button>
    <button class="tab" onclick="showTab('strategies')">Bots</button>
    <button class="tab" onclick="showTab('greglab')">Greg Lab</button>
    <button class="tab" onclick="showTab('health')">System Health</button>
    <button class="tab" onclick="showTab('chat')">Chat</button>
  </div>

  <!-- DASHBOARD TAB (formerly Live Agent) -->
  <div id="tab-live" class="tab-content active">
    <div class="section">
      <h2>Live Market Sensors</h2>
      <div style="margin-bottom: 8px;">
        <button id="dashboard-refresh-sensors-btn" style="background:#2a2a2a;border:1px solid #444;color:#ccc;padding:6px 12px;border-radius:4px;cursor:pointer;">Refresh Sensors</button>
        <label style="margin-left: 12px; font-size: 0.9em; color:#ccc;">
          <input type="checkbox" id="dashboard-show-sensor-debug" />
          Show debug inputs
        </label>
      </div>
      <div style="overflow-x:auto;">
        <table class="steps-table">
          <thead>
            <tr>
              <th>Sensor</th>
              <th>BTC</th>
              <th>ETH</th>
            </tr>
          </thead>
          <tbody id="dashboard-sensors-body">
            <tr>
              <td colspan="3" style="text-align:center;color:#666;">Loading...</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
    
    <div class="section">
      <h2>Market Overview</h2>
      <div class="status-grid" id="status-cards">
        <div class="status-card">
          <div class="label">BTC Price</div>
          <div class="value" id="btc-price">--</div>
        </div>
        <div class="status-card">
          <div class="label">ETH Price</div>
          <div class="value" id="eth-price">--</div>
        </div>
        <div class="status-card">
          <div class="label">Portfolio</div>
          <div class="value" id="portfolio-value">--</div>
        </div>
        <div class="status-card">
          <div class="label">Margin Used</div>
          <div class="value" id="margin-used">--</div>
        </div>
        <div class="status-card">
          <div class="label">Net Delta</div>
          <div class="value" id="net-delta">--</div>
        </div>
        <div class="status-card">
          <div class="label">Positions</div>
          <div class="value" id="positions-count">--</div>
        </div>
      </div>
      <div class="last-update" id="last-update">Last update: --</div>
    </div>

    <div class="section" id="strategy-status-section">
      <h2>Strategy & Safeguards</h2>
      <div id="strategy-status-box" style="border:1px solid #333;border-radius:8px;padding:12px;background:#1e1e1e;">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
          <div style="display:flex;align-items:center;gap:12px;">
            <div id="strategy-headline" style="font-weight:bold;font-size:1rem;">Loading...</div>
            <div id="strategy-badges"></div>
          </div>
          <button onclick="showRulesModal()" style="background:#2a2a2a;border:1px solid #444;color:#888;padding:4px 10px;border-radius:4px;font-size:0.8em;cursor:pointer;">View full rules</button>
        </div>
        <div style="display:grid;grid-template-columns:repeat(3, 1fr);gap:12px;font-size:0.85em;">
          <div style="background:#2a2a2a;border-radius:6px;padding:8px;">
            <div style="color:#7c4dff;font-weight:600;margin-bottom:4px;">Training Mode</div>
            <ul id="training-rules-compact" style="list-style:none;padding:0;margin:0;color:#ccc;"></ul>
          </div>
          <div style="background:#2a2a2a;border-radius:6px;padding:8px;">
            <div style="color:#00bcd4;font-weight:600;margin-bottom:4px;">Live Mode</div>
            <ul id="live-rules-compact" style="list-style:none;padding:0;margin:0;color:#ccc;"></ul>
          </div>
          <div style="background:#2a2a2a;border-radius:6px;padding:8px;">
            <div style="color:#4caf50;font-weight:600;margin-bottom:4px;">Safeguards</div>
            <div id="safeguards-compact" style="display:flex;flex-wrap:wrap;gap:4px;"></div>
          </div>
        </div>
      </div>
    </div>
    
    <!-- Rules Modal -->
    <div id="rules-modal" class="modal" style="display:none;">
      <div class="modal-content" style="max-width:800px;background:#1e1e1e;color:#fff;">
        <span class="modal-close" onclick="closeRulesModal()" style="color:#888;">&times;</span>
        <h3 style="margin-top:0;">Full Rules & Safeguards</h3>
        <div style="display:flex;gap:16px;flex-wrap:wrap;">
          <div style="flex:1;min-width:280px;">
            <h4 style="color:#7c4dff;margin-bottom:8px;">Training Mode Rules</h4>
            <p id="training-rules-desc" style="font-size:0.9em;color:#aaa;margin-bottom:8px;"></p>
            <ul id="training-rules-notes" style="list-style:disc;padding-left:20px;margin:0;font-size:0.85em;color:#ccc;"></ul>
          </div>
          <div style="flex:1;min-width:280px;">
            <h4 style="color:#00bcd4;margin-bottom:8px;">Live Mode Rules</h4>
            <p id="live-rules-desc" style="font-size:0.9em;color:#aaa;margin-bottom:8px;"></p>
            <ul id="live-rules-notes" style="list-style:disc;padding-left:20px;margin:0;font-size:0.85em;color:#ccc;"></ul>
          </div>
        </div>
        <div style="margin-top:16px;">
          <h4 style="color:#4caf50;margin-bottom:8px;">Active Safeguards</h4>
          <div id="safeguards-full" style="display:flex;flex-wrap:wrap;gap:8px;"></div>
        </div>
      </div>
    </div>

    <div class="section">
      <h2>Latest Decision</h2>
      <div class="decision-card" id="latest-decision">
        <h3 id="decision-time">Waiting for first decision...</h3>
        <div class="action" id="decision-action">--</div>
        <div class="details" id="decision-details">--</div>
        <div class="reasoning" id="decision-reasoning" style="font-size:0.9em;max-height:60px;overflow:hidden;text-overflow:ellipsis;">Agent is starting up...</div>
      </div>
      <div style="margin-top:12px;">
        <h4 style="font-size:0.9rem;margin-bottom:8px;color:#888;">Recent Decisions</h4>
        <div id="mini-timeline" style="display:flex;flex-wrap:wrap;gap:6px;"></div>
      </div>
    </div>

    <div class="section">
      <h2>Bot Positions</h2>
      <div id="positions-pnl-summary" style="font-size:0.9rem;color:#888;margin-bottom:0.5rem;"></div>
      
      <div class="subsection">
        <h3 style="margin:0 0 8px 0;color:#ff9800;">Test Positions (DRY_RUN)</h3>
        <div style="overflow-x: auto; max-height: 220px; overflow-y: auto;">
          <table class="steps-table">
            <thead>
              <tr>
                <th>Underlying</th>
                <th>Strategy</th>
                <th>Symbol</th>
                <th>Qty</th>
                <th>Entry</th>
                <th>Mark</th>
                <th>Unreal. PnL</th>
                <th>Unreal. %</th>
                <th>DTE</th>
                <th>Rolls</th>
                <th>Entry Mode</th>
              </tr>
            </thead>
            <tbody id="dashboard-test-positions-body">
              <tr><td colspan="11" style="text-align:center;color:#666;">Loading...</td></tr>
            </tbody>
          </table>
        </div>
      </div>

      <div class="subsection" style="margin-top:16px;">
        <h3 style="margin:0 0 8px 0;color:#4caf50;">Live Positions</h3>
        <div style="overflow-x: auto; max-height: 220px; overflow-y: auto;">
          <table class="steps-table">
            <thead>
              <tr>
                <th>Underlying</th>
                <th>Strategy</th>
                <th>Symbol</th>
                <th>Qty</th>
                <th>Entry</th>
                <th>Mark</th>
                <th>Unreal. PnL</th>
                <th>Unreal. %</th>
                <th>DTE</th>
                <th>Rolls</th>
                <th>Entry Mode</th>
              </tr>
            </thead>
            <tbody id="dashboard-live-positions-body">
              <tr><td colspan="11" style="text-align:center;color:#666;">Loading...</td></tr>
            </tbody>
          </table>
        </div>
      </div>
      
      <details style="margin-top:16px;">
        <summary style="cursor:pointer;color:#888;font-size:0.9em;">Closed Chains</summary>
        <div style="overflow-x: auto; max-height: 260px; overflow-y: auto; margin-top:8px;">
          <table class="steps-table">
            <thead>
              <tr>
                <th>Closed At</th>
                <th>Underlying</th>
                <th>Type</th>
                <th>Strategy</th>
                <th>Symbol</th>
                <th>Legs</th>
                <th>Rolls</th>
                <th>Real. PnL</th>
                <th>Real. %</th>
                <th>Max DD %</th>
                <th>Mode</th>
              </tr>
            </thead>
            <tbody id="live-closed-positions-body">
              <tr><td colspan="11" style="text-align:center;color:#666;">No closed chains yet</td></tr>
            </tbody>
          </table>
        </div>
      </details>
    </div>

    <div class="section">
      <h2>Recent Decisions</h2>
      <div style="overflow-x: auto;">
        <table class="decisions-table">
          <thead>
            <tr>
              <th>Time</th>
              <th>Source</th>
              <th>Proposed</th>
              <th>Final</th>
              <th>Risk</th>
              <th>Execution</th>
            </tr>
          </thead>
          <tbody id="decisions-tbody">
            <tr><td colspan="6" style="text-align:center;color:#666;">Loading decisions...</td></tr>
          </tbody>
        </table>
      </div>
    </div>

    <div class="section">
      <details>
        <summary style="cursor: pointer; font-weight: 600;">Full Status JSON</summary>
        <pre id="status-box">Loading...</pre>
      </details>
    </div>
  </div>

  <!-- UNIFIED BACKTESTING TAB -->
  <div id="tab-backtesting" class="tab-content">
    <!-- Collapsible Card: Backtesting Lab -->
    <div class="card collapsible-section" style="margin-bottom: 1.5rem;">
      <div class="card-header" style="display:flex;justify-content:space-between;align-items:center;padding:12px 16px;background:#2a2a2a;border-radius:8px 8px 0 0;cursor:pointer;" onclick="toggleBacktestSection('lab')">
        <h3 style="margin:0;font-size:1.1rem;color:#4fc3f7;">Backtesting Lab - Configure &amp; Run</h3>
        <span id="lab-toggle-icon" style="font-size:1.2rem;color:#888;">&#9660;</span>
      </div>
      <div id="backtest-lab-body" class="card-body collapsible-body" style="display:block;padding:16px;background:#1e1e1e;border-radius:0 0 8px 8px;">
        <div class="section" style="margin:0;">
          <h4 style="margin-top:0;">Backtest Configuration</h4>
      <div class="form-row">
        <div class="form-group">
          <label>Underlying</label>
          <select id="bt-underlying">
            <option value="BTC">BTC</option>
            <option value="ETH">ETH</option>
          </select>
        </div>
        <div class="form-group">
          <label>Start Date</label>
          <input type="date" id="bt-start" value="2024-09-01">
        </div>
        <div class="form-group">
          <label>End Date</label>
          <input type="date" id="bt-end" value="2024-11-01">
        </div>
      </div>
      <div class="form-row">
        <div class="form-group">
          <label>Timeframe</label>
          <select id="bt-timeframe">
            <option value="1h">1 hour</option>
            <option value="4h">4 hours</option>
            <option value="1d" selected>1 day</option>
          </select>
        </div>
        <div class="form-group">
          <label>Decision Interval</label>
          <select id="bt-interval">
            <option value="1">Every bar</option>
            <option value="4">Every 4 bars</option>
            <option value="24" selected>Daily</option>
          </select>
        </div>
        <div class="form-group">
          <label>Exit Style</label>
          <select id="bt-exit-style">
            <option value="hold_to_expiry">Hold to Expiry</option>
            <option value="tp_and_roll">Take Profit &amp; Roll</option>
            <option value="both">Both (compare)</option>
          </select>
        </div>
      </div>
      <div class="form-row">
        <div class="form-group">
          <label>DTE Range (days)</label>
          <div style="display:flex;gap:0.5rem;align-items:center;">
            <input type="number" id="bt-min-dte" value="3" min="1" max="365" style="width:70px;">
            <span>to</span>
            <input type="number" id="bt-max-dte" value="21" min="1" max="365" style="width:70px;">
          </div>
        </div>
        <div class="form-group">
          <label>Delta Range</label>
          <div style="display:flex;gap:0.5rem;align-items:center;">
            <input type="number" id="bt-delta-min" value="0.15" min="0.05" max="0.9" step="0.05" style="width:70px;">
            <span>to</span>
            <input type="number" id="bt-delta-max" value="0.35" min="0.05" max="0.9" step="0.05" style="width:70px;">
          </div>
        </div>
      </div>
      <div class="form-row">
        <div class="form-group">
          <label>Option Type</label>
          <select id="bt-margin-type">
            <option value="inverse" selected>Inverse (coin-settled, pre-2025)</option>
            <option value="linear">Linear (USDC-settled, Aug 2025+)</option>
          </select>
        </div>
        <div class="form-group">
          <label>Settlement Currency</label>
          <select id="bt-settlement-ccy">
            <option value="ANY" selected>Any</option>
            <option value="USDC">USDC only</option>
            <option value="BTC">BTC only</option>
            <option value="ETH">ETH only</option>
          </select>
        </div>
      </div>
      <div class="form-row">
        <div class="form-group">
          <label>Target DTE</label>
          <input type="number" id="bt-dte" value="7" min="1" max="90">
        </div>
        <div class="form-group">
          <label>Target Delta</label>
          <input type="number" id="bt-delta" value="0.25" min="0.05" max="0.5" step="0.05">
        </div>
      </div>
      <div class="form-row">
        <div class="form-group" style="flex:2;">
          <label>Synthetic Mode</label>
          <select id="bt-synthetic-mode" onchange="updateSyntheticModeDescription()">
            <option value="pure_synthetic" selected>Pure Synthetic (RV-based)</option>
            <option value="live_iv_synthetic">Live IV, Synthetic Grid</option>
            <option value="live_chain">Live Chain + Live IV</option>
          </select>
          <small id="bt-synthetic-mode-desc" style="display:block;margin-top:4px;color:#888;font-size:0.8rem;">Uses realized volatility with multiplier to price synthetic options on a generated strike grid.</small>
        </div>
      </div>
      <div class="form-row">
        <div class="form-group" style="flex:2;">
          <label>Selector / Strategy</label>
          <select id="bt-selector-name">
            <option value="generic_covered_call" selected>Generic - Covered Call Agent</option>
            <option value="greg_vrp_harvester">GregBot - VRP Harvester</option>
          </select>
        </div>
      </div>
      <div style="display:flex;gap:0.5rem;">
        <button id="bt-start-stop-btn" onclick="startBacktest()">Start Backtest</button>
        <button id="bt-pause-resume-btn" class="secondary" onclick="togglePause()" style="display:none;">Pause</button>
      </div>
    </div>

    <div class="section">
      <h2>Live Progress</h2>
      <div class="bt-status-header">
        <span>Status:</span>
        <span class="bt-status-indicator bt-status-idle" id="bt-status-text">IDLE</span>
        <span id="bt-phase-label" style="display:none;">Phase:</span>
        <span id="bt-current-phase" style="display:none;font-weight:600;"></span>
        <span>Decisions:</span>
        <span id="bt-decisions">0 / 0</span>
      </div>
      <div id="bt-error" class="error-text" style="display:none;"></div>
      <div class="progress-bar">
        <div class="progress-bar-inner" id="bt-progress-inner" style="width:0%">0%</div>
      </div>
      
      <div id="bt-metrics-live" style="margin-top:1rem;display:none;">
        <h3>Results</h3>
        <pre id="bt-metrics-json" style="background:#f1f3f4;padding:1rem;border-radius:6px;font-size:0.85rem;"></pre>
      </div>
      
      <h3>Recent Steps</h3>
      <div style="overflow-x: auto; max-height: 300px; overflow-y: auto;">
        <table class="steps-table">
          <thead>
            <tr>
              <th>Time</th>
              <th>Candidates</th>
              <th>Best Score</th>
              <th>Traded?</th>
              <th>Exit Style</th>
            </tr>
          </thead>
          <tbody id="bt-steps-body">
            <tr><td colspan="5" style="text-align:center;color:#666;">No steps yet</td></tr>
          </tbody>
        </table>
      </div>
      
      <h3>Recent Chains (TP &amp; Roll)</h3>
      <div style="overflow-x: auto; max-height: 300px; overflow-y: auto;">
        <table class="steps-table">
          <thead>
            <tr>
              <th>Decision Time</th>
              <th>Underlying</th>
              <th>Legs</th>
              <th>Rolls</th>
              <th>Total PnL</th>
              <th>Max DD%</th>
              <th>Details</th>
            </tr>
          </thead>
          <tbody id="bt-chains-body">
            <tr><td colspan="7" style="text-align:center;color:#666;">No chains yet</td></tr>
          </tbody>
        </table>
      </div>
    </div>
    
    <!-- Chain Details Modal -->
    <div id="chain-modal" class="modal" style="display:none;">
      <div class="modal-content">
        <span class="modal-close" onclick="closeChainModal()">&times;</span>
        <h2>Chain Details</h2>
        <div id="chain-modal-summary"></div>
        <h3>Legs</h3>
        <div style="overflow-x:auto;">
          <table class="decisions-table">
            <thead>
              <tr>
                <th>#</th>
                <th>Open</th>
                <th>Close</th>
                <th>Strike</th>
                <th>DTE</th>
                <th>PnL</th>
                <th>Trigger</th>
              </tr>
            </thead>
            <tbody id="chain-legs-body"></tbody>
          </table>
        </div>
      </div>
    </div>

    <div class="section" id="backtest-results" style="display:none;">
      <h2>Backtest Results - TradingView Summary</h2>
      
      <!-- TradingView-style Summary Panel -->
      <div class="tv-summary-panel">
        <div class="tv-summary-header">
          <div class="tv-main-stat" id="tv-net-profit">
            <span class="tv-value">$0</span>
            <span class="tv-pct">(0.00%)</span>
          </div>
          <div class="tv-secondary-stats">
            <div class="tv-stat">
              <span class="tv-label">vs HODL</span>
              <span class="tv-value" id="tv-vs-hodl">$0</span>
            </div>
            <div class="tv-stat">
              <span class="tv-label">HODL Return</span>
              <span class="tv-value" id="tv-hodl-return">0%</span>
            </div>
          </div>
        </div>
        
        <div class="tv-metrics-grid">
          <div class="tv-metric-group">
            <h4>Profit &amp; Loss</h4>
            <div class="tv-metric-row">
              <span class="tv-metric-label">Gross Profit</span>
              <span class="tv-metric-value positive" id="tv-gross-profit">$0</span>
            </div>
            <div class="tv-metric-row">
              <span class="tv-metric-label">Gross Loss</span>
              <span class="tv-metric-value negative" id="tv-gross-loss">$0</span>
            </div>
            <div class="tv-metric-row">
              <span class="tv-metric-label">Profit Factor</span>
              <span class="tv-metric-value" id="tv-profit-factor">0</span>
            </div>
          </div>
          
          <div class="tv-metric-group">
            <h4>Trade Statistics</h4>
            <div class="tv-metric-row">
              <span class="tv-metric-label">Total Trades</span>
              <span class="tv-metric-value" id="tv-num-trades">0</span>
            </div>
            <div class="tv-metric-row">
              <span class="tv-metric-label">Win Rate</span>
              <span class="tv-metric-value" id="tv-win-rate">0%</span>
            </div>
            <div class="tv-metric-row">
              <span class="tv-metric-label">Avg Trade</span>
              <span class="tv-metric-value" id="tv-avg-trade">$0</span>
            </div>
          </div>
          
          <div class="tv-metric-group">
            <h4>Avg Win/Loss</h4>
            <div class="tv-metric-row">
              <span class="tv-metric-label">Avg Winner</span>
              <span class="tv-metric-value positive" id="tv-avg-winner">$0</span>
            </div>
            <div class="tv-metric-row">
              <span class="tv-metric-label">Avg Loser</span>
              <span class="tv-metric-value negative" id="tv-avg-loser">$0</span>
            </div>
          </div>
          
          <div class="tv-metric-group">
            <h4>Risk Metrics</h4>
            <div class="tv-metric-row">
              <span class="tv-metric-label">Max Drawdown</span>
              <span class="tv-metric-value negative" id="tv-max-dd">0%</span>
            </div>
            <div class="tv-metric-row">
              <span class="tv-metric-label">Sharpe Ratio</span>
              <span class="tv-metric-value" id="tv-sharpe">0</span>
            </div>
            <div class="tv-metric-row">
              <span class="tv-metric-label">Sortino Ratio</span>
              <span class="tv-metric-value" id="tv-sortino">0</span>
            </div>
          </div>
        </div>
      </div>
      
      <h3>Equity Curve (Strategy vs HODL)</h3>
      <div class="chart-container" style="height: 350px;">
        <canvas id="equity-chart"></canvas>
      </div>
      <div class="chart-legend">
        <span class="legend-item"><span class="legend-color" style="background:#1565c0;"></span>Strategy</span>
        <span class="legend-item"><span class="legend-color" style="background:#ff9800;"></span>HODL</span>
      </div>
      
      <h3>Sample Trades</h3>
      <div style="overflow-x: auto;">
        <table class="decisions-table" id="trades-table">
          <thead>
            <tr>
              <th>Instrument</th>
              <th>Open</th>
              <th>Close</th>
              <th>PnL</th>
              <th>vs HODL</th>
              <th>Max DD</th>
            </tr>
          </thead>
          <tbody id="trades-tbody"></tbody>
        </table>
      </div>
      
      <div style="margin-top:1rem;">
        <button class="secondary" onclick="getInsights()">Get AI Insights</button>
      </div>
      <div class="insights-box" id="insights-box" style="display:none;"></div>
      
      <!-- Live Chain Debug Samples -->
      <div id="debug-samples-section" style="display:none; margin-top:1.5rem;">
        <h3>Live Chain Debug Samples</h3>
        <p style="color:#666;font-size:0.9rem;margin-bottom:0.5rem;">
          Compares Deribit mark prices vs engine-calculated prices. When multiplier=1.0, abs_diff_pct should be ~0.
        </p>
        <div style="overflow-x: auto;">
          <table class="decisions-table" id="debug-samples-table">
            <thead>
              <tr>
                <th>Instrument</th>
                <th>DTE</th>
                <th>Strike</th>
                <th>Deribit Mark</th>
                <th>Engine Price</th>
                <th>Diff %</th>
              </tr>
            </thead>
            <tbody id="debug-samples-tbody"></tbody>
          </table>
        </div>
      </div>
    </div>
    
    <!-- SELECTOR FREQUENCY SCAN -->
    <div class="section" style="margin-top:2rem;">
      <h2>Selector Frequency Scan (Synthetic)</h2>
      <p style="color:#666;margin-bottom:1rem;">
        Analyze how often a selector's rules (e.g. GregBot Phase 1) would allow trading in a synthetic universe.
        Threshold overrides are backtest-only and do not affect the live bot.
      </p>
      
      <div class="card" id="selector-scan-panel">
        <div class="form-row" style="flex-wrap:wrap;gap:12px;margin-bottom:12px;">
          <div class="form-group">
            <label>Selector</label>
            <select id="selector-id-select">
              <option value="greg">GregBot – VRP Harvester</option>
            </select>
          </div>
          <div class="form-group">
            <label>Underlyings</label>
            <div style="display:flex;gap:12px;">
              <label style="font-weight:normal;"><input type="checkbox" id="selector-underlying-btc" checked> BTC</label>
              <label style="font-weight:normal;"><input type="checkbox" id="selector-underlying-eth" checked> ETH</label>
            </div>
          </div>
          <div class="form-group">
            <label>Horizon (days)</label>
            <input type="number" id="selector-horizon-input" value="365" style="width:80px;">
          </div>
          <div class="form-group">
            <label>Decision interval (days)</label>
            <input type="number" id="selector-interval-input" value="1" step="0.25" style="width:80px;">
          </div>
        </div>
        
        <details style="margin-bottom:12px;">
          <summary style="cursor:pointer;font-weight:600;color:#4fc3f7;">Threshold Overrides (optional)</summary>
          <p style="color:#888;font-size:0.9rem;margin:8px 0;">Overrides apply only to this scan; they do not affect live trading.</p>
          <div class="form-row" style="flex-wrap:wrap;gap:12px;">
            <div class="form-group">
              <label>Min VRP 30d</label>
              <input type="number" id="selector-vrp-min-input" step="0.5" placeholder="default" style="width:80px;">
            </div>
            <div class="form-group">
              <label>Max Chop Factor 7d</label>
              <input type="number" id="selector-chop-max-input" step="0.05" placeholder="default" style="width:80px;">
            </div>
            <div class="form-group">
              <label>Max ADX 14d</label>
              <input type="number" id="selector-adx-max-input" step="1" placeholder="default" style="width:80px;">
            </div>
            <div class="form-group">
              <label>Min RSI 14d</label>
              <input type="number" id="selector-rsi-min-input" step="1" placeholder="default" style="width:80px;">
            </div>
            <div class="form-group">
              <label>Max RSI 14d</label>
              <input type="number" id="selector-rsi-max-input" step="1" placeholder="default" style="width:80px;">
            </div>
            <div class="form-group">
              <label>Min IV Rank 6m</label>
              <input type="number" id="selector-ivrank-min-input" step="0.05" placeholder="default" style="width:80px;">
            </div>
          </div>
        </details>
        
        <button id="selector-scan-run-btn" style="background:#2196f3;color:#fff;border:none;padding:8px 16px;border-radius:4px;cursor:pointer;">Run Selector Scan</button>
        <div id="selector-scan-status" aria-live="polite" style="margin-top:8px;font-size:0.9rem;"></div>
        
        <table id="selector-scan-results-table" class="steps-table" style="margin-top:1rem;display:none;">
          <thead>
            <tr>
              <th>Strategy</th>
              <th>Underlying</th>
              <th>Pass Count</th>
              <th>Total Steps</th>
              <th>Pass %</th>
            </tr>
          </thead>
          <tbody id="selector-scan-results-body"></tbody>
        </table>
      </div>
    </div>
    
    <!-- INTRADAY DATA & SCRAPER STATUS -->
    <div class="section" style="margin-top:2rem;">
      <h2>Intraday Data & Scraper Status (Deribit)</h2>
      <p style="color:#666;margin-bottom:1rem;">
        Monitor the Deribit intraday data harvester. This shows how much data has been collected and whether the scraper is actively running.
      </p>
      
      <div class="card" id="intraday-scraper-panel" style="background:#fff;">
        <div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(180px, 1fr));gap:12px;margin-bottom:12px;">
          <div style="background:#f5f5f5;padding:10px;border-radius:6px;">
            <div style="color:#666;font-size:0.85rem;">Source</div>
            <div id="scraper-source" style="font-weight:600;color:#333;">Loading...</div>
          </div>
          <div style="background:#f5f5f5;padding:10px;border-radius:6px;">
            <div style="color:#666;font-size:0.85rem;">Backend</div>
            <div id="scraper-backend" style="font-weight:600;color:#333;">Loading...</div>
          </div>
          <div style="background:#f5f5f5;padding:10px;border-radius:6px;">
            <div style="color:#666;font-size:0.85rem;">Rows Total</div>
            <div id="scraper-rows" style="font-weight:600;color:#333;">Loading...</div>
          </div>
          <div style="background:#f5f5f5;padding:10px;border-radius:6px;">
            <div style="color:#666;font-size:0.85rem;">Days Covered</div>
            <div id="scraper-days" style="font-weight:600;color:#333;">Loading...</div>
          </div>
          <div style="background:#f5f5f5;padding:10px;border-radius:6px;">
            <div style="color:#666;font-size:0.85rem;">First / Last Timestamp</div>
            <div id="scraper-range" style="font-weight:600;font-size:0.9rem;color:#333;">Loading...</div>
          </div>
          <div style="background:#f5f5f5;padding:10px;border-radius:6px;">
            <div style="color:#666;font-size:0.85rem;">Approx DB Size</div>
            <div id="scraper-size" style="font-weight:600;color:#333;">Loading...</div>
          </div>
          <div style="background:#f5f5f5;padding:10px;border-radius:6px;">
            <div style="color:#666;font-size:0.85rem;">Target Update Interval</div>
            <div id="scraper-interval" style="font-weight:600;color:#333;">Loading...</div>
          </div>
          <div style="background:#f5f5f5;padding:10px;border-radius:6px;">
            <div style="color:#666;font-size:0.85rem;">Status</div>
            <div id="scraper-running" style="font-weight:600;">Unknown</div>
          </div>
        </div>
        
        <button id="refresh-scraper-status-btn" style="background:#2196f3;color:#fff;border:none;padding:8px 16px;border-radius:4px;cursor:pointer;">Refresh Scraper Status</button>
        <span id="scraper-status-message" aria-live="polite" style="margin-left:12px;font-size:0.9rem;"></span>
      </div>
    </div>
    
    <!-- SELECTOR HEATMAP -->
    <div class="section" style="margin-top:2rem;">
      <h2>Selector Heatmap (Synthetic)</h2>
      <p style="color:#666;margin-bottom:1rem;">
        Explore how strictness of two threshold metrics affects trade frequency in the synthetic universe.
        Choose a strategy, two metrics to sweep, and generate a 2D pass% heatmap.
      </p>
      
      <div class="card" id="selector-heatmap-panel">
        <div class="form-row" style="flex-wrap:wrap;gap:12px;margin-bottom:12px;">
          <div class="form-group">
            <label>Selector</label>
            <select id="heatmap-selector-id-select">
              <option value="greg">GregBot – VRP Harvester</option>
            </select>
          </div>
          <div class="form-group">
            <label>Underlying</label>
            <select id="heatmap-underlying-select">
              <option value="BTC">BTC</option>
              <option value="ETH">ETH</option>
            </select>
          </div>
          <div class="form-group">
            <label>Strategy</label>
            <select id="heatmap-strategy-select">
              <option value="STRATEGY_A_STRADDLE">ATM Straddle</option>
              <option value="STRATEGY_A_STRANGLE">OTM Strangle</option>
              <option value="STRATEGY_B_CALENDAR">Calendar Spread</option>
              <option value="STRATEGY_C_SHORT_PUT">Short Put</option>
              <option value="STRATEGY_D_IRON_BUTTERFLY">Iron Butterfly</option>
              <option value="STRATEGY_F_BULL_PUT_SPREAD">Bull Put Spread</option>
              <option value="STRATEGY_F_BEAR_CALL_SPREAD">Bear Call Spread</option>
            </select>
          </div>
          <div class="form-group">
            <label>Horizon (days)</label>
            <input type="number" id="heatmap-horizon-input" value="180" style="width:80px;">
          </div>
          <div class="form-group">
            <label>Decision interval</label>
            <input type="number" id="heatmap-interval-input" value="1" step="0.25" style="width:80px;">
          </div>
        </div>
        
        <details style="margin-bottom:12px;" open>
          <summary style="cursor:pointer;font-weight:600;color:#4fc3f7;">Axes & Grids</summary>
          <div class="form-row" style="flex-wrap:wrap;gap:12px;margin-top:8px;">
            <div class="form-group">
              <label>X Metric</label>
              <select id="heatmap-metric-x-select">
                <option value="vrp_30d_min">Min VRP 30d</option>
                <option value="adx_14d_max">Max ADX 14d</option>
                <option value="chop_factor_7d_max">Max Chop Factor 7d</option>
                <option value="rsi_14d_min">Min RSI 14d</option>
                <option value="rsi_14d_max">Max RSI 14d</option>
                <option value="iv_rank_6m_min">Min IV Rank 6m</option>
              </select>
            </div>
            <div class="form-group">
              <label>X Start</label>
              <input type="number" id="heatmap-x-start-input" value="5" step="1" style="width:60px;">
            </div>
            <div class="form-group">
              <label>X Step</label>
              <input type="number" id="heatmap-x-step-input" value="5" step="1" style="width:60px;">
            </div>
            <div class="form-group">
              <label>X Points</label>
              <input type="number" id="heatmap-x-count-input" value="5" style="width:60px;">
            </div>
          </div>
          <div class="form-row" style="flex-wrap:wrap;gap:12px;margin-top:8px;">
            <div class="form-group">
              <label>Y Metric</label>
              <select id="heatmap-metric-y-select">
                <option value="adx_14d_max">Max ADX 14d</option>
                <option value="vrp_30d_min">Min VRP 30d</option>
                <option value="chop_factor_7d_max">Max Chop Factor 7d</option>
                <option value="rsi_14d_min">Min RSI 14d</option>
                <option value="rsi_14d_max">Max RSI 14d</option>
                <option value="iv_rank_6m_min">Min IV Rank 6m</option>
              </select>
            </div>
            <div class="form-group">
              <label>Y Start</label>
              <input type="number" id="heatmap-y-start-input" value="15" step="1" style="width:60px;">
            </div>
            <div class="form-group">
              <label>Y Step</label>
              <input type="number" id="heatmap-y-step-input" value="5" step="1" style="width:60px;">
            </div>
            <div class="form-group">
              <label>Y Points</label>
              <input type="number" id="heatmap-y-count-input" value="5" style="width:60px;">
            </div>
          </div>
        </details>
        
        <button id="selector-heatmap-run-btn" style="background:#9c27b0;color:#fff;border:none;padding:8px 16px;border-radius:4px;cursor:pointer;">Run Heatmap</button>
        <div id="selector-heatmap-status" aria-live="polite" style="margin-top:8px;font-size:0.9rem;"></div>
        
        <div id="selector-heatmap-container" style="margin-top:1rem;display:none;">
          <table id="selector-heatmap-table" class="steps-table">
            <thead id="selector-heatmap-thead"></thead>
            <tbody id="selector-heatmap-tbody"></tbody>
          </table>
          <p style="font-size:0.85rem;color:#888;margin-top:8px;">
            Cell color reflects pass% (darker green = more frequent). All runs use the synthetic universe (backtest-only).
          </p>
        </div>
      </div>
    </div>
    
    <!-- ENVIRONMENT HEATMAP -->
    <div class="section" style="margin-top:2rem;">
      <h2>Environment Heatmap (Synthetic)</h2>
      <p style="color:#666;margin-bottom:1rem;">
        Explore where the market actually spends time in the synthetic universe, for any pair of metrics (no selector / strategy applied).
        This counts opportunities in the environment, not trades or PnL.
      </p>
      
      <div class="card" id="env-heatmap-panel">
        <div class="form-row" style="flex-wrap:wrap;gap:12px;margin-bottom:12px;">
          <div class="form-group">
            <label>Underlying</label>
            <select id="env-underlying-select">
              <option value="BTC">BTC</option>
              <option value="ETH">ETH</option>
            </select>
          </div>
          <div class="form-group">
            <label>Horizon (days)</label>
            <input type="number" id="env-horizon-input" value="365" min="1" max="3650" style="width:80px;">
          </div>
          <div class="form-group">
            <label>Decision interval (days)</label>
            <input type="number" id="env-decision-interval-input" value="1" min="1" max="30" style="width:80px;">
          </div>
        </div>
        
        <details style="margin-bottom:12px;" open>
          <summary style="cursor:pointer;font-weight:600;color:#4fc3f7;">Axes & Grids</summary>
          <div class="form-row" style="flex-wrap:wrap;gap:12px;margin-top:8px;">
            <div class="form-group">
              <label>X Metric</label>
              <select id="env-x-metric-select">
                <option value="vrp_30d">VRP 30d</option>
                <option value="adx_14d">ADX 14d</option>
                <option value="chop_factor_7d">Chop Factor 7d</option>
                <option value="iv_rank_6m">IV Rank 6m</option>
                <option value="term_structure_spread">Term Spread</option>
                <option value="skew_25d">Skew 25d</option>
                <option value="rsi_14d">RSI 14d</option>
                <option value="price_vs_ma200">Price vs MA200</option>
              </select>
            </div>
            <div class="form-group">
              <label>X Start</label>
              <input type="number" id="env-x-start-input" value="0" step="1" style="width:60px;">
            </div>
            <div class="form-group">
              <label>X Step</label>
              <input type="number" id="env-x-step-input" value="5" step="1" style="width:60px;">
            </div>
            <div class="form-group">
              <label>X Points</label>
              <input type="number" id="env-x-points-input" value="5" min="2" max="20" style="width:60px;">
            </div>
          </div>
          <div class="form-row" style="flex-wrap:wrap;gap:12px;margin-top:8px;">
            <div class="form-group">
              <label>Y Metric</label>
              <select id="env-y-metric-select">
                <option value="adx_14d">ADX 14d</option>
                <option value="vrp_30d">VRP 30d</option>
                <option value="chop_factor_7d">Chop Factor 7d</option>
                <option value="iv_rank_6m">IV Rank 6m</option>
                <option value="term_structure_spread">Term Spread</option>
                <option value="skew_25d">Skew 25d</option>
                <option value="rsi_14d">RSI 14d</option>
                <option value="price_vs_ma200">Price vs MA200</option>
              </select>
            </div>
            <div class="form-group">
              <label>Y Start</label>
              <input type="number" id="env-y-start-input" value="15" step="1" style="width:60px;">
            </div>
            <div class="form-group">
              <label>Y Step</label>
              <input type="number" id="env-y-step-input" value="5" step="1" style="width:60px;">
            </div>
            <div class="form-group">
              <label>Y Points</label>
              <input type="number" id="env-y-points-input" value="5" min="2" max="20" style="width:60px;">
            </div>
          </div>
        </details>
        
        <button id="env-heatmap-run-btn" style="background:#00897b;color:#fff;border:none;padding:8px 16px;border-radius:4px;cursor:pointer;">Run Environment Heatmap</button>
        <div id="env-heatmap-status" aria-live="polite" style="margin-top:8px;font-size:0.9rem;"></div>
        
        <div id="env-heatmap-container" style="margin-top:1rem;display:none;">
          <table id="env-heatmap-table" class="steps-table">
            <thead id="env-heatmap-thead"></thead>
            <tbody id="env-heatmap-tbody"></tbody>
          </table>
          <p style="font-size:0.85rem;color:#888;margin-top:8px;">
            Cell value = % of decision steps where environment fell into that bucket. All runs use the synthetic universe.
          </p>
        </div>
      </div>
      
      <div class="card" id="greg-sweetspots-panel">
        <h3 style="margin-top:0;">Greg Environment Sweet Spots</h3>
        <p style="color:#666;font-size:0.9rem;margin-bottom:12px;">
          Shows regions where Greg's strategies pass AND the environment spends time.
          Click "Run Greg Sweet Spot Scan" to analyze synthetic data across all metric pairs.
        </p>
        <div style="display:flex;gap:8px;flex-wrap:wrap;">
          <button id="greg-sweetspots-run-btn" style="background:#9c27b0;color:#fff;border:none;padding:8px 16px;border-radius:4px;cursor:pointer;">Run Greg Sweet Spot Scan</button>
          <button id="greg-sweetspots-refresh-btn" style="background:#ff9800;color:#fff;border:none;padding:8px 16px;border-radius:4px;cursor:pointer;">Refresh Sweet Spots</button>
        </div>
        <div id="greg-sweetspots-status" aria-live="polite" style="margin-top:8px;font-size:0.9rem;"></div>
        <div id="greg-sweetspots-content" style="margin-top:1rem;"></div>
      </div>
    </div>

    <div class="loading" id="backtest-loading" style="display:none;">
      <div class="spinner"></div>
      <p>Running backtest... This may take a minute.</p>
    </div>
        </div>
      </div>
    </div>
    
    <!-- Collapsible Card: Backtest Runs History -->
    <div class="card collapsible-section">
      <div class="card-header" style="display:flex;justify-content:space-between;align-items:center;padding:12px 16px;background:#2a2a2a;border-radius:8px 8px 0 0;cursor:pointer;" onclick="toggleBacktestSection('runs')">
        <h3 style="margin:0;font-size:1.1rem;color:#81c784;">Backtest Runs - History</h3>
        <span id="runs-toggle-icon" style="font-size:1.2rem;color:#888;">&#9654;</span>
      </div>
      <div id="backtest-runs-body" class="card-body collapsible-body" style="display:none;padding:16px;background:#1e1e1e;border-radius:0 0 8px 8px;">
        <p style="color: #888; margin-bottom: 15px;">View all saved backtest runs with performance metrics. Each run is saved automatically when completed.</p>
        
        <div style="margin-bottom: 12px;">
          <button onclick="fetchBacktestRuns()" style="background:#2196f3;color:#fff;border:none;padding:8px 16px;border-radius:4px;cursor:pointer;">Refresh Runs</button>
        </div>
        
        <div style="overflow-x:auto;">
          <table class="steps-table" id="runs-table">
            <thead>
              <tr>
                <th>Run ID</th>
                <th>Created</th>
                <th>Underlying</th>
                <th>Date Range</th>
                <th>Status</th>
                <th>Exit Style</th>
                <th>Net PnL %</th>
                <th>Max DD %</th>
                <th>Sharpe</th>
                <th>Trades</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody id="runs-table-body">
              <tr><td colspan="11" style="text-align:center;color:#666;">Loading...</td></tr>
            </tbody>
          </table>
        </div>
        
        <!-- Run Detail Modal -->
        <div id="run-detail-modal" class="modal" style="display:none;">
          <div class="modal-content" style="max-width:900px;background:#1e1e1e;color:#fff;max-height:90vh;overflow-y:auto;">
            <span class="modal-close" onclick="closeRunDetailModal()" style="color:#888;">&times;</span>
            <h3 id="run-detail-title" style="margin-top:0;">Backtest Run Details</h3>
            
            <div id="run-detail-config" style="margin-bottom:16px;padding:12px;background:#2a2a2a;border-radius:6px;">
              <h4 style="margin:0 0 8px 0;color:#4fc3f7;">Configuration</h4>
              <div id="run-config-grid" style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px;font-size:0.85em;"></div>
            </div>
            
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px;">
              <div id="run-metrics-hte" style="padding:12px;background:#2a2a2a;border-radius:6px;">
                <h4 style="margin:0 0 8px 0;color:#81c784;">Hold to Expiry</h4>
                <div id="run-metrics-hte-content"></div>
              </div>
              <div id="run-metrics-tpr" style="padding:12px;background:#2a2a2a;border-radius:6px;">
                <h4 style="margin:0 0 8px 0;color:#ffb74d;">TP & Roll</h4>
                <div id="run-metrics-tpr-content"></div>
              </div>
            </div>
            
            <div style="margin-bottom:16px;">
              <h4 style="margin:0 0 8px 0;color:#ce93d8;">Recent Chains (TP & Roll)</h4>
              <div style="overflow-x:auto;max-height:200px;overflow-y:auto;">
                <table class="steps-table">
                  <thead>
                    <tr>
                      <th>Decision Time</th>
                      <th>Underlying</th>
                      <th>Legs</th>
                      <th>Rolls</th>
                      <th>Total PnL</th>
                      <th>Max DD %</th>
                    </tr>
                  </thead>
                  <tbody id="run-chains-body">
                    <tr><td colspan="6" style="text-align:center;color:#666;">No chains</td></tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- CALIBRATION TAB -->
  <div id="tab-calibration" class="tab-content">
    <div class="section">
      <h2>Calibration vs Deribit</h2>
      <p style="color:#666;margin-bottom:1rem;">Compare synthetic Black-Scholes prices with live Deribit mark prices for near-dated calls.</p>
      
      <div class="card">
        <div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-bottom:8px;">
          <label>
            Underlying:
            <select id="calib-underlying" style="padding:0.3rem;">
              <option value="BTC">BTC</option>
              <option value="ETH">ETH</option>
            </select>
          </label>
          <label>
            DTE Range:
            <input id="calib-min-dte" type="number" value="3" style="width:60px;padding:0.3rem;"> -
            <input id="calib-max-dte" type="number" value="10" style="width:60px;padding:0.3rem;"> days
          </label>
          <label>
            IV Multiplier:
            <input id="calib-iv-mult" type="number" step="0.1" value="1.0" style="width:70px;padding:0.3rem;">
          </label>
          <button id="calib-run-btn" onclick="runCalibration()">Run Calibration</button>
          <button id="calib-use-latest-btn" onclick="useLatestCalibration()" style="background:#9c27b0;color:#fff;">Use Latest Recommended</button>
        </div>
        
        <div id="calib-override-status" style="font-size:0.85rem;margin-bottom:8px;color:#9c27b0;display:none;"></div>
        
        <div id="calib-summary" style="font-size:0.9rem;margin-bottom:8px;color:#555;">
          Click "Run Calibration" to compare synthetic vs Deribit prices.
        </div>
        
        <div style="overflow-x:auto;max-height:320px;overflow-y:auto;">
          <table class="steps-table">
            <thead>
              <tr>
                <th>Instrument</th>
                <th>DTE</th>
                <th>Strike</th>
                <th>Mark Price</th>
                <th>Synthetic</th>
                <th>Diff %</th>
              </tr>
            </thead>
            <tbody id="calib-rows-body">
              <tr><td colspan="6" style="text-align:center;color:#666;">No data</td></tr>
            </tbody>
          </table>
        </div>
      </div>
      
      <!-- Calibration Coverage Panel -->
      <div class="card" style="margin-top:1rem;border-left:4px solid #9c27b0;">
        <h3 style="margin-top:0;color:#9c27b0;">Calibration Coverage</h3>
        <div id="calib-coverage-explanation" style="background:#f3e5f5;padding:12px;border-radius:6px;margin-bottom:16px;font-size:0.9rem;color:#555;">
          Calibration measures how well synthetic prices match market prices across option types (calls, puts) and expiry buckets (weekly, monthly, quarterly).
        </div>
        
        <!-- Option Types Coverage -->
        <div style="margin-bottom:16px;">
          <h4 style="margin:0 0 8px 0;color:#7b1fa2;">Option Types Used</h4>
          <div id="calib-option-types-status" style="font-size:0.9rem;color:#333;margin-bottom:8px;">
            <span style="background:#e1bee7;padding:4px 10px;border-radius:12px;font-size:0.85rem;" id="calib-types-badge">Calls only</span>
          </div>
        </div>
        
        <!-- Calls vs Puts Metrics Table -->
        <div id="calib-by-type-section" style="margin-bottom:16px;">
          <h4 style="margin:0 0 8px 0;color:#7b1fa2;">Metrics by Option Type</h4>
          <div style="overflow-x:auto;">
            <table class="steps-table" style="font-size:0.85rem;">
              <thead>
                <tr>
                  <th>Option Type</th>
                  <th>Count</th>
                  <th>MAE %</th>
                  <th>Bias %</th>
                  <th>MAE Vol Pts</th>
                  <th>Vega-Weighted MAE %</th>
                </tr>
              </thead>
              <tbody id="calib-by-type-body">
                <tr><td colspan="6" style="text-align:center;color:#666;">Run calibration to see metrics</td></tr>
              </tbody>
            </table>
          </div>
        </div>
        
        <!-- Term Buckets Table -->
        <div id="calib-term-buckets-section">
          <h4 style="margin:0 0 8px 0;color:#7b1fa2;">Term Structure (DTE Buckets)</h4>
          <div style="overflow-x:auto;max-height:320px;overflow-y:auto;">
            <table class="steps-table" style="font-size:0.85rem;">
              <thead>
                <tr>
                  <th>Band</th>
                  <th>DTE Range</th>
                  <th>Option Type</th>
                  <th>Count</th>
                  <th>MAE %</th>
                  <th>Vega-Wtd MAE %</th>
                  <th>Bias %</th>
                  <th>Rec. IV Mult</th>
                </tr>
              </thead>
              <tbody id="calib-term-buckets-body">
                <tr><td colspan="8" style="text-align:center;color:#666;">Run calibration to see term structure</td></tr>
              </tbody>
            </table>
          </div>
        </div>
        
        <!-- Skew Fit Panel -->
        <div id="calib-skew-fit-section" style="margin-top:16px;display:none;">
          <h4 style="margin:0 0 8px 0;color:#7b1fa2;">Skew Fit Analysis</h4>
          <div style="background:#f3e5f5;padding:12px;border-radius:6px;margin-bottom:12px;font-size:0.9rem;color:#555;">
            Compares the fitted skew anchor ratios (OTM IV / ATM IV) against the currently configured ratios.
            A positive diff means the market has higher OTM IV than our model assumes.
          </div>
          <div style="overflow-x:auto;">
            <table class="steps-table" style="font-size:0.85rem;">
              <thead>
                <tr>
                  <th>Delta Bucket</th>
                  <th>Current Ratio</th>
                  <th>Recommended Ratio</th>
                  <th>Diff</th>
                </tr>
              </thead>
              <tbody id="calib-skew-fit-body">
              </tbody>
            </table>
          </div>
          <div id="skew-fit-summary" style="margin-top:10px;font-size:0.9rem;"></div>
        </div>
      </div>
      
      <!-- Data Health & Reproducibility Panel -->
      <div class="card" style="margin-top:1rem;border-left:4px solid #00bcd4;">
        <h3 style="margin-top:0;color:#00bcd4;">Data Health & Reproducibility</h3>
        
        <div style="background:#e0f7fa;padding:12px;border-radius:6px;margin-bottom:16px;font-size:0.9rem;color:#555;">
          Historical calibrations use harvested Deribit data from the specified time window.
          For each run, we record the dataset path, time period, configuration, and data-quality checks so that the results are reproducible and you can see whether they are based on reliable data.
        </div>
        
        <!-- Data Health Summary -->
        <div style="margin-bottom:16px;">
          <h4 style="margin:0 0 8px 0;color:#0097a7;">Harvested Data Health</h4>
          <div id="data-health-status" style="display:flex;align-items:center;gap:12px;margin-bottom:12px;">
            <span id="data-health-badge" style="background:#c8e6c9;color:#2e7d32;padding:6px 14px;border-radius:16px;font-weight:600;font-size:0.9rem;">OK</span>
            <span id="data-health-summary" style="font-size:0.9rem;color:#333;">No calibration run yet</span>
          </div>
          
          <div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(140px, 1fr));gap:12px;margin-bottom:12px;">
            <div style="background:#f5f5f5;padding:10px;border-radius:6px;text-align:center;">
              <div style="font-size:0.75rem;color:#666;margin-bottom:4px;">Snapshots</div>
              <div id="dh-num-snapshots" style="font-size:1.1rem;font-weight:600;color:#333;">-</div>
            </div>
            <div style="background:#f5f5f5;padding:10px;border-radius:6px;text-align:center;">
              <div style="font-size:0.75rem;color:#666;margin-bottom:4px;">Schema Issues</div>
              <div id="dh-schema-issues" style="font-size:1.1rem;font-weight:600;color:#333;">-</div>
            </div>
            <div style="background:#f5f5f5;padding:10px;border-radius:6px;text-align:center;">
              <div style="font-size:0.75rem;color:#666;margin-bottom:4px;">Low-Quality</div>
              <div id="dh-low-quality" style="font-size:1.1rem;font-weight:600;color:#333;">-</div>
            </div>
            <div style="background:#f5f5f5;padding:10px;border-radius:6px;text-align:center;">
              <div style="font-size:0.75rem;color:#666;margin-bottom:4px;">Core Completeness</div>
              <div id="dh-completeness" style="font-size:1.1rem;font-weight:600;color:#333;">-</div>
            </div>
          </div>
          
          <div id="dh-issues-list" style="display:none;background:#fff3e0;padding:10px;border-radius:6px;margin-bottom:12px;">
            <h5 style="margin:0 0 6px 0;color:#e65100;font-size:0.85rem;">Issues Detected:</h5>
            <ul id="dh-issues-ul" style="margin:0;padding-left:20px;font-size:0.85rem;color:#bf360c;"></ul>
          </div>
        </div>
        
        <!-- Reproducibility Info -->
        <div id="repro-section" style="display:none;">
          <h4 style="margin:0 0 8px 0;color:#0097a7;">Last Historical Calibration</h4>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:12px;">
            <div style="background:#e0f2f1;padding:10px;border-radius:6px;">
              <div style="font-size:0.75rem;color:#666;margin-bottom:4px;">Run Time</div>
              <div id="repro-timestamp" style="font-size:0.9rem;color:#333;">-</div>
            </div>
            <div style="background:#e0f2f1;padding:10px;border-radius:6px;">
              <div style="font-size:0.75rem;color:#666;margin-bottom:4px;">Underlying</div>
              <div id="repro-underlying" style="font-size:0.9rem;color:#333;">-</div>
            </div>
          </div>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:12px;">
            <div style="background:#e0f2f1;padding:10px;border-radius:6px;">
              <div style="font-size:0.75rem;color:#666;margin-bottom:4px;">Harvest Period</div>
              <div id="repro-period" style="font-size:0.9rem;color:#333;">-</div>
            </div>
            <div style="background:#e0f2f1;padding:10px;border-radius:6px;">
              <div style="font-size:0.75rem;color:#666;margin-bottom:4px;">Config Hash</div>
              <div id="repro-config-hash" style="font-size:0.9rem;color:#333;font-family:monospace;">-</div>
            </div>
          </div>
          <div style="background:#e0f2f1;padding:10px;border-radius:6px;margin-bottom:12px;">
            <div style="font-size:0.75rem;color:#666;margin-bottom:4px;">Regime Model</div>
            <div id="repro-regimes" style="font-size:0.9rem;color:#333;">-</div>
          </div>
          
          <button id="view-raw-metadata-btn" onclick="toggleRawMetadata()" style="background:#00acc1;color:#fff;border:none;padding:6px 12px;border-radius:4px;cursor:pointer;font-size:0.85rem;">
            View Raw Metadata
          </button>
          <div id="raw-metadata-container" style="display:none;margin-top:12px;background:#263238;color:#b2ebf2;padding:12px;border-radius:6px;font-family:monospace;font-size:0.75rem;max-height:200px;overflow:auto;white-space:pre-wrap;"></div>
        </div>
      </div>
      
      <!-- Update Policy Panel -->
      <div class="card" style="margin-top:1rem;border-left:4px solid #4caf50;">
        <h3 style="margin-top:0;color:#4caf50;">IV Calibration Update Policy</h3>
        
        <div id="policy-explanation-box" style="background:#f5f5f5;padding:12px;border-radius:6px;margin-bottom:16px;font-size:0.9rem;color:#555;">
          Loading policy explanation...
        </div>
        
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px;">
          <!-- Current Applied Multipliers -->
          <div style="background:#e8f5e9;padding:12px;border-radius:6px;">
            <h4 style="margin:0 0 8px 0;color:#2e7d32;font-size:0.95rem;">Current Applied Multipliers</h4>
            <table class="steps-table" style="font-size:0.85rem;">
              <thead>
                <tr>
                  <th>Scope</th>
                  <th>Multiplier</th>
                  <th>Last Updated</th>
                </tr>
              </thead>
              <tbody id="current-multipliers-body">
                <tr><td colspan="3" style="text-align:center;color:#666;">Loading...</td></tr>
              </tbody>
            </table>
          </div>
          
          <!-- Latest Calibration Run -->
          <div style="background:#e3f2fd;padding:12px;border-radius:6px;">
            <h4 style="margin:0 0 8px 0;color:#1565c0;font-size:0.95rem;">Latest Calibration Run</h4>
            <div id="latest-run-info" style="font-size:0.85rem;color:#333;">
              <p style="margin:4px 0;"><strong>Source:</strong> <span id="latest-run-source">-</span></p>
              <p style="margin:4px 0;"><strong>Recommended:</strong> <span id="latest-run-recommended">-</span></p>
              <p style="margin:4px 0;"><strong>Smoothed:</strong> <span id="latest-run-smoothed">-</span></p>
              <p style="margin:4px 0;"><strong>Status:</strong> <span id="latest-run-status">-</span></p>
              <p style="margin:4px 0;"><strong>Reason:</strong> <span id="latest-run-reason">-</span></p>
            </div>
          </div>
        </div>
        
        <!-- Actions -->
        <div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:16px;">
          <button id="run-policy-calibration-btn" onclick="runPolicyCalibration(false)" style="background:#2196f3;color:#fff;border:none;padding:8px 16px;border-radius:4px;cursor:pointer;">
            Run Calibration (With Policy)
          </button>
          <button id="force-apply-btn" onclick="runPolicyCalibration(true)" style="background:#ff9800;color:#fff;border:none;padding:8px 16px;border-radius:4px;cursor:pointer;">
            Force-Apply Latest
          </button>
          <button id="refresh-policy-btn" onclick="refreshPolicyUI()" style="background:#9e9e9e;color:#fff;border:none;padding:8px 16px;border-radius:4px;cursor:pointer;">
            Refresh
          </button>
          <select id="policy-underlying-select" style="padding:6px 12px;border-radius:4px;border:1px solid #ccc;">
            <option value="BTC">BTC</option>
            <option value="ETH">ETH</option>
          </select>
          <select id="policy-source-select" style="padding:6px 12px;border-radius:4px;border:1px solid #ccc;">
            <option value="live">Live API</option>
            <option value="harvested">Harvested Data</option>
          </select>
        </div>
        
        <div id="policy-action-status" aria-live="polite" style="margin-bottom:12px;font-size:0.9rem;"></div>
        
        <!-- Recent Calibration Runs History -->
        <h4 style="margin:0 0 8px 0;color:#555;">Recent Calibration Runs</h4>
        <div style="overflow-x:auto;max-height:240px;overflow-y:auto;">
          <table class="steps-table" style="font-size:0.85rem;">
            <thead>
              <tr>
                <th>Time</th>
                <th>Source</th>
                <th>Recommended</th>
                <th>Smoothed</th>
                <th>Samples</th>
                <th>Applied?</th>
                <th>Reason</th>
              </tr>
            </thead>
            <tbody id="policy-runs-body">
              <tr><td colspan="7" style="text-align:center;color:#666;">No runs yet</td></tr>
            </tbody>
          </table>
        </div>
      </div>
      
      <div class="card" style="margin-top:1rem;">
        <h3 style="margin-top:0;">Calibration History (Auto-Calibrate)</h3>
        <p style="color:#666;font-size:0.9rem;margin-bottom:8px;">
          Auto-calibrations use harvested Deribit data and the same calibration engine as the live UI.
          Each run is scored as OK, Degraded, or Failed based on data quality, error magnitude, and multiplier sanity.
          Failed runs are recorded for debugging only and are not used to update the vol surface.
        </p>
        <p style="color:#888;font-size:0.85rem;margin-bottom:12px;">
          Run <code>scripts/auto_calibrate_iv.py --underlying BTC</code> or <code>--underlying ETH</code> to add new entries.
        </p>
        <div style="display:flex;gap:8px;margin-bottom:12px;">
          <button id="calib-history-refresh-btn" onclick="fetchCalibrationHistory()" style="background:#2196f3;color:#fff;border:none;padding:6px 12px;border-radius:4px;cursor:pointer;">Refresh History</button>
        </div>
        <div style="overflow-x:auto;max-height:320px;overflow-y:auto;">
          <table class="steps-table" style="font-size:0.85rem;">
            <thead>
              <tr>
                <th>Date</th>
                <th>Status</th>
                <th>DTE Range</th>
                <th>Lookback</th>
                <th>Multiplier</th>
                <th>vMAE %</th>
                <th>MAE %</th>
                <th>Samples</th>
                <th style="max-width:200px;">Reason</th>
              </tr>
            </thead>
            <tbody id="calib-history-body">
              <tr><td colspan="9" style="text-align:center;color:#666;">No history data</td></tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  </div>

  <!-- BOTS TAB -->
  <div id="tab-strategies" class="tab-content">
    <div class="section">
      <!-- Header with Test/Live Toggle -->
      <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
        <h2 style="margin: 0;">Bots</h2>
        <div id="bots-env-toggle" style="display: flex; gap: 0; border-radius: 6px; overflow: hidden; border: 2px solid #7c4dff;">
          <button id="bots-env-test" onclick="switchBotsEnv('test')" style="padding: 0.5rem 1.25rem; font-weight: 600; border: none; cursor: pointer; background: #7c4dff; color: white;">TEST</button>
          <button id="bots-env-live" onclick="switchBotsEnv('live')" style="padding: 0.5rem 1.25rem; font-weight: 600; border: none; cursor: pointer; background: #f5f5f5; color: #333;">LIVE</button>
        </div>
      </div>
      <p style="color: #666; margin-bottom: 1rem;">View and configure expert trading bots. In TEST mode, you can override thresholds and risk settings.</p>
      
      <!-- Bot Tabs -->
      <div id="bots-bot-tabs" style="display: flex; gap: 0.5rem; margin-bottom: 1rem; border-bottom: 2px solid #e0e0e0; padding-bottom: 0.5rem;">
        <button class="bots-bot-tab active" data-bot-id="gregbot" onclick="selectBotTab('gregbot')" style="padding: 0.5rem 1rem; background: #7c4dff; color: white; border: none; border-radius: 4px 4px 0 0; cursor: pointer; font-weight: 600;">GregBot</button>
      </div>
      
      <!-- Main Layout: Sidebar + Content -->
      <div id="bots-main-layout" style="display: flex; gap: 1.5rem;">
        <!-- Left Sidebar -->
        <div id="bots-sidebar" style="min-width: 180px; flex-shrink: 0;">
          <div style="background: #f8f9fa; border-radius: 8px; padding: 0.5rem;">
            <button class="bots-sidebar-item active" data-section="overview" onclick="selectBotsSection('overview')" style="display: block; width: 100%; text-align: left; padding: 0.75rem 1rem; border: none; background: #7c4dff; color: white; border-radius: 4px; cursor: pointer; margin-bottom: 0.25rem; font-weight: 500;">Overview</button>
            <button class="bots-sidebar-item" data-section="entry_rules" onclick="selectBotsSection('entry_rules')" style="display: block; width: 100%; text-align: left; padding: 0.75rem 1rem; border: none; background: transparent; color: #333; border-radius: 4px; cursor: pointer; margin-bottom: 0.25rem;">Entry Rules</button>
            <button class="bots-sidebar-item" data-section="global_risk" onclick="selectBotsSection('global_risk')" style="display: block; width: 100%; text-align: left; padding: 0.75rem 1rem; border: none; background: transparent; color: #333; border-radius: 4px; cursor: pointer; margin-bottom: 0.25rem;">Global Risk</button>
            <button class="bots-sidebar-item" data-section="bot_risk" onclick="selectBotsSection('bot_risk')" style="display: block; width: 100%; text-align: left; padding: 0.75rem 1rem; border: none; background: transparent; color: #333; border-radius: 4px; cursor: pointer;">Bot Risk</button>
          </div>
        </div>
        
        <!-- Right Content Pane -->
        <div id="bots-content-pane" style="flex: 1; min-width: 0;">
          
          <!-- OVERVIEW Section -->
          <div id="bots-section-overview" class="bots-section" style="display: block;">
            <!-- Live Market Sensors -->
            <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 8px; border-left: 4px solid #1565c0; margin-bottom: 1.5rem;">
              <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                <h3 style="margin: 0; color: #1565c0; font-size: 1.1rem;">Live Market Sensors</h3>
                <div style="display: flex; align-items: center; gap: 1rem;">
                  <label style="display: flex; align-items: center; gap: 0.5rem; font-size: 0.85rem; cursor: pointer;">
                    <input type="checkbox" id="bots-debug-toggle">
                    Show debug inputs
                  </label>
                  <button onclick="refreshBotsSensors()" style="padding: 0.5rem 1rem; background: #1565c0; color: white; border: none; border-radius: 4px; cursor: pointer;">
                    Refresh Sensors
                  </button>
                </div>
              </div>
              <div id="bots-live-sensors" style="overflow-x: auto;">
                <p style="color: #666; font-style: italic;">Loading live market sensors...</p>
              </div>
              <details id="bots-debug-panel" style="margin-top: 0.75rem; display: none;">
                <summary style="cursor: pointer; color: #1565c0; font-weight: 500;">Debug: raw sensor inputs</summary>
                <pre id="bots-debug-output" style="max-height: 400px; overflow: auto; background: #fff; padding: 1rem; border: 1px solid #ddd; border-radius: 4px; font-size: 0.8rem; white-space: pre-wrap; margin-top: 0.5rem;"></pre>
              </details>
            </div>
            
            <!-- Strategy Matches (All Bots) -->
            <div style="background: #e8f5e9; padding: 1.5rem; border-radius: 8px; border-left: 4px solid #2e7d32; margin-bottom: 1.5rem;">
              <h3 style="margin: 0 0 1rem 0; color: #2e7d32; font-size: 1.1rem;">Strategy Matches (All Bots)</h3>
              <div id="bots-strategy-matches" style="overflow-x: auto;">
                <p style="color: #666; font-style: italic;">Loading strategy matches...</p>
              </div>
            </div>
            
            <!-- Expert Bots Table -->
            <div style="background: #f3e5f5; padding: 1.5rem; border-radius: 8px; border-left: 4px solid #7c4dff; margin-bottom: 1.5rem;">
              <h3 style="margin: 0 0 1rem 0; color: #7c4dff; font-size: 1.1rem;">Strategy Evaluations</h3>
              
              <div id="bots-expert-tabs" style="margin-bottom: 1rem; display: none;">
                <button class="bots-expert-tab active" data-expert-id="greg_mandolini" onclick="selectExpertTab('greg_mandolini')" style="padding: 0.5rem 1rem; background: #7c4dff; color: white; border: none; border-radius: 4px; cursor: pointer; margin-right: 0.5rem;">
                  GregBot
                </button>
              </div>
        
        <!-- Filters -->
        <div id="bots-expert-filters" style="margin-bottom: 1rem; display: flex; gap: 1rem; flex-wrap: wrap; font-size: 0.85rem;">
          <label style="display: flex; align-items: center; gap: 0.3rem; cursor: pointer;">
            <input type="checkbox" id="filter-show-pass" checked onchange="renderExpertTable()">
            <span style="color: #2e7d32;">Show PASS</span>
          </label>
          <label style="display: flex; align-items: center; gap: 0.3rem; cursor: pointer;">
            <input type="checkbox" id="filter-show-blocked" checked onchange="renderExpertTable()">
            <span style="color: #c62828;">Show BLOCKED</span>
          </label>
          <label style="display: flex; align-items: center; gap: 0.3rem; cursor: pointer;">
            <input type="checkbox" id="filter-show-nodata" checked onchange="renderExpertTable()">
            <span style="color: #666;">Show NO DATA</span>
          </label>
        </div>
        
        <div id="bots-expert-table" style="overflow-x: auto;">
          <p style="color: #666; font-style: italic;">Loading expert strategies...</p>
        </div>
        
        <!-- Greg Calibration Panel -->
        <details id="greg-calibration-panel" style="margin-top: 1rem; background: white; padding: 1rem; border-radius: 6px; border: 1px solid #ddd;">
          <summary style="cursor: pointer; font-weight: 500; color: #7c4dff;">Greg Calibration (v1)</summary>
          <div style="margin-top: 0.75rem;">
            <div id="greg-calibration-status" style="font-size: 0.9rem; color: #666; margin-bottom: 0.5rem;">Loading...</div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 0.5rem; font-size: 0.8rem;">
              <div id="greg-calibration-values"></div>
            </div>
            <button onclick="refreshGregCalibration()" style="margin-top: 0.75rem; padding: 0.4rem 0.8rem; font-size: 0.8rem; background: #7c4dff; color: white; border: none; border-radius: 4px; cursor: pointer;">
              Refresh Calibration
            </button>
          </div>
        </details>
        
        <!-- Greg Position Management Panel -->
        <div id="greg-management-panel" style="margin-top: 1rem; background: #fff8e1; padding: 1rem; border-radius: 6px; border: 1px solid #ffe082;">
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
            <div style="display: flex; align-items: center; gap: 0.5rem;">
              <h4 id="greg-pm-title" style="margin: 0; color: #f57c00; font-size: 1rem;">Position Management</h4>
              <span id="greg-mode-badge" style="padding: 0.2rem 0.5rem; font-size: 0.7rem; border-radius: 4px; background: #e0e0e0; color: #555;">ADVICE ONLY</span>
            </div>
            <div style="display: flex; gap: 0.5rem;">
              <button onclick="openGregModeSettings()" style="padding: 0.4rem 0.8rem; font-size: 0.8rem; background: #666; color: white; border: none; border-radius: 4px; cursor: pointer;">
                Mode Settings
              </button>
              <button onclick="loadMockGregManagement()" style="padding: 0.4rem 0.8rem; font-size: 0.8rem; background: #ffa726; color: white; border: none; border-radius: 4px; cursor: pointer;">
                Load Demo
              </button>
              <button onclick="refreshGregManagement()" style="padding: 0.4rem 0.8rem; font-size: 0.8rem; background: #f57c00; color: white; border: none; border-radius: 4px; cursor: pointer;">
                Refresh
              </button>
            </div>
          </div>
          <p id="greg-pm-subtitle" style="font-size: 0.8rem; color: #666; margin: 0 0 0.75rem 0;">
            Greg-style management suggestions for open positions. <strong>Advisory only</strong> - no real orders sent.
          </p>
          <div id="greg-management-table-container" style="overflow-x: auto;">
            <table id="greg-management-table" style="width: 100%; border-collapse: collapse; font-size: 0.85rem;">
              <thead>
                <tr style="background: #f5f5f5;">
                  <th style="padding: 0.5rem; text-align: left; border-bottom: 2px solid #ddd;">Type</th>
                  <th style="padding: 0.5rem; text-align: left; border-bottom: 2px solid #ddd;">Underlying</th>
                  <th style="padding: 0.5rem; text-align: left; border-bottom: 2px solid #ddd;">Strategy</th>
                  <th style="padding: 0.5rem; text-align: left; border-bottom: 2px solid #ddd;">Position</th>
                  <th style="padding: 0.5rem; text-align: left; border-bottom: 2px solid #ddd;">Suggested Action</th>
                  <th style="padding: 0.5rem; text-align: left; border-bottom: 2px solid #ddd;">Key Metrics</th>
                  <th style="padding: 0.5rem; text-align: left; border-bottom: 2px solid #ddd;">Execute</th>
                </tr>
              </thead>
              <tbody id="greg-management-tbody">
                <tr><td colspan="7" style="padding: 1rem; color: #666; font-style: italic; text-align: center;">No management suggestions loaded. Click "Load Demo" or "Refresh" to see suggestions.</td></tr>
              </tbody>
            </table>
          </div>
          <div id="greg-management-status" style="font-size: 0.8rem; color: #666; margin-top: 0.5rem;" aria-live="polite"></div>
        </div>
        
        <!-- Greg Mode Settings Modal -->
        <div id="greg-mode-modal" style="display: none; position: fixed; top: 0; left: 0; right: 0; bottom: 0; background: rgba(0,0,0,0.5); z-index: 1000; align-items: center; justify-content: center;">
          <div style="background: white; border-radius: 8px; padding: 1.5rem; max-width: 500px; width: 90%; max-height: 80vh; overflow-y: auto;">
            <h3 style="margin: 0 0 1rem 0; color: #f57c00;">Greg Trading Mode Settings</h3>
            <div style="margin-bottom: 1rem;">
              <label style="font-weight: 600; display: block; margin-bottom: 0.5rem;">Trading Mode</label>
              <select id="greg-mode-select" style="width: 100%; padding: 0.5rem; border: 1px solid #ddd; border-radius: 4px;">
                <option value="advice_only">Advice Only (No orders)</option>
                <option value="paper">Paper Mode (Testnet/DRY_RUN)</option>
                <option value="live">Live Mode (Orders allowed)</option>
              </select>
            </div>
            <div id="greg-live-settings" style="display: none; margin-bottom: 1rem; padding: 1rem; background: #fff3e0; border-radius: 4px;">
              <label style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
                <input type="checkbox" id="greg-enable-live">
                <span>Enable Live Execution (Master Switch)</span>
              </label>
              <div style="margin-top: 0.75rem;">
                <strong style="font-size: 0.9rem;">Notional Limits (Safety Guardrails):</strong>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; margin-top: 0.5rem;">
                  <label style="font-size: 0.85rem;">
                    Max $/Position:
                    <input type="number" id="greg-max-notional-pos" min="100" max="10000" step="100" style="width: 100%; padding: 0.3rem; margin-top: 0.25rem;">
                  </label>
                  <label style="font-size: 0.85rem;">
                    Max $/Underlying:
                    <input type="number" id="greg-max-notional-und" min="500" max="50000" step="500" style="width: 100%; padding: 0.3rem; margin-top: 0.25rem;">
                  </label>
                </div>
              </div>
              <div style="margin-top: 0.75rem;">
                <strong style="font-size: 0.9rem;">Per-Strategy Toggles:</strong>
                <div id="greg-strategy-toggles" style="margin-top: 0.5rem; font-size: 0.85rem;"></div>
              </div>
            </div>
            <div id="greg-live-confirm" style="display: none; margin-bottom: 1rem; padding: 1rem; background: #ffebee; border-radius: 4px;">
              <p style="color: #c62828; margin: 0 0 0.5rem 0; font-weight: 600;">Confirm LIVE Mode</p>
              <p style="font-size: 0.85rem; color: #666; margin: 0 0 0.5rem 0;">Type "LIVE" to confirm switching to live trading mode.</p>
              <input type="text" id="greg-live-confirm-input" placeholder="Type LIVE" style="width: 100%; padding: 0.5rem; border: 1px solid #ddd; border-radius: 4px;">
            </div>
            <div style="display: flex; gap: 0.5rem; justify-content: flex-end;">
              <button onclick="closeGregModeSettings()" style="padding: 0.5rem 1rem; background: #e0e0e0; border: none; border-radius: 4px; cursor: pointer;">Cancel</button>
              <button onclick="saveGregModeSettings()" style="padding: 0.5rem 1rem; background: #f57c00; color: white; border: none; border-radius: 4px; cursor: pointer;">Save</button>
            </div>
          </div>
        </div>
        
        <!-- Delta Hedging Engine Panel -->
        <div id="greg-hedging-panel" style="margin-top: 1rem; background: #e3f2fd; padding: 1rem; border-radius: 6px; border: 1px solid #90caf9;">
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
            <h4 style="margin: 0; color: #1565c0; font-size: 1rem;">Delta Hedging Engine</h4>
            <div style="display: flex; gap: 0.5rem; align-items: center;">
              <span id="hedge-dry-run-badge" style="padding: 0.25rem 0.5rem; font-size: 0.75rem; background: #fff3e0; color: #e65100; border-radius: 4px;">DRY RUN</span>
              <button onclick="evaluateHedging()" style="padding: 0.4rem 0.8rem; font-size: 0.8rem; background: #42a5f5; color: white; border: none; border-radius: 4px; cursor: pointer;">
                Evaluate Hedges
              </button>
              <button onclick="refreshHedgingStatus()" style="padding: 0.4rem 0.8rem; font-size: 0.8rem; background: #1565c0; color: white; border: none; border-radius: 4px; cursor: pointer;">
                Refresh
              </button>
            </div>
          </div>
          <p style="font-size: 0.8rem; color: #666; margin: 0 0 0.75rem 0;">
            Delta-neutral hedging for short-vol strategies (straddles, strangles, iron flies). Uses perpetual futures to restore delta to target.
          </p>
          
          <!-- Hedge Rules Summary -->
          <details style="margin-bottom: 0.75rem;">
            <summary style="cursor: pointer; font-weight: bold; color: #1565c0; font-size: 0.85rem;">Hedging Rules by Strategy</summary>
            <div id="hedge-rules-container" style="margin-top: 0.5rem; padding: 0.5rem; background: white; border-radius: 4px;">
              <table style="width: 100%; border-collapse: collapse; font-size: 0.8rem;">
                <thead>
                  <tr style="background: #f5f5f5;">
                    <th style="padding: 0.4rem; text-align: left; border-bottom: 1px solid #ddd;">Strategy</th>
                    <th style="padding: 0.4rem; text-align: left; border-bottom: 1px solid #ddd;">Hedge Mode</th>
                    <th style="padding: 0.4rem; text-align: left; border-bottom: 1px solid #ddd;">Delta Threshold</th>
                  </tr>
                </thead>
                <tbody id="hedge-rules-tbody">
                  <tr><td colspan="3" style="padding: 0.5rem; color: #666; font-style: italic;">Loading...</td></tr>
                </tbody>
              </table>
            </div>
          </details>
          
          <!-- Proposed Hedges Table -->
          <div id="hedge-proposals-container" style="overflow-x: auto;">
            <table id="hedge-proposals-table" style="width: 100%; border-collapse: collapse; font-size: 0.85rem;">
              <thead>
                <tr style="background: #f5f5f5;">
                  <th style="padding: 0.5rem; text-align: left; border-bottom: 2px solid #ddd;">Position</th>
                  <th style="padding: 0.5rem; text-align: left; border-bottom: 2px solid #ddd;">Strategy</th>
                  <th style="padding: 0.5rem; text-align: left; border-bottom: 2px solid #ddd;">Net Delta</th>
                  <th style="padding: 0.5rem; text-align: left; border-bottom: 2px solid #ddd;">Threshold</th>
                  <th style="padding: 0.5rem; text-align: left; border-bottom: 2px solid #ddd;">Proposed Hedge</th>
                </tr>
              </thead>
              <tbody id="hedge-proposals-tbody">
                <tr><td colspan="5" style="padding: 1rem; color: #666; font-style: italic; text-align: center;">Click "Evaluate Hedges" to analyze positions.</td></tr>
              </tbody>
            </table>
          </div>
          <div id="hedge-status" style="font-size: 0.8rem; color: #666; margin-top: 0.5rem;" aria-live="polite"></div>
          
          <!-- Hedge History -->
          <details style="margin-top: 0.75rem;">
            <summary style="cursor: pointer; font-weight: bold; color: #1565c0; font-size: 0.85rem;">Recent Hedge History</summary>
            <div id="hedge-history-container" style="margin-top: 0.5rem; max-height: 200px; overflow-y: auto;">
              <table style="width: 100%; border-collapse: collapse; font-size: 0.8rem;">
                <thead>
                  <tr style="background: #f5f5f5;">
                    <th style="padding: 0.4rem; text-align: left; border-bottom: 1px solid #ddd;">Time</th>
                    <th style="padding: 0.4rem; text-align: left; border-bottom: 1px solid #ddd;">Position</th>
                    <th style="padding: 0.4rem; text-align: left; border-bottom: 1px solid #ddd;">Order</th>
                    <th style="padding: 0.4rem; text-align: left; border-bottom: 1px solid #ddd;">Status</th>
                  </tr>
                </thead>
                <tbody id="hedge-history-tbody">
                  <tr><td colspan="4" style="padding: 0.5rem; color: #666; font-style: italic;">No hedge history yet.</td></tr>
                </tbody>
              </table>
            </div>
          </details>
        </div>
            </div>
          </div>
          <!-- End of OVERVIEW Section -->
          
          <!-- ENTRY RULES Section -->
          <div id="bots-section-entry_rules" class="bots-section" style="display: none;">
            <div style="background: #fff3e0; padding: 1.5rem; border-radius: 8px; border-left: 4px solid #ff9800;">
              <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                <h3 style="margin: 0; color: #e65100; font-size: 1.1rem;">Entry Rule Thresholds</h3>
                <div id="entry-rules-mode-badge" style="padding: 0.25rem 0.75rem; border-radius: 4px; font-size: 0.75rem; font-weight: 600; background: #7c4dff; color: white;">TEST MODE</div>
              </div>
              <p style="color: #666; font-size: 0.9rem; margin-bottom: 1rem;">Configure the sensor thresholds that determine when strategies are eligible for entry.</p>
              
              <div id="entry-rules-override-toggle" style="margin-bottom: 1rem; padding: 0.75rem; background: #f5f5f5; border-radius: 4px;">
                <label style="display: flex; align-items: center; gap: 0.5rem; cursor: pointer;">
                  <input type="checkbox" id="entry-rules-use-overrides" onchange="toggleEntryRulesOverrides()">
                  <span style="font-weight: 500;">Enable TEST Mode Overrides</span>
                </label>
                <p style="color: #888; font-size: 0.8rem; margin: 0.5rem 0 0 1.5rem;">When enabled, you can modify thresholds for testing without affecting production values.</p>
              </div>
              
              <div id="entry-rules-form" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem;">
                <p style="color: #666; font-style: italic;">Loading entry rules...</p>
              </div>
              
              <div id="entry-rules-actions" style="margin-top: 1rem; display: none;">
                <button onclick="saveEntryRules()" style="padding: 0.5rem 1rem; background: #ff9800; color: white; border: none; border-radius: 4px; cursor: pointer; margin-right: 0.5rem;">Save Overrides</button>
                <button onclick="resetEntryRules()" style="padding: 0.5rem 1rem; background: #e0e0e0; color: #333; border: none; border-radius: 4px; cursor: pointer;">Reset to Defaults</button>
              </div>
              <div id="entry-rules-status" style="margin-top: 0.5rem; font-size: 0.85rem;"></div>
            </div>
          </div>
          
          <!-- GLOBAL RISK Section -->
          <div id="bots-section-global_risk" class="bots-section" style="display: none;">
            <div style="background: #ffebee; padding: 1.5rem; border-radius: 8px; border-left: 4px solid #c62828;">
              <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                <h3 style="margin: 0; color: #c62828; font-size: 1.1rem;">Global Risk Settings</h3>
                <div id="global-risk-mode-badge" style="padding: 0.25rem 0.75rem; border-radius: 4px; font-size: 0.75rem; font-weight: 600; background: #7c4dff; color: white;">TEST MODE</div>
              </div>
              <p style="color: #666; font-size: 0.9rem; margin-bottom: 1rem;">System-wide risk parameters that apply to all bots and strategies.</p>
              
              <div id="global-risk-override-toggle" style="margin-bottom: 1rem; padding: 0.75rem; background: #f5f5f5; border-radius: 4px;">
                <label style="display: flex; align-items: center; gap: 0.5rem; cursor: pointer;">
                  <input type="checkbox" id="global-risk-use-overrides" onchange="toggleGlobalRiskOverrides()">
                  <span style="font-weight: 500;">Enable TEST Mode Overrides</span>
                </label>
              </div>
              
              <div id="global-risk-form" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem;">
                <p style="color: #666; font-style: italic;">Loading global risk settings...</p>
              </div>
              
              <div id="global-risk-actions" style="margin-top: 1rem; display: none;">
                <button onclick="saveGlobalRisk()" style="padding: 0.5rem 1rem; background: #c62828; color: white; border: none; border-radius: 4px; cursor: pointer; margin-right: 0.5rem;">Save Overrides</button>
                <button onclick="resetGlobalRisk()" style="padding: 0.5rem 1rem; background: #e0e0e0; color: #333; border: none; border-radius: 4px; cursor: pointer;">Reset to Defaults</button>
              </div>
              <div id="global-risk-status" style="margin-top: 0.5rem; font-size: 0.85rem;"></div>
            </div>
          </div>
          
          <!-- BOT RISK Section -->
          <div id="bots-section-bot_risk" class="bots-section" style="display: none;">
            <div style="background: #e3f2fd; padding: 1.5rem; border-radius: 8px; border-left: 4px solid #1565c0;">
              <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                <h3 style="margin: 0; color: #1565c0; font-size: 1.1rem;">Per-Bot Risk Settings</h3>
                <div id="bot-risk-mode-badge" style="padding: 0.25rem 0.75rem; border-radius: 4px; font-size: 0.75rem; font-weight: 600; background: #7c4dff; color: white;">TEST MODE</div>
              </div>
              <p style="color: #666; font-size: 0.9rem; margin-bottom: 1rem;">Risk parameters specific to the currently selected bot (GregBot).</p>
              
              <div id="bot-risk-override-toggle" style="margin-bottom: 1rem; padding: 0.75rem; background: #f5f5f5; border-radius: 4px;">
                <label style="display: flex; align-items: center; gap: 0.5rem; cursor: pointer;">
                  <input type="checkbox" id="bot-risk-use-overrides" onchange="toggleBotRiskOverrides()">
                  <span style="font-weight: 500;">Enable TEST Mode Overrides</span>
                </label>
              </div>
              
              <div id="bot-risk-form" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem;">
                <p style="color: #666; font-style: italic;">Loading bot risk settings...</p>
              </div>
              
              <div id="bot-risk-actions" style="margin-top: 1rem; display: none;">
                <button onclick="saveBotRisk()" style="padding: 0.5rem 1rem; background: #1565c0; color: white; border: none; border-radius: 4px; cursor: pointer; margin-right: 0.5rem;">Save Overrides</button>
                <button onclick="resetBotRisk()" style="padding: 0.5rem 1rem; background: #e0e0e0; color: #333; border: none; border-radius: 4px; cursor: pointer;">Reset to Defaults</button>
              </div>
              <div id="bot-risk-status" style="margin-top: 0.5rem; font-size: 0.85rem;"></div>
            </div>
          </div>
          
        </div>
        <!-- End of Right Content Pane -->
      </div>
      <!-- End of Main Layout -->
      
      <p style="color: #888; font-size: 0.85rem; margin-top: 1rem;">
        <strong>Phase 1 (Read-Only):</strong> Computes sensors, runs decision tree, displays recommendation. No orders placed.
      </p>
    </div>
  </div>

  <!-- GREG LAB TAB -->
  <div id="tab-greglab" class="tab-content">
    <div class="section">
      <h2>Greg Lab - Position Management</h2>
      
      <!-- Mode Banner -->
      <div id="greg-mode-banner" style="padding: 0.75rem 1rem; border-radius: 8px; margin-bottom: 1.5rem; display: flex; align-items: center; justify-content: space-between; background: #e3f2fd; border-left: 4px solid #1565c0;">
        <div>
          <span id="greg-mode-pill" style="padding: 0.25rem 0.75rem; border-radius: 4px; font-weight: bold; background: #fff3e0; color: #e65100;">ADVICE ONLY</span>
          <span id="greg-mode-desc" style="margin-left: 0.75rem; color: #333;">No orders will be sent.</span>
        </div>
        <div id="greg-env-badge" style="font-size: 0.8rem; color: #666;"></div>
      </div>
      
      <!-- Sandbox Summary -->
      <div id="greg-sandbox-summary" style="display: none; padding: 0.75rem 1rem; border-radius: 8px; margin-bottom: 1rem; background: #fff8e1; border-left: 4px solid #ffc107;">
        <strong style="color: #f57c00;">Sandbox Run:</strong> <span id="greg-sandbox-run-id"></span>
        <span id="greg-sandbox-counts" style="margin-left: 1rem;"></span>
      </div>
      
      <!-- Filters -->
      <div style="display: flex; gap: 1rem; align-items: center; margin-bottom: 1rem; flex-wrap: wrap;">
        <div style="display: flex; gap: 0.5rem;">
          <button class="greg-filter-btn" data-underlying="" onclick="filterGregPositions(this, '')" style="padding: 0.4rem 0.75rem; border-radius: 4px; background: #1565c0; color: white; border: none; cursor: pointer;">All</button>
          <button class="greg-filter-btn" data-underlying="BTC" onclick="filterGregPositions(this, 'BTC')" style="padding: 0.4rem 0.75rem; border-radius: 4px; background: #e0e0e0; border: none; cursor: pointer;">BTC</button>
          <button class="greg-filter-btn" data-underlying="ETH" onclick="filterGregPositions(this, 'ETH')" style="padding: 0.4rem 0.75rem; border-radius: 4px; background: #e0e0e0; border: none; cursor: pointer;">ETH</button>
        </div>
        <select id="greg-sandbox-filter" onchange="loadGregPositions()" style="padding: 0.4rem 0.5rem; border-radius: 4px;">
          <option value="all">All Positions</option>
          <option value="sandbox_only">Sandbox Only</option>
          <option value="non_sandbox">Non-Sandbox</option>
        </select>
        <button onclick="loadGregPositions()" style="padding: 0.4rem 0.75rem;">Refresh</button>
      </div>
      
      <!-- Positions Table -->
      <div style="overflow-x: auto;">
        <table style="width: 100%; border-collapse: collapse; font-size: 0.85rem;">
          <thead>
            <tr style="background: #f5f5f5;">
              <th style="padding: 0.6rem; text-align: left; border-bottom: 2px solid #ddd;">Badge</th>
              <th style="padding: 0.6rem; text-align: left; border-bottom: 2px solid #ddd;">Underlying</th>
              <th style="padding: 0.6rem; text-align: left; border-bottom: 2px solid #ddd;">Strategy</th>
              <th style="padding: 0.6rem; text-align: right; border-bottom: 2px solid #ddd;">Size</th>
              <th style="padding: 0.6rem; text-align: right; border-bottom: 2px solid #ddd;">PnL %</th>
              <th style="padding: 0.6rem; text-align: right; border-bottom: 2px solid #ddd;">PnL USD</th>
              <th style="padding: 0.6rem; text-align: right; border-bottom: 2px solid #ddd;">DTE</th>
              <th style="padding: 0.6rem; text-align: left; border-bottom: 2px solid #ddd;">Action</th>
              <th style="padding: 0.6rem; text-align: center; border-bottom: 2px solid #ddd;">Actions</th>
            </tr>
          </thead>
          <tbody id="greg-positions-tbody">
            <tr><td colspan="9" style="padding: 1rem; text-align: center; color: #666;">Loading positions...</td></tr>
          </tbody>
        </table>
      </div>
      <div id="greg-positions-count" style="margin-top: 0.5rem; font-size: 0.85rem; color: #666;"></div>
    </div>
    
    <!-- Position Log Timeline Panel -->
    <div id="greg-log-panel" style="display: none; margin-top: 1.5rem;" class="section">
      <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
        <h3 id="greg-log-title" style="margin: 0;">Management Log</h3>
        <button onclick="closeGregLogPanel()" style="padding: 0.25rem 0.5rem;">Close</button>
      </div>
      <div style="overflow-x: auto;">
        <table style="width: 100%; border-collapse: collapse; font-size: 0.8rem;">
          <thead>
            <tr style="background: #f5f5f5;">
              <th style="padding: 0.5rem; text-align: left; border-bottom: 1px solid #ddd;">Time</th>
              <th style="padding: 0.5rem; text-align: left; border-bottom: 1px solid #ddd;">Action</th>
              <th style="padding: 0.5rem; text-align: center; border-bottom: 1px solid #ddd;">Suggested</th>
              <th style="padding: 0.5rem; text-align: center; border-bottom: 1px solid #ddd;">Executed</th>
              <th style="padding: 0.5rem; text-align: left; border-bottom: 1px solid #ddd;">Reason</th>
              <th style="padding: 0.5rem; text-align: right; border-bottom: 1px solid #ddd;">PnL%</th>
              <th style="padding: 0.5rem; text-align: left; border-bottom: 1px solid #ddd;">Sensors</th>
            </tr>
          </thead>
          <tbody id="greg-log-tbody">
            <tr><td colspan="7" style="padding: 0.5rem; color: #666;">No logs yet.</td></tr>
          </tbody>
        </table>
      </div>
    </div>
    
    <!-- Observer Notes Stub -->
    <div class="section" style="margin-top: 1.5rem;">
      <details>
        <summary style="cursor: pointer; font-weight: bold; color: #1565c0;">Observer Notes (Coming Soon)</summary>
        <div style="padding: 1rem; background: #f5f5f5; border-radius: 4px; margin-top: 0.5rem;">
          <p style="color: #666; font-style: italic;">TODO: Daily Greg Observer LLM summary will be displayed here.</p>
          <ul style="color: #666; font-size: 0.85rem;">
            <li>Key market observations</li>
            <li>Strategy performance notes</li>
            <li>Risk alerts</li>
          </ul>
        </div>
      </details>
    </div>
  </div>

  <!-- SYSTEM HEALTH TAB -->
  <div id="tab-health" class="tab-content">
    <div class="section">
      <h2>System Controls & Health</h2>
      <p style="color: #666; margin-bottom: 1rem;">Run diagnostics and test system components. These are dry-run tests that do not execute trades.</p>
      
      <!-- Health Guard Status (Agent Pause Indicator) -->
      <div id="health-guard-status"></div>
      
      <!-- Overall Health Status Badge -->
      <div id="health-status-badge" style="display: flex; align-items: center; gap: 1rem; padding: 1rem; background: #f5f5f5; border-radius: 8px; margin-bottom: 1.5rem;">
        <div id="health-badge" style="padding: 0.5rem 1rem; border-radius: 4px; font-weight: bold; background: #e0e0e0; color: #666;">PENDING</div>
        <div style="flex: 1;">
          <div id="health-summary" style="font-size: 0.9rem; color: #333;">Healthcheck not run yet</div>
          <div id="health-last-run" style="font-size: 0.8rem; color: #666;">Last run: never</div>
        </div>
        <button onclick="runHealthcheck()" style="padding: 0.5rem 1rem;">Run Healthcheck</button>
      </div>
      
      <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1.5rem;">
        
        <!-- LLM & Decision Mode -->
        <div style="background: #f8f9fa; padding: 1.25rem; border-radius: 8px; border-left: 4px solid #1565c0;">
          <h3 style="margin: 0 0 0.75rem 0; color: #1565c0; font-size: 1rem;">LLM & Decision Mode</h3>
          <div id="llm-status-line" style="font-size: 0.85rem; color: #666; margin-bottom: 1rem;">Loading...</div>
          <button id="llm-test-btn" onclick="testLlmDecision()" style="width: 100%; margin-bottom: 0.5rem;">Test LLM Decision Pipeline</button>
          <div id="llm-result" style="font-size: 0.85rem; min-height: 2rem; padding: 0.5rem; background: white; border-radius: 4px;"></div>
        </div>
        
        <!-- Position Reconciliation -->
        <div style="background: #f8f9fa; padding: 1.25rem; border-radius: 8px; border-left: 4px solid #ff9800;">
          <h3 style="margin: 0 0 0.75rem 0; color: #ff9800; font-size: 1rem;">Position Reconciliation</h3>
          <div id="reconcile-status-line" style="font-size: 0.85rem; color: #666; margin-bottom: 1rem;">Last check: not run yet</div>
          <button onclick="runReconciliation()" style="width: 100%; margin-bottom: 0.5rem;">Run Reconciliation Now</button>
          <div id="reconcile-result" style="font-size: 0.85rem; min-height: 2rem; padding: 0.5rem; background: white; border-radius: 4px;"></div>
        </div>
        
        <!-- Risk Limits & Kill Switch -->
        <div style="background: #f8f9fa; padding: 1.25rem; border-radius: 8px; border-left: 4px solid #c62828;">
          <h3 style="margin: 0 0 0.75rem 0; color: #c62828; font-size: 1rem;">Risk Limits & Kill Switch</h3>
          <div id="risk-status-line" style="font-size: 0.85rem; color: #666; margin-bottom: 1rem;">Loading...</div>
          <button onclick="testKillSwitch()" style="width: 100%; margin-bottom: 0.5rem;">Test Risk Checks / Kill Switch</button>
          <div id="risk-result" style="font-size: 0.85rem; min-height: 2rem; padding: 0.5rem; background: white; border-radius: 4px;"></div>
        </div>
        
        <!-- Agent Healthcheck -->
        <div style="background: #f8f9fa; padding: 1.25rem; border-radius: 8px; border-left: 4px solid #2e7d32;">
          <h3 style="margin: 0 0 0.75rem 0; color: #2e7d32; font-size: 1rem;">Agent Healthcheck</h3>
          <div id="healthcheck-status-line" style="font-size: 0.85rem; color: #666; margin-bottom: 1rem;">Last check: not run yet</div>
          <button onclick="runHealthcheck()" style="width: 100%; margin-bottom: 0.5rem;">Run Full Agent Healthcheck</button>
          <div id="healthcheck-result" style="font-size: 0.85rem; min-height: 2rem; padding: 0.5rem; background: white; border-radius: 4px;"></div>
          <details id="healthcheck-details" style="margin-top: 0.5rem; display: none;">
            <summary style="cursor: pointer; font-size: 0.8rem; color: #666;">Show details</summary>
            <pre id="healthcheck-details-content" style="font-size: 0.75rem; background: white; padding: 0.5rem; border-radius: 4px; overflow-x: auto; margin-top: 0.5rem;"></pre>
          </details>
        </div>
        
      </div>
      
      <!-- AI Steward (Project Brain) Section -->
      <div style="margin-top: 2rem;">
        <h2 style="margin-bottom: 0.5rem;">AI Steward (Project Brain)</h2>
        <p style="color: #666; margin-bottom: 1.5rem;">Use the AI Steward to scan ROADMAP, BACKLOG, UI gaps, and healthcheck docs, and suggest the next tasks.</p>
        
        <div id="steward-panel" style="background: #e8f5e9; padding: 1.5rem; border-radius: 8px; border-left: 4px solid #43a047;">
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
            <div>
              <strong>Last run:</strong> <span id="steward-last-run" style="color: #666;">Never</span>
            </div>
            <button id="steward-run-btn" onclick="runStewardCheck()" style="padding: 0.5rem 1rem;">Run Steward Check</button>
          </div>
          
          <div id="steward-status" aria-live="polite" style="font-size: 0.85rem; min-height: 1.5rem; margin-bottom: 1rem;"></div>
          
          <div style="background: white; padding: 1rem; border-radius: 6px; margin-bottom: 1rem;">
            <h4 style="margin: 0 0 0.5rem 0; color: #333;">Summary</h4>
            <p id="steward-summary" style="margin: 0; color: #666; font-size: 0.9rem;">No report yet.</p>
          </div>
          
          <div style="background: white; padding: 1rem; border-radius: 6px; margin-bottom: 1rem;">
            <h4 style="margin: 0 0 0.5rem 0; color: #333;">Top Suggestions</h4>
            <ol id="steward-top-items" style="margin: 0; padding-left: 1.5rem; color: #666; font-size: 0.9rem;"></ol>
          </div>
          
          <div style="background: white; padding: 1rem; border-radius: 6px;">
            <h4 style="margin: 0 0 0.5rem 0; color: #333;">Builder Prompt (copy & paste)</h4>
            <textarea id="steward-builder-prompt" rows="6" style="width: 100%; font-size: 0.85rem; font-family: monospace; border: 1px solid #ddd; border-radius: 4px; padding: 0.5rem; resize: vertical;"></textarea>
          </div>
        </div>
      </div>
      
      <!-- Runtime Controls Section -->
      <div style="margin-top: 2rem;">
        <h2 style="margin-bottom: 0.5rem;">Runtime Controls</h2>
        <p style="color: #666; margin-bottom: 1.5rem;">Adjust safety and operational settings. Changes apply immediately but do not persist across restarts.</p>
        
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1.5rem;">
          
          <!-- Kill Switch Toggle -->
          <div style="background: #fff3e0; padding: 1.25rem; border-radius: 8px; border-left: 4px solid #e65100;">
            <h3 style="margin: 0 0 0.75rem 0; color: #e65100; font-size: 1rem;">Global Kill Switch</h3>
            <p style="font-size: 0.8rem; color: #666; margin-bottom: 1rem;">When enabled, blocks all trading actions except DO_NOTHING.</p>
            <div style="display: flex; align-items: center; gap: 1rem;">
              <label class="switch">
                <input type="checkbox" id="kill-switch-toggle" onchange="updateKillSwitch(this.checked)">
                <span class="slider round"></span>
              </label>
              <span id="kill-switch-label" style="font-weight: 600; color: #333;">OFF</span>
            </div>
            <div id="kill-switch-feedback" style="font-size: 0.8rem; min-height: 1.5rem; margin-top: 0.75rem;"></div>
          </div>
          
          <!-- Daily Drawdown Limit -->
          <div style="background: #fce4ec; padding: 1.25rem; border-radius: 8px; border-left: 4px solid #c62828;">
            <h3 style="margin: 0 0 0.75rem 0; color: #c62828; font-size: 1rem;">Daily Drawdown Limit %</h3>
            <p style="font-size: 0.8rem; color: #666; margin-bottom: 1rem;">Max daily peak-to-trough equity loss. Set 0 to disable.</p>
            <div style="display: flex; align-items: center; gap: 0.5rem;">
              <input type="number" id="drawdown-limit-input" min="0" max="100" step="0.1" style="width: 80px; padding: 0.5rem; border: 1px solid #ccc; border-radius: 4px;">
              <span>%</span>
              <button onclick="updateDrawdownLimit()" style="padding: 0.5rem 1rem;">Save</button>
            </div>
            <div id="drawdown-limit-feedback" style="font-size: 0.8rem; min-height: 1.5rem; margin-top: 0.75rem;"></div>
          </div>
          
          <!-- Decision Mode -->
          <div style="background: #e3f2fd; padding: 1.25rem; border-radius: 8px; border-left: 4px solid #1565c0;">
            <h3 style="margin: 0 0 0.75rem 0; color: #1565c0; font-size: 1rem;">Decision Mode</h3>
            <p style="font-size: 0.8rem; color: #666; margin-bottom: 1rem;">Choose how trading decisions are made.</p>
            <select id="decision-mode-select" onchange="updateDecisionMode(this.value)" style="width: 100%; padding: 0.5rem; border: 1px solid #ccc; border-radius: 4px;">
              <option value="rule_only">Rule Only</option>
              <option value="llm_only">LLM Only</option>
              <option value="hybrid_shadow">Hybrid (LLM Shadow)</option>
            </select>
            <div id="decision-mode-feedback" style="font-size: 0.8rem; min-height: 1.5rem; margin-top: 0.75rem;"></div>
          </div>
          
          <!-- Dry Run Toggle -->
          <div style="background: #e8f5e9; padding: 1.25rem; border-radius: 8px; border-left: 4px solid #2e7d32;">
            <h3 style="margin: 0 0 0.75rem 0; color: #2e7d32; font-size: 1rem;">Dry Run Mode</h3>
            <p style="font-size: 0.8rem; color: #666; margin-bottom: 1rem;">When enabled, simulates trades without placing real orders.</p>
            <div style="display: flex; align-items: center; gap: 1rem;">
              <label class="switch">
                <input type="checkbox" id="dry-run-toggle" onchange="updateDryRun(this.checked)">
                <span class="slider round"></span>
              </label>
              <span id="dry-run-label" style="font-weight: 600; color: #333;">OFF</span>
            </div>
            <div id="dry-run-feedback" style="font-size: 0.8rem; min-height: 1.5rem; margin-top: 0.75rem;"></div>
          </div>
          
          <!-- Position Reconcile Action -->
          <div style="background: #f3e5f5; padding: 1.25rem; border-radius: 8px; border-left: 4px solid #7b1fa2;">
            <h3 style="margin: 0 0 0.75rem 0; color: #7b1fa2; font-size: 1rem;">On Position Mismatch</h3>
            <p style="font-size: 0.8rem; color: #666; margin-bottom: 1rem;">Action when local positions differ from exchange.</p>
            <select id="reconcile-action-select" onchange="updateReconcileAction(this.value)" style="width: 100%; padding: 0.5rem; border: 1px solid #ccc; border-radius: 4px;">
              <option value="halt">Halt</option>
              <option value="auto_heal">Auto-Heal</option>
            </select>
            <div id="reconcile-action-feedback" style="font-size: 0.8rem; min-height: 1.5rem; margin-top: 0.75rem;"></div>
          </div>
          
        </div>
      </div>
      
      <!-- LLM & Strategy Tuning Section -->
      <div style="margin-top: 2rem;">
        <h2 style="margin-bottom: 0.5rem;">LLM & Strategy Tuning</h2>
        <p style="color: #666; margin-bottom: 1.5rem;">Adjust LLM, strategy thresholds, and risk limits. Changes are runtime-only and will reset on restart.</p>
        
        <div id="llm-strategy-panel">
          <div style="display: flex; gap: 1rem; margin-bottom: 1rem; flex-wrap: wrap;">
            <div style="padding: 0.75rem; background: #e8eaf6; border-radius: 6px;">
              <strong>Mode:</strong> <span id="llm-mode-label">Loading...</span>
            </div>
            <div style="padding: 0.75rem; background: #e8eaf6; border-radius: 6px;">
              <strong>Deribit:</strong> <span id="llm-deribit-label">Loading...</span>
            </div>
          </div>
          
          <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1.5rem;">
            
            <!-- LLM Enabled Toggle -->
            <div style="background: #e3f2fd; padding: 1.25rem; border-radius: 8px; border-left: 4px solid #1976d2;">
              <h3 style="margin: 0 0 0.75rem 0; color: #1976d2; font-size: 1rem;">LLM Enabled</h3>
              <p style="font-size: 0.8rem; color: #666; margin-bottom: 1rem;">Toggle LLM-powered decision making on/off.</p>
              <div style="display: flex; align-items: center; gap: 1rem;">
                <label class="switch">
                  <input type="checkbox" id="llm-enabled-toggle" onchange="updateLLMEnabled(this.checked)">
                  <span class="slider round"></span>
                </label>
                <span id="llm-enabled-label" style="font-weight: 600; color: #333;">OFF</span>
              </div>
              <div id="llm-enabled-feedback" style="font-size: 0.8rem; min-height: 1.5rem; margin-top: 0.75rem;"></div>
            </div>
            
            <!-- Explore Probability -->
            <div style="background: #f3e5f5; padding: 1.25rem; border-radius: 8px; border-left: 4px solid #8e24aa;">
              <h3 style="margin: 0 0 0.75rem 0; color: #8e24aa; font-size: 1rem;">Explore Probability</h3>
              <p style="font-size: 0.8rem; color: #666; margin-bottom: 1rem;">Chance of exploration vs. best-score action.</p>
              <div style="display: flex; align-items: center; gap: 0.5rem;">
                <input type="range" id="explore-prob-slider" min="0" max="100" step="5" style="flex: 1;" oninput="updateExploreProbLabel(this.value)">
                <span id="explore-prob-label" style="font-weight: 600; min-width: 40px;">0%</span>
              </div>
              <button onclick="saveExploreProb()" style="margin-top: 0.75rem; width: 100%;">Save Explore %</button>
              <div id="explore-prob-feedback" style="font-size: 0.8rem; min-height: 1.5rem; margin-top: 0.5rem;"></div>
            </div>
            
            <!-- Training Profile Mode -->
            <div style="background: #e8f5e9; padding: 1.25rem; border-radius: 8px; border-left: 4px solid #388e3c;">
              <h3 style="margin: 0 0 0.75rem 0; color: #388e3c; font-size: 1rem;">Training Profile</h3>
              <p style="font-size: 0.8rem; color: #666; margin-bottom: 1rem;">Single = one config; Ladder = sweep parameters.</p>
              <select id="training-profile-select" onchange="updateTrainingProfile(this.value)" style="width: 100%; padding: 0.5rem; border: 1px solid #ccc; border-radius: 4px;">
                <option value="single">Single</option>
                <option value="ladder">Ladder</option>
              </select>
              <div id="training-profile-feedback" style="font-size: 0.8rem; min-height: 1.5rem; margin-top: 0.75rem;"></div>
            </div>
            
          </div>
          
          <!-- Strategy Thresholds Fieldset -->
          <fieldset style="margin-top: 1.5rem; padding: 1.25rem; border: 2px solid #00bcd4; border-radius: 8px;">
            <legend style="color: #00838f; font-weight: 600; padding: 0 0.5rem;">Strategy Thresholds (effective for current mode)</legend>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem;">
              <div>
                <label style="display: block; font-size: 0.85rem; color: #555; margin-bottom: 0.25rem;">Min IV/RV Ratio</label>
                <input type="number" id="ivrv-min-input" step="0.1" min="0" style="width: 100%; padding: 0.5rem; border: 1px solid #ccc; border-radius: 4px;">
              </div>
              <div>
                <label style="display: block; font-size: 0.85rem; color: #555; margin-bottom: 0.25rem;">Delta Min</label>
                <input type="number" id="delta-min-input" step="0.01" min="0" max="1" style="width: 100%; padding: 0.5rem; border: 1px solid #ccc; border-radius: 4px;">
              </div>
              <div>
                <label style="display: block; font-size: 0.85rem; color: #555; margin-bottom: 0.25rem;">Delta Max</label>
                <input type="number" id="delta-max-input" step="0.01" min="0" max="1" style="width: 100%; padding: 0.5rem; border: 1px solid #ccc; border-radius: 4px;">
              </div>
              <div>
                <label style="display: block; font-size: 0.85rem; color: #555; margin-bottom: 0.25rem;">DTE Min (days)</label>
                <input type="number" id="dte-min-input" min="0" style="width: 100%; padding: 0.5rem; border: 1px solid #ccc; border-radius: 4px;">
              </div>
              <div>
                <label style="display: block; font-size: 0.85rem; color: #555; margin-bottom: 0.25rem;">DTE Max (days)</label>
                <input type="number" id="dte-max-input" min="0" style="width: 100%; padding: 0.5rem; border: 1px solid #ccc; border-radius: 4px;">
              </div>
            </div>
            <button onclick="saveStrategyThresholds()" style="margin-top: 1rem;">Save Strategy Thresholds</button>
            <div id="strategy-thresholds-feedback" style="font-size: 0.8rem; min-height: 1.5rem; margin-top: 0.5rem;"></div>
          </fieldset>
          
          <!-- Risk Limits Fieldset -->
          <fieldset style="margin-top: 1.5rem; padding: 1.25rem; border: 2px solid #e53935; border-radius: 8px;">
            <legend style="color: #c62828; font-weight: 600; padding: 0 0.5rem;">Risk Limits</legend>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem;">
              <div>
                <label style="display: block; font-size: 0.85rem; color: #555; margin-bottom: 0.25rem;">Max Margin Used (%)</label>
                <input type="number" id="max-margin-input" min="0" max="100" step="1" style="width: 100%; padding: 0.5rem; border: 1px solid #ccc; border-radius: 4px;">
              </div>
              <div>
                <label style="display: block; font-size: 0.85rem; color: #555; margin-bottom: 0.25rem;">Max Net Delta (abs)</label>
                <input type="number" id="max-net-delta-input" min="0" step="0.1" style="width: 100%; padding: 0.5rem; border: 1px solid #ccc; border-radius: 4px;">
              </div>
              <div>
                <label style="display: block; font-size: 0.85rem; color: #555; margin-bottom: 0.25rem;">Max Bid/Ask Spread (%)</label>
                <input type="number" id="liquidity-max-spread-input" min="0" max="100" step="0.1" style="width: 100%; padding: 0.5rem; border: 1px solid #ccc; border-radius: 4px;">
              </div>
              <div>
                <label style="display: block; font-size: 0.85rem; color: #555; margin-bottom: 0.25rem;">Min Open Interest</label>
                <input type="number" id="liquidity-min-oi-input" min="0" step="1" style="width: 100%; padding: 0.5rem; border: 1px solid #ccc; border-radius: 4px;">
              </div>
            </div>
            <p style="font-size: 0.8rem; color: #666; margin-top: 0.75rem; margin-bottom: 0;">
              Liquidity guard: max spread <span id="liquidity-max-spread-label">--</span>%, min OI <span id="liquidity-min-oi-label">--</span> contracts.
            </p>
            <button onclick="saveRiskLimits()" style="margin-top: 1rem;">Save Risk Limits</button>
            <div id="risk-limits-feedback" style="font-size: 0.8rem; min-height: 1.5rem; margin-top: 0.5rem;"></div>
          </fieldset>
          
          <!-- Position Reconciliation Fieldset -->
          <fieldset style="margin-top: 1.5rem; padding: 1.25rem; border: 2px solid #ff9800; border-radius: 8px;">
            <legend style="color: #e65100; font-weight: 600; padding: 0 0.5rem;">Position Reconciliation</legend>
            <p style="font-size: 0.85rem; color: #666; margin-bottom: 1rem;">
              Configure how the bot reacts if local positions ever diverge from Deribit.
              These settings are runtime-only and reset on restart.
            </p>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem;">
              <div>
                <label style="display: block; font-size: 0.85rem; color: #555; margin-bottom: 0.25rem;">On Position Mismatch</label>
                <select id="reconcile-action-config-select" style="width: 100%; padding: 0.5rem; border: 1px solid #ccc; border-radius: 4px;">
                  <option value="halt">Halt trading (safe)</option>
                  <option value="auto_heal">Auto-heal from exchange</option>
                </select>
              </div>
              <div>
                <label style="display: block; font-size: 0.85rem; color: #555; margin-bottom: 0.25rem;">Mismatch Tolerance (USD)</label>
                <input type="number" id="reconcile-tolerance-input" min="0" step="1" style="width: 100%; padding: 0.5rem; border: 1px solid #ccc; border-radius: 4px;">
              </div>
              <div style="display: flex; flex-direction: column; gap: 0.5rem; justify-content: center;">
                <label style="display: flex; align-items: center; gap: 0.5rem; font-size: 0.85rem; color: #555; cursor: pointer;">
                  <input type="checkbox" id="reconcile-on-startup-checkbox">
                  Run on startup
                </label>
                <label style="display: flex; align-items: center; gap: 0.5rem; font-size: 0.85rem; color: #555; cursor: pointer;">
                  <input type="checkbox" id="reconcile-on-each-loop-checkbox">
                  Run on each loop
                </label>
              </div>
            </div>
            <button onclick="saveReconciliationConfig()" style="margin-top: 1rem;">Apply Reconciliation Settings</button>
            <div id="reconcile-feedback" style="font-size: 0.8rem; min-height: 1.5rem; margin-top: 0.5rem;"></div>
          </fieldset>
          
        </div>
      </div>
    </div>
  </div>

  <!-- CHAT TAB -->
  <div id="tab-chat" class="tab-content">
    <div class="section">
      <h2>Chat with Agent</h2>
      <p style="color: #888; margin-bottom: 15px;">Ask questions about the bot's current state, positions, decisions, trading rules, or architecture.</p>
      
      <div id="chat-messages" style="height: 400px; overflow-y: auto; border: 1px solid #ddd; border-radius: 8px; padding: 15px; margin-bottom: 15px; background: #ffffff;">
        <div id="chat-log" style="display: flex; flex-direction: column; gap: 12px;">
          <div style="text-align: center; color: #888; padding: 20px;">Start a conversation by asking a question below...</div>
        </div>
      </div>
      
      <div style="display: flex; gap: 10px; align-items: flex-end;">
        <div style="flex: 1;">
          <textarea id="question" placeholder="e.g., What strategy are you running? Why is my PnL negative? Will you roll my 94k calls?" style="resize: none; height: 60px;"></textarea>
        </div>
        <div style="display: flex; flex-direction: column; gap: 5px;">
          <button id="ask-btn" onclick="sendQuestion()" style="height: 40px;">Ask Agent</button>
          <button onclick="clearChat()" style="height: 20px; font-size: 11px; background: #444; padding: 2px 8px;">Clear Chat</button>
        </div>
      </div>
    </div>
  </div>

  <script>
    let backtestResult = null;
    
    function showTab(name) {{
      document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
      document.querySelector(`[onclick="showTab('${{name}}')"]`).classList.add('active');
      document.getElementById(`tab-${{name}}`).classList.add('active');
      
      if (name === 'chat') {{
        loadChatHistory();
      }}
      if (name === 'backtesting') {{
        fetchBacktestRuns();
      }}
      if (name === 'health') {{
        loadSystemHealthStatus();
      }}
      if (name === 'strategies') {{
        loadBotsTab();
      }}
      if (name === 'greglab') {{
        loadGregPositions();
      }}
    }}
    
    function toggleBacktestSection(which) {{
      const bodyId = which === 'lab' ? 'backtest-lab-body' : 'backtest-runs-body';
      const iconId = which === 'lab' ? 'lab-toggle-icon' : 'runs-toggle-icon';
      const el = document.getElementById(bodyId);
      const iconEl = document.getElementById(iconId);
      if (!el) return;
      const isHidden = el.style.display === 'none';
      el.style.display = isHidden ? 'block' : 'none';
      if (iconEl) {{
        iconEl.innerHTML = isHidden ? '&#9660;' : '&#9654;';
      }}
    }}
    
    // ==============================================
    // BOTS TAB FUNCTIONS
    // ==============================================
    
    let botsStrategiesData = [];
    let currentExpertId = 'greg_mandolini';
    
    function formatSensorValue(val, decimals = 2) {{
      if (val === null || val === undefined) return '--';
      return typeof val === 'number' ? val.toFixed(decimals) : String(val);
    }}
    
    function classifySensorValue(sensorName, value) {{
      if (value === null || value === undefined || isNaN(value)) {{
        return "sensor-neutral";
      }}
      const v = Number(value);
      switch (sensorName) {{
        case "vrp_30d":
          if (v >= 10) return "sensor-good";
          if (v <= 0) return "sensor-bad";
          return "sensor-neutral";
        case "chop_factor_7d":
          if (v <= 0.6) return "sensor-good";
          if (v >= 0.9) return "sensor-bad";
          return "sensor-neutral";
        case "adx_14d":
          if (v <= 20) return "sensor-good";
          if (v >= 30) return "sensor-bad";
          return "sensor-neutral";
        case "rsi_14d":
          if (v < 30 || v > 70) return "sensor-bad";
          return "sensor-good";
        case "price_vs_ma200":
          if (v <= -15 || v >= 25) return "sensor-bad";
          return "sensor-neutral";
        case "iv_rank_6m":
          if (v >= 0.7) return "sensor-good";
          if (v <= 0.3) return "sensor-bad";
          return "sensor-neutral";
        case "term_structure_spread":
          if (v >= 5) return "sensor-bad";
          return "sensor-neutral";
        case "skew_25d":
          if (Math.abs(v) >= 5) return "sensor-bad";
          return "sensor-neutral";
        default:
          return "sensor-neutral";
      }}
    }}
    
    function formatCriterion(crit) {{
      const metric = crit.metric || "metric";
      const value = crit.value;
      const min = crit.min;
      const max = crit.max;
      const note = crit.note || "";
      
      if (value === null || value === undefined || (typeof value === 'number' && isNaN(value))) {{
        return `<span class="criterion-missing">❓ ${{metric}} missing${{note ? " (" + note + ")" : ""}}</span>`;
      }}
      
      const v = Number(value);
      let comparison = "";
      if (min != null && max != null) {{
        comparison = `${{v.toFixed(2)}} ∈ [${{min}}, ${{max}}]`;
      }} else if (min != null) {{
        const symbol = crit.ok ? "≥" : "<";
        comparison = `${{v.toFixed(2)}} ${{symbol}} ${{min}}`;
      }} else if (max != null) {{
        const symbol = crit.ok ? "≤" : ">";
        comparison = `${{v.toFixed(2)}} ${{symbol}} ${{max}}`;
      }} else {{
        comparison = v.toFixed(2);
      }}
      
      if (crit.ok) {{
        return `<span class="criterion-ok">✅ ${{metric}} ${{comparison}}</span>`;
      }} else {{
        return `<span class="criterion-bad">❌ ${{metric}} ${{comparison}}</span>`;
      }}
    }}
    
    async function refreshBotsSensors() {{
      const container = document.getElementById('bots-live-sensors');
      const debugPanel = document.getElementById('bots-debug-panel');
      const debugOutput = document.getElementById('bots-debug-output');
      const debugToggle = document.getElementById('bots-debug-toggle');
      
      container.innerHTML = '<p style="color: #666; font-style: italic;">Loading...</p>';
      
      const debugMode = debugToggle && debugToggle.checked ? '1' : '0';
      
      try {{
        const res = await fetch(`/api/bots/market_sensors?debug=${{debugMode}}`);
        const data = await res.json();
        
        if (data.ok) {{
          const sensors = data.sensors || {{}};
          const underlyings = Object.keys(sensors);
          
          let html = `<table style="width: 100%; border-collapse: collapse; font-size: 0.9rem;">
            <thead>
              <tr style="border-bottom: 2px solid #ddd;">
                <th style="text-align: left; padding: 0.5rem;">Sensor</th>
                ${{underlyings.map(u => `<th style="text-align: right; padding: 0.5rem;">${{u}}</th>`).join('')}}
              </tr>
            </thead>
            <tbody>`;
          
          const sensorNames = ['iv_30d', 'rv_30d', 'vrp_30d', 'chop_factor_7d', 'iv_rank_6m', 'term_structure_spread', 'skew_25d', 'adx_14d', 'rsi_14d', 'price_vs_ma200'];
          const sensorLabels = {{'iv_30d': 'IV 30d (DVOL)', 'rv_30d': 'RV 30d (Deribit)', 'vrp_30d': 'VRP 30d (IV-RV)', 'chop_factor_7d': 'Chop Factor 7d', 'iv_rank_6m': 'IV Rank 6m', 'term_structure_spread': 'Term Spread', 'skew_25d': 'Skew 25d', 'adx_14d': 'ADX 14d', 'rsi_14d': 'RSI 14d', 'price_vs_ma200': 'Price vs MA200'}};
          
          for (const name of sensorNames) {{
            html += `<tr style="border-bottom: 1px solid #eee;">
              <td style="padding: 0.5rem; color: #666;">${{sensorLabels[name] || name}}</td>
              ${{underlyings.map(u => {{
                const val = sensors[u] ? sensors[u][name] : null;
                const display = formatSensorValue(val, name === 'chop_factor_7d' ? 3 : 2);
                const sensorClass = classifySensorValue(name, val);
                return `<td style="text-align: right; padding: 0.5rem;" class="${{sensorClass}}">${{display}}</td>`;
              }}).join('')}}
            </tr>`;
          }}
          
          html += '</tbody></table>';
          container.innerHTML = html;
          
          // Handle debug panel
          if (data.debug_inputs && debugPanel && debugOutput) {{
            const debugLines = [];
            for (const [underlying, sensorDebug] of Object.entries(data.debug_inputs || {{}})) {{
              debugLines.push(`== ${{underlying}} ==`);
              for (const [name, payload] of Object.entries(sensorDebug || {{}})) {{
                const v = payload.value;
                const formula = payload.formula || '';
                const inputs = payload.inputs || {{}};
                const valStr = v !== null && v !== undefined ? Number(v).toFixed(4) : 'null';
                debugLines.push(`${{name}}: ${{valStr}}`);
                if (formula) debugLines.push(`  formula: ${{formula}}`);
                for (const [k, iv] of Object.entries(inputs)) {{
                  if (iv === null || iv === undefined) continue;
                  const ivStr = typeof iv === 'number' ? iv.toFixed(4) : String(iv);
                  debugLines.push(`  ${{k}}: ${{ivStr}}`);
                }}
              }}
              debugLines.push('');
            }}
            debugOutput.textContent = debugLines.join('\\n');
            debugPanel.style.display = 'block';
          }} else if (debugPanel) {{
            debugPanel.style.display = 'none';
            if (debugOutput) debugOutput.textContent = '';
          }}
        }} else {{
          container.innerHTML = `<p style="color: #c62828;">Error: ${{data.error}}</p>`;
        }}
      }} catch (e) {{
        container.innerHTML = `<p style="color: #c62828;">Error: ${{e.message}}</p>`;
      }}
    }}
    
    async function refreshBotsStrategies() {{
      const matchesContainer = document.getElementById('bots-strategy-matches');
      const expertContainer = document.getElementById('bots-expert-table');
      
      matchesContainer.innerHTML = '<p style="color: #666; font-style: italic;">Loading...</p>';
      expertContainer.innerHTML = '<p style="color: #666; font-style: italic;">Loading...</p>';
      
      try {{
        const res = await fetch('/api/bots/strategies');
        const data = await res.json();
        
        if (data.ok) {{
          botsStrategiesData = data.strategies || [];
          
          // Render Strategy Matches (passing strategies, with special handling for NO_TRADE)
          const passing = botsStrategiesData.filter(s => s.status === 'pass');
          
          if (passing.length === 0) {{
            matchesContainer.innerHTML = '<p style="color: #666; font-style: italic;">No strategies currently passing.</p>';
          }} else {{
            let html = `<table style="width: 100%; border-collapse: collapse; font-size: 0.9rem;">
              <thead>
                <tr style="border-bottom: 2px solid #ddd;">
                  <th style="text-align: left; padding: 0.5rem;">Bot</th>
                  <th style="text-align: left; padding: 0.5rem;">Underlying</th>
                  <th style="text-align: left; padding: 0.5rem;">Recommendation</th>
                  <th style="text-align: center; padding: 0.5rem;">Status</th>
                </tr>
              </thead>
              <tbody>`;
            
            for (const s of passing) {{
              const isNoTrade = s.strategy_key === 'NO_TRADE';
              const tooltip = isNoTrade 
                ? (s.debug?.description || s.summary || 'No favorable setup detected')
                : s.criteria.map(c => `${{c.metric}}: ${{formatSensorValue(c.value)}} (${{c.note || 'ok'}})`).join('; ');
              
              const statusColor = isNoTrade ? '#ff9800' : '#2e7d32';
              const statusLabel = isNoTrade ? 'CAUTION' : 'PREFERRED';
              const statusBg = isNoTrade ? '#fff3e0' : '#e8f5e9';
              const label = isNoTrade ? 'NO TRADE' : s.label;
              
              html += `<tr style="border-bottom: 1px solid #eee; background: ${{statusBg}};" title="${{tooltip}}">
                <td style="padding: 0.5rem;">${{s.bot_name}}</td>
                <td style="padding: 0.5rem;">${{s.underlying}}</td>
                <td style="padding: 0.5rem; font-weight: 600;">${{label}}</td>
                <td style="padding: 0.5rem; text-align: center;">
                  <span style="display: inline-block; padding: 0.2rem 0.5rem; border-radius: 4px; background: ${{statusColor}}; color: white; font-size: 0.75rem; font-weight: 600;">${{statusLabel}}</span>
                </td>
              </tr>`;
            }}
            
            html += '</tbody></table>';
            matchesContainer.innerHTML = html;
          }}
          
          // Render Expert Table
          renderExpertTable();
        }} else {{
          matchesContainer.innerHTML = `<p style="color: #c62828;">Error: ${{data.error}}</p>`;
          expertContainer.innerHTML = `<p style="color: #c62828;">Error: ${{data.error}}</p>`;
        }}
      }} catch (e) {{
        matchesContainer.innerHTML = `<p style="color: #c62828;">Error: ${{e.message}}</p>`;
        expertContainer.innerHTML = `<p style="color: #c62828;">Error: ${{e.message}}</p>`;
      }}
    }}
    
    function selectExpertTab(expertId) {{
      currentExpertId = expertId;
      
      // Update tab styling
      document.querySelectorAll('.bots-expert-tab').forEach(btn => {{
        if (btn.dataset.expertId === expertId) {{
          btn.style.background = '#7c4dff';
          btn.style.color = 'white';
        }} else {{
          btn.style.background = '#e0e0e0';
          btn.style.color = '#333';
        }}
      }});
      
      renderExpertTable();
    }}
    
    function renderExpertTable() {{
      const container = document.getElementById('bots-expert-table');
      
      // Get filter states
      const showPass = document.getElementById('filter-show-pass')?.checked ?? true;
      const showBlocked = document.getElementById('filter-show-blocked')?.checked ?? true;
      const showNoData = document.getElementById('filter-show-nodata')?.checked ?? true;
      
      // Filter by expert and status
      let filtered = botsStrategiesData.filter(s => s.expert_id === currentExpertId);
      filtered = filtered.filter(s => {{
        if (s.status === 'pass' && !showPass) return false;
        if (s.status === 'blocked' && !showBlocked) return false;
        if (s.status === 'no_data' && !showNoData) return false;
        return true;
      }});
      
      if (filtered.length === 0) {{
        container.innerHTML = '<p style="color: #666; font-style: italic;">No strategies match current filters.</p>';
        return;
      }}
      
      let html = `<table style="width: 100%; border-collapse: collapse; font-size: 0.9rem;">
        <thead>
          <tr style="border-bottom: 2px solid #ddd;">
            <th style="text-align: left; padding: 0.5rem;">Strategy</th>
            <th style="text-align: left; padding: 0.5rem;">Underlying</th>
            <th style="text-align: center; padding: 0.5rem;">Status</th>
            <th style="text-align: left; padding: 0.5rem;">Details</th>
          </tr>
        </thead>
        <tbody>`;
      
      for (const s of filtered) {{
        const isNoTrade = s.strategy_key === 'NO_TRADE';
        const statusColors = {{'pass': '#2e7d32', 'blocked': '#c62828', 'no_data': '#666'}};
        const statusLabels = {{'pass': 'PASS', 'blocked': 'BLOCKED', 'no_data': 'NO DATA'}};
        
        // Build detailed tooltip
        let tooltipLines = [];
        if (s.criteria && s.criteria.length > 0) {{
          for (const c of s.criteria) {{
            const valStr = c.value == null ? 'missing' : Number(c.value).toFixed(2);
            const min = c.min != null ? `min=${{c.min}}` : '';
            const max = c.max != null ? `max=${{c.max}}` : '';
            const status = c.ok ? 'PASS' : (c.value == null ? 'NO DATA' : 'FAIL');
            tooltipLines.push(`${{c.metric}}: ${{valStr}} ${{min}} ${{max}} – ${{status}}`);
          }}
        }} else if (isNoTrade) {{
          tooltipLines.push(s.summary || 'No favorable conditions detected');
        }}
        const tooltip = tooltipLines.join('\\n');
        
        // Build detailed HTML using formatCriterion
        const detailsHtml = s.criteria && s.criteria.length > 0 
          ? s.criteria.map(formatCriterion).join('; ')
          : (s.summary ? s.summary.substring(0, 50) : '');
        
        const displayLabel = isNoTrade ? 'No Trade' : s.label;
        const rowBg = isNoTrade && s.status === 'pass' ? '#fff3e0' : 'transparent';
        
        html += `<tr style="border-bottom: 1px solid #eee; background: ${{rowBg}};" title="${{tooltip}}">
          <td style="padding: 0.5rem;">${{displayLabel}}</td>
          <td style="padding: 0.5rem;">${{s.underlying}}</td>
          <td style="padding: 0.5rem; text-align: center; color: ${{statusColors[s.status]}}; font-weight: 600;">${{statusLabels[s.status]}}</td>
          <td style="padding: 0.5rem; font-size: 0.8rem;">${{detailsHtml}}</td>
        </tr>`;
      }}
      
      html += '</tbody></table>';
      container.innerHTML = html;
    }}
    
    // State for Bots tab navigation
    let currentBotsEnv = 'test';
    let currentBotId = 'gregbot';
    let currentBotsSection = 'overview';
    
    function loadBotsTab() {{
      refreshBotsSensors();
      refreshBotsStrategies();
      refreshGregCalibration();
      refreshGregManagement();
      updateBotsEnvUI();
      updateBotsSectionUI();
    }}
    
    function switchBotsEnv(env) {{
      currentBotsEnv = env;
      updateBotsEnvUI();
      // Reload current section data
      if (currentBotsSection === 'entry_rules') {{
        loadEntryRules();
      }} else if (currentBotsSection === 'global_risk') {{
        loadGlobalRisk();
      }} else if (currentBotsSection === 'bot_risk') {{
        loadBotRisk();
      }}
    }}
    
    function updateBotsEnvUI() {{
      const testBtn = document.getElementById('bots-env-test');
      const liveBtn = document.getElementById('bots-env-live');
      if (testBtn && liveBtn) {{
        if (currentBotsEnv === 'test') {{
          testBtn.style.background = '#7c4dff';
          testBtn.style.color = 'white';
          liveBtn.style.background = '#f5f5f5';
          liveBtn.style.color = '#333';
        }} else {{
          testBtn.style.background = '#f5f5f5';
          testBtn.style.color = '#333';
          liveBtn.style.background = '#7c4dff';
          liveBtn.style.color = 'white';
        }}
      }}
      // Update mode badges
      const entryBadge = document.getElementById('entry-rules-mode-badge');
      const globalBadge = document.getElementById('global-risk-mode-badge');
      const botBadge = document.getElementById('bot-risk-mode-badge');
      const badgeText = currentBotsEnv === 'test' ? 'TEST MODE' : 'LIVE MODE';
      const badgeBg = currentBotsEnv === 'test' ? '#7c4dff' : '#2e7d32';
      [entryBadge, globalBadge, botBadge].forEach(b => {{
        if (b) {{
          b.textContent = badgeText;
          b.style.background = badgeBg;
        }}
      }});
      // Hide override toggles and actions in LIVE mode
      const overrideToggles = ['entry-rules-override-toggle', 'global-risk-override-toggle', 'bot-risk-override-toggle'];
      const actionContainers = ['entry-rules-actions', 'global-risk-actions', 'bot-risk-actions'];
      overrideToggles.forEach(id => {{
        const el = document.getElementById(id);
        if (el) el.style.display = currentBotsEnv === 'test' ? 'block' : 'none';
      }});
      if (currentBotsEnv === 'live') {{
        actionContainers.forEach(id => {{
          const el = document.getElementById(id);
          if (el) el.style.display = 'none';
        }});
      }}
    }}
    
    function selectBotTab(botId) {{
      currentBotId = botId;
      // Update tab styling
      document.querySelectorAll('.bots-bot-tab').forEach(btn => {{
        if (btn.dataset.botId === botId) {{
          btn.style.background = '#7c4dff';
          btn.style.color = 'white';
          btn.classList.add('active');
        }} else {{
          btn.style.background = '#f5f5f5';
          btn.style.color = '#333';
          btn.classList.remove('active');
        }}
      }});
      // Reload section if it's bot-specific
      if (currentBotsSection === 'entry_rules') loadEntryRules();
      if (currentBotsSection === 'bot_risk') loadBotRisk();
    }}
    
    function selectBotsSection(section) {{
      currentBotsSection = section;
      updateBotsSectionUI();
      // Load data for the selected section
      if (section === 'entry_rules') loadEntryRules();
      if (section === 'global_risk') loadGlobalRisk();
      if (section === 'bot_risk') loadBotRisk();
    }}
    
    function updateBotsSectionUI() {{
      // Hide all sections
      document.querySelectorAll('.bots-section').forEach(sec => {{
        sec.style.display = 'none';
      }});
      // Show selected section
      const sectionEl = document.getElementById(`bots-section-${{currentBotsSection}}`);
      if (sectionEl) sectionEl.style.display = 'block';
      // Update sidebar styling
      document.querySelectorAll('.bots-sidebar-item').forEach(btn => {{
        if (btn.dataset.section === currentBotsSection) {{
          btn.style.background = '#7c4dff';
          btn.style.color = 'white';
          btn.classList.add('active');
        }} else {{
          btn.style.background = 'transparent';
          btn.style.color = '#333';
          btn.classList.remove('active');
        }}
      }});
    }}
    
    // Entry Rules functions
    async function loadEntryRules() {{
      const form = document.getElementById('entry-rules-form');
      const checkbox = document.getElementById('entry-rules-use-overrides');
      const actions = document.getElementById('entry-rules-actions');
      const status = document.getElementById('entry-rules-status');
      if (!form) return;
      
      form.innerHTML = '<p style="color: #666; font-style: italic;">Loading...</p>';
      if (status) status.textContent = '';
      
      try {{
        const res = await fetch(`/api/bots/${{currentBotId}}/entry_rules?env=${{currentBotsEnv}}`);
        const data = await res.json();
        
        if (data.ok) {{
          const rules = data.rules || [];
          const useOverrides = data.use_overrides || false;
          const isTest = currentBotsEnv === 'test';
          const canEdit = isTest && useOverrides;
          
          if (checkbox) checkbox.checked = useOverrides;
          if (actions) actions.style.display = canEdit ? 'block' : 'none';
          
          let html = '';
          if (!isTest) {{
            html += '<p style="color: #2e7d32; font-weight: 500; margin-bottom: 1rem; padding: 0.5rem; background: #e8f5e9; border-radius: 4px;">Read-only view of LIVE production values</p>';
          }}
          for (const r of rules) {{
            const inputDisabled = !canEdit ? 'disabled' : '';
            const inputStyle = !canEdit 
              ? 'width: 100%; padding: 0.5rem; border: 1px solid #ccc; border-radius: 4px; background: #f5f5f5; color: #666;'
              : 'width: 100%; padding: 0.5rem; border: 1px solid #ccc; border-radius: 4px;';
            html += `
              <div style="background: white; padding: 0.75rem; border-radius: 4px; border: 1px solid #ddd;">
                <label style="font-weight: 500; font-size: 0.9rem; display: block; margin-bottom: 0.25rem;">${{r.label}}</label>
                <input type="number" step="any" id="entry-${{r.key}}" data-key="${{r.key}}" value="${{r.current_value}}" ${{inputDisabled}}
                  style="${{inputStyle}}">
                <span style="font-size: 0.75rem; color: #888;">Default: ${{r.default_value}} ${{r.unit}}</span>
              </div>
            `;
          }}
          form.innerHTML = html || '<p style="color: #666;">No entry rules defined.</p>';
        }} else {{
          form.innerHTML = `<p style="color: #c62828;">Error: ${{data.error}}</p>`;
        }}
      }} catch (e) {{
        form.innerHTML = `<p style="color: #c62828;">Error: ${{e.message}}</p>`;
      }}
    }}
    
    async function saveEntryRules() {{
      const status = document.getElementById('entry-rules-status');
      const checkbox = document.getElementById('entry-rules-use-overrides');
      if (status) status.textContent = 'Saving...';
      
      const thresholds = {{}};
      document.querySelectorAll('#entry-rules-form input[data-key]').forEach(inp => {{
        thresholds[inp.dataset.key] = parseFloat(inp.value) || 0;
      }});
      
      try {{
        const res = await fetch(`/api/bots/${{currentBotId}}/entry_rules`, {{
          method: 'POST',
          headers: {{'Content-Type': 'application/json'}},
          body: JSON.stringify({{
            use_overrides: checkbox ? checkbox.checked : true,
            thresholds: thresholds
          }})
        }});
        const data = await res.json();
        if (status) {{
          status.textContent = data.ok ? 'Saved successfully!' : 'Save failed: ' + (data.error || 'Unknown error');
          status.style.color = data.ok ? '#2e7d32' : '#c62828';
        }}
      }} catch (e) {{
        if (status) {{
          status.textContent = 'Error: ' + e.message;
          status.style.color = '#c62828';
        }}
      }}
    }}
    
    async function resetEntryRules() {{
      const status = document.getElementById('entry-rules-status');
      const checkbox = document.getElementById('entry-rules-use-overrides');
      if (status) status.textContent = 'Resetting...';
      
      try {{
        const res = await fetch(`/api/bots/${{currentBotId}}/entry_rules`, {{
          method: 'POST',
          headers: {{'Content-Type': 'application/json'}},
          body: JSON.stringify({{
            use_overrides: false,
            thresholds: {{}}
          }})
        }});
        const data = await res.json();
        if (data.ok) {{
          if (checkbox) checkbox.checked = false;
          loadEntryRules();
          if (status) {{
            status.textContent = 'Reset to defaults.';
            status.style.color = '#666';
          }}
        }}
      }} catch (e) {{
        if (status) {{
          status.textContent = 'Error: ' + e.message;
          status.style.color = '#c62828';
        }}
      }}
    }}
    
    function toggleEntryRulesOverrides() {{
      const checkbox = document.getElementById('entry-rules-use-overrides');
      const actions = document.getElementById('entry-rules-actions');
      const useOverrides = checkbox ? checkbox.checked : false;
      
      // Enable/disable inputs
      document.querySelectorAll('#entry-rules-form input[data-key]').forEach(inp => {{
        inp.disabled = !useOverrides;
      }});
      // Show/hide action buttons
      if (actions) actions.style.display = useOverrides ? 'block' : 'none';
    }}
    
    // Global Risk functions
    async function loadGlobalRisk() {{
      const form = document.getElementById('global-risk-form');
      const checkbox = document.getElementById('global-risk-use-overrides');
      const actions = document.getElementById('global-risk-actions');
      const status = document.getElementById('global-risk-status');
      if (!form) return;
      
      form.innerHTML = '<p style="color: #666; font-style: italic;">Loading...</p>';
      if (status) status.textContent = '';
      
      try {{
        const res = await fetch(`/api/bots/global_risk?env=${{currentBotsEnv}}`);
        const data = await res.json();
        
        if (data.ok) {{
          const fields = data.fields || [];
          const useOverrides = data.use_overrides || false;
          const isTest = currentBotsEnv === 'test';
          const canEdit = isTest && useOverrides;
          
          if (checkbox) checkbox.checked = useOverrides;
          if (actions) actions.style.display = canEdit ? 'block' : 'none';
          
          let html = '';
          if (!isTest) {{
            html += '<p style="color: #2e7d32; font-weight: 500; margin-bottom: 1rem; padding: 0.5rem; background: #e8f5e9; border-radius: 4px;">Read-only view of LIVE production values</p>';
          }}
          for (const f of fields) {{
            const inputDisabled = !canEdit ? 'disabled' : '';
            const inputStyle = !canEdit 
              ? 'width: 100%; padding: 0.5rem; border: 1px solid #ccc; border-radius: 4px; background: #f5f5f5; color: #666;'
              : 'width: 100%; padding: 0.5rem; border: 1px solid #ccc; border-radius: 4px;';
            html += `
              <div style="background: white; padding: 0.75rem; border-radius: 4px; border: 1px solid #ddd;">
                <label style="font-weight: 500; font-size: 0.9rem; display: block; margin-bottom: 0.25rem;">${{f.label}}</label>
                <input type="number" step="any" id="global-${{f.key}}" data-key="${{f.key}}" value="${{f.current_value}}" ${{inputDisabled}}
                  style="${{inputStyle}}">
                <span style="font-size: 0.75rem; color: #888;">Default: ${{f.default_value}} ${{f.unit}}</span>
              </div>
            `;
          }}
          form.innerHTML = html || '<p style="color: #666;">No global risk settings defined.</p>';
        }} else {{
          form.innerHTML = `<p style="color: #c62828;">Error: ${{data.error}}</p>`;
        }}
      }} catch (e) {{
        form.innerHTML = `<p style="color: #c62828;">Error: ${{e.message}}</p>`;
      }}
    }}
    
    async function saveGlobalRisk() {{
      const status = document.getElementById('global-risk-status');
      const checkbox = document.getElementById('global-risk-use-overrides');
      if (status) status.textContent = 'Saving...';
      
      const fields = {{}};
      document.querySelectorAll('#global-risk-form input[data-key]').forEach(inp => {{
        fields[inp.dataset.key] = parseFloat(inp.value) || 0;
      }});
      
      try {{
        const res = await fetch('/api/bots/global_risk', {{
          method: 'POST',
          headers: {{'Content-Type': 'application/json'}},
          body: JSON.stringify({{
            use_overrides: checkbox ? checkbox.checked : true,
            fields: fields
          }})
        }});
        const data = await res.json();
        if (status) {{
          status.textContent = data.ok ? 'Saved successfully!' : 'Save failed: ' + (data.error || 'Unknown error');
          status.style.color = data.ok ? '#2e7d32' : '#c62828';
        }}
      }} catch (e) {{
        if (status) {{
          status.textContent = 'Error: ' + e.message;
          status.style.color = '#c62828';
        }}
      }}
    }}
    
    async function resetGlobalRisk() {{
      const status = document.getElementById('global-risk-status');
      const checkbox = document.getElementById('global-risk-use-overrides');
      if (status) status.textContent = 'Resetting...';
      
      try {{
        const res = await fetch('/api/bots/global_risk', {{
          method: 'POST',
          headers: {{'Content-Type': 'application/json'}},
          body: JSON.stringify({{
            use_overrides: false,
            fields: {{}}
          }})
        }});
        const data = await res.json();
        if (data.ok) {{
          if (checkbox) checkbox.checked = false;
          loadGlobalRisk();
          if (status) {{
            status.textContent = 'Reset to defaults.';
            status.style.color = '#666';
          }}
        }}
      }} catch (e) {{
        if (status) {{
          status.textContent = 'Error: ' + e.message;
          status.style.color = '#c62828';
        }}
      }}
    }}
    
    function toggleGlobalRiskOverrides() {{
      const checkbox = document.getElementById('global-risk-use-overrides');
      const actions = document.getElementById('global-risk-actions');
      const useOverrides = checkbox ? checkbox.checked : false;
      
      document.querySelectorAll('#global-risk-form input[data-key]').forEach(inp => {{
        inp.disabled = !useOverrides;
      }});
      if (actions) actions.style.display = useOverrides ? 'block' : 'none';
    }}
    
    // Bot Risk functions
    async function loadBotRisk() {{
      const form = document.getElementById('bot-risk-form');
      const checkbox = document.getElementById('bot-risk-use-overrides');
      const actions = document.getElementById('bot-risk-actions');
      const status = document.getElementById('bot-risk-status');
      if (!form) return;
      
      form.innerHTML = '<p style="color: #666; font-style: italic;">Loading...</p>';
      if (status) status.textContent = '';
      
      try {{
        const res = await fetch(`/api/bots/${{currentBotId}}/risk?env=${{currentBotsEnv}}`);
        const data = await res.json();
        
        if (data.ok) {{
          const fields = data.fields || [];
          const useOverrides = data.use_overrides || false;
          const isTest = currentBotsEnv === 'test';
          const canEdit = isTest && useOverrides;
          
          if (checkbox) checkbox.checked = useOverrides;
          if (actions) actions.style.display = canEdit ? 'block' : 'none';
          
          let html = '';
          if (!isTest) {{
            html += '<p style="color: #2e7d32; font-weight: 500; margin-bottom: 1rem; padding: 0.5rem; background: #e8f5e9; border-radius: 4px;">Read-only view of LIVE production values</p>';
          }}
          for (const f of fields) {{
            const inputDisabled = !canEdit ? 'disabled' : '';
            const inputStyle = !canEdit 
              ? 'width: 100%; padding: 0.5rem; border: 1px solid #ccc; border-radius: 4px; background: #f5f5f5; color: #666;'
              : 'width: 100%; padding: 0.5rem; border: 1px solid #ccc; border-radius: 4px;';
            html += `
              <div style="background: white; padding: 0.75rem; border-radius: 4px; border: 1px solid #ddd;">
                <label style="font-weight: 500; font-size: 0.9rem; display: block; margin-bottom: 0.25rem;">${{f.label}}</label>
                <input type="number" step="any" id="botrisk-${{f.key}}" data-key="${{f.key}}" value="${{f.current_value}}" ${{inputDisabled}}
                  style="${{inputStyle}}">
                <span style="font-size: 0.75rem; color: #888;">Default: ${{f.default_value}} ${{f.unit}}</span>
              </div>
            `;
          }}
          form.innerHTML = html || '<p style="color: #666;">No bot risk settings defined.</p>';
        }} else {{
          form.innerHTML = `<p style="color: #c62828;">Error: ${{data.error}}</p>`;
        }}
      }} catch (e) {{
        form.innerHTML = `<p style="color: #c62828;">Error: ${{e.message}}</p>`;
      }}
    }}
    
    async function saveBotRisk() {{
      const status = document.getElementById('bot-risk-status');
      const checkbox = document.getElementById('bot-risk-use-overrides');
      if (status) status.textContent = 'Saving...';
      
      const fields = {{}};
      document.querySelectorAll('#bot-risk-form input[data-key]').forEach(inp => {{
        fields[inp.dataset.key] = parseFloat(inp.value) || 0;
      }});
      
      try {{
        const res = await fetch(`/api/bots/${{currentBotId}}/risk`, {{
          method: 'POST',
          headers: {{'Content-Type': 'application/json'}},
          body: JSON.stringify({{
            use_overrides: checkbox ? checkbox.checked : true,
            fields: fields
          }})
        }});
        const data = await res.json();
        if (status) {{
          status.textContent = data.ok ? 'Saved successfully!' : 'Save failed: ' + (data.error || 'Unknown error');
          status.style.color = data.ok ? '#2e7d32' : '#c62828';
        }}
      }} catch (e) {{
        if (status) {{
          status.textContent = 'Error: ' + e.message;
          status.style.color = '#c62828';
        }}
      }}
    }}
    
    async function resetBotRisk() {{
      const status = document.getElementById('bot-risk-status');
      const checkbox = document.getElementById('bot-risk-use-overrides');
      if (status) status.textContent = 'Resetting...';
      
      try {{
        const res = await fetch(`/api/bots/${{currentBotId}}/risk`, {{
          method: 'POST',
          headers: {{'Content-Type': 'application/json'}},
          body: JSON.stringify({{
            use_overrides: false,
            fields: {{}}
          }})
        }});
        const data = await res.json();
        if (data.ok) {{
          if (checkbox) checkbox.checked = false;
          loadBotRisk();
          if (status) {{
            status.textContent = 'Reset to defaults.';
            status.style.color = '#666';
          }}
        }}
      }} catch (e) {{
        if (status) {{
          status.textContent = 'Error: ' + e.message;
          status.style.color = '#c62828';
        }}
      }}
    }}
    
    function toggleBotRiskOverrides() {{
      const checkbox = document.getElementById('bot-risk-use-overrides');
      const actions = document.getElementById('bot-risk-actions');
      const useOverrides = checkbox ? checkbox.checked : false;
      
      document.querySelectorAll('#bot-risk-form input[data-key]').forEach(inp => {{
        inp.disabled = !useOverrides;
      }});
      if (actions) actions.style.display = useOverrides ? 'block' : 'none';
    }}
    
    async function refreshGregCalibration() {{
      const statusEl = document.getElementById('greg-calibration-status');
      const valuesEl = document.getElementById('greg-calibration-values');
      
      if (!statusEl || !valuesEl) return;
      
      statusEl.textContent = 'Loading...';
      valuesEl.innerHTML = '';
      
      try {{
        const res = await fetch('/api/greg/calibration');
        const data = await res.json();
        
        if (data.ok) {{
          const version = data.version || 'unknown';
          const calibVersion = data.calibration_version || 'unknown';
          statusEl.innerHTML = `<strong>Version:</strong> ${{version}} | <strong>Calibration:</strong> ${{calibVersion}}`;
          
          const calib = data.calibration || {{}};
          const keyGroups = {{
            'Core': ['skew_neutral_threshold', 'min_vrp_floor'],
            'Safety': ['safety_adx_high', 'safety_chop_high'],
            'Straddle': ['straddle_vrp_min', 'straddle_adx_max', 'straddle_chop_max'],
            'Strangle': ['strangle_vrp_min', 'strangle_adx_max', 'strangle_chop_max'],
            'Calendar': ['calendar_term_spread_min', 'calendar_front_rv_iv_ratio_max'],
            'Iron Fly': ['iron_fly_iv_rank_min', 'iron_fly_vrp_min'],
            'Directional': ['short_put_iv_rank_min', 'short_put_price_vs_ma200_min', 'bull_put_rsi_max', 'bear_call_rsi_min'],
          }};
          
          let html = '<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem;">';
          for (const [group, keys] of Object.entries(keyGroups)) {{
            html += `<div><strong style="color: #7c4dff; font-size: 0.85rem;">${{group}}</strong><ul style="margin: 0.25rem 0 0; padding-left: 1rem; font-size: 0.8rem;">`;
            for (const k of keys) {{
              const v = calib[k];
              if (v !== undefined) {{
                html += `<li style="color: #555;"><code>${{k.replace(/_/g, ' ')}}</code>: ${{v}}</li>`;
              }}
            }}
            html += '</ul></div>';
          }}
          html += '</div>';
          valuesEl.innerHTML = html;
        }} else {{
          statusEl.textContent = 'Error: ' + (data.error || 'Unknown error');
        }}
      }} catch (e) {{
        statusEl.textContent = 'Error: ' + e.message;
      }}
    }}
    
    // ==============================================
    // GREG POSITION MANAGEMENT FUNCTIONS
    // ==============================================
    
    let gregTradingMode = 'advice_only';
    let gregEnableLive = false;
    let gregStrategyFlags = {{}};
    let gregMaxNotionalPos = 500;
    let gregMaxNotionalUnd = 2000;
    
    async function loadGregTradingMode() {{
      try {{
        const res = await fetch('/api/greg/trading_mode');
        const data = await res.json();
        if (data.ok) {{
          gregTradingMode = data.mode;
          gregEnableLive = data.enable_live_execution;
          gregStrategyFlags = data.strategy_live_enabled || {{}};
          gregMaxNotionalPos = data.max_notional_per_position || 500;
          gregMaxNotionalUnd = data.max_notional_per_underlying || 2000;
          updateGregModeUI();
        }}
      }} catch (e) {{
        console.error('Failed to load Greg trading mode:', e);
      }}
    }}
    
    function updateGregModeUI() {{
      const badge = document.getElementById('greg-mode-badge');
      const subtitle = document.getElementById('greg-pm-subtitle');
      
      if (badge) {{
        const modeConfig = {{
          'advice_only': {{ text: 'ADVICE ONLY', bg: '#e0e0e0', color: '#555' }},
          'paper': {{ text: 'PAPER MODE', bg: '#e3f2fd', color: '#1565c0' }},
          'live': {{ text: 'LIVE MODE', bg: '#ffebee', color: '#c62828' }},
        }};
        const cfg = modeConfig[gregTradingMode] || modeConfig['advice_only'];
        badge.textContent = cfg.text;
        badge.style.background = cfg.bg;
        badge.style.color = cfg.color;
      }}
      
      if (subtitle) {{
        const subtitles = {{
          'advice_only': 'Greg-style management suggestions for open positions. <strong>Advisory only</strong> - no real orders sent.',
          'paper': 'Paper mode active. Execute buttons will send orders to <strong>testnet/DRY_RUN</strong> pipeline.',
          'live': '<strong style="color: #c62828;">LIVE MODE</strong> - Execute buttons will send real orders. Trade carefully!',
        }};
        subtitle.innerHTML = subtitles[gregTradingMode] || subtitles['advice_only'];
      }}
    }}
    
    function openGregModeSettings() {{
      loadGregTradingMode().then(() => {{
        document.getElementById('greg-mode-modal').style.display = 'flex';
        document.getElementById('greg-mode-select').value = gregTradingMode;
        document.getElementById('greg-enable-live').checked = gregEnableLive;
        document.getElementById('greg-max-notional-pos').value = gregMaxNotionalPos;
        document.getElementById('greg-max-notional-und').value = gregMaxNotionalUnd;
        updateModeSettingsUI();
        
        let togglesHtml = '';
        for (const [strat, enabled] of Object.entries(gregStrategyFlags)) {{
          const label = strat.replace(/_/g, ' ').replace('STRATEGY ', '');
          togglesHtml += `<label style="display: flex; align-items: center; gap: 0.5rem; margin: 0.25rem 0;">
            <input type="checkbox" class="greg-strat-toggle" data-strategy="${{strat}}" ${{enabled ? 'checked' : ''}}>
            <span>${{label}}</span>
          </label>`;
        }}
        document.getElementById('greg-strategy-toggles').innerHTML = togglesHtml;
      }});
    }}
    
    function updateModeSettingsUI() {{
      const mode = document.getElementById('greg-mode-select').value;
      document.getElementById('greg-live-settings').style.display = mode === 'live' ? 'block' : 'none';
      document.getElementById('greg-live-confirm').style.display = mode === 'live' && gregTradingMode !== 'live' ? 'block' : 'none';
    }}
    
    document.getElementById('greg-mode-select')?.addEventListener('change', updateModeSettingsUI);
    
    function closeGregModeSettings() {{
      document.getElementById('greg-mode-modal').style.display = 'none';
    }}
    
    async function saveGregModeSettings() {{
      const mode = document.getElementById('greg-mode-select').value;
      const enableLive = document.getElementById('greg-enable-live').checked;
      const confirmText = document.getElementById('greg-live-confirm-input').value;
      const maxNotionalPos = parseFloat(document.getElementById('greg-max-notional-pos').value) || 500;
      const maxNotionalUnd = parseFloat(document.getElementById('greg-max-notional-und').value) || 2000;
      
      const strategyFlags = {{}};
      document.querySelectorAll('.greg-strat-toggle').forEach(cb => {{
        strategyFlags[cb.dataset.strategy] = cb.checked;
      }});
      
      try {{
        const res = await fetch('/api/greg/trading_mode', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify({{
            mode: mode,
            enable_live_execution: enableLive,
            strategy_live_enabled: strategyFlags,
            max_notional_per_position: maxNotionalPos,
            max_notional_per_underlying: maxNotionalUnd,
            confirmation_text: confirmText,
          }}),
        }});
        const data = await res.json();
        
        if (data.ok) {{
          gregTradingMode = data.current_mode;
          gregEnableLive = data.current_enable_live;
          gregStrategyFlags = data.current_strategy_flags;
          gregMaxNotionalPos = data.max_notional_per_position || gregMaxNotionalPos;
          gregMaxNotionalUnd = data.max_notional_per_underlying || gregMaxNotionalUnd;
          updateGregModeUI();
          closeGregModeSettings();
          alert('Settings saved successfully');
        }} else {{
          alert('Error: ' + (data.error || 'Unknown error'));
        }}
      }} catch (e) {{
        alert('Error: ' + e.message);
      }}
    }}
    
    async function refreshGregManagement() {{
      const tbody = document.getElementById('greg-management-tbody');
      const statusEl = document.getElementById('greg-management-status');
      
      if (!tbody || !statusEl) return;
      
      statusEl.textContent = 'Loading...';
      await loadGregTradingMode();
      
      try {{
        const res = await fetch('/api/bots/greg/management');
        const data = await res.json();
        
        if (data.ok) {{
          renderGregManagementTable(data.suggestions || [], tbody, false);
          const count = data.count || 0;
          const updatedAt = data.updated_at ? new Date(data.updated_at).toLocaleTimeString() : 'N/A';
          statusEl.innerHTML = `Loaded ${{count}} suggestion(s). Rules v${{data.rules_version || 'unknown'}}. Updated: ${{updatedAt}}`;
        }} else {{
          statusEl.textContent = 'Error: ' + (data.error || 'Unknown error');
          tbody.innerHTML = '<tr><td colspan="7" style="padding: 1rem; color: #c62828; text-align: center;">Error loading suggestions</td></tr>';
        }}
      }} catch (e) {{
        statusEl.textContent = 'Error: ' + e.message;
        tbody.innerHTML = '<tr><td colspan="7" style="padding: 1rem; color: #c62828; text-align: center;">Error: ' + e.message + '</td></tr>';
      }}
    }}
    
    async function loadMockGregManagement() {{
      const tbody = document.getElementById('greg-management-tbody');
      const statusEl = document.getElementById('greg-management-status');
      
      if (!tbody || !statusEl) return;
      
      statusEl.textContent = 'Loading demo positions...';
      await loadGregTradingMode();
      
      try {{
        const res = await fetch('/api/bots/greg/management/mock', {{ method: 'POST' }});
        const data = await res.json();
        
        if (data.ok) {{
          renderGregManagementTable(data.suggestions || [], tbody, true);
          statusEl.innerHTML = `<span style="color: #f57c00;">Demo mode:</span> Loaded ${{data.count || 0}} sample suggestion(s).`;
        }} else {{
          statusEl.textContent = 'Error: ' + (data.error || 'Unknown error');
        }}
      }} catch (e) {{
        statusEl.textContent = 'Error: ' + e.message;
      }}
    }}
    
    function renderGregManagementTable(suggestions, tbody, isDemo = false) {{
      if (!suggestions || suggestions.length === 0) {{
        tbody.innerHTML = '<tr><td colspan="7" style="padding: 1rem; color: #666; font-style: italic; text-align: center;">No management suggestions available.</td></tr>';
        return;
      }}
      
      const actionColors = {{
        'HEDGE': {{ bg: '#fff3e0', color: '#e65100', label: 'Hedge', priority: 'medium' }},
        'TAKE_PROFIT': {{ bg: '#e8f5e9', color: '#2e7d32', label: 'Take Profit', priority: 'medium' }},
        'ROLL': {{ bg: '#fff8e1', color: '#f9a825', label: 'Roll', priority: 'low' }},
        'ASSIGN': {{ bg: '#fce4ec', color: '#c62828', label: 'Assign', priority: 'medium' }},
        'CLOSE': {{ bg: '#ffebee', color: '#c62828', label: 'Close', priority: 'high' }},
        'HOLD': {{ bg: '#f5f5f5', color: '#666', label: 'Hold', priority: 'low' }},
      }};
      
      const priorityColors = {{
        'high': {{ border: '#c62828', tooltip: 'HIGH - must act now' }},
        'medium': {{ border: '#ff9800', tooltip: 'MED - suggested soon' }},
        'low': {{ border: '#4caf50', tooltip: 'LOW - informational' }},
      }};
      
      const strategyLabels = {{
        'STRATEGY_A_STRADDLE': 'ATM Straddle (A)',
        'STRATEGY_A_STRANGLE': 'ATM Strangle (A)',
        'STRATEGY_B_CALENDAR': 'Calendar Spread (B)',
        'STRATEGY_C_SHORT_PUT': 'Short Put (C)',
        'STRATEGY_D_IRON_BUTTERFLY': 'Iron Butterfly (D)',
        'STRATEGY_F_BULL_PUT_SPREAD': 'Bull Put Spread (F)',
        'STRATEGY_F_BEAR_CALL_SPREAD': 'Bear Call Spread (F)',
      }};
      
      const canExecute = gregTradingMode !== 'advice_only';
      
      let html = '';
      for (const s of suggestions) {{
        const actionStyle = actionColors[s.action] || {{ bg: '#f5f5f5', color: '#333', label: s.action, priority: 'low' }};
        const stratLabel = strategyLabels[s.strategy_code] || s.strategy_code;
        
        const isPositionDemo = s.position_id.startsWith('demo:');
        const typeBadge = isPositionDemo 
          ? '<span style="background: #e0e0e0; color: #666; padding: 0.15rem 0.4rem; border-radius: 3px; font-size: 0.7rem;">DEMO</span>'
          : '<span style="background: #e3f2fd; color: #1565c0; padding: 0.15rem 0.4rem; border-radius: 3px; font-size: 0.7rem;">LIVE</span>';
        
        const priority = determinePriority(s, actionStyle.priority);
        const prioStyle = priorityColors[priority];
        
        const metrics = s.metrics || {{}};
        let metricsHtml = [];
        
        if (metrics.net_delta !== undefined) {{
          metricsHtml.push(`<span title="Net Delta">Δ=${{metrics.net_delta.toFixed(2)}}</span>`);
        }}
        if (metrics.target_delta_abs !== undefined) {{
          metricsHtml.push(`<span title="Target Delta Threshold">(tgt: ${{metrics.target_delta_abs}})</span>`);
        }}
        if (metrics.dte !== undefined) {{
          metricsHtml.push(`<span title="Days to Expiry">DTE=${{metrics.dte}}</span>`);
        }}
        if (metrics.profit_pct !== undefined) {{
          const pnlColor = metrics.profit_pct >= 0 ? '#2e7d32' : '#c62828';
          metricsHtml.push(`<span style="color: ${{pnlColor}};" title="Profit %">PnL=${{(metrics.profit_pct * 100).toFixed(1)}}%</span>`);
        }}
        if (metrics.delta_abs !== undefined) {{
          metricsHtml.push(`<span title="Absolute Delta">|Δ|=${{metrics.delta_abs.toFixed(2)}}</span>`);
        }}
        
        const executeBtn = canExecute && s.action !== 'HOLD'
          ? `<button onclick="executeSuggestion('${{s.position_id}}', '${{s.action}}', '${{s.strategy_code}}', '${{s.underlying}}')" 
               style="padding: 0.3rem 0.6rem; font-size: 0.75rem; background: ${{actionStyle.color}}; color: white; border: none; border-radius: 3px; cursor: pointer;">
               Execute
             </button>`
          : '<span style="color: #999; font-size: 0.75rem;">-</span>';
        
        const hedgeLink = s.action === 'HEDGE' 
          ? `<a href="#greg-hedging-panel" onclick="scrollToHedgeRow('${{s.position_id}}')" style="font-size: 0.7rem; color: #1565c0; margin-left: 0.25rem;">View Hedge</a>`
          : '';
        
        html += `<tr style="border-bottom: 1px solid #eee; border-left: 3px solid ${{prioStyle.border}};" title="${{prioStyle.tooltip}}: ${{s.reason}}">
          <td style="padding: 0.5rem;">${{typeBadge}}</td>
          <td style="padding: 0.5rem; font-weight: 600;">${{s.underlying}}</td>
          <td style="padding: 0.5rem;">${{stratLabel}}</td>
          <td style="padding: 0.5rem; font-size: 0.8rem; color: #666;">${{s.position_id.substring(0, 25)}}...</td>
          <td style="padding: 0.5rem;">
            <span style="background: ${{actionStyle.bg}}; color: ${{actionStyle.color}}; padding: 0.2rem 0.5rem; border-radius: 4px; font-weight: 600; font-size: 0.8rem;">
              ${{actionStyle.label}}
            </span>
            ${{hedgeLink}}
            <div style="font-size: 0.75rem; color: #666; margin-top: 0.25rem; max-width: 180px;">${{s.summary}}</div>
          </td>
          <td style="padding: 0.5rem; font-size: 0.8rem;">${{metricsHtml.join(' | ')}}</td>
          <td style="padding: 0.5rem;">${{executeBtn}}</td>
        </tr>`;
      }}
      
      tbody.innerHTML = html;
    }}
    
    function determinePriority(suggestion, defaultPriority) {{
      const metrics = suggestion.metrics || {{}};
      const action = suggestion.action;
      
      if (action === 'CLOSE' || metrics.loss_pct >= 2.0) return 'high';
      if (metrics.net_delta !== undefined && Math.abs(metrics.net_delta) > 0.3) return 'high';
      
      if (action === 'TAKE_PROFIT' && metrics.profit_pct >= 0.6) return 'medium';
      if (action === 'ASSIGN' && metrics.delta_abs >= 0.8) return 'medium';
      if (action === 'HEDGE') return 'medium';
      
      return defaultPriority || 'low';
    }}
    
    async function executeSuggestion(positionId, action, strategyType, underlying) {{
      if (!confirm(`Execute ${{action}} for ${{underlying}} position?\\n\\nMode: ${{gregTradingMode.toUpperCase()}}\\nPosition: ${{positionId}}`)) {{
        return;
      }}
      
      try {{
        const res = await fetch('/api/bots/greg/execute_suggestion', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify({{
            position_id: positionId,
            suggested_action: action,
            strategy_type: strategyType,
            underlying: underlying,
          }}),
        }});
        const data = await res.json();
        
        if (data.ok) {{
          alert(`Executed ${{action}} successfully!\\nMode: ${{data.mode}}\\nResult: ${{JSON.stringify(data.result, null, 2)}}`);
          refreshGregManagement();
        }} else {{
          alert('Execution failed: ' + (data.error || 'Unknown error'));
        }}
      }} catch (e) {{
        alert('Execution error: ' + e.message);
      }}
    }}
    
    function scrollToHedgeRow(positionId) {{
      const hedgePanel = document.getElementById('greg-hedging-panel');
      if (hedgePanel) {{
        hedgePanel.scrollIntoView({{ behavior: 'smooth' }});
        evaluateHedging();
      }}
    }}
    
    // ==============================================
    // DELTA HEDGING ENGINE FUNCTIONS
    // ==============================================
    
    async function refreshHedgingStatus() {{
      const rulesTbody = document.getElementById('hedge-rules-tbody');
      const historyTbody = document.getElementById('hedge-history-tbody');
      const statusEl = document.getElementById('hedge-status');
      const dryRunBadge = document.getElementById('hedge-dry-run-badge');
      
      try {{
        const res = await fetch('/api/bots/greg/hedging');
        const data = await res.json();
        
        if (data.ok) {{
          // Update dry run badge
          if (data.dry_run) {{
            dryRunBadge.textContent = 'DRY RUN';
            dryRunBadge.style.background = '#fff3e0';
            dryRunBadge.style.color = '#e65100';
          }} else {{
            dryRunBadge.textContent = 'LIVE';
            dryRunBadge.style.background = '#ffebee';
            dryRunBadge.style.color = '#c62828';
          }}
          
          // Render rules table
          if (rulesTbody && data.strategies) {{
            const hedgeModeColors = {{
              'DYNAMIC_DELTA': {{ bg: '#e3f2fd', color: '#1565c0', label: 'Dynamic Delta' }},
              'LIGHT_DELTA': {{ bg: '#e8f5e9', color: '#2e7d32', label: 'Light Delta' }},
              'LOOSE_DELTA': {{ bg: '#fff8e1', color: '#f9a825', label: 'Loose Delta' }},
              'NONE': {{ bg: '#f5f5f5', color: '#666', label: 'None' }},
            }};
            
            let rulesHtml = '';
            for (const s of data.strategies) {{
              const modeStyle = hedgeModeColors[s.hedge_mode] || {{ bg: '#f5f5f5', color: '#333', label: s.hedge_mode }};
              const threshold = s.delta_threshold !== null ? s.delta_threshold.toFixed(2) : '-';
              rulesHtml += `<tr>
                <td style="padding: 0.4rem;">${{s.display_name || s.strategy}}</td>
                <td style="padding: 0.4rem;">
                  <span style="background: ${{modeStyle.bg}}; color: ${{modeStyle.color}}; padding: 0.15rem 0.4rem; border-radius: 3px; font-size: 0.75rem;">
                    ${{modeStyle.label}}
                  </span>
                </td>
                <td style="padding: 0.4rem;">${{threshold}}</td>
              </tr>`;
            }}
            rulesTbody.innerHTML = rulesHtml || '<tr><td colspan="3" style="padding: 0.5rem; color: #666;">No strategies found.</td></tr>';
          }}
          
          // Render history table
          if (historyTbody && data.history) {{
            if (data.history.length === 0) {{
              historyTbody.innerHTML = '<tr><td colspan="4" style="padding: 0.5rem; color: #666; font-style: italic;">No hedge history yet.</td></tr>';
            }} else {{
              let historyHtml = '';
              for (const h of data.history) {{
                const order = h.order || {{}};
                const time = h.order?.timestamp ? new Date(h.order.timestamp).toLocaleTimeString() : 'N/A';
                const orderStr = order.side ? `${{order.side.toUpperCase()}} ${{order.size}} ${{order.instrument}}` : '-';
                const statusBadge = h.dry_run ? 
                  '<span style="background: #fff3e0; color: #e65100; padding: 0.1rem 0.3rem; border-radius: 3px; font-size: 0.7rem;">DRY RUN</span>' :
                  (h.executed ? '<span style="background: #e8f5e9; color: #2e7d32; padding: 0.1rem 0.3rem; border-radius: 3px; font-size: 0.7rem;">EXECUTED</span>' : 
                   '<span style="background: #f5f5f5; color: #666; padding: 0.1rem 0.3rem; border-radius: 3px; font-size: 0.7rem;">PENDING</span>');
                
                historyHtml += `<tr>
                  <td style="padding: 0.4rem;">${{time}}</td>
                  <td style="padding: 0.4rem; font-size: 0.75rem;">${{order.strategy_position_id || '-'}}</td>
                  <td style="padding: 0.4rem; font-size: 0.75rem;">${{orderStr}}</td>
                  <td style="padding: 0.4rem;">${{statusBadge}}</td>
                </tr>`;
              }}
              historyTbody.innerHTML = historyHtml;
            }}
          }}
          
          statusEl.textContent = `Loaded. Instruments: ${{Object.keys(data.hedge_instruments || {{}}).join(', ')}}`;
        }} else {{
          statusEl.textContent = 'Error: ' + (data.error || 'Unknown error');
        }}
      }} catch (e) {{
        statusEl.textContent = 'Error: ' + e.message;
      }}
    }}
    
    async function evaluateHedging() {{
      const tbody = document.getElementById('hedge-proposals-tbody');
      const statusEl = document.getElementById('hedge-status');
      
      statusEl.textContent = 'Evaluating hedge needs...';
      
      try {{
        const res = await fetch('/api/bots/greg/hedging/evaluate', {{ method: 'POST' }});
        const data = await res.json();
        
        if (data.ok) {{
          const results = data.results || [];
          
          if (results.length === 0) {{
            tbody.innerHTML = '<tr><td colspan="5" style="padding: 1rem; color: #666; font-style: italic; text-align: center;">No positions to evaluate.</td></tr>';
          }} else {{
            let html = '';
            for (const r of results) {{
              const order = r.proposed_order;
              const netDelta = r.net_delta !== undefined ? r.net_delta.toFixed(4) : '-';
              const threshold = r.threshold !== undefined ? r.threshold.toFixed(2) : '-';
              
              let orderCell = '';
              if (order) {{
                const deltaChange = `${{order.net_delta_before.toFixed(4)}} → ${{order.net_delta_after.toFixed(4)}}`;
                orderCell = `<span style="background: #e3f2fd; color: #1565c0; padding: 0.2rem 0.5rem; border-radius: 4px; font-size: 0.8rem;">
                  ${{order.side.toUpperCase()}} ${{order.size}} ${{order.instrument}}
                </span><br/><span style="font-size: 0.7rem; color: #666;">Δ: ${{deltaChange}}</span>`;
              }} else {{
                orderCell = '<span style="color: #2e7d32; font-size: 0.8rem;">No hedge needed</span>';
              }}
              
              html += `<tr style="border-bottom: 1px solid #eee;">
                <td style="padding: 0.5rem; font-size: 0.8rem;">${{r.position_id}}</td>
                <td style="padding: 0.5rem;">${{r.strategy_type}}</td>
                <td style="padding: 0.5rem; font-family: monospace;">${{netDelta}}</td>
                <td style="padding: 0.5rem;">${{threshold}}</td>
                <td style="padding: 0.5rem;">${{orderCell}}</td>
              </tr>`;
            }}
            tbody.innerHTML = html;
          }}
          
          statusEl.innerHTML = `Evaluated ${{data.positions_evaluated}} position(s). <strong>${{data.hedges_proposed}} hedge(s) proposed.</strong> ${{data.dry_run ? '(Dry Run)' : ''}}`;
        }} else {{
          statusEl.textContent = 'Error: ' + (data.error || 'Unknown error');
          tbody.innerHTML = '<tr><td colspan="5" style="padding: 1rem; color: #c62828; text-align: center;">Error evaluating hedges</td></tr>';
        }}
      }} catch (e) {{
        statusEl.textContent = 'Error: ' + e.message;
        tbody.innerHTML = '<tr><td colspan="5" style="padding: 1rem; color: #c62828; text-align: center;">Error: ' + e.message + '</td></tr>';
      }}
    }}
    
    // ==============================================
    // GREG LAB TAB FUNCTIONS
    // ==============================================
    
    let gregFilteredUnderlying = '';
    let gregPositionsCache = [];
    
    async function loadGregPositions() {{
      const tbody = document.getElementById('greg-positions-tbody');
      const sandboxFilter = document.getElementById('greg-sandbox-filter')?.value || 'all';
      
      tbody.innerHTML = '<tr><td colspan="9" style="padding: 1rem; text-align: center; color: #666;">Loading...</td></tr>';
      
      try {{
        let url = `/api/greg/positions?sandbox_filter=${{sandboxFilter}}`;
        if (gregFilteredUnderlying) {{
          url += `&underlying=${{gregFilteredUnderlying}}`;
        }}
        const resp = await fetch(url);
        const data = await resp.json();
        
        if (!data.ok) {{
          tbody.innerHTML = `<tr><td colspan="9" style="padding: 1rem; text-align: center; color: #c62828;">Error: ${{data.error}}</td></tr>`;
          return;
        }}
        
        gregPositionsCache = data.positions || [];
        
        // Update mode banner
        const modePill = document.getElementById('greg-mode-pill');
        const modeDesc = document.getElementById('greg-mode-desc');
        const envBadge = document.getElementById('greg-env-badge');
        
        if (data.mode === 'live' && data.enable_live_execution) {{
          modePill.textContent = 'LIVE EXECUTION';
          modePill.style.background = '#ffcdd2';
          modePill.style.color = '#c62828';
          modeDesc.textContent = 'Orders will be sent to exchange!';
        }} else {{
          modePill.textContent = 'ADVICE ONLY';
          modePill.style.background = '#fff3e0';
          modePill.style.color = '#e65100';
          modeDesc.textContent = 'No orders will be sent.';
        }}
        
        envBadge.textContent = `Env: ${{data.deribit_env || 'testnet'}}`;
        
        // Update sandbox summary
        const sandboxSummary = document.getElementById('greg-sandbox-summary');
        if (data.sandbox_summary) {{
          sandboxSummary.style.display = 'block';
          document.getElementById('greg-sandbox-run-id').textContent = data.sandbox_summary.run_id;
          document.getElementById('greg-sandbox-counts').textContent = 
            `BTC: ${{data.sandbox_summary.btc_count}} | ETH: ${{data.sandbox_summary.eth_count}} | PnL: ${{data.sandbox_summary.total_pnl_pct.toFixed(2)}}%`;
        }} else {{
          sandboxSummary.style.display = 'none';
        }}
        
        // Render positions
        if (gregPositionsCache.length === 0) {{
          tbody.innerHTML = '<tr><td colspan="9" style="padding: 1.5rem; text-align: center; color: #666;">No Greg positions found. Use the sandbox script to create test positions.</td></tr>';
          document.getElementById('greg-positions-count').textContent = '0 positions';
          return;
        }}
        
        let html = '';
        for (const pos of gregPositionsCache) {{
          const badgeColor = pos.badge === 'LIVE' ? '#c62828' : 
                             pos.badge === 'SANDBOX' ? '#ffc107' :
                             pos.badge === 'DEMO' ? '#1565c0' : '#757575';
          const badgeBg = pos.badge === 'LIVE' ? '#ffcdd2' : 
                          pos.badge === 'SANDBOX' ? '#fff8e1' :
                          pos.badge === 'DEMO' ? '#e3f2fd' : '#e0e0e0';
          
          const pnlColor = pos.pnl_pct >= 0 ? '#2e7d32' : '#c62828';
          const urgencyColor = pos.urgency === 'HIGH' ? '#c62828' : 
                               pos.urgency === 'MEDIUM' ? '#ff9800' : '#2e7d32';
          
          html += `<tr style="border-bottom: 1px solid #eee;" data-position-id="${{pos.position_id}}">
            <td style="padding: 0.6rem;">
              <span style="padding: 0.2rem 0.5rem; border-radius: 4px; font-size: 0.75rem; font-weight: bold; background: ${{badgeBg}}; color: ${{badgeColor}};">
                ${{pos.badge}}
              </span>
            </td>
            <td style="padding: 0.6rem; font-weight: bold;">${{pos.underlying}}</td>
            <td style="padding: 0.6rem;">${{pos.human_readable_name}}</td>
            <td style="padding: 0.6rem; text-align: right; font-family: monospace;">${{pos.size.toFixed(2)}}</td>
            <td style="padding: 0.6rem; text-align: right; font-weight: bold; color: ${{pnlColor}};">${{pos.pnl_pct.toFixed(2)}}%</td>
            <td style="padding: 0.6rem; text-align: right; color: ${{pnlColor}};">$${{pos.pnl_usd.toFixed(2)}}</td>
            <td style="padding: 0.6rem; text-align: right;">${{pos.dte}}d</td>
            <td style="padding: 0.6rem;">
              <span style="padding: 0.15rem 0.4rem; border-radius: 3px; font-size: 0.75rem; background: #f5f5f5; color: ${{urgencyColor}};">
                ${{pos.suggested_action}}
              </span>
            </td>
            <td style="padding: 0.6rem; text-align: center;">
              <button onclick="viewGregPositionLogs('${{pos.position_id}}')" style="padding: 0.25rem 0.5rem; font-size: 0.75rem;">Logs</button>
            </td>
          </tr>`;
        }}
        tbody.innerHTML = html;
        document.getElementById('greg-positions-count').textContent = `${{gregPositionsCache.length}} position(s)`;
        
      }} catch (e) {{
        tbody.innerHTML = `<tr><td colspan="9" style="padding: 1rem; text-align: center; color: #c62828;">Error: ${{e.message}}</td></tr>`;
      }}
    }}
    
    function filterGregPositions(btn, underlying) {{
      gregFilteredUnderlying = underlying;
      document.querySelectorAll('.greg-filter-btn').forEach(b => {{
        b.style.background = '#e0e0e0';
        b.style.color = 'black';
      }});
      btn.style.background = '#1565c0';
      btn.style.color = 'white';
      loadGregPositions();
    }}
    
    async function viewGregPositionLogs(positionId) {{
      const panel = document.getElementById('greg-log-panel');
      const title = document.getElementById('greg-log-title');
      const tbody = document.getElementById('greg-log-tbody');
      
      panel.style.display = 'block';
      title.textContent = `Management Log: ${{positionId}}`;
      tbody.innerHTML = '<tr><td colspan="7" style="padding: 0.5rem; color: #666;">Loading...</td></tr>';
      
      try {{
        const resp = await fetch(`/api/greg/positions/${{positionId}}/logs`);
        const data = await resp.json();
        
        if (!data.ok) {{
          tbody.innerHTML = `<tr><td colspan="7" style="padding: 0.5rem; color: #c62828;">Error: ${{data.error}}</td></tr>`;
          return;
        }}
        
        if (data.logs.length === 0) {{
          tbody.innerHTML = '<tr><td colspan="7" style="padding: 0.5rem; color: #666; font-style: italic;">No decision logs for this position yet.</td></tr>';
          return;
        }}
        
        let html = '';
        for (const log of data.logs) {{
          const suggestedIcon = log.suggested ? '✓' : '';
          const executedIcon = log.executed ? '✓' : '';
          const sensors = [];
          if (log.vrp_30d !== null) sensors.push(`VRP: ${{log.vrp_30d.toFixed(1)}}`);
          if (log.adx_14d !== null) sensors.push(`ADX: ${{log.adx_14d.toFixed(1)}}`);
          if (log.net_delta !== null) sensors.push(`Δ: ${{log.net_delta.toFixed(4)}}`);
          
          html += `<tr style="border-bottom: 1px solid #eee;">
            <td style="padding: 0.5rem; font-size: 0.75rem;">${{log.timestamp ? new Date(log.timestamp).toLocaleString() : '--'}}</td>
            <td style="padding: 0.5rem;">${{log.action_type}}</td>
            <td style="padding: 0.5rem; text-align: center; color: #1565c0;">${{suggestedIcon}}</td>
            <td style="padding: 0.5rem; text-align: center; color: #2e7d32;">${{executedIcon}}</td>
            <td style="padding: 0.5rem; max-width: 200px; overflow: hidden; text-overflow: ellipsis;" title="${{log.reason || ''}}">${{log.reason || '--'}}</td>
            <td style="padding: 0.5rem; text-align: right;">${{log.pnl_pct !== null ? log.pnl_pct.toFixed(2) + '%' : '--'}}</td>
            <td style="padding: 0.5rem; font-size: 0.7rem; color: #666;">${{sensors.join(' | ')}}</td>
          </tr>`;
        }}
        tbody.innerHTML = html;
        
      }} catch (e) {{
        tbody.innerHTML = `<tr><td colspan="7" style="padding: 0.5rem; color: #c62828;">Error: ${{e.message}}</td></tr>`;
      }}
    }}
    
    function closeGregLogPanel() {{
      document.getElementById('greg-log-panel').style.display = 'none';
    }}
    
    // ==============================================
    // SYSTEM HEALTH TAB FUNCTIONS
    // ==============================================
    
    let lastHealthcheckTime = 0;
    const HEALTHCHECK_THROTTLE_MS = 60000; // Don't auto-run within 60 seconds
    
    async function loadSystemHealthStatus() {{
      // Load agent health guard status
      try {{
        const healthStatusRes = await fetch('/api/system_health/status');
        const healthStatus = await healthStatusRes.json();
        
        const healthGuardEl = document.getElementById('health-guard-status');
        if (healthGuardEl && healthStatus.ok) {{
          let guardHtml = '';
          if (healthStatus.agent_paused_due_to_health) {{
            guardHtml = `<div style="background: #fce4ec; border: 1px solid #c62828; padding: 1rem; margin-bottom: 1rem; border-radius: 4px;">
              <strong style="color: #c62828;">⛔ AGENT PAUSED</strong>
              <span style="color: #666; margin-left: 1rem;">Trading suspended due to health failure</span>
            </div>`;
          }} else if (healthStatus.overall_status === 'FAIL') {{
            guardHtml = `<div style="background: #fff3e0; border: 1px solid #e65100; padding: 0.75rem; margin-bottom: 1rem; border-radius: 4px;">
              <strong style="color: #e65100;">⚠️ Health FAIL</strong>
              <span style="color: #666; margin-left: 0.5rem;">${{healthStatus.summary || 'Check healthcheck results'}}</span>
            </div>`;
          }} else if (healthStatus.overall_status === 'WARN') {{
            guardHtml = `<div style="background: #fffde7; border: 1px solid #f9a825; padding: 0.75rem; margin-bottom: 1rem; border-radius: 4px;">
              <strong style="color: #f9a825;">⚠️ Health WARN</strong>
              <span style="color: #666; margin-left: 0.5rem;">${{healthStatus.summary || 'Check healthcheck results'}}</span>
            </div>`;
          }} else if (healthStatus.overall_status === 'OK') {{
            guardHtml = `<div style="background: #e8f5e9; border: 1px solid #2e7d32; padding: 0.5rem; margin-bottom: 1rem; border-radius: 4px;">
              <strong style="color: #2e7d32;">✓ Health OK</strong>
              <span style="color: #666; margin-left: 0.5rem;">${{healthStatus.last_run_at ? 'Last checked: ' + new Date(healthStatus.last_run_at).toLocaleTimeString() : ''}}</span>
            </div>`;
          }}
          healthGuardEl.innerHTML = guardHtml;
        }}
      }} catch (e) {{
        console.error('Error loading health guard status:', e);
      }}
      
      // Load LLM status and check LLM readiness
      try {{
        const [llmRes, readinessRes] = await Promise.all([
          fetch('/api/llm_status'),
          fetch('/api/llm_readiness')
        ]);
        const llmData = await llmRes.json();
        const readinessData = await readinessRes.json();
        
        if (llmData.ok) {{
          const llmStatus = `Decision: ${{llmData.decision_mode}} | Env: ${{llmData.deribit_env}} | LLM: ${{llmData.llm_enabled ? 'enabled' : 'disabled'}}`;
          document.getElementById('llm-status-line').textContent = llmStatus;
        }} else {{
          document.getElementById('llm-status-line').textContent = 'Error loading LLM status';
        }}
        
        // Gate LLM test button based on readiness
        const llmTestBtn = document.getElementById('llm-test-btn');
        if (llmTestBtn) {{
          if (readinessData.ready) {{
            llmTestBtn.disabled = false;
            llmTestBtn.title = 'Run LLM decision pipeline test';
            llmTestBtn.style.opacity = '1';
          }} else {{
            llmTestBtn.disabled = true;
            llmTestBtn.title = readinessData.reason || 'LLM not ready';
            llmTestBtn.style.opacity = '0.5';
            document.getElementById('llm-result').innerHTML = `<span style="color: #ff9800; font-size: 0.8rem;">⚠️ ${{readinessData.reason || 'LLM not configured'}}</span>`;
          }}
        }}
      }} catch (e) {{
        document.getElementById('llm-status-line').textContent = 'Error: ' + e.message;
      }}
      
      // Load risk limits
      try {{
        const riskRes = await fetch('/api/risk_limits');
        const riskData = await riskRes.json();
        if (riskData.ok) {{
          const ks = riskData.kill_switch_enabled ? 'ON' : 'OFF';
          const dd = riskData.daily_drawdown_limit_pct || 0;
          const riskStatus = `Max Margin: ${{riskData.max_margin_used_pct}}% | Max Δ: ${{riskData.max_net_delta_abs}} | DD limit: ${{dd}}% | Kill switch: ${{ks}}`;
          document.getElementById('risk-status-line').textContent = riskStatus;
        }} else {{
          document.getElementById('risk-status-line').textContent = 'Error loading risk limits';
        }}
      }} catch (e) {{
        document.getElementById('risk-status-line').textContent = 'Error: ' + e.message;
      }}
      
      // Load runtime config for controls
      await loadRuntimeConfig();
      
      // Load LLM & Strategy tuning config
      await loadLLMStrategyConfig();
      
      // Auto-trigger healthcheck if not run recently
      const now = Date.now();
      if (now - lastHealthcheckTime > HEALTHCHECK_THROTTLE_MS) {{
        lastHealthcheckTime = now;
        runHealthcheck();
      }}
    }}
    
    async function loadRuntimeConfig() {{
      try {{
        const res = await fetch('/api/system/runtime-config');
        const data = await res.json();
        if (data.ok) {{
          // Kill switch toggle
          const killSwitchToggle = document.getElementById('kill-switch-toggle');
          const killSwitchLabel = document.getElementById('kill-switch-label');
          if (killSwitchToggle && killSwitchLabel) {{
            killSwitchToggle.checked = data.kill_switch_enabled;
            killSwitchLabel.textContent = data.kill_switch_enabled ? 'ON' : 'OFF';
            killSwitchLabel.style.color = data.kill_switch_enabled ? '#c62828' : '#333';
          }}
          
          // Daily drawdown limit
          const drawdownInput = document.getElementById('drawdown-limit-input');
          if (drawdownInput) {{
            drawdownInput.value = data.daily_drawdown_limit_pct || 0;
          }}
          
          // Decision mode
          const decisionModeSelect = document.getElementById('decision-mode-select');
          if (decisionModeSelect) {{
            decisionModeSelect.value = data.decision_mode;
          }}
          
          // Dry run toggle
          const dryRunToggle = document.getElementById('dry-run-toggle');
          const dryRunLabel = document.getElementById('dry-run-label');
          if (dryRunToggle && dryRunLabel) {{
            dryRunToggle.checked = data.dry_run;
            dryRunLabel.textContent = data.dry_run ? 'ON' : 'OFF';
            dryRunLabel.style.color = data.dry_run ? '#e65100' : '#333';
          }}
          
          // Position reconcile action
          const reconcileActionSelect = document.getElementById('reconcile-action-select');
          if (reconcileActionSelect) {{
            reconcileActionSelect.value = data.position_reconcile_action;
          }}
        }}
      }} catch (e) {{
        console.error('Error loading runtime config:', e);
      }}
    }}
    
    async function updateKillSwitch(enabled) {{
      const feedbackEl = document.getElementById('kill-switch-feedback');
      const labelEl = document.getElementById('kill-switch-label');
      feedbackEl.innerHTML = '<span style="color: #666;">Updating...</span>';
      try {{
        const res = await fetch('/api/system/runtime-config', {{
          method: 'POST',
          headers: {{'Content-Type': 'application/json'}},
          body: JSON.stringify({{kill_switch_enabled: enabled}})
        }});
        const data = await res.json();
        if (data.ok) {{
          labelEl.textContent = enabled ? 'ON' : 'OFF';
          labelEl.style.color = enabled ? '#c62828' : '#333';
          feedbackEl.innerHTML = `<span style="color: #2e7d32;">✓ Kill switch ${{enabled ? 'enabled' : 'disabled'}}</span>`;
          setTimeout(() => {{ feedbackEl.innerHTML = ''; }}, 3000);
          loadSystemHealthStatus();
        }} else {{
          feedbackEl.innerHTML = `<span style="color: #c62828;">✗ ${{data.errors?.join(', ') || 'Update failed'}}</span>`;
          document.getElementById('kill-switch-toggle').checked = !enabled;
        }}
      }} catch (e) {{
        feedbackEl.innerHTML = `<span style="color: #c62828;">✗ Error: ${{e.message}}</span>`;
        document.getElementById('kill-switch-toggle').checked = !enabled;
      }}
    }}
    
    async function updateDrawdownLimit() {{
      const inputEl = document.getElementById('drawdown-limit-input');
      const feedbackEl = document.getElementById('drawdown-limit-feedback');
      const value = parseFloat(inputEl.value) || 0;
      feedbackEl.innerHTML = '<span style="color: #666;">Updating...</span>';
      try {{
        const res = await fetch('/api/system/runtime-config', {{
          method: 'POST',
          headers: {{'Content-Type': 'application/json'}},
          body: JSON.stringify({{daily_drawdown_limit_pct: value}})
        }});
        const data = await res.json();
        if (data.ok) {{
          feedbackEl.innerHTML = `<span style="color: #2e7d32;">✓ Drawdown limit updated to ${{value}}%</span>`;
          setTimeout(() => {{ feedbackEl.innerHTML = ''; }}, 3000);
          loadSystemHealthStatus();
        }} else {{
          feedbackEl.innerHTML = `<span style="color: #c62828;">✗ ${{data.errors?.join(', ') || 'Update failed'}}</span>`;
        }}
      }} catch (e) {{
        feedbackEl.innerHTML = `<span style="color: #c62828;">✗ Error: ${{e.message}}</span>`;
      }}
    }}
    
    async function updateDecisionMode(mode) {{
      const feedbackEl = document.getElementById('decision-mode-feedback');
      feedbackEl.innerHTML = '<span style="color: #666;">Updating...</span>';
      const modeLabels = {{'rule_only': 'Rule Only', 'llm_only': 'LLM Only', 'hybrid_shadow': 'Hybrid (LLM Shadow)'}};
      try {{
        const res = await fetch('/api/system/runtime-config', {{
          method: 'POST',
          headers: {{'Content-Type': 'application/json'}},
          body: JSON.stringify({{decision_mode: mode}})
        }});
        const data = await res.json();
        if (data.ok) {{
          feedbackEl.innerHTML = `<span style="color: #2e7d32;">✓ Decision mode changed to: ${{modeLabels[mode] || mode}}</span>`;
          setTimeout(() => {{ feedbackEl.innerHTML = ''; }}, 3000);
          loadSystemHealthStatus();
        }} else {{
          feedbackEl.innerHTML = `<span style="color: #c62828;">✗ ${{data.errors?.join(', ') || 'Update failed'}}</span>`;
        }}
      }} catch (e) {{
        feedbackEl.innerHTML = `<span style="color: #c62828;">✗ Error: ${{e.message}}</span>`;
      }}
    }}
    
    async function updateDryRun(enabled) {{
      const feedbackEl = document.getElementById('dry-run-feedback');
      const labelEl = document.getElementById('dry-run-label');
      feedbackEl.innerHTML = '<span style="color: #666;">Updating...</span>';
      try {{
        const res = await fetch('/api/system/runtime-config', {{
          method: 'POST',
          headers: {{'Content-Type': 'application/json'}},
          body: JSON.stringify({{dry_run: enabled}})
        }});
        const data = await res.json();
        if (data.ok) {{
          labelEl.textContent = enabled ? 'ON' : 'OFF';
          labelEl.style.color = enabled ? '#e65100' : '#333';
          feedbackEl.innerHTML = `<span style="color: #2e7d32;">✓ Dry run mode is now ${{enabled ? 'ON' : 'OFF'}}</span>`;
          setTimeout(() => {{ feedbackEl.innerHTML = ''; }}, 3000);
          loadSystemHealthStatus();
        }} else {{
          feedbackEl.innerHTML = `<span style="color: #c62828;">✗ ${{data.errors?.join(', ') || 'Update failed'}}</span>`;
          document.getElementById('dry-run-toggle').checked = !enabled;
        }}
      }} catch (e) {{
        feedbackEl.innerHTML = `<span style="color: #c62828;">✗ Error: ${{e.message}}</span>`;
        document.getElementById('dry-run-toggle').checked = !enabled;
      }}
    }}
    
    async function updateReconcileAction(action) {{
      const feedbackEl = document.getElementById('reconcile-action-feedback');
      feedbackEl.innerHTML = '<span style="color: #666;">Updating...</span>';
      const actionLabels = {{'halt': 'Halt', 'auto_heal': 'Auto-Heal'}};
      try {{
        const res = await fetch('/api/system/runtime-config', {{
          method: 'POST',
          headers: {{'Content-Type': 'application/json'}},
          body: JSON.stringify({{position_reconcile_action: action}})
        }});
        const data = await res.json();
        if (data.ok) {{
          feedbackEl.innerHTML = `<span style="color: #2e7d32;">✓ Reconciliation behavior updated to: ${{actionLabels[action] || action}}</span>`;
          setTimeout(() => {{ feedbackEl.innerHTML = ''; }}, 3000);
        }} else {{
          feedbackEl.innerHTML = `<span style="color: #c62828;">✗ ${{data.errors?.join(', ') || 'Update failed'}}</span>`;
        }}
      }} catch (e) {{
        feedbackEl.innerHTML = `<span style="color: #c62828;">✗ Error: ${{e.message}}</span>`;
      }}
    }}
    
    // ==============================================
    // LLM & STRATEGY TUNING FUNCTIONS
    // ==============================================
    
    async function loadLLMStrategyConfig() {{
      // Load LLM status
      try {{
        const llmRes = await fetch('/api/llm_status');
        const llmData = await llmRes.json();
        if (llmData.ok) {{
          document.getElementById('llm-mode-label').textContent = llmData.mode + ' / ' + llmData.decision_mode;
          document.getElementById('llm-deribit-label').textContent = llmData.deribit_env;
          
          const llmEnabledToggle = document.getElementById('llm-enabled-toggle');
          const llmEnabledLabel = document.getElementById('llm-enabled-label');
          if (llmEnabledToggle) {{
            llmEnabledToggle.checked = llmData.llm_enabled;
            llmEnabledLabel.textContent = llmData.llm_enabled ? 'ON' : 'OFF';
            llmEnabledLabel.style.color = llmData.llm_enabled ? '#1976d2' : '#333';
          }}
          
          const exploreSlider = document.getElementById('explore-prob-slider');
          if (exploreSlider) {{
            exploreSlider.value = Math.round((llmData.explore_prob || 0) * 100);
            document.getElementById('explore-prob-label').textContent = exploreSlider.value + '%';
          }}
        }}
      }} catch (e) {{
        console.error('Error loading LLM status:', e);
      }}
      
      // Load strategy thresholds
      try {{
        const stratRes = await fetch('/api/strategy_thresholds');
        const stratData = await stratRes.json();
        if (stratData.ok) {{
          const eff = stratData.effective;
          document.getElementById('ivrv-min-input').value = eff.ivrv_min;
          document.getElementById('delta-min-input').value = eff.delta_min;
          document.getElementById('delta-max-input').value = eff.delta_max;
          document.getElementById('dte-min-input').value = eff.dte_min;
          document.getElementById('dte-max-input').value = eff.dte_max;
          
          const trainingProfileSelect = document.getElementById('training-profile-select');
          if (trainingProfileSelect) {{
            trainingProfileSelect.value = stratData.training_profile_mode || 'single';
          }}
        }}
      }} catch (e) {{
        console.error('Error loading strategy thresholds:', e);
      }}
      
      // Load risk limits
      try {{
        const riskRes = await fetch('/api/risk_limits');
        const riskData = await riskRes.json();
        if (riskData.ok) {{
          document.getElementById('max-margin-input').value = riskData.max_margin_used_pct;
          document.getElementById('max-net-delta-input').value = riskData.max_net_delta_abs;
          document.getElementById('liquidity-max-spread-input').value = riskData.liquidity_max_spread_pct;
          document.getElementById('liquidity-min-oi-input').value = riskData.liquidity_min_open_interest;
          document.getElementById('liquidity-max-spread-label').textContent = riskData.liquidity_max_spread_pct.toFixed(1);
          document.getElementById('liquidity-min-oi-label').textContent = riskData.liquidity_min_open_interest;
        }}
      }} catch (e) {{
        console.error('Error loading risk limits:', e);
      }}
      
      // Load reconciliation config
      try {{
        const reconcileRes = await fetch('/api/reconciliation_config');
        const reconcileData = await reconcileRes.json();
        if (reconcileData.ok) {{
          document.getElementById('reconcile-action-config-select').value = reconcileData.position_reconcile_action;
          document.getElementById('reconcile-on-startup-checkbox').checked = !!reconcileData.position_reconcile_on_startup;
          document.getElementById('reconcile-on-each-loop-checkbox').checked = !!reconcileData.position_reconcile_on_each_loop;
          document.getElementById('reconcile-tolerance-input').value = reconcileData.position_reconcile_tolerance_usd;
        }}
      }} catch (e) {{
        console.error('Error loading reconciliation config:', e);
      }}
    }}
    
    function updateExploreProbLabel(value) {{
      document.getElementById('explore-prob-label').textContent = value + '%';
    }}
    
    async function updateLLMEnabled(enabled) {{
      const feedbackEl = document.getElementById('llm-enabled-feedback');
      const labelEl = document.getElementById('llm-enabled-label');
      feedbackEl.innerHTML = '<span style="color: #666;">Updating...</span>';
      try {{
        const res = await fetch('/api/llm_status', {{
          method: 'POST',
          headers: {{'Content-Type': 'application/json'}},
          body: JSON.stringify({{llm_enabled: enabled}})
        }});
        const data = await res.json();
        if (data.ok) {{
          labelEl.textContent = enabled ? 'ON' : 'OFF';
          labelEl.style.color = enabled ? '#1976d2' : '#333';
          feedbackEl.innerHTML = `<span style="color: #2e7d32;">✓ LLM ${{enabled ? 'enabled' : 'disabled'}}</span>`;
          setTimeout(() => {{ feedbackEl.innerHTML = ''; }}, 3000);
          loadSystemHealthStatus();
        }} else {{
          feedbackEl.innerHTML = `<span style="color: #c62828;">✗ ${{data.error || 'Update failed'}}</span>`;
          document.getElementById('llm-enabled-toggle').checked = !enabled;
        }}
      }} catch (e) {{
        feedbackEl.innerHTML = `<span style="color: #c62828;">✗ Error: ${{e.message}}</span>`;
        document.getElementById('llm-enabled-toggle').checked = !enabled;
      }}
    }}
    
    async function saveExploreProb() {{
      const feedbackEl = document.getElementById('explore-prob-feedback');
      const sliderValue = parseInt(document.getElementById('explore-prob-slider').value);
      const exploreProb = sliderValue / 100.0;
      feedbackEl.innerHTML = '<span style="color: #666;">Updating...</span>';
      try {{
        const res = await fetch('/api/llm_status', {{
          method: 'POST',
          headers: {{'Content-Type': 'application/json'}},
          body: JSON.stringify({{explore_prob: exploreProb}})
        }});
        const data = await res.json();
        if (data.ok) {{
          feedbackEl.innerHTML = `<span style="color: #2e7d32;">✓ Explore probability set to ${{sliderValue}}%</span>`;
          setTimeout(() => {{ feedbackEl.innerHTML = ''; }}, 3000);
        }} else {{
          feedbackEl.innerHTML = `<span style="color: #c62828;">✗ ${{data.error || 'Update failed'}}</span>`;
        }}
      }} catch (e) {{
        feedbackEl.innerHTML = `<span style="color: #c62828;">✗ Error: ${{e.message}}</span>`;
      }}
    }}
    
    async function updateTrainingProfile(mode) {{
      const feedbackEl = document.getElementById('training-profile-feedback');
      feedbackEl.innerHTML = '<span style="color: #666;">Updating...</span>';
      try {{
        const res = await fetch('/api/strategy_thresholds', {{
          method: 'POST',
          headers: {{'Content-Type': 'application/json'}},
          body: JSON.stringify({{training_profile_mode: mode}})
        }});
        const data = await res.json();
        if (data.ok) {{
          feedbackEl.innerHTML = `<span style="color: #2e7d32;">✓ Training profile set to: ${{mode}}</span>`;
          setTimeout(() => {{ feedbackEl.innerHTML = ''; }}, 3000);
        }} else {{
          feedbackEl.innerHTML = `<span style="color: #c62828;">✗ ${{data.error || 'Update failed'}}</span>`;
        }}
      }} catch (e) {{
        feedbackEl.innerHTML = `<span style="color: #c62828;">✗ Error: ${{e.message}}</span>`;
      }}
    }}
    
    async function saveStrategyThresholds() {{
      const feedbackEl = document.getElementById('strategy-thresholds-feedback');
      feedbackEl.innerHTML = '<span style="color: #666;">Saving thresholds...</span>';
      
      const payload = {{
        ivrv_min: parseFloat(document.getElementById('ivrv-min-input').value),
        delta_min: parseFloat(document.getElementById('delta-min-input').value),
        delta_max: parseFloat(document.getElementById('delta-max-input').value),
        dte_min: parseInt(document.getElementById('dte-min-input').value),
        dte_max: parseInt(document.getElementById('dte-max-input').value)
      }};
      
      try {{
        const res = await fetch('/api/strategy_thresholds', {{
          method: 'POST',
          headers: {{'Content-Type': 'application/json'}},
          body: JSON.stringify(payload)
        }});
        const data = await res.json();
        if (data.ok) {{
          feedbackEl.innerHTML = '<span style="color: #2e7d32;">✓ Strategy thresholds updated</span>';
          setTimeout(() => {{ feedbackEl.innerHTML = ''; }}, 3000);
        }} else {{
          feedbackEl.innerHTML = `<span style="color: #c62828;">✗ ${{data.error || 'Update failed'}}</span>`;
        }}
      }} catch (e) {{
        feedbackEl.innerHTML = `<span style="color: #c62828;">✗ Error: ${{e.message}}</span>`;
      }}
    }}
    
    async function saveRiskLimits() {{
      const feedbackEl = document.getElementById('risk-limits-feedback');
      feedbackEl.innerHTML = '<span style="color: #666;">Saving risk limits...</span>';
      
      const payload = {{
        max_margin_used_pct: parseFloat(document.getElementById('max-margin-input').value),
        max_net_delta_abs: parseFloat(document.getElementById('max-net-delta-input').value),
        liquidity_max_spread_pct: parseFloat(document.getElementById('liquidity-max-spread-input').value),
        liquidity_min_open_interest: parseInt(document.getElementById('liquidity-min-oi-input').value, 10)
      }};
      
      try {{
        const res = await fetch('/api/risk_limits', {{
          method: 'POST',
          headers: {{'Content-Type': 'application/json'}},
          body: JSON.stringify(payload)
        }});
        const data = await res.json();
        if (data.ok) {{
          feedbackEl.innerHTML = '<span style="color: #2e7d32;">✓ Risk limits updated (runtime only)</span>';
          document.getElementById('liquidity-max-spread-label').textContent = data.liquidity_max_spread_pct.toFixed(1);
          document.getElementById('liquidity-min-oi-label').textContent = data.liquidity_min_open_interest;
          setTimeout(() => {{ feedbackEl.innerHTML = ''; }}, 3000);
          loadSystemHealthStatus();
        }} else {{
          feedbackEl.innerHTML = `<span style="color: #c62828;">✗ ${{data.error || 'Update failed'}}</span>`;
        }}
      }} catch (e) {{
        feedbackEl.innerHTML = `<span style="color: #c62828;">✗ Error: ${{e.message}}</span>`;
      }}
    }}
    
    async function saveReconciliationConfig() {{
      const feedbackEl = document.getElementById('reconcile-feedback');
      feedbackEl.innerHTML = '<span style="color: #666;">Applying reconciliation settings...</span>';
      
      const payload = {{
        position_reconcile_action: document.getElementById('reconcile-action-config-select').value,
        position_reconcile_on_startup: document.getElementById('reconcile-on-startup-checkbox').checked,
        position_reconcile_on_each_loop: document.getElementById('reconcile-on-each-loop-checkbox').checked,
        position_reconcile_tolerance_usd: parseFloat(document.getElementById('reconcile-tolerance-input').value)
      }};
      
      try {{
        const res = await fetch('/api/reconciliation_config', {{
          method: 'POST',
          headers: {{'Content-Type': 'application/json'}},
          body: JSON.stringify(payload)
        }});
        const data = await res.json();
        if (data.ok) {{
          feedbackEl.innerHTML = '<span style="color: #2e7d32;">✓ Reconciliation settings updated (runtime only)</span>';
          setTimeout(() => {{ feedbackEl.innerHTML = ''; }}, 3000);
        }} else {{
          feedbackEl.innerHTML = `<span style="color: #c62828;">✗ ${{data.error || 'Update failed'}}</span>`;
        }}
      }} catch (e) {{
        feedbackEl.innerHTML = `<span style="color: #c62828;">✗ Error: ${{e.message}}</span>`;
      }}
    }}
    
    async function testLlmDecision() {{
      const el = document.getElementById('llm-result');
      el.innerHTML = '<span style="color: #666;">Testing LLM pipeline...</span>';
      try {{
        const res = await fetch('/api/test_llm_decision', {{ method: 'POST' }});
        const data = await res.json();
        if (data.ok) {{
          el.innerHTML = `<span style="color: #2e7d32;">✅ LLM OK: ${{data.action}}</span><br><span style="color: #666; font-size: 0.8em;">${{data.reasoning || ''}}</span>`;
        }} else {{
          el.innerHTML = `<span style="color: #ff9800;">⚠️ ${{data.error}}</span>`;
        }}
      }} catch (e) {{
        el.innerHTML = `<span style="color: #c62828;">❌ Request error: ${{e.message}}</span>`;
      }}
    }}
    
    async function runReconciliation() {{
      const el = document.getElementById('reconcile-result');
      const statusEl = document.getElementById('reconcile-status-line');
      el.innerHTML = '<span style="color: #666;">Running reconciliation...</span>';
      try {{
        const res = await fetch('/api/reconcile_positions', {{ method: 'POST' }});
        const data = await res.json();
        if (data.ok) {{
          const s = data.summary;
          if (data.is_clean) {{
            el.innerHTML = `<span style="color: #2e7d32;">✅ Reconciliation OK: ${{s.deribit_positions}} / ${{s.tracked_positions}} positions match.</span>`;
            statusEl.textContent = `Last check: OK (${{s.deribit_positions}} / ${{s.tracked_positions}} aligned)`;
          }} else {{
            const issues = data.details.join(', ');
            el.innerHTML = `<span style="color: #ff9800;">⚠️ Reconciliation WARN: ${{issues}}</span>`;
            statusEl.textContent = `Last check: WARN - ${{issues}}`;
          }}
        }} else {{
          el.innerHTML = `<span style="color: #c62828;">❌ Error: ${{data.error}}</span>`;
        }}
      }} catch (e) {{
        el.innerHTML = `<span style="color: #c62828;">❌ Request error: ${{e.message}}</span>`;
      }}
    }}
    
    async function testKillSwitch() {{
      const el = document.getElementById('risk-result');
      el.innerHTML = '<span style="color: #666;">Testing risk engine...</span>';
      try {{
        const res = await fetch('/api/test_kill_switch', {{ method: 'POST' }});
        const data = await res.json();
        if (data.ok) {{
          if (data.allowed) {{
            el.innerHTML = `<span style="color: #2e7d32;">✅ Risk engine ALLOWS OPEN_COVERED_CALL under current config.</span>`;
          }} else {{
            const reasons = data.reasons.join('; ');
            el.innerHTML = `<span style="color: #c62828;">❌ Risk engine BLOCKS OPEN_COVERED_CALL: ${{reasons}}</span>`;
          }}
        }} else {{
          el.innerHTML = `<span style="color: #c62828;">❌ Test failed: ${{data.error}}</span>`;
        }}
      }} catch (e) {{
        el.innerHTML = `<span style="color: #c62828;">❌ Request error: ${{e.message}}</span>`;
      }}
    }}
    
    async function runHealthcheck() {{
      const el = document.getElementById('healthcheck-result');
      const statusEl = document.getElementById('healthcheck-status-line');
      const detailsEl = document.getElementById('healthcheck-details');
      const detailsContent = document.getElementById('healthcheck-details-content');
      const badge = document.getElementById('health-badge');
      const summary = document.getElementById('health-summary');
      const lastRun = document.getElementById('health-last-run');
      
      if (el) el.innerHTML = '<span style="color: #666;">Running healthcheck...</span>';
      if (detailsEl) detailsEl.style.display = 'none';
      if (badge) {{ badge.textContent = 'CHECKING...'; badge.style.background = '#e0e0e0'; badge.style.color = '#666'; }}
      
      try {{
        const res = await fetch('/api/agent_healthcheck', {{ method: 'POST' }});
        const data = await res.json();
        const now = new Date().toLocaleTimeString();
        
        if (data.ok !== false) {{
          const status = data.overall_status;
          const checksStr = data.results.map(r => `${{r.name}}=${{r.status}}`).join(', ');
          const summaryText = data.summary || checksStr;
          
          if (status === 'OK') {{
            if (el) el.innerHTML = `<span style="color: #2e7d32;">✅ Healthcheck OK – ${{checksStr}}</span>`;
            if (statusEl) statusEl.textContent = 'Last check: OK';
            if (badge) {{ badge.textContent = 'OK'; badge.style.background = '#c8e6c9'; badge.style.color = '#2e7d32'; }}
            if (summary) summary.textContent = 'All systems operational';
          }} else if (status === 'WARN') {{
            if (el) el.innerHTML = `<span style="color: #ff9800;">⚠️ Healthcheck WARN – ${{checksStr}}</span>`;
            if (statusEl) statusEl.textContent = 'Last check: WARN';
            if (badge) {{ badge.textContent = 'WARN'; badge.style.background = '#fff3e0'; badge.style.color = '#e65100'; }}
            if (summary) summary.textContent = summaryText;
          }} else {{
            if (el) el.innerHTML = `<span style="color: #c62828;">❌ Healthcheck FAIL – ${{checksStr}}</span>`;
            if (statusEl) statusEl.textContent = 'Last check: FAIL';
            if (badge) {{ badge.textContent = 'FAIL'; badge.style.background = '#ffcdd2'; badge.style.color = '#c62828'; }}
            if (summary) summary.textContent = summaryText;
          }}
          if (lastRun) lastRun.textContent = `Last run: ${{now}}`;
          
          // Show details
          if (detailsContent) detailsContent.textContent = data.results.map(r => `${{r.name}}: ${{r.status}} - ${{r.detail}}`).join('\\n');
          if (detailsEl) detailsEl.style.display = 'block';
        }} else {{
          if (el) el.innerHTML = `<span style="color: #c62828;">❌ Healthcheck failed: ${{data.error || data.overall_status}}</span>`;
          if (statusEl) statusEl.textContent = 'Last check: ERROR';
          if (badge) {{ badge.textContent = 'ERROR'; badge.style.background = '#ffcdd2'; badge.style.color = '#c62828'; }}
          if (summary) summary.textContent = data.error || 'Request failed';
          if (lastRun) lastRun.textContent = `Last run: ${{now}}`;
        }}
      }} catch (e) {{
        if (el) el.innerHTML = `<span style="color: #c62828;">❌ Request error: ${{e.message}}</span>`;
        if (statusEl) statusEl.textContent = 'Last check: ERROR';
        if (badge) {{ badge.textContent = 'ERROR'; badge.style.background = '#ffcdd2'; badge.style.color = '#c62828'; }}
        if (summary) summary.textContent = 'Network error';
      }}
    }}
    
    function renderStewardReport(data) {{
      const lastRunEl = document.getElementById('steward-last-run');
      const statusEl = document.getElementById('steward-status');
      const summaryEl = document.getElementById('steward-summary');
      const listEl = document.getElementById('steward-top-items');
      const promptEl = document.getElementById('steward-builder-prompt');
      
      if (!data || data.ok === false) {{
        statusEl.textContent = data && data.error ? `Error: ${{data.error}}` : 'Error: failed to load Steward report.';
        statusEl.style.color = '#c62828';
        return;
      }}
      
      statusEl.textContent = data.llm_used ? 'Steward report generated (LLM used).' : 'Steward report generated (fallback, no LLM).';
      statusEl.style.color = data.llm_used ? '#2e7d32' : '#ff9800';
      
      lastRunEl.textContent = data.generated_at || 'Never';
      summaryEl.textContent = data.summary || '';
      
      listEl.innerHTML = '';
      (data.top_items || []).forEach((item) => {{
        const li = document.createElement('li');
        li.innerHTML = `<strong>[${{item.priority}}]</strong> <span style="color: #1565c0;">${{item.area}}</span> – ${{item.suggested_change}}`;
        li.style.marginBottom = '0.5rem';
        listEl.appendChild(li);
      }});
      
      if ((data.top_items || []).length === 0) {{
        const li = document.createElement('li');
        li.textContent = 'No suggestions available.';
        li.style.color = '#999';
        listEl.appendChild(li);
      }}
      
      promptEl.value = data.builder_prompt || '';
    }}
    
    async function runStewardCheck() {{
      const statusEl = document.getElementById('steward-status');
      statusEl.textContent = 'Running Steward check...';
      statusEl.style.color = '#666';
      
      try {{
        const res = await fetch('/api/steward/run', {{ method: 'POST' }});
        const data = await res.json();
        renderStewardReport(data);
      }} catch (err) {{
        statusEl.textContent = `Error running Steward: ${{err.message}}`;
        statusEl.style.color = '#c62828';
      }}
    }}
    
    async function loadStewardReport() {{
      try {{
        const res = await fetch('/api/steward/report');
        const data = await res.json();
        renderStewardReport(data);
      }} catch (err) {{
        // Silent fail on initial load
      }}
    }}
    
    function formatNumber(num) {{
      if (num === undefined || num === null) return '--';
      if (Math.abs(num) >= 1000000) return '$' + (num / 1000000).toFixed(2) + 'M';
      if (Math.abs(num) >= 1000) return '$' + (num / 1000).toFixed(1) + 'K';
      return '$' + num.toFixed(0);
    }}
    
    function formatTime(iso) {{
      if (!iso) return '--';
      const d = new Date(iso);
      return d.toLocaleTimeString([], {{hour: '2-digit', minute:'2-digit'}});
    }}

    async function fetchStatus() {{
      try {{
        const res = await fetch('/status');
        const data = await res.json();
        
        const spot = data.state?.spot || {{}};
        const portfolio = data.state?.portfolio || {{}};
        
        document.getElementById('btc-price').innerText = spot.BTC ? '$' + spot.BTC.toLocaleString() : '--';
        document.getElementById('eth-price').innerText = spot.ETH ? '$' + spot.ETH.toLocaleString() : '--';
        document.getElementById('portfolio-value').innerText = formatNumber(portfolio.equity_usd);
        document.getElementById('margin-used').innerText = (portfolio.margin_used_pct || 0).toFixed(1) + '%';
        document.getElementById('net-delta').innerText = (portfolio.net_delta || 0).toFixed(3);
        document.getElementById('positions-count').innerText = portfolio.positions_count || 0;
        
        document.getElementById('status-box').innerText = JSON.stringify(data, null, 2);
        document.getElementById('last-update').innerText = 'Last update: ' + new Date().toLocaleTimeString();
        
        updateOpenPositions();
        updateClosedPositions();
        updateStrategyStatus();
      }} catch (err) {{
        console.error('Status fetch error:', err);
      }}
    }}
    
    function showRulesModal() {{
      document.getElementById('rules-modal').style.display = 'flex';
    }}
    
    function closeRulesModal() {{
      document.getElementById('rules-modal').style.display = 'none';
    }}
    
    async function updateStrategyStatus() {{
      try {{
        const res = await fetch('/api/strategy-status');
        const s = await res.json();
        
        const modeLabel = s.training_mode ? `Training (${{s.mode}})` : s.mode.charAt(0).toUpperCase() + s.mode.slice(1);
        const headline = `${{modeLabel}} on ${{s.network.charAt(0).toUpperCase() + s.network.slice(1)}}`;
        document.getElementById('strategy-headline').textContent = headline;
        
        const badgesHtml = [
          `<span style="display:inline-block;background:${{s.training_mode ? '#7c4dff' : '#00bcd4'}};color:#fff;font-size:0.7em;padding:2px 6px;border-radius:999px;">${{s.training_mode ? 'TRAINING' : 'LIVE'}}</span>`,
          `<span style="display:inline-block;background:#333;color:#eee;font-size:0.7em;padding:2px 6px;border-radius:999px;">${{s.network}}</span>`,
          `<span style="display:inline-block;background:${{s.dry_run ? '#ff9800' : '#4caf50'}};color:#fff;font-size:0.7em;padding:2px 6px;border-radius:999px;">${{s.dry_run ? 'DRY' : 'LIVE'}}</span>`,
        ].join(' ');
        document.getElementById('strategy-badges').innerHTML = badgesHtml;
        
        const trainingCompact = s.training_rules.notes.slice(0, 3).map(n => `<li style="margin:2px 0;font-size:0.8em;">${{n.length > 40 ? n.slice(0, 40) + '...' : n}}</li>`).join('');
        document.getElementById('training-rules-compact').innerHTML = trainingCompact;
        
        const liveCompact = s.live_rules.notes.slice(0, 3).map(n => `<li style="margin:2px 0;font-size:0.8em;">${{n.length > 40 ? n.slice(0, 40) + '...' : n}}</li>`).join('');
        document.getElementById('live-rules-compact').innerHTML = liveCompact;
        
        const safeguardsCompact = s.safeguards.slice(0, 4).map(sg => {{
          const bgColor = sg.status === 'ON' ? '#4caf50' : '#f44336';
          return `<span style="display:inline-flex;align-items:center;gap:3px;background:#333;padding:2px 6px;border-radius:4px;font-size:0.75em;color:#fff;">
            <span style="width:6px;height:6px;border-radius:50%;background:${{bgColor}};"></span>
            ${{sg.name}}
          </span>`;
        }}).join('');
        document.getElementById('safeguards-compact').innerHTML = safeguardsCompact;
        
        document.getElementById('training-rules-desc').textContent = s.training_rules.description;
        document.getElementById('training-rules-notes').innerHTML = s.training_rules.notes.map(n => `<li style="margin-bottom:4px;">${{n}}</li>`).join('');
        
        document.getElementById('live-rules-desc').textContent = s.live_rules.description;
        document.getElementById('live-rules-notes').innerHTML = s.live_rules.notes.map(n => `<li style="margin-bottom:4px;">${{n}}</li>`).join('');
        
        const safeguardsFull = s.safeguards.map(sg => {{
          const bgColor = sg.status === 'ON' ? '#4caf50' : '#f44336';
          return `<div style="background:#2a2a2a;border-radius:4px;padding:4px 8px;font-size:0.85em;">
            <span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${{bgColor}};margin-right:4px;"></span>
            <strong>${{sg.name}}</strong>: ${{sg.status}}
            <div style="font-size:0.85em;color:#888;">${{sg.details}}</div>
          </div>`;
        }}).join('');
        document.getElementById('safeguards-full').innerHTML = safeguardsFull;
      }} catch (err) {{
        console.error('Strategy status fetch error:', err);
      }}
    }}
    
    async function updateOpenPositions() {{
      await updateDashboardPositions();
    }}
    
    async function updateClosedPositions() {{
      try {{
        const res = await fetch('/api/positions/closed');
        const data = await res.json();
        const tbody = document.getElementById('live-closed-positions-body');
        const chains = data.chains || [];

        if (chains.length === 0) {{
          tbody.innerHTML = '<tr><td colspan="11" style="text-align:center;color:#666;">No closed chains yet</td></tr>';
          return;
        }}

        tbody.innerHTML = chains.slice(-50).reverse().map(chain => {{
          const t = new Date(chain.close_time).toISOString().replace('T', ' ').slice(0, 19);
          const typeLabel = 'SHORT ' + (chain.option_type || 'CALL');
          const stratLabel = (chain.strategy_type || '').replace(/_/g, ' ');
          const pnlClass = chain.realized_pnl >= 0 ? 'traded-yes' : 'traded-no';

          return `<tr>
            <td>${{t}}</td>
            <td>${{chain.underlying}}</td>
            <td>${{typeLabel}}</td>
            <td>${{stratLabel}}</td>
            <td>${{chain.symbol}}</td>
            <td>${{chain.num_legs}}</td>
            <td>${{chain.num_rolls}}</td>
            <td class="${{pnlClass}}">${{chain.realized_pnl.toFixed(2)}}</td>
            <td class="${{pnlClass}}">${{chain.realized_pnl_pct.toFixed(1)}}%</td>
            <td>${{chain.max_drawdown_pct.toFixed(1)}}%</td>
            <td>${{chain.mode}}</td>
          </tr>`;
        }}).join('');
      }} catch (err) {{
        console.error('Closed positions fetch error:', err);
      }}
    }}
    
    async function updateDashboardSensors() {{
      const tbody = document.getElementById('dashboard-sensors-body');
      const debugToggle = document.getElementById('dashboard-show-sensor-debug');
      const debugMode = debugToggle && debugToggle.checked ? '1' : '0';
      
      try {{
        const res = await fetch(`/api/bots/market_sensors?debug=${{debugMode}}`);
        const data = await res.json();
        
        if (!data.ok) {{
          tbody.innerHTML = '<tr><td colspan="3" style="text-align:center;color:#f44336;">Failed to load sensors</td></tr>';
          return;
        }}
        
        const sensors = data.sensors || {{}};
        const sensorNames = ['iv_30d', 'rv_30d', 'vrp_30d', 'chop_factor_7d', 'iv_rank_6m', 'term_structure_spread', 'skew_25d', 'adx_14d', 'rsi_14d', 'price_vs_ma200'];
        const sensorLabels = {{'iv_30d': 'IV 30d (DVOL)', 'rv_30d': 'RV 30d (Deribit)', 'vrp_30d': 'VRP 30d (IV-RV)', 'chop_factor_7d': 'Chop Factor 7d', 'iv_rank_6m': 'IV Rank 6m', 'term_structure_spread': 'Term Spread', 'skew_25d': 'Skew 25d', 'adx_14d': 'ADX 14d', 'rsi_14d': 'RSI 14d', 'price_vs_ma200': 'Price vs MA200'}};
        
        const btcSensors = sensors['BTC'] || {{}};
        const ethSensors = sensors['ETH'] || {{}};
        
        let html = '';
        for (const name of sensorNames) {{
          const btcVal = btcSensors[name];
          const ethVal = ethSensors[name];
          const btcDisplay = btcVal !== null && btcVal !== undefined ? Number(btcVal).toFixed(name === 'chop_factor_7d' ? 3 : 2) : '--';
          const ethDisplay = ethVal !== null && ethVal !== undefined ? Number(ethVal).toFixed(name === 'chop_factor_7d' ? 3 : 2) : '--';
          html += `<tr>
            <td>${{sensorLabels[name] || name}}</td>
            <td>${{btcDisplay}}</td>
            <td>${{ethDisplay}}</td>
          </tr>`;
        }}
        tbody.innerHTML = html;
        console.log('Dashboard sensors updated:', sensorNames.length, 'sensors');
      }} catch (err) {{
        console.error('Dashboard sensors fetch error:', err);
        tbody.innerHTML = '<tr><td colspan="3" style="text-align:center;color:#f44336;">Error loading sensors</td></tr>';
      }}
    }}
    
    async function updateDashboardPositions() {{
      const testTbody = document.getElementById('dashboard-test-positions-body');
      const liveTbody = document.getElementById('dashboard-live-positions-body');
      const summaryEl = document.getElementById('positions-pnl-summary');
      
      try {{
        const res = await fetch('/api/positions/open');
        const data = await res.json();
        const positions = data.positions || [];
        const totals = data.totals || {{}};
        
        const testPositions = positions.filter(p => p.mode === 'DRY_RUN');
        const livePositions = positions.filter(p => p.mode !== 'DRY_RUN');
        
        const totalPnl = totals.unrealized_pnl || 0;
        const pnlColor = totalPnl >= 0 ? '#26a69a' : '#ef5350';
        summaryEl.innerHTML = `Total Unrealized: <span style="color:${{pnlColor}};font-weight:600;">${{totalPnl >= 0 ? '+' : ''}}${{totalPnl.toFixed(2)}}</span>`;
        
        const renderPositionRow = (pos) => {{
          const stratLabel = (pos.strategy_type || '').replace(/_/g, ' ');
          const pnlClass = pos.unrealized_pnl >= 0 ? 'traded-yes' : 'traded-no';
          const entryMode = pos.entry_mode || 'NATURAL';
          return `<tr>
            <td>${{pos.underlying}}</td>
            <td>${{stratLabel}}</td>
            <td>${{pos.symbol}}</td>
            <td>${{pos.quantity.toFixed(3)}}</td>
            <td>${{pos.entry_price.toFixed(6)}}</td>
            <td>${{pos.mark_price.toFixed(6)}}</td>
            <td class="${{pnlClass}}">${{pos.unrealized_pnl.toFixed(2)}}</td>
            <td class="${{pnlClass}}">${{pos.unrealized_pnl_pct.toFixed(1)}}%</td>
            <td>${{Math.max(0, pos.dte).toFixed(1)}}</td>
            <td>${{pos.num_rolls}}</td>
            <td>${{entryMode}}</td>
          </tr>`;
        }};
        
        if (testPositions.length === 0) {{
          testTbody.innerHTML = '<tr><td colspan="11" style="text-align:center;color:#666;">No test positions</td></tr>';
        }} else {{
          testTbody.innerHTML = testPositions.map(renderPositionRow).join('');
        }}
        
        if (livePositions.length === 0) {{
          liveTbody.innerHTML = '<tr><td colspan="11" style="text-align:center;color:#666;">No live positions</td></tr>';
        }} else {{
          liveTbody.innerHTML = livePositions.map(renderPositionRow).join('');
        }}
      }} catch (err) {{
        console.error('Dashboard positions fetch error:', err);
        testTbody.innerHTML = '<tr><td colspan="11" style="text-align:center;color:#f44336;">Error loading positions</td></tr>';
        liveTbody.innerHTML = '<tr><td colspan="11" style="text-align:center;color:#f44336;">Error loading positions</td></tr>';
      }}
    }}
    
    function initDashboard() {{
      const sensorsBtn = document.getElementById('dashboard-refresh-sensors-btn');
      if (sensorsBtn) {{
        sensorsBtn.addEventListener('click', updateDashboardSensors);
      }}
      const debugToggle = document.getElementById('dashboard-show-sensor-debug');
      if (debugToggle) {{
        debugToggle.addEventListener('change', updateDashboardSensors);
      }}
      updateDashboardSensors();
      updateDashboardPositions();
      setInterval(updateDashboardSensors, 30000);
      setInterval(updateDashboardPositions, 30000);
    }}
    
    async function fetchDecisions() {{
      try {{
        const res = await fetch('/api/agent/decisions');
        const data = await res.json();
        
        const decisions = data.decisions || [];
        
        if (data.last_update) {{
          const lastUpdate = new Date(data.last_update);
          const now = new Date();
          const diffMin = (now - lastUpdate) / 60000;
          const badge = document.getElementById('agent-status-badge');
          if (diffMin < 10) {{
            badge.style.background = '#e8f5e9';
            badge.style.color = '#2e7d32';
            badge.innerText = 'Active';
          }} else {{
            badge.style.background = '#fff3e0';
            badge.style.color = '#e65100';
            badge.innerText = 'Idle';
          }}
        }}
        
        if (decisions.length > 0) {{
          const latest = decisions[0];
          const proposed = latest.proposed_action || {{}};
          const final = latest.final_action || {{}};
          const risk = latest.risk_check || {{}};
          const exec = latest.execution || {{}};
          
          document.getElementById('decision-time').innerText = 'Last Decision - ' + formatTime(latest.timestamp);
          document.getElementById('decision-action').innerText = (final.action || proposed.action || '--').replace(/_/g, ' ');
          
          const params = final.params || proposed.params || {{}};
          let details = '';
          if (params.symbol) details += params.symbol + ' ';
          if (params.to_symbol) details += '-> ' + params.to_symbol + ' ';
          if (params.size) details += 'x' + params.size;
          document.getElementById('decision-details').innerText = details || 'No params';
          
          const reasoning = proposed.reasoning || final.reasoning || 'No reasoning provided';
          document.getElementById('decision-reasoning').innerText = reasoning.length > 150 ? reasoning.slice(0, 150) + '...' : reasoning;
        }}
        
        const miniTimeline = document.getElementById('mini-timeline');
        const recentDecs = decisions.slice(0, 5);
        if (recentDecs.length === 0) {{
          miniTimeline.innerHTML = '<span style="color:#666;font-size:0.85em;">No recent decisions</span>';
        }} else {{
          miniTimeline.innerHTML = recentDecs.map(d => {{
            const action = (d.final_action || {{}}).action || (d.proposed_action || {{}}).action || 'UNKNOWN';
            const isDoNothing = action === 'DO_NOTHING';
            const isTrade = action.includes('OPEN') || action.includes('ROLL') || action.includes('CLOSE');
            const params = (d.final_action || {{}}).params || (d.proposed_action || {{}}).params || {{}};
            const symbol = params.symbol || '';
            const shortSymbol = symbol ? symbol.split('-').slice(0, 2).join('-') : '';
            
            let bgColor = '#555';
            let textColor = '#ccc';
            if (isTrade) {{ bgColor = '#2e7d32'; textColor = '#fff'; }}
            else if (isDoNothing) {{ bgColor = '#424242'; textColor = '#888'; }}
            
            const source = (d.decision_source || '').toUpperCase();
            const sourceIcon = source === 'LLM' ? 'AI' : source === 'RULE_BASED' ? 'RB' : source === 'TRAINING_MODE' ? 'TR' : '';
            
            return `<div style="display:inline-flex;align-items:center;gap:4px;background:${{bgColor}};color:${{textColor}};padding:4px 8px;border-radius:4px;font-size:0.75em;">
              <span style="opacity:0.7;">${{formatTime(d.timestamp)}}</span>
              <span style="font-weight:600;">${{action.replace(/_/g, ' ').slice(0, 12)}}</span>
              ${{shortSymbol ? `<span style="opacity:0.7;">${{shortSymbol}}</span>` : ''}}
              ${{sourceIcon ? `<span style="background:rgba(255,255,255,0.2);padding:1px 4px;border-radius:2px;font-size:0.85em;">${{sourceIcon}}</span>` : ''}}
            </div>`;
          }}).join('');
        }}
        
        const tbody = document.getElementById('decisions-tbody');
        if (decisions.length === 0) {{
          tbody.innerHTML = '<tr><td colspan="6" style="text-align:center;color:#666;">No decisions yet</td></tr>';
        }} else {{
          const filteredDecisions = [];
          let lastWasDoNothing = false;
          for (const d of decisions) {{
            const final = d.final_action || {{}};
            const action = final.action || (d.proposed_action || {{}}).action || '';
            const isDoNothing = action === 'DO_NOTHING';
            
            if (isDoNothing && lastWasDoNothing) {{
              continue;
            }}
            filteredDecisions.push(d);
            lastWasDoNothing = isDoNothing;
          }}
          
          tbody.innerHTML = filteredDecisions.slice(0, 20).map(d => {{
            const proposed = d.proposed_action || {{}};
            const final = d.final_action || {{}};
            const risk = d.risk_check || {{}};
            const exec = d.execution || {{}};
            
            const riskClass = risk.allowed ? 'pass' : 'blocked';
            const riskText = risk.allowed ? 'PASS' : 'BLOCKED';
            
            return `<tr>
              <td>${{formatTime(d.timestamp)}}</td>
              <td>${{(d.decision_source || '--').toUpperCase()}}</td>
              <td>${{(proposed.action || '--').replace(/_/g, ' ')}}</td>
              <td>${{(final.action || '--').replace(/_/g, ' ')}}</td>
              <td class="${{riskClass}}">${{riskText}}</td>
              <td>${{exec.status || '--'}}</td>
            </tr>`;
          }}).join('');
        }}
      }} catch (err) {{
        console.error('Decisions fetch error:', err);
      }}
    }}

    function renderChatMessages(messages) {{
      const chatLog = document.getElementById('chat-log');
      if (!messages || messages.length === 0) {{
        chatLog.innerHTML = '<div style="text-align: center; color: #888; padding: 20px;">Start a conversation by asking a question below...</div>';
        return;
      }}
      
      chatLog.innerHTML = messages.map(msg => {{
        const isUser = msg.role === 'user';
        const bgColor = isUser ? '#e3f2fd' : '#f5f5f5';
        const textColor = '#333';
        const align = isUser ? 'flex-end' : 'flex-start';
        const label = isUser ? 'You' : 'Agent';
        const labelColor = isUser ? '#1565c0' : '#2e7d32';
        return `
          <div style="display: flex; justify-content: ${{align}};">
            <div style="max-width: 85%; background: ${{bgColor}}; padding: 10px 14px; border-radius: 12px; color: ${{textColor}};">
              <div style="font-size: 11px; color: ${{labelColor}}; margin-bottom: 4px; font-weight: 600;">${{label}}</div>
              <div style="white-space: pre-wrap; line-height: 1.5;">${{msg.content.replace(/</g, '&lt;').replace(/>/g, '&gt;')}}</div>
            </div>
          </div>
        `;
      }}).join('');
      
      const chatContainer = document.getElementById('chat-messages');
      chatContainer.scrollTop = chatContainer.scrollHeight;
    }}
    
    async function fetchBacktestRuns() {{
      try {{
        const res = await fetch('/api/backtests');
        const runs = await res.json();
        renderBacktestRuns(runs);
      }} catch (err) {{
        console.error('Failed to fetch backtest runs:', err);
        document.getElementById('runs-table-body').innerHTML = '<tr><td colspan="11" style="text-align:center;color:#ff5252;">Error loading runs</td></tr>';
      }}
    }}
    
    function renderBacktestRuns(runs) {{
      const tbody = document.getElementById('runs-table-body');
      if (!runs || runs.length === 0) {{
        tbody.innerHTML = '<tr><td colspan="11" style="text-align:center;color:#666;">No backtest runs yet. Run a backtest from the Backtesting Lab to see results here.</td></tr>';
        return;
      }}
      
      tbody.innerHTML = runs.map(run => {{
        const shortId = run.run_id.length > 20 ? run.run_id.substring(0, 20) + '...' : run.run_id;
        const createdDate = new Date(run.created_at).toLocaleString();
        const startDate = run.start_date ? run.start_date.split('T')[0] : '--';
        const endDate = run.end_date ? run.end_date.split('T')[0] : '--';
        const dateRange = startDate + ' to ' + endDate;
        
        let statusColor = '#888';
        if (run.status === 'finished') statusColor = '#4caf50';
        else if (run.status === 'running') statusColor = '#2196f3';
        else if (run.status === 'failed') statusColor = '#f44336';
        else if (run.status === 'queued') statusColor = '#ff9800';
        
        const netPnl = (run.net_profit_pct || 0).toFixed(2);
        const netPnlColor = parseFloat(netPnl) >= 0 ? '#4caf50' : '#f44336';
        const maxDD = (run.max_drawdown_pct || 0).toFixed(2);
        const sharpe = (run.sharpe_ratio || 0).toFixed(2);
        const numTrades = run.num_trades || 0;
        
        return `
          <tr>
            <td title="${{run.run_id}}">${{shortId}}</td>
            <td>${{createdDate}}</td>
            <td>${{run.underlying || '--'}}</td>
            <td style="font-size:0.85em;">${{dateRange}}</td>
            <td><span style="color:${{statusColor}};font-weight:600;">${{run.status}}</span></td>
            <td>${{run.primary_exit_style || '--'}}</td>
            <td style="color:${{netPnlColor}};font-weight:600;">${{netPnl}}%</td>
            <td style="color:#f44336;">${{maxDD}}%</td>
            <td>${{sharpe}}</td>
            <td>${{numTrades}}</td>
            <td>
              <button onclick="viewRunDetail('${{run.run_id}}')" style="background:#2196f3;color:#fff;border:none;padding:4px 8px;border-radius:3px;cursor:pointer;margin-right:4px;font-size:0.8em;">View</button>
              <a href="/api/backtests/${{run.run_id}}/download" style="background:#4caf50;color:#fff;border:none;padding:4px 8px;border-radius:3px;cursor:pointer;text-decoration:none;font-size:0.8em;">Download</a>
            </td>
          </tr>
        `;
      }}).join('');
    }}
    
    async function viewRunDetail(runId) {{
      try {{
        const res = await fetch('/api/backtests/' + runId);
        if (!res.ok) {{
          alert('Failed to load run details');
          return;
        }}
        const data = await res.json();
        showRunDetailModal(data);
      }} catch (err) {{
        alert('Error loading run: ' + err);
      }}
    }}
    
    function showRunDetailModal(run) {{
      document.getElementById('run-detail-title').innerText = 'Run: ' + run.run_id;
      
      const configGrid = document.getElementById('run-config-grid');
      const cfg = run.config || {{}};
      configGrid.innerHTML = `
        <div><span style="color:#888;">Underlying:</span> ${{cfg.underlying || '--'}}</div>
        <div><span style="color:#888;">Start:</span> ${{cfg.start_date || '--'}}</div>
        <div><span style="color:#888;">End:</span> ${{cfg.end_date || '--'}}</div>
        <div><span style="color:#888;">Timeframe:</span> ${{cfg.timeframe || '--'}}</div>
        <div><span style="color:#888;">Exit Style:</span> ${{cfg.exit_style || '--'}}</div>
        <div><span style="color:#888;">Target DTE:</span> ${{cfg.target_dte || '--'}}</div>
        <div><span style="color:#888;">Target Delta:</span> ${{cfg.target_delta || '--'}}</div>
        <div><span style="color:#888;">Status:</span> ${{run.status}}</div>
      `;
      
      const renderMetrics = (m) => {{
        if (!m) return '<div style="color:#666;">No data</div>';
        return `
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:4px;font-size:0.85em;">
            <div><span style="color:#888;">Net PnL:</span> ${{(m.net_profit_pct || 0).toFixed(2)}}%</div>
            <div><span style="color:#888;">Net USD:</span> $${{(m.net_profit_usd || 0).toFixed(2)}}</div>
            <div><span style="color:#888;">Max DD:</span> ${{(m.max_drawdown_pct || 0).toFixed(2)}}%</div>
            <div><span style="color:#888;">Trades:</span> ${{m.num_trades || 0}}</div>
            <div><span style="color:#888;">Win Rate:</span> ${{(m.win_rate || 0).toFixed(1)}}%</div>
            <div><span style="color:#888;">Sharpe:</span> ${{(m.sharpe_ratio || 0).toFixed(2)}}</div>
            <div><span style="color:#888;">Sortino:</span> ${{(m.sortino_ratio || 0).toFixed(2)}}</div>
            <div><span style="color:#888;">PF:</span> ${{(m.profit_factor || 0).toFixed(2)}}</div>
          </div>
        `;
      }};
      
      const metrics = run.metrics || {{}};
      document.getElementById('run-metrics-hte-content').innerHTML = renderMetrics(metrics.hold_to_expiry || metrics);
      document.getElementById('run-metrics-tpr-content').innerHTML = renderMetrics(metrics.tp_and_roll || metrics);
      
      const chains = (run.recent_chains || {{}}).tp_and_roll || [];
      const chainsBody = document.getElementById('run-chains-body');
      if (chains.length === 0) {{
        chainsBody.innerHTML = '<tr><td colspan="6" style="text-align:center;color:#666;">No chains recorded</td></tr>';
      }} else {{
        chainsBody.innerHTML = chains.slice(0, 20).map(c => `
          <tr>
            <td>${{c.decision_time}}</td>
            <td>${{c.underlying}}</td>
            <td>${{c.num_legs}}</td>
            <td>${{c.num_rolls}}</td>
            <td style="color:${{c.total_pnl >= 0 ? '#4caf50' : '#f44336'}};">${{c.total_pnl.toFixed(2)}}</td>
            <td>${{c.max_drawdown_pct.toFixed(2)}}%</td>
          </tr>
        `).join('');
      }}
      
      document.getElementById('run-detail-modal').style.display = 'block';
    }}
    
    function closeRunDetailModal() {{
      document.getElementById('run-detail-modal').style.display = 'none';
    }}
    
    async function loadChatHistory() {{
      try {{
        const res = await fetch('/chat/messages');
        const data = await res.json();
        renderChatMessages(data.messages || []);
      }} catch (err) {{
        console.error('Failed to load chat history:', err);
      }}
    }}
    
    async function sendQuestion() {{
      const q = document.getElementById('question').value;
      if (!q.trim()) {{
        alert('Please enter a question first.');
        return;
      }}
      
      const btn = document.getElementById('ask-btn');
      btn.disabled = true;
      btn.innerText = 'Thinking...';
      
      const chatLog = document.getElementById('chat-log');
      const thinkingDiv = document.createElement('div');
      thinkingDiv.style.cssText = 'text-align: center; color: #888; padding: 10px;';
      thinkingDiv.innerHTML = '<span style="animation: pulse 1.5s infinite;">Analyzing state and generating response...</span>';
      thinkingDiv.id = 'thinking-indicator';
      chatLog.appendChild(thinkingDiv);
      document.getElementById('chat-messages').scrollTop = document.getElementById('chat-messages').scrollHeight;

      try {{
        const res = await fetch('/chat', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify({{ question: q }})
        }});
        const data = await res.json();
        
        if (data.error) {{
          alert('Error: ' + data.error);
        }} else {{
          renderChatMessages(data.messages || []);
          document.getElementById('question').value = '';
        }}
      }} catch (err) {{
        alert('Error: ' + err);
      }} finally {{
        const indicator = document.getElementById('thinking-indicator');
        if (indicator) indicator.remove();
        btn.disabled = false;
        btn.innerText = 'Ask Agent';
      }}
    }}
    
    async function clearChat() {{
      try {{
        await fetch('/chat/clear', {{ method: 'POST' }});
        renderChatMessages([]);
      }} catch (err) {{
        console.error('Failed to clear chat:', err);
      }}
    }}
    
    function updateCalibrationCoverage(data) {{
      // Update option types badge
      const typesBadge = document.getElementById('calib-types-badge');
      const typesUsed = data.option_types_used || ['C'];
      let typesLabel = 'Calls only';
      if (typesUsed.includes('C') && typesUsed.includes('P')) {{
        typesLabel = 'Calls + Puts';
      }} else if (typesUsed.includes('P')) {{
        typesLabel = 'Puts only';
      }}
      typesBadge.textContent = typesLabel;
      
      // Update explanation text
      const explanationEl = document.getElementById('calib-coverage-explanation');
      const bandsInfo = data.bands ? `across ${{data.bands.length}} term buckets` : 'across weekly, monthly, quarterly buckets';
      if (typesUsed.includes('C') && typesUsed.includes('P')) {{
        explanationEl.innerHTML = `Calibration now measures both <strong>calls</strong> and <strong>puts</strong> ${{bandsInfo}}. Differences between call and put errors can indicate skew or crash risk mismatches.`;
      }} else if (typesUsed.includes('P')) {{
        explanationEl.innerHTML = `Calibration measures <strong>puts only</strong> ${{bandsInfo}} for skew/crash analysis.`;
      }} else {{
        explanationEl.innerHTML = `Calibration measures <strong>calls only</strong> ${{bandsInfo}}.`;
      }}
      
      // Update by-type metrics table
      const byTypeBody = document.getElementById('calib-by-type-body');
      const byOptionType = data.by_option_type || {{}};
      const typeKeys = Object.keys(byOptionType);
      
      if (typeKeys.length === 0) {{
        byTypeBody.innerHTML = '<tr><td colspan="6" style="text-align:center;color:#666;">No per-type metrics available</td></tr>';
      }} else {{
        byTypeBody.innerHTML = typeKeys.map(typeCode => {{
          const m = byOptionType[typeCode];
          const typeName = typeCode === 'C' ? 'Calls (C)' : 'Puts (P)';
          return `<tr>
            <td style="font-weight:600;">${{typeName}}</td>
            <td>${{m.count || 0}}</td>
            <td>${{(m.mae_pct || 0).toFixed(2)}}%</td>
            <td>${{(m.bias_pct || 0).toFixed(2)}}%</td>
            <td>${{m.mae_vol_points ? m.mae_vol_points.toFixed(3) : '-'}}</td>
            <td>${{m.vega_weighted_mae_pct ? m.vega_weighted_mae_pct.toFixed(2) + '%' : '-'}}</td>
          </tr>`;
        }}).join('');
      }}
      
      // Update term buckets table
      const termBody = document.getElementById('calib-term-buckets-body');
      
      // Use term_structure_bands if available (weekly/monthly/quarterly from broader DTE range)
      let bandRows = [];
      if (data.term_structure_bands && data.term_structure_bands.length > 0) {{
        bandRows = data.term_structure_bands.map(b => ({{
          name: b.band_name,
          dte_range: b.dte_range,
          option_type: b.option_type || 'Calls',
          count: b.count,
          mae_pct: b.mae_pct,
          vega_weighted_mae_pct: b.vega_weighted_mae_pct,
          bias_pct: b.bias_pct,
          rec_mult: b.recommended_iv_multiplier,
        }}));
      }} else {{
        // Fallback: collect per-type bands
        for (const typeCode of typeKeys) {{
          const typeData = byOptionType[typeCode];
          if (typeData.bands && typeData.bands.length > 0) {{
            for (const band of typeData.bands) {{
              bandRows.push({{
                name: band.name,
                dte_range: `${{band.min_dte}}-${{band.max_dte}}`,
                option_type: typeCode === 'C' ? 'Calls' : 'Puts',
                count: band.count,
                mae_pct: band.mae_pct,
                vega_weighted_mae_pct: band.vega_weighted_mae_pct,
                bias_pct: band.bias_pct,
                rec_mult: band.recommended_iv_multiplier,
              }});
            }}
          }}
        }}
        
        // If still no bands, fall back to global bands
        if (bandRows.length === 0 && data.bands && data.bands.length > 0) {{
          bandRows = data.bands.map(b => ({{
            name: b.band_name,
            dte_range: b.dte_range,
            option_type: 'All',
            count: b.count,
            mae_pct: b.mae_pct,
            vega_weighted_mae_pct: b.vega_weighted_mae_pct,
            bias_pct: b.bias_pct,
            rec_mult: b.recommended_iv_multiplier,
          }}));
        }}
      }}
      
      if (bandRows.length === 0) {{
        termBody.innerHTML = '<tr><td colspan="8" style="text-align:center;color:#666;">No term buckets available</td></tr>';
      }} else {{
        termBody.innerHTML = bandRows.map(b => {{
          const vegaWtd = b.vega_weighted_mae_pct != null ? (b.vega_weighted_mae_pct).toFixed(2) + '%' : '-';
          return `<tr>
            <td>${{b.name}}</td>
            <td>${{b.dte_range}}d</td>
            <td>${{b.option_type}}</td>
            <td>${{b.count}}</td>
            <td>${{(b.mae_pct || 0).toFixed(2)}}%</td>
            <td>${{vegaWtd}}</td>
            <td>${{(b.bias_pct || 0).toFixed(2)}}%</td>
            <td>${{b.rec_mult ? b.rec_mult.toFixed(4) : '-'}}</td>
          </tr>`;
        }}).join('');
      }}
      
      // Update skew fit panel
      const skewSection = document.getElementById('calib-skew-fit-section');
      const skewBody = document.getElementById('calib-skew-fit-body');
      const skewSummary = document.getElementById('skew-fit-summary');
      
      if (data.skew_fit && data.skew_fit.recommended_skew) {{
        skewSection.style.display = 'block';
        const rec = data.skew_fit.recommended_skew;
        const cur = data.skew_fit.current_skew;
        const misfit = data.skew_fit.skew_misfit;
        
        const deltaLabels = {{'0.15': '15-delta OTM', '0.25': '25-delta OTM', '0.35': '35-delta OTM'}};
        let rowsHtml = '';
        
        for (const [delta, recRatio] of Object.entries(rec.anchor_ratios || {{}})) {{
          const curRatio = cur && cur.anchor_ratios ? (cur.anchor_ratios[delta] || 1.0) : 1.0;
          const diff = misfit && misfit.anchor_diffs ? misfit.anchor_diffs[delta] : null;
          const diffStr = diff != null ? (diff > 0 ? '+' : '') + diff.toFixed(4) : '-';
          const diffColor = diff != null ? (Math.abs(diff) > 0.02 ? '#e65100' : '#2e7d32') : '#333';
          
          rowsHtml += `<tr>
            <td>${{deltaLabels[delta] || delta}}</td>
            <td>${{curRatio.toFixed(4)}}</td>
            <td>${{recRatio.toFixed(4)}}</td>
            <td style="color:${{diffColor}};font-weight:600;">${{diffStr}}</td>
          </tr>`;
        }}
        
        skewBody.innerHTML = rowsHtml || '<tr><td colspan="4" style="text-align:center;color:#666;">No skew data</td></tr>';
        
        // Summary
        if (misfit && misfit.max_abs_diff != null) {{
          const maxDiff = misfit.max_abs_diff;
          const status = maxDiff > 0.05 ? 'Significant misfit' : maxDiff > 0.02 ? 'Minor misfit' : 'Good fit';
          const statusColor = maxDiff > 0.05 ? '#c62828' : maxDiff > 0.02 ? '#ef6c00' : '#2e7d32';
          skewSummary.innerHTML = `<span style="background:${{statusColor}};color:#fff;padding:4px 10px;border-radius:12px;font-size:0.85rem;">${{status}}</span> <span style="color:#666;margin-left:8px;">Max absolute diff: ${{maxDiff.toFixed(4)}}</span>`;
        }} else {{
          skewSummary.innerHTML = '';
        }}
      }} else {{
        skewSection.style.display = 'none';
      }}
    }}
    
    function updateDataHealthPanel(data) {{
      // Get data_quality from the response
      const dq = data.data_quality;
      
      if (!dq) {{
        // No data quality info available
        document.getElementById('data-health-summary').textContent = 'No data quality info (live calibration)';
        return;
      }}
      
      // Update status badge
      const badge = document.getElementById('data-health-badge');
      const status = (dq.status || 'ok').toLowerCase();
      
      if (status === 'ok') {{
        badge.style.background = '#c8e6c9';
        badge.style.color = '#2e7d32';
        badge.textContent = 'OK';
      }} else if (status === 'degraded') {{
        badge.style.background = '#ffe0b2';
        badge.style.color = '#e65100';
        badge.textContent = 'DEGRADED';
      }} else {{
        badge.style.background = '#ffcdd2';
        badge.style.color = '#c62828';
        badge.textContent = 'FAILED';
      }}
      
      // Update summary text
      const completeness = ((dq.overall_non_null_core_fraction || 0) * 100).toFixed(0);
      const summaryText = `${{dq.num_snapshots || 0}} snapshots checked, ${{completeness}}% core-field completeness`;
      document.getElementById('data-health-summary').textContent = summaryText;
      
      // Update metrics
      document.getElementById('dh-num-snapshots').textContent = dq.num_snapshots || 0;
      document.getElementById('dh-schema-issues').textContent = dq.num_schema_failures || 0;
      document.getElementById('dh-low-quality').textContent = dq.num_low_quality_snapshots || 0;
      document.getElementById('dh-completeness').textContent = `${{completeness}}%`;
      
      // Update issues list
      const issuesList = document.getElementById('dh-issues-list');
      const issuesUl = document.getElementById('dh-issues-ul');
      
      if (dq.issues && dq.issues.length > 0) {{
        issuesList.style.display = 'block';
        issuesUl.innerHTML = dq.issues.map(issue => `<li>${{issue}}</li>`).join('');
      }} else {{
        issuesList.style.display = 'none';
      }}
    }}
    
    let lastReproducibilityMetadata = null;
    let lastCalibrationResult = null;
    
    function updateReproducibilityPanel(data) {{
      const repro = data.reproducibility;
      const reproSection = document.getElementById('repro-section');
      
      if (!repro) {{
        reproSection.style.display = 'none';
        return;
      }}
      
      reproSection.style.display = 'block';
      lastReproducibilityMetadata = repro;
      
      // Update timestamp
      const timestamp = data.timestamp || data.time_range_end;
      if (timestamp) {{
        const dt = new Date(timestamp);
        document.getElementById('repro-timestamp').textContent = dt.toLocaleString();
      }} else {{
        document.getElementById('repro-timestamp').textContent = '-';
      }}
      
      // Update underlying
      document.getElementById('repro-underlying').textContent = data.underlying || '-';
      
      // Update harvest period
      const hc = repro.harvest_config;
      if (hc) {{
        const startStr = hc.start_time ? new Date(hc.start_time).toLocaleDateString() : 'start';
        const endStr = hc.end_time ? new Date(hc.end_time).toLocaleDateString() : 'end';
        document.getElementById('repro-period').textContent = `${{startStr}} to ${{endStr}}`;
      }} else {{
        document.getElementById('repro-period').textContent = '-';
      }}
      
      // Update config hash
      document.getElementById('repro-config-hash').textContent = repro.calibration_config_hash || '-';
      
      // Update regimes info
      const rv = repro.greg_regimes_version;
      if (rv) {{
        const lastMod = rv.last_modified ? new Date(rv.last_modified).toLocaleDateString() : 'unknown';
        document.getElementById('repro-regimes').textContent = `greg_regimes.json (modified ${{lastMod}}, hash: ${{rv.hash || 'n/a'}})`;
      }} else {{
        document.getElementById('repro-regimes').textContent = 'Not available';
      }}
      
      // Hide raw metadata container on new data
      document.getElementById('raw-metadata-container').style.display = 'none';
    }}
    
    function toggleRawMetadata() {{
      const container = document.getElementById('raw-metadata-container');
      
      if (container.style.display === 'none') {{
        container.style.display = 'block';
        container.textContent = JSON.stringify(lastReproducibilityMetadata, null, 2);
      }} else {{
        container.style.display = 'none';
      }}
    }}
    
    async function runCalibration() {{
      const btn = document.getElementById('calib-run-btn');
      const underlying = document.getElementById('calib-underlying').value || 'BTC';
      const minDte = parseFloat(document.getElementById('calib-min-dte').value || '3');
      const maxDte = parseFloat(document.getElementById('calib-max-dte').value || '10');
      const ivMult = parseFloat(document.getElementById('calib-iv-mult').value || '1.0');

      const summaryEl = document.getElementById('calib-summary');
      const tbody = document.getElementById('calib-rows-body');

      summaryEl.textContent = 'Running calibration...';
      tbody.innerHTML = '<tr><td colspan="6" style="text-align:center;color:#666;">Loading...</td></tr>';
      btn.disabled = true;

      try {{
        const params = new URLSearchParams({{
          underlying,
          min_dte: String(minDte),
          max_dte: String(maxDte),
          iv_multiplier: String(ivMult),
        }});

        const res = await fetch(`/api/calibration?${{params.toString()}}`);
        if (!res.ok) {{
          throw new Error(`HTTP ${{res.status}}`);
        }}
        const data = await res.json();

        const mae = (data.mae_pct ?? 0).toFixed(2);
        const bias = (data.bias_pct ?? 0).toFixed(2);
        const count = data.count ?? 0;
        const spot = data.spot ?? 0;
        const rv = data.rv_annualized;
        const atmIv = data.atm_iv;
        const recMult = data.recommended_iv_multiplier;

        let line1 =
          `Underlying ${{data.underlying}} @ ${{spot.toFixed ? spot.toFixed(2) : spot}} USD - ` +
          `${{count}} options in [${{data.min_dte}}d, ${{data.max_dte}}d], ` +
          `MAE ~ ${{mae}}% of mark, bias ~ ${{bias}}%.`;

        let line2 = '';

        if (rv && rv > 0) {{
          const rvPct = (rv * 100).toFixed(1);
          line2 += `RV_7d = ${{rvPct}}%`;
        }}

        if (atmIv && atmIv > 0) {{
          const atmIvPct = (atmIv * 100).toFixed(1);
          line2 += (line2 ? ' | ' : '') + `ATM IV = ${{atmIvPct}}%`;
        }}

        if (recMult && recMult > 0) {{
          const recStr = recMult.toFixed(3);
          line2 += (line2 ? ' | ' : '') + `Recommended iv_multiplier = ${{recStr}}`;
        }}

        summaryEl.innerHTML = line2
          ? `${{line1}}<br><span style="font-size:0.85rem;color:#666;">${{line2}}</span>`
          : line1;

        const rows = (data.rows || []).slice(0, 50);
        if (rows.length === 0) {{
          tbody.innerHTML = '<tr><td colspan="6" style="text-align:center;color:#666;">No options found for this range</td></tr>';
        }} else {{
          tbody.innerHTML = rows.map(row => {{
            const diffClass = row.diff_pct > 0 ? 'traded-yes' : 'traded-no';
            return `<tr>
              <td>${{row.instrument}}</td>
              <td>${{row.dte.toFixed(2)}}</td>
              <td>${{row.strike.toFixed(0)}}</td>
              <td>${{row.mark_price.toFixed(4)}}</td>
              <td>${{row.syn_price.toFixed(4)}}</td>
              <td class="${{diffClass}}">${{row.diff_pct.toFixed(2)}}%</td>
            </tr>`;
          }}).join('');
        }}
        
        // Update Calibration Coverage panels
        updateCalibrationCoverage(data);
        
        // Update Data Health & Reproducibility panels
        updateDataHealthPanel(data);
        updateReproducibilityPanel(data);
        
        // Store the calibration result for "Use Latest Recommended" button
        if (recMult && recMult > 0) {{
          lastCalibrationResult = {{
            underlying: underlying,
            dte_min: minDte,
            dte_max: maxDte,
            multiplier: recMult,
            mae_pct: parseFloat(mae),
            num_samples: count
          }};
        }} else {{
          lastCalibrationResult = null;
        }}
      }} catch (err) {{
        console.error('Calibration error:', err);
        summaryEl.textContent = 'Calibration failed. Check console/logs.';
        tbody.innerHTML = '<tr><td colspan="6" style="text-align:center;color:#c00;">Error</td></tr>';
        lastCalibrationResult = null;
      }} finally {{
        btn.disabled = false;
      }}
    }}
    
    async function fetchCalibrationHistory() {{
      const underlying = document.getElementById('calib-underlying').value || 'BTC';
      const tbody = document.getElementById('calib-history-body');
      
      tbody.innerHTML = '<tr><td colspan="9" style="text-align:center;color:#666;">Loading...</td></tr>';
      
      try {{
        const res = await fetch(`/api/calibration/history?underlying=${{underlying}}&limit=20`);
        if (!res.ok) throw new Error(`HTTP ${{res.status}}`);
        const data = await res.json();
        
        const entries = data.entries || [];
        if (entries.length === 0) {{
          tbody.innerHTML = '<tr><td colspan="9" style="text-align:center;color:#666;">No calibration history. Run scripts/auto_calibrate_iv.py --underlying BTC</td></tr>';
          return;
        }}
        
        tbody.innerHTML = entries.map(e => {{
          const dt = e.created_at ? new Date(e.created_at).toLocaleString() : 'N/A';
          const status = e.status || 'ok';
          const statusColor = status === 'ok' ? '#4caf50' : (status === 'degraded' ? '#ff9800' : '#f44336');
          const statusBadge = `<span style="background:${{statusColor}};color:#fff;padding:2px 6px;border-radius:3px;font-size:0.75rem;text-transform:uppercase;">${{status}}</span>`;
          const rowStyle = status === 'failed' ? 'opacity:0.6;' : '';
          const vMAE = e.vega_weighted_mae_pct != null ? e.vega_weighted_mae_pct.toFixed(2) + '%' : '-';
          const mae = e.mae_pct != null ? e.mae_pct.toFixed(2) + '%' : '-';
          const reason = e.reason || '-';
          return `<tr style="${{rowStyle}}">
            <td style="font-size:0.8rem;white-space:nowrap;">${{dt}}</td>
            <td>${{statusBadge}}</td>
            <td>${{e.dte_min}}-${{e.dte_max}}d</td>
            <td>${{e.lookback_days}}d</td>
            <td style="font-weight:600;">${{e.multiplier.toFixed(4)}}</td>
            <td>${{vMAE}}</td>
            <td>${{mae}}</td>
            <td>${{e.num_samples.toLocaleString()}}</td>
            <td style="max-width:200px;font-size:0.75rem;white-space:normal;word-wrap:break-word;color:#666;">${{reason}}</td>
          </tr>`;
        }}).join('');
      }} catch (err) {{
        console.error('Failed to fetch calibration history:', err);
        tbody.innerHTML = '<tr><td colspan="9" style="text-align:center;color:#c00;">Error loading history</td></tr>';
      }}
    }}
    
    async function useLatestCalibration() {{
      const btn = document.getElementById('calib-use-latest-btn');
      const underlying = document.getElementById('calib-underlying').value || 'BTC';
      const dteMin = parseInt(document.getElementById('calib-min-dte').value || '3');
      const dteMax = parseInt(document.getElementById('calib-max-dte').value || '10');
      const overrideStatus = document.getElementById('calib-override-status');
      
      // First try to use the result from the last "Run Calibration" 
      if (lastCalibrationResult && lastCalibrationResult.underlying === underlying 
          && lastCalibrationResult.dte_min === dteMin && lastCalibrationResult.dte_max === dteMax) {{
        // Apply the current calibration result directly
        btn.disabled = true;
        btn.innerText = 'Applying...';
        
        try {{
          const res = await fetch('/api/calibration/apply_direct', {{
            method: 'POST',
            headers: {{ 'Content-Type': 'application/json' }},
            body: JSON.stringify(lastCalibrationResult)
          }});
          
          const data = await res.json();
          
          if (!res.ok) {{
            alert(data.message || data.error || 'Failed to apply calibration');
            overrideStatus.style.display = 'none';
            return;
          }}
          
          const msg = `Applied ${{underlying}} calibration: multiplier=${{lastCalibrationResult.multiplier.toFixed(4)}} (from current run)`;
          overrideStatus.textContent = msg;
          overrideStatus.style.display = 'block';
          overrideStatus.style.color = '#4caf50';
          
          document.getElementById('calib-iv-mult').value = lastCalibrationResult.multiplier.toFixed(4);
        }} catch (err) {{
          console.error('Failed to apply calibration:', err);
          alert('Error applying calibration: ' + err.message);
          overrideStatus.style.display = 'none';
        }} finally {{
          btn.disabled = false;
          btn.innerText = 'Use Latest Recommended';
        }}
        return;
      }}
      
      // Fallback: fetch from history (skip failed entries)
      btn.disabled = true;
      btn.innerText = 'Applying...';
      
      try {{
        const res = await fetch('/api/calibration/use_latest', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify({{ underlying, dte_min: dteMin, dte_max: dteMax }})
        }});
        
        const data = await res.json();
        
        if (!res.ok) {{
          alert(data.message || data.error || 'Failed to apply calibration. Run a new calibration first.');
          overrideStatus.style.display = 'none';
          return;
        }}
        
        const msg = `Applied ${{underlying}} calibration: multiplier=${{data.multiplier.toFixed(4)}} (MAE ${{data.mae_pct.toFixed(2)}}%, ${{data.num_samples}} samples)`;
        overrideStatus.textContent = msg;
        overrideStatus.style.display = 'block';
        overrideStatus.style.color = '#4caf50';
        
        document.getElementById('calib-iv-mult').value = data.multiplier.toFixed(4);
        
        fetchCalibrationHistory();
      }} catch (err) {{
        console.error('Failed to apply calibration:', err);
        alert('Error applying calibration: ' + err.message);
        overrideStatus.style.display = 'none';
      }} finally {{
        btn.disabled = false;
        btn.innerText = 'Use Latest Recommended';
      }}
    }}
    
    document.getElementById('calib-underlying').addEventListener('change', function() {{
      fetchCalibrationHistory();
    }});
    
    // ===== Calibration Update Policy UI Functions =====
    
    async function refreshPolicyUI() {{
      await Promise.all([
        fetchPolicy(),
        fetchCurrentMultipliers(),
        fetchPolicyRuns()
      ]);
    }}
    
    async function fetchPolicy() {{
      try {{
        const res = await fetch('/api/calibration/policy');
        if (!res.ok) throw new Error(`HTTP ${{res.status}}`);
        const data = await res.json();
        
        const box = document.getElementById('policy-explanation-box');
        box.innerHTML = `
          <p style="margin:0 0 8px 0;font-weight:600;color:#333;">How the Update Policy Works:</p>
          <p style="margin:0;">${{data.explanation}}</p>
          <p style="margin:8px 0 0 0;font-size:0.85rem;color:#666;">
            Thresholds: min_delta=${{data.min_delta_global}}, min_samples=${{data.min_sample_size}}, 
            min_vega=${{data.min_vega_sum}}, smoothing=${{data.smoothing_window_days}} days
          </p>
        `;
      }} catch (err) {{
        console.error('Failed to fetch policy:', err);
        document.getElementById('policy-explanation-box').textContent = 'Failed to load policy info.';
      }}
    }}
    
    async function fetchCurrentMultipliers() {{
      const underlying = document.getElementById('policy-underlying-select')?.value || 'BTC';
      const tbody = document.getElementById('current-multipliers-body');
      
      try {{
        const res = await fetch(`/api/calibration/current_multipliers?underlying=${{underlying}}`);
        if (!res.ok) throw new Error(`HTTP ${{res.status}}`);
        const data = await res.json();
        
        let rows = `<tr>
          <td style="font-weight:600;">Global</td>
          <td>${{data.global_multiplier.toFixed(4)}}</td>
          <td style="font-size:0.85rem;">${{data.last_updated ? new Date(data.last_updated).toLocaleDateString() : 'Never'}}</td>
        </tr>`;
        
        if (data.band_multipliers && data.band_multipliers.length > 0) {{
          rows += data.band_multipliers.map(b => `<tr>
            <td>${{b.name}} (${{b.min_dte}}-${{b.max_dte}}d)</td>
            <td>${{b.iv_multiplier.toFixed(4)}}</td>
            <td style="font-size:0.85rem;">${{data.last_updated ? new Date(data.last_updated).toLocaleDateString() : '-'}}</td>
          </tr>`).join('');
        }}
        
        tbody.innerHTML = rows;
      }} catch (err) {{
        console.error('Failed to fetch current multipliers:', err);
        tbody.innerHTML = '<tr><td colspan="3" style="text-align:center;color:#c00;">Error loading</td></tr>';
      }}
    }}
    
    async function fetchPolicyRuns() {{
      const underlying = document.getElementById('policy-underlying-select')?.value || 'BTC';
      const tbody = document.getElementById('policy-runs-body');
      
      tbody.innerHTML = '<tr><td colspan="7" style="text-align:center;color:#666;">Loading...</td></tr>';
      
      try {{
        const res = await fetch(`/api/calibration/runs?underlying=${{underlying}}&limit=10`);
        if (!res.ok) throw new Error(`HTTP ${{res.status}}`);
        const data = await res.json();
        
        const runs = data.runs || [];
        if (runs.length === 0) {{
          tbody.innerHTML = '<tr><td colspan="7" style="text-align:center;color:#666;">No calibration runs. Run calibration to see history.</td></tr>';
          
          // Update latest run info
          document.getElementById('latest-run-source').textContent = '-';
          document.getElementById('latest-run-recommended').textContent = '-';
          document.getElementById('latest-run-smoothed').textContent = '-';
          document.getElementById('latest-run-status').innerHTML = '<span style="color:#888;">No runs</span>';
          document.getElementById('latest-run-reason').textContent = '-';
          return;
        }}
        
        // Update latest run info
        const latest = runs[0];
        document.getElementById('latest-run-source').textContent = latest.source;
        document.getElementById('latest-run-recommended').textContent = latest.recommended_iv_multiplier?.toFixed(4) || '-';
        document.getElementById('latest-run-smoothed').textContent = latest.smoothed_global_multiplier?.toFixed(4) || '-';
        document.getElementById('latest-run-status').innerHTML = latest.applied 
          ? '<span style="background:#4caf50;color:#fff;padding:2px 8px;border-radius:4px;font-size:0.8rem;">Applied</span>'
          : '<span style="background:#9e9e9e;color:#fff;padding:2px 8px;border-radius:4px;font-size:0.8rem;">Not Applied</span>';
        document.getElementById('latest-run-reason').textContent = latest.applied_reason || '-';
        
        // Populate runs table
        tbody.innerHTML = runs.map(r => {{
          const ts = r.timestamp ? new Date(r.timestamp).toLocaleString() : 'N/A';
          const appliedBadge = r.applied
            ? '<span style="background:#4caf50;color:#fff;padding:1px 6px;border-radius:3px;font-size:0.75rem;">Yes</span>'
            : '<span style="background:#9e9e9e;color:#fff;padding:1px 6px;border-radius:3px;font-size:0.75rem;">No</span>';
          
          return `<tr>
            <td style="font-size:0.8rem;">${{ts}}</td>
            <td>${{r.source}}</td>
            <td>${{r.recommended_iv_multiplier?.toFixed(4) || '-'}}</td>
            <td style="font-weight:600;">${{r.smoothed_global_multiplier?.toFixed(4) || '-'}}</td>
            <td>${{r.sample_size}}</td>
            <td>${{appliedBadge}}</td>
            <td style="font-size:0.8rem;max-width:200px;overflow:hidden;text-overflow:ellipsis;">${{r.applied_reason || '-'}}</td>
          </tr>`;
        }}).join('');
      }} catch (err) {{
        console.error('Failed to fetch policy runs:', err);
        tbody.innerHTML = '<tr><td colspan="7" style="text-align:center;color:#c00;">Error loading runs</td></tr>';
      }}
    }}
    
    async function runPolicyCalibration(force = false) {{
      const underlying = document.getElementById('policy-underlying-select')?.value || 'BTC';
      const source = document.getElementById('policy-source-select')?.value || 'live';
      const statusEl = document.getElementById('policy-action-status');
      const runBtn = document.getElementById('run-policy-calibration-btn');
      const forceBtn = document.getElementById('force-apply-btn');
      
      runBtn.disabled = true;
      forceBtn.disabled = true;
      statusEl.innerHTML = '<span style="color:#2196f3;">Running calibration...</span>';
      
      try {{
        const endpoint = force ? '/api/calibration/force_apply' : '/api/calibration/run_with_policy';
        const res = await fetch(endpoint, {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify({{ underlying, source, min_dte: 3, max_dte: 30 }})
        }});
        
        const data = await res.json();
        
        if (!res.ok) {{
          statusEl.innerHTML = `<span style="color:#c00;">Error: ${{data.message || data.error}}</span>`;
          return;
        }}
        
        const appliedMsg = data.applied
          ? `<span style="color:#4caf50;font-weight:600;">Applied!</span> Smoothed multiplier: ${{data.smoothed_iv_multiplier?.toFixed(4)}}`
          : `<span style="color:#ff9800;font-weight:600;">Not Applied:</span> ${{data.applied_reason}}`;
        
        statusEl.innerHTML = appliedMsg + ` (samples: ${{data.sample_size}})`;
        
        // Refresh all displays
        await refreshPolicyUI();
        
      }} catch (err) {{
        console.error('Failed to run policy calibration:', err);
        statusEl.innerHTML = `<span style="color:#c00;">Error: ${{err.message}}</span>`;
      }} finally {{
        runBtn.disabled = false;
        forceBtn.disabled = false;
      }}
    }}
    
    // Initialize policy UI when calibration tab is shown
    document.addEventListener('DOMContentLoaded', function() {{
      const calibTab = document.querySelector('button[onclick*="calibration"]');
      if (calibTab) {{
        calibTab.addEventListener('click', function() {{
          setTimeout(refreshPolicyUI, 100);
        }});
      }}
      
      // Also refresh when underlying changes
      const policyUnderlyingSelect = document.getElementById('policy-underlying-select');
      if (policyUnderlyingSelect) {{
        policyUnderlyingSelect.addEventListener('change', refreshPolicyUI);
      }}
    }});
    
    // ===== End Calibration Update Policy UI =====
    
    async function toggleTraining(enable) {{
      try {{
        const res = await fetch('/api/training/toggle', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify({{ enable }})
        }});
        const data = await res.json();
        if (!res.ok) {{
          alert('Failed to toggle training mode: ' + (data.error || res.statusText));
          document.getElementById('training-toggle').checked = !enable;
          return;
        }}
        const badge = document.getElementById('training-badge');
        if (data.enabled) {{
          badge.style.display = 'inline-block';
        }} else {{
          badge.style.display = 'none';
        }}
      }} catch (err) {{
        alert('Error toggling training mode: ' + err.message);
        document.getElementById('training-toggle').checked = !enable;
      }}
    }}
    
    function setBacktestInputsDisabled(disabled) {{
      const inputs = ['bt-underlying', 'bt-start', 'bt-end', 'bt-timeframe', 'bt-interval', 'bt-exit-style', 'bt-dte', 'bt-delta', 'bt-min-dte', 'bt-max-dte', 'bt-delta-min', 'bt-delta-max', 'bt-margin-type', 'bt-settlement-ccy'];
      inputs.forEach(id => {{
        const el = document.getElementById(id);
        if (el) el.disabled = disabled;
      }});
    }}
    
    function getSyntheticModeParams() {{
      const mode = document.getElementById('bt-synthetic-mode').value;
      const presets = {{
        'pure_synthetic': {{ sigma_mode: 'rv_x_multiplier', chain_mode: 'synthetic_grid' }},
        'live_iv_synthetic': {{ sigma_mode: 'atm_iv_x_multiplier', chain_mode: 'synthetic_grid' }},
        'live_chain': {{ sigma_mode: 'mark_iv_x_multiplier', chain_mode: 'live_chain' }},
      }};
      return presets[mode] || presets['pure_synthetic'];
    }}
    
    function updateSyntheticModeDescription() {{
      const mode = document.getElementById('bt-synthetic-mode').value;
      const desc = document.getElementById('bt-synthetic-mode-desc');
      const descriptions = {{
        'pure_synthetic': 'Uses realized volatility with multiplier to price synthetic options on a generated strike grid.',
        'live_iv_synthetic': 'Uses live ATM IV from Deribit with multiplier to price synthetic options on a generated strike grid.',
        'live_chain': 'Uses actual Deribit option chains with live mark IV for each strike. Most realistic backtesting.',
      }};
      desc.textContent = descriptions[mode] || descriptions['pure_synthetic'];
    }}
    
    async function startBacktest() {{
      const underlying = document.getElementById('bt-underlying').value;
      const start = document.getElementById('bt-start').value;
      const end = document.getElementById('bt-end').value;
      const timeframe = document.getElementById('bt-timeframe').value;
      const intervalHours = parseInt(document.getElementById('bt-interval').value, 10);
      const exitStyle = document.getElementById('bt-exit-style').value;
      const dte = parseInt(document.getElementById('bt-dte').value);
      const delta = parseFloat(document.getElementById('bt-delta').value);
      const minDte = parseInt(document.getElementById('bt-min-dte').value, 10);
      const maxDte = parseInt(document.getElementById('bt-max-dte').value, 10);
      const deltaMin = parseFloat(document.getElementById('bt-delta-min').value);
      const deltaMax = parseFloat(document.getElementById('bt-delta-max').value);
      const marginType = document.getElementById('bt-margin-type').value;
      const settlementCcy = document.getElementById('bt-settlement-ccy').value;
      const syntheticParams = getSyntheticModeParams();
      
      const selectorName = document.getElementById('bt-selector-name').value;
      
      const payload = {{
        underlying,
        start: start + 'T00:00:00Z',
        end: end + 'T00:00:00Z',
        timeframe,
        decision_interval_hours: intervalHours,
        exit_style: exitStyle,
        target_dte: dte,
        target_delta: delta,
        min_dte: minDte,
        max_dte: maxDte,
        delta_min: deltaMin,
        delta_max: deltaMax,
        margin_type: marginType,
        settlement_ccy: settlementCcy,
        sigma_mode: syntheticParams.sigma_mode,
        chain_mode: syntheticParams.chain_mode,
        selector_name: selectorName,
      }};
      
      document.getElementById('bt-error').style.display = 'none';
      
      try {{
        const res = await fetch('/api/backtest/start', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify(payload),
        }});
        const data = await res.json();
        if (!res.ok || data.started === false) {{
          alert('Failed to start backtest: ' + (data.error || res.statusText));
        }}
      }} catch (err) {{
        alert('Backtest start error: ' + err.message);
      }}
    }}
    
    async function togglePause() {{
      const btn = document.getElementById('bt-pause-resume-btn');
      const isPaused = btn.textContent === 'Resume';
      
      try {{
        if (isPaused) {{
          await fetch('/api/backtest/resume', {{ method: 'POST' }});
        }} else {{
          await fetch('/api/backtest/pause', {{ method: 'POST' }});
        }}
      }} catch (err) {{
        console.error('Pause/Resume error:', err);
      }}
    }}
    
    async function stopBacktest() {{
      try {{
        await fetch('/api/backtest/stop', {{ method: 'POST' }});
      }} catch (err) {{
        console.error('Stop backtest error:', err);
      }}
    }}
    
    async function refreshBacktestStatus() {{
      try {{
        const res = await fetch('/api/backtest/status');
        if (!res.ok) return;
        const st = await res.json();
        
        const statusTextEl = document.getElementById('bt-status-text');
        const progressBarInner = document.getElementById('bt-progress-inner');
        const button = document.getElementById('bt-start-stop-btn');
        const pauseBtn = document.getElementById('bt-pause-resume-btn');
        const errorEl = document.getElementById('bt-error');
        const phaseLabel = document.getElementById('bt-phase-label');
        const phaseEl = document.getElementById('bt-current-phase');
        
        statusTextEl.className = 'bt-status-indicator';
        if (st.error) {{
          statusTextEl.textContent = 'ERROR';
          statusTextEl.classList.add('bt-status-error');
          errorEl.textContent = st.error;
          errorEl.style.display = 'block';
        }} else {{
          errorEl.style.display = 'none';
          if (st.paused) {{
            statusTextEl.textContent = 'PAUSED';
            statusTextEl.classList.add('bt-status-paused');
          }} else if (st.running) {{
            statusTextEl.textContent = 'RUNNING';
            statusTextEl.classList.add('bt-status-running');
          }} else if (st.finished_at) {{
            statusTextEl.textContent = 'FINISHED';
            statusTextEl.classList.add('bt-status-finished');
          }} else {{
            statusTextEl.textContent = 'IDLE';
            statusTextEl.classList.add('bt-status-idle');
          }}
        }}
        
        if (st.current_phase) {{
          phaseLabel.style.display = 'inline';
          phaseEl.style.display = 'inline';
          phaseEl.textContent = st.current_phase;
        }} else {{
          phaseLabel.style.display = 'none';
          phaseEl.style.display = 'none';
        }}
        
        const pct = Math.round((st.progress_pct || 0) * 100);
        progressBarInner.style.width = pct + '%';
        progressBarInner.textContent = pct + '%';
        
        if (st.running) {{
          button.textContent = 'Stop Backtest';
          button.onclick = stopBacktest;
          setBacktestInputsDisabled(true);
          pauseBtn.style.display = 'inline-block';
          pauseBtn.textContent = st.paused ? 'Resume' : 'Pause';
        }} else {{
          button.textContent = 'Start Backtest';
          button.onclick = startBacktest;
          setBacktestInputsDisabled(false);
          pauseBtn.style.display = 'none';
        }}
        
        document.getElementById('bt-decisions').textContent =
          (st.decisions_processed || 0) + ' / ' + (st.total_decisions || 0);
        
        const metricsLive = document.getElementById('bt-metrics-live');
        const metricsJson = document.getElementById('bt-metrics-json');
        if (st.metrics && Object.keys(st.metrics).length > 0) {{
          metricsLive.style.display = 'block';
          metricsJson.textContent = JSON.stringify(st.metrics, null, 2);
        }} else {{
          metricsLive.style.display = 'none';
        }}
        
        const tbody = document.getElementById('bt-steps-body');
        const steps = st.recent_steps || [];
        if (steps.length === 0) {{
          tbody.innerHTML = '<tr><td colspan="5" style="text-align:center;color:#666;">No steps yet</td></tr>';
        }} else {{
          tbody.innerHTML = steps.slice(-20).reverse().map(step => {{
            const t = new Date(step.time).toISOString().replace('T', ' ').slice(0, 19);
            const tradedClass = step.traded ? 'traded-yes' : 'traded-no';
            return `<tr>
              <td>${{t}}</td>
              <td>${{step.candidates}}</td>
              <td>${{step.best_score.toFixed(2)}}</td>
              <td class="${{tradedClass}}">${{step.traded ? 'Yes' : 'No'}}</td>
              <td>${{step.exit_style}}</td>
            </tr>`;
          }}).join('');
        }}
        
        const chainsBody = document.getElementById('bt-chains-body');
        const chains = st.recent_chains || [];
        window.__recentChains = chains;
        if (chains.length === 0) {{
          chainsBody.innerHTML = '<tr><td colspan="7" style="text-align:center;color:#666;">No chains yet</td></tr>';
        }} else {{
          chainsBody.innerHTML = chains.slice(-20).reverse().map((chain, idx) => {{
            const t = new Date(chain.decision_time).toISOString().replace('T', ' ').slice(0, 19);
            const pnlClass = chain.total_pnl >= 0 ? 'traded-yes' : 'traded-no';
            const realIdx = chains.length - 1 - idx;
            return `<tr>
              <td>${{t}}</td>
              <td>${{chain.underlying}}</td>
              <td>${{chain.num_legs}}</td>
              <td>${{chain.num_rolls}}</td>
              <td class="${{pnlClass}}">${{chain.total_pnl.toFixed(4)}}</td>
              <td>${{chain.max_drawdown_pct.toFixed(2)}}%</td>
              <td><button class="view-btn" onclick="showChainDetails(${{realIdx}})">View</button></td>
            </tr>`;
          }}).join('');
        }}
        
        // Render debug samples if available
        const debugSamplesSection = document.getElementById('debug-samples-section');
        const debugSamplesTbody = document.getElementById('debug-samples-tbody');
        const debugSamples = st.live_chain_debug_samples || [];
        if (debugSamples.length > 0) {{
          debugSamplesSection.style.display = 'block';
          debugSamplesTbody.innerHTML = debugSamples.map(sample => {{
            const diffClass = Math.abs(sample.abs_diff_pct) < 0.1 ? 'traded-yes' : (Math.abs(sample.abs_diff_pct) < 1 ? '' : 'traded-no');
            return `<tr>
              <td>${{sample.instrument_name}}</td>
              <td>${{sample.dte_days}}</td>
              <td>${{sample.strike.toLocaleString()}}</td>
              <td>${{sample.deribit_mark_price.toFixed(6)}}</td>
              <td>${{sample.engine_price.toFixed(6)}}</td>
              <td class="${{diffClass}}">${{sample.abs_diff_pct.toFixed(4)}}%</td>
            </tr>`;
          }}).join('');
        }} else {{
          debugSamplesSection.style.display = 'none';
        }}
      }} catch (err) {{
        console.error('Backtest status fetch error:', err);
      }}
    }}
    
    function showChainDetails(idx) {{
      const chains = window.__recentChains || [];
      if (idx < 0 || idx >= chains.length) return;
      const chain = chains[idx];
      
      const summaryEl = document.getElementById('chain-modal-summary');
      summaryEl.innerHTML = `
        <p><strong>Decision Time:</strong> ${{new Date(chain.decision_time).toISOString().replace('T', ' ').slice(0, 19)}}</p>
        <p><strong>Underlying:</strong> ${{chain.underlying}} | <strong>Legs:</strong> ${{chain.num_legs}} | <strong>Rolls:</strong> ${{chain.num_rolls}}</p>
        <p><strong>Total PnL:</strong> <span class="${{chain.total_pnl >= 0 ? 'traded-yes' : 'traded-no'}}">${{chain.total_pnl.toFixed(4)}}</span> | <strong>Max DD:</strong> ${{chain.max_drawdown_pct.toFixed(2)}}%</p>
      `;
      
      const legsBody = document.getElementById('chain-legs-body');
      const legs = chain.legs || [];
      legsBody.innerHTML = legs.map(leg => {{
        const openT = new Date(leg.open_time).toISOString().replace('T', ' ').slice(0, 16);
        const closeT = new Date(leg.close_time).toISOString().replace('T', ' ').slice(0, 16);
        const pnlClass = leg.pnl >= 0 ? 'traded-yes' : 'traded-no';
        let triggerClass = '';
        if (leg.trigger === 'tp_roll') triggerClass = 'trigger-tp';
        else if (leg.trigger === 'defensive_roll') triggerClass = 'trigger-defensive';
        else if (leg.trigger === 'expiry') triggerClass = 'trigger-expiry';
        return `<tr>
          <td>${{leg.index + 1}}</td>
          <td>${{openT}}</td>
          <td>${{closeT}}</td>
          <td>${{leg.strike.toLocaleString()}}</td>
          <td>${{leg.dte_open.toFixed(1)}}</td>
          <td class="${{pnlClass}}">${{leg.pnl.toFixed(4)}}</td>
          <td class="${{triggerClass}}">${{leg.trigger.replace('_', ' ')}}</td>
        </tr>`;
      }}).join('');
      
      document.getElementById('chain-modal').style.display = 'flex';
    }}
    
    function closeChainModal() {{
      document.getElementById('chain-modal').style.display = 'none';
    }}
    
    async function runBacktest() {{
      const underlying = document.getElementById('bt-underlying').value;
      const start = document.getElementById('bt-start').value;
      const end = document.getElementById('bt-end').value;
      const timeframe = document.getElementById('bt-timeframe').value;
      const interval = parseInt(document.getElementById('bt-interval').value);
      const dte = parseInt(document.getElementById('bt-dte').value);
      const delta = parseFloat(document.getElementById('bt-delta').value);
      
      document.getElementById('backtest-results').style.display = 'none';
      document.getElementById('backtest-loading').style.display = 'block';
      document.getElementById('insights-box').style.display = 'none';
      
      try {{
        const res = await fetch('/api/backtest/run', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify({{
            underlying,
            start: start + 'T00:00:00Z',
            end: end + 'T00:00:00Z',
            timeframe,
            decision_interval_bars: interval,
            target_dte: dte,
            target_delta: delta,
          }})
        }});
        
        if (!res.ok) {{
          const err = await res.json();
          throw new Error(err.detail || 'Backtest failed');
        }}
        
        backtestResult = await res.json();
        displayBacktestResults(backtestResult);
        
      }} catch (err) {{
        alert('Backtest error: ' + err.message);
      }} finally {{
        document.getElementById('backtest-loading').style.display = 'none';
      }}
    }}
    
    function displayBacktestResults(result) {{
      document.getElementById('backtest-results').style.display = 'block';
      
      const m = result.metrics || {{}};
      
      // Update TradingView-style summary panel
      const netProfit = m.net_profit_usd || 0;
      const netProfitPct = m.net_profit_pct || 0;
      const mainStat = document.getElementById('tv-net-profit');
      mainStat.className = 'tv-main-stat ' + (netProfit >= 0 ? 'positive' : 'negative');
      mainStat.innerHTML = `
        <span class="tv-value">$${{formatMoney(netProfit)}}</span>
        <span class="tv-pct">(${{netProfitPct >= 0 ? '+' : ''}}${{netProfitPct.toFixed(2)}}%)</span>
      `;
      
      const vsHodl = m.final_pnl_vs_hodl || 0;
      document.getElementById('tv-vs-hodl').innerText = (vsHodl >= 0 ? '+' : '') + formatMoney(vsHodl);
      document.getElementById('tv-vs-hodl').style.color = vsHodl >= 0 ? '#26a69a' : '#ef5350';
      
      document.getElementById('tv-hodl-return').innerText = (m.hodl_profit_pct >= 0 ? '+' : '') + (m.hodl_profit_pct || 0).toFixed(2) + '%';
      document.getElementById('tv-hodl-return').style.color = (m.hodl_profit_pct || 0) >= 0 ? '#26a69a' : '#ef5350';
      
      // Profit & Loss
      document.getElementById('tv-gross-profit').innerText = '$' + formatMoney(m.gross_profit || 0);
      document.getElementById('tv-gross-loss').innerText = '-$' + formatMoney(Math.abs(m.gross_loss || 0));
      document.getElementById('tv-profit-factor').innerText = (m.profit_factor || 0).toFixed(2);
      
      // Trade Statistics
      document.getElementById('tv-num-trades').innerText = m.num_trades || 0;
      document.getElementById('tv-win-rate').innerText = (m.win_rate || 0).toFixed(1) + '%';
      document.getElementById('tv-avg-trade').innerText = (m.avg_trade_usd >= 0 ? '+$' : '-$') + formatMoney(Math.abs(m.avg_trade_usd || 0));
      document.getElementById('tv-avg-trade').style.color = (m.avg_trade_usd || 0) >= 0 ? '#26a69a' : '#ef5350';
      
      // Avg Win/Loss
      document.getElementById('tv-avg-winner').innerText = '+$' + formatMoney(m.avg_winner || 0);
      document.getElementById('tv-avg-loser').innerText = '-$' + formatMoney(Math.abs(m.avg_loser || 0));
      
      // Risk Metrics
      document.getElementById('tv-max-dd').innerText = '-' + (m.max_drawdown_pct || 0).toFixed(2) + '%';
      document.getElementById('tv-sharpe').innerText = (m.sharpe_ratio || 0).toFixed(2);
      document.getElementById('tv-sortino').innerText = (m.sortino_ratio || 0).toFixed(2);
      
      // Draw equity chart with HODL comparison
      drawEquityChart(result.equity_curve || []);
      
      const trades = result.trades_sample || [];
      const tbody = document.getElementById('trades-tbody');
      if (trades.length === 0) {{
        tbody.innerHTML = '<tr><td colspan="6" style="text-align:center;">No trades</td></tr>';
      }} else {{
        tbody.innerHTML = trades.map(t => `
          <tr>
            <td>${{t.instrument_name}}</td>
            <td>${{t.open_time?.split('T')[0] || '--'}}</td>
            <td>${{t.close_time?.split('T')[0] || '--'}}</td>
            <td style="color:${{t.pnl >= 0 ? '#2e7d32' : '#c62828'}}">${{t.pnl?.toFixed(4)}}</td>
            <td style="color:${{t.pnl_vs_hodl >= 0 ? '#2e7d32' : '#c62828'}}">${{t.pnl_vs_hodl?.toFixed(4)}}</td>
            <td>${{t.max_drawdown_pct?.toFixed(2)}}%</td>
          </tr>
        `).join('');
      }}
    }}
    
    function formatMoney(num) {{
      if (Math.abs(num) >= 1000000) return (num / 1000000).toFixed(2) + 'M';
      if (Math.abs(num) >= 1000) return (num / 1000).toFixed(1) + 'K';
      return num.toFixed(2);
    }}
    
    function drawEquityChart(equityCurve) {{
      const canvas = document.getElementById('equity-chart');
      const ctx = canvas.getContext('2d');
      
      const container = canvas.parentElement;
      canvas.width = container.clientWidth;
      canvas.height = container.clientHeight;
      
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      if (equityCurve.length < 2) {{
        ctx.fillStyle = '#666';
        ctx.font = '14px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('Not enough data points', canvas.width / 2, canvas.height / 2);
        return;
      }}
      
      // Handle both old format [time, equity] and new format {{time, equity, hodl_equity}}
      const hasHodl = equityCurve[0].hodl_equity !== undefined;
      const strategyValues = equityCurve.map(p => hasHodl ? p.equity : p[1]);
      const hodlValues = hasHodl ? equityCurve.map(p => p.hodl_equity) : [];
      
      const allValues = hasHodl ? [...strategyValues, ...hodlValues] : strategyValues;
      const minVal = Math.min(...allValues);
      const maxVal = Math.max(...allValues);
      const range = maxVal - minVal || 1;
      
      const padding = {{ top: 20, right: 20, bottom: 30, left: 70 }};
      const chartWidth = canvas.width - padding.left - padding.right;
      const chartHeight = canvas.height - padding.top - padding.bottom;
      
      // Draw grid lines
      ctx.strokeStyle = '#e9ecef';
      ctx.lineWidth = 1;
      for (let i = 0; i <= 4; i++) {{
        const y = padding.top + (chartHeight * i / 4);
        ctx.beginPath();
        ctx.moveTo(padding.left, y);
        ctx.lineTo(canvas.width - padding.right, y);
        ctx.stroke();
        
        const val = maxVal - (range * i / 4);
        ctx.fillStyle = '#666';
        ctx.font = '11px sans-serif';
        ctx.textAlign = 'right';
        ctx.fillText('$' + formatMoney(val), padding.left - 5, y + 4);
      }}
      
      // Draw HODL line first (background)
      if (hasHodl && hodlValues.length > 0) {{
        ctx.beginPath();
        ctx.strokeStyle = '#ff9800';
        ctx.lineWidth = 2;
        ctx.setLineDash([5, 3]);
        
        equityCurve.forEach((point, i) => {{
          const x = padding.left + (chartWidth * i / (equityCurve.length - 1));
          const y = padding.top + chartHeight - ((point.hodl_equity - minVal) / range * chartHeight);
          
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }});
        
        ctx.stroke();
        ctx.setLineDash([]);
      }}
      
      // Draw Strategy line (foreground)
      ctx.beginPath();
      ctx.strokeStyle = '#1565c0';
      ctx.lineWidth = 2.5;
      
      equityCurve.forEach((point, i) => {{
        const x = padding.left + (chartWidth * i / (equityCurve.length - 1));
        const val = hasHodl ? point.equity : point[1];
        const y = padding.top + chartHeight - ((val - minVal) / range * chartHeight);
        
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }});
      
      ctx.stroke();
      
      // Fill under strategy curve
      ctx.fillStyle = 'rgba(21, 101, 192, 0.08)';
      const lastPoint = equityCurve[equityCurve.length - 1];
      const lastVal = hasHodl ? lastPoint.equity : lastPoint[1];
      const lastY = padding.top + chartHeight - ((lastVal - minVal) / range * chartHeight);
      ctx.lineTo(padding.left + chartWidth, lastY);
      ctx.lineTo(padding.left + chartWidth, padding.top + chartHeight);
      ctx.lineTo(padding.left, padding.top + chartHeight);
      ctx.closePath();
      ctx.fill();
    }}
    
    async function getInsights() {{
      if (!backtestResult) {{
        alert('Run a backtest first');
        return;
      }}
      
      const box = document.getElementById('insights-box');
      box.style.display = 'block';
      box.innerText = 'Generating insights...';
      
      try {{
        const res = await fetch('/api/backtest/insights', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify({{
            metrics: backtestResult.metrics,
            trades_sample: backtestResult.trades_sample,
            config: backtestResult.config,
          }})
        }});
        
        const data = await res.json();
        box.innerText = data.insights || 'No insights generated';
        
      }} catch (err) {{
        box.innerText = 'Error generating insights: ' + err.message;
      }}
    }}
    
    // Selector Frequency Scan
    document.getElementById('selector-scan-run-btn').addEventListener('click', runSelectorScan);
    
    async function runSelectorScan() {{
      const statusDiv = document.getElementById('selector-scan-status');
      const tableDiv = document.getElementById('selector-scan-results-table');
      const tbody = document.getElementById('selector-scan-results-body');
      
      statusDiv.innerHTML = '<span style="color:#4fc3f7;">Running scan...</span>';
      tableDiv.style.display = 'none';
      
      const underlyings = [];
      if (document.getElementById('selector-underlying-btc').checked) underlyings.push('BTC');
      if (document.getElementById('selector-underlying-eth').checked) underlyings.push('ETH');
      
      const thresholdOverrides = {{}};
      const vrpMin = document.getElementById('selector-vrp-min-input').value;
      const chopMax = document.getElementById('selector-chop-max-input').value;
      const adxMax = document.getElementById('selector-adx-max-input').value;
      const rsiMin = document.getElementById('selector-rsi-min-input').value;
      const rsiMax = document.getElementById('selector-rsi-max-input').value;
      const ivrankMin = document.getElementById('selector-ivrank-min-input').value;
      
      if (vrpMin !== '') thresholdOverrides['vrp_30d_min'] = parseFloat(vrpMin);
      if (chopMax !== '') thresholdOverrides['chop_factor_7d_max'] = parseFloat(chopMax);
      if (adxMax !== '') thresholdOverrides['adx_14d_max'] = parseFloat(adxMax);
      if (rsiMin !== '') thresholdOverrides['rsi_14d_min'] = parseFloat(rsiMin);
      if (rsiMax !== '') thresholdOverrides['rsi_14d_max'] = parseFloat(rsiMax);
      if (ivrankMin !== '') thresholdOverrides['iv_rank_6m_min'] = parseFloat(ivrankMin);
      
      const payload = {{
        selector_id: document.getElementById('selector-id-select').value,
        underlyings: underlyings,
        horizon_days: parseInt(document.getElementById('selector-horizon-input').value) || 365,
        decision_interval_days: parseFloat(document.getElementById('selector-interval-input').value) || 1.0,
        threshold_overrides: thresholdOverrides,
      }};
      
      try {{
        const res = await fetch('/api/backtest/selector_scan', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify(payload)
        }});
        const data = await res.json();
        
        if (data.ok) {{
          statusDiv.innerHTML = '<span style="color:#4caf50;">Scan complete</span>';
          
          tbody.innerHTML = '';
          for (const [underlying, strategies] of Object.entries(data.summary)) {{
            for (const [strategyKey, stats] of Object.entries(strategies)) {{
              const row = document.createElement('tr');
              const passPct = (stats.pass_pct * 100).toFixed(1);
              row.innerHTML = `
                <td>${{strategyKey}}</td>
                <td>${{underlying}}</td>
                <td>${{Math.round(stats.pass_count)}}</td>
                <td>${{Math.round(stats.total_steps)}}</td>
                <td style="color:${{stats.pass_pct > 0.5 ? '#4caf50' : '#ff9800'}};">${{passPct}}%</td>
              `;
              tbody.appendChild(row);
            }}
          }}
          tableDiv.style.display = 'table';
        }} else {{
          statusDiv.innerHTML = `<span style="color:#f44336;">Error: ${{data.error}}</span>`;
        }}
      }} catch (err) {{
        statusDiv.innerHTML = `<span style="color:#f44336;">Error: ${{err.message}}</span>`;
      }}
    }}
    
    // Intraday Scraper Status
    async function fetchIntradayScraperStatus() {{
      const statusMsg = document.getElementById("scraper-status-message");
      if (statusMsg) {{
        statusMsg.textContent = "Loading scraper status...";
        statusMsg.style.color = "#4fc3f7";
      }}

      try {{
        const resp = await fetch("/api/data_status/intraday");
        const data = await resp.json();

        if (!data.ok) {{
          if (statusMsg) {{
            statusMsg.textContent = data.error || "Unknown error";
            statusMsg.style.color = "#ff9800";
          }}
        }}

        document.getElementById("scraper-source").textContent = data.source || "Deribit intraday";
        document.getElementById("scraper-backend").textContent = data.backend || "unknown";
        document.getElementById("scraper-rows").textContent =
          (data.rows_total != null && data.rows_total.toLocaleString)
            ? data.rows_total.toLocaleString()
            : (data.rows_total ?? "0");
        document.getElementById("scraper-days").textContent = data.days_covered ?? "0";

        const first = data.first_timestamp ? data.first_timestamp.replace("T", " ").slice(0, 19) : "n/a";
        const last = data.last_timestamp ? data.last_timestamp.replace("T", " ").slice(0, 19) : "n/a";
        document.getElementById("scraper-range").textContent = first + " → " + last;

        if (data.approx_size_mb != null) {{
          document.getElementById("scraper-size").textContent = data.approx_size_mb.toFixed(1) + " MB";
        }} else {{
          document.getElementById("scraper-size").textContent = "n/a";
        }}

        if (data.target_interval_sec != null) {{
          document.getElementById("scraper-interval").textContent = data.target_interval_sec + " sec";
        }} else {{
          document.getElementById("scraper-interval").textContent = "n/a";
        }}

        const runningEl = document.getElementById("scraper-running");
        if (runningEl) {{
          if (data.is_running) {{
            runningEl.textContent = "RUNNING";
            runningEl.style.color = "#4caf50";
          }} else {{
            runningEl.textContent = "STALE / STOPPED";
            runningEl.style.color = "#f44336";
          }}
        }}

        if (statusMsg && data.ok) {{
          statusMsg.textContent = "Updated";
          statusMsg.style.color = "#4caf50";
        }}
      }} catch (err) {{
        if (statusMsg) {{
          statusMsg.textContent = "Error: " + err.message;
          statusMsg.style.color = "#f44336";
        }}
      }}
    }}
    
    document.getElementById("refresh-scraper-status-btn").addEventListener("click", fetchIntradayScraperStatus);
    fetchIntradayScraperStatus();
    
    // Selector Heatmap
    document.getElementById('selector-heatmap-run-btn').addEventListener('click', runSelectorHeatmap);
    
    async function runSelectorHeatmap() {{
      const statusDiv = document.getElementById('selector-heatmap-status');
      const container = document.getElementById('selector-heatmap-container');
      const thead = document.getElementById('selector-heatmap-thead');
      const tbody = document.getElementById('selector-heatmap-tbody');
      
      statusDiv.innerHTML = '<span style="color:#ce93d8;">Running heatmap scan...</span>';
      container.style.display = 'none';
      
      const metricX = document.getElementById('heatmap-metric-x-select').value;
      const metricY = document.getElementById('heatmap-metric-y-select').value;
      const xStart = parseFloat(document.getElementById('heatmap-x-start-input').value) || 5;
      const xStep = parseFloat(document.getElementById('heatmap-x-step-input').value) || 5;
      const xCount = parseInt(document.getElementById('heatmap-x-count-input').value) || 5;
      const yStart = parseFloat(document.getElementById('heatmap-y-start-input').value) || 15;
      const yStep = parseFloat(document.getElementById('heatmap-y-step-input').value) || 5;
      const yCount = parseInt(document.getElementById('heatmap-y-count-input').value) || 5;
      
      const gridX = [];
      for (let i = 0; i < xCount; i++) gridX.push(xStart + i * xStep);
      const gridY = [];
      for (let i = 0; i < yCount; i++) gridY.push(yStart + i * yStep);
      
      const payload = {{
        selector_id: document.getElementById('heatmap-selector-id-select').value,
        underlying: document.getElementById('heatmap-underlying-select').value,
        strategy_key: document.getElementById('heatmap-strategy-select').value,
        metric_x: metricX,
        metric_y: metricY,
        grid_x: gridX,
        grid_y: gridY,
        horizon_days: parseInt(document.getElementById('heatmap-horizon-input').value) || 180,
        decision_interval_days: parseFloat(document.getElementById('heatmap-interval-input').value) || 1.0,
      }};
      
      try {{
        const res = await fetch('/api/backtest/selector_heatmap', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify(payload)
        }});
        const data = await res.json();
        
        if (data.ok) {{
          statusDiv.innerHTML = '<span style="color:#4caf50;">Heatmap complete</span>';
          
          // Build header row
          const headerRow = document.createElement('tr');
          headerRow.innerHTML = `<th>${{metricY}} \\ ${{metricX}}</th>`;
          data.grid_x.forEach(xVal => {{
            headerRow.innerHTML += `<th>${{xVal}}</th>`;
          }});
          thead.innerHTML = '';
          thead.appendChild(headerRow);
          
          // Build data rows
          tbody.innerHTML = '';
          data.values.forEach((row, yIdx) => {{
            const tr = document.createElement('tr');
            tr.innerHTML = `<th>${{data.grid_y[yIdx]}}</th>`;
            row.forEach(val => {{
              const pct = (val * 100).toFixed(0);
              const alpha = Math.min(val, 1);
              const bgColor = `rgba(76, 175, 80, ${{alpha.toFixed(2)}})`;
              tr.innerHTML += `<td style="background:${{bgColor}};text-align:center;font-weight:600;color:${{alpha > 0.5 ? '#fff' : '#333'}};">${{pct}}%</td>`;
            }});
            tbody.appendChild(tr);
          }});
          
          container.style.display = 'block';
        }} else {{
          statusDiv.innerHTML = `<span style="color:#f44336;">Error: ${{data.error}}</span>`;
        }}
      }} catch (err) {{
        statusDiv.innerHTML = `<span style="color:#f44336;">Error: ${{err.message}}</span>`;
      }}
    }}
    
    // Environment Heatmap
    document.getElementById('env-heatmap-run-btn').addEventListener('click', runEnvHeatmap);
    
    async function runEnvHeatmap() {{
      const statusDiv = document.getElementById('env-heatmap-status');
      const container = document.getElementById('env-heatmap-container');
      const thead = document.getElementById('env-heatmap-thead');
      const tbody = document.getElementById('env-heatmap-tbody');
      
      statusDiv.innerHTML = '<span style="color:#26a69a;">Running environment heatmap...</span>';
      container.style.display = 'none';
      
      const xMetric = document.getElementById('env-x-metric-select').value;
      const yMetric = document.getElementById('env-y-metric-select').value;
      const xStart = parseFloat(document.getElementById('env-x-start-input').value) || 0;
      const xStep = parseFloat(document.getElementById('env-x-step-input').value) || 5;
      const xPoints = parseInt(document.getElementById('env-x-points-input').value) || 5;
      const yStart = parseFloat(document.getElementById('env-y-start-input').value) || 15;
      const yStep = parseFloat(document.getElementById('env-y-step-input').value) || 5;
      const yPoints = parseInt(document.getElementById('env-y-points-input').value) || 5;
      
      const payload = {{
        underlying: document.getElementById('env-underlying-select').value,
        horizon_days: parseInt(document.getElementById('env-horizon-input').value) || 365,
        decision_interval_days: parseInt(document.getElementById('env-decision-interval-input').value) || 1,
        x_metric: xMetric,
        y_metric: yMetric,
        x_start: xStart,
        x_step: xStep,
        x_points: xPoints,
        y_start: yStart,
        y_step: yStep,
        y_points: yPoints,
      }};
      
      try {{
        const res = await fetch('/api/environment_heatmap', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify(payload)
        }});
        const data = await res.json();
        
        if (data.ok) {{
          statusDiv.innerHTML = '<span style="color:#4caf50;">Heatmap complete. Cell value = % of decision steps (occupancy).</span>';
          
          // Build header row
          const headerRow = document.createElement('tr');
          headerRow.innerHTML = `<th>${{yMetric}} \\ ${{xMetric}}</th>`;
          data.x_labels.forEach(xVal => {{
            headerRow.innerHTML += `<th>${{xVal}}</th>`;
          }});
          thead.innerHTML = '';
          thead.appendChild(headerRow);
          
          // Build data rows
          tbody.innerHTML = '';
          data.grid.forEach((row, yIdx) => {{
            const tr = document.createElement('tr');
            tr.innerHTML = `<th>${{data.y_labels[yIdx]}}</th>`;
            row.forEach(val => {{
              const alpha = Math.min(val / 100, 1);
              const bgColor = `rgba(76, 175, 80, ${{alpha.toFixed(2)}})`;
              tr.innerHTML += `<td style="background:${{bgColor}};text-align:center;font-weight:600;color:${{alpha > 0.5 ? '#fff' : '#333'}};">${{val}}%</td>`;
            }});
            tbody.appendChild(tr);
          }});
          
          container.style.display = 'block';
        }} else {{
          statusDiv.innerHTML = `<span style="color:#f44336;">Error: ${{data.error}}</span>`;
        }}
      }} catch (err) {{
        statusDiv.innerHTML = `<span style="color:#f44336;">Error: ${{err.message}}</span>`;
      }}
    }}

    // Greg Environment Sweet Spots panel
    const sweetPanel = document.getElementById('greg-sweetspots-panel');
    if (sweetPanel) {{
      const runBtn = document.getElementById('greg-sweetspots-run-btn');
      const refreshBtn = document.getElementById('greg-sweetspots-refresh-btn');
      const sweetStatus = document.getElementById('greg-sweetspots-status');
      const sweetContent = document.getElementById('greg-sweetspots-content');
      
      function renderSweetSpots(payload) {{
        if (!payload.ok) {{
          sweetStatus.textContent = payload.error || 'No sweet spot data.';
          sweetStatus.style.color = '#f44336';
          sweetContent.innerHTML = '';
          return;
        }}
        
        const data = payload.data || [];
        if (!Array.isArray(data) || data.length === 0) {{
          sweetStatus.textContent = 'No sweet spot entries found. Run a scan to generate data.';
          sweetStatus.style.color = '#ff9800';
          sweetContent.innerHTML = '';
          return;
        }}
        
        sweetStatus.textContent = 'Loaded sweet spots from latest run.';
        sweetStatus.style.color = '#4caf50';
        
        const groups = {{}};
        data.forEach((entry) => {{
          const key = entry.underlying + ' / ' + entry.strategy;
          if (!groups[key]) groups[key] = [];
          const spots = entry.sweet_spots || [];
          spots.slice(0, 3).forEach(spot => {{
            groups[key].push({{...spot, x_metric: entry.x_metric, y_metric: entry.y_metric}});
          }});
        }});
        
        let html = '';
        Object.keys(groups).slice(0, 10).forEach((key) => {{
          html += '<div style="margin-bottom:12px;"><strong>' + key + '</strong><ul style="margin:4px 0 0 16px;padding:0;">';
          groups[key].slice(0, 5).forEach((spot, idx) => {{
            const occ = (spot.occupancy_frac * 100).toFixed(1);
            const pass = (spot.strategy_pass_frac * 100).toFixed(1);
            const sweet = spot.sweetness != null ? spot.sweetness.toFixed(4) : '';
            html += '<li style="font-size:0.9rem;color:#333;">';
            html += spot.x_metric + ' \\u2208 [' + spot.x_low.toFixed(1) + ', ' + spot.x_high.toFixed(1) + '], ';
            html += spot.y_metric + ' \\u2208 [' + spot.y_low.toFixed(1) + ', ' + spot.y_high.toFixed(1) + ']';
            html += ' &mdash; occ: ' + occ + '%, pass: ' + pass + '%';
            if (sweet) html += ', score: ' + sweet;
            html += '</li>';
          }});
          html += '</ul></div>';
        }});
        
        sweetContent.innerHTML = html || '<p style="color:#888;">No sweet spots to display.</p>';
      }}
      
      function fetchSweetSpots() {{
        sweetStatus.textContent = 'Loading sweet spots...';
        sweetStatus.style.color = '';
        
        fetch('/api/greg_sweetspots')
          .then(r => r.json())
          .then(renderSweetSpots)
          .catch(err => {{
            sweetStatus.textContent = 'Error loading sweet spots: ' + err;
            sweetStatus.style.color = '#f44336';
            sweetContent.innerHTML = '';
          }});
      }}
      
      function runSweetSpotScan() {{
        sweetStatus.textContent = 'Running Greg sweet spot scan... This may take a minute.';
        sweetStatus.style.color = '#9c27b0';
        if (runBtn) runBtn.disabled = true;
        if (refreshBtn) refreshBtn.disabled = true;
        
        fetch('/api/greg_sweetspots/run', {{ method: 'POST' }})
          .then(r => r.json())
          .then(payload => {{
            if (!payload.ok) {{
              sweetStatus.textContent = payload.error || 'Sweet spot scan failed.';
              sweetStatus.style.color = '#f44336';
            }} else {{
              sweetStatus.textContent = payload.message || 'Sweet spot scan completed.';
              sweetStatus.style.color = '#4caf50';
              fetchSweetSpots();
            }}
          }})
          .catch(err => {{
            sweetStatus.textContent = 'Error running sweet spot scan: ' + err;
            sweetStatus.style.color = '#f44336';
          }})
          .finally(() => {{
            if (runBtn) runBtn.disabled = false;
            if (refreshBtn) refreshBtn.disabled = false;
          }});
      }}
      
      if (runBtn) runBtn.addEventListener('click', runSweetSpotScan);
      if (refreshBtn) refreshBtn.addEventListener('click', fetchSweetSpots);
      fetchSweetSpots();
    }}

    fetchStatus();
    fetchDecisions();
    refreshBacktestStatus();
    loadStewardReport();
    fetchCalibrationHistory();
    initDashboard();
    setInterval(fetchStatus, 5000);
    setInterval(fetchDecisions, 10000);
    setInterval(refreshBacktestStatus, 3000);
    
    document.getElementById('question').addEventListener('keydown', function(e) {{
      if (e.key === 'Enter' && !e.shiftKey) {{
        e.preventDefault();
        sendQuestion();
      }}
    }});
  </script>
</body>
</html>
"""
