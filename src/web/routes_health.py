"""
Health, risk limits, strategy thresholds, and system configuration endpoints.

Extracted from src/web_app.py for modular organization.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import httpx

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.config import settings

router = APIRouter()


def _ops_health_run_authorized(request: Request) -> bool:
    secret = os.environ.get("OPS_HEALTH_RUN_SECRET")
    if not secret:
        return True
    return request.headers.get("X-OPS-HEALTH-SECRET", "") == secret


class LLMConfigUpdate(BaseModel):
    """Request model for updating LLM configuration."""
    llm_enabled: Optional[bool] = None
    decision_mode: Optional[str] = None
    explore_prob: Optional[float] = None
    llm_shadow_enabled: Optional[bool] = None
    llm_validation_strict: Optional[bool] = None


class RiskLimitsUpdate(BaseModel):
    """Request model for updating risk limits."""
    max_margin_used_pct: Optional[float] = None
    max_net_delta_abs: Optional[float] = None
    daily_drawdown_limit_pct: Optional[float] = None
    kill_switch_enabled: Optional[bool] = None
    liquidity_max_spread_pct: Optional[float] = None
    liquidity_min_open_interest: Optional[int] = None


class StrategyThresholdsUpdate(BaseModel):
    """Request model for updating strategy thresholds."""
    ivrv_min: Optional[float] = None
    delta_min: Optional[float] = None
    delta_max: Optional[float] = None
    dte_min: Optional[int] = None
    dte_max: Optional[int] = None
    training_profile_mode: Optional[str] = None


class RuntimeConfigUpdate(BaseModel):
    """Request model for updating runtime configuration."""
    kill_switch_enabled: Optional[bool] = None
    daily_drawdown_limit_pct: Optional[float] = None
    decision_mode: Optional[str] = None
    dry_run: Optional[bool] = None
    position_reconcile_action: Optional[str] = None
    trade_mode: Optional[str] = None
    heartbeat_timeout_sec: Optional[int] = None
    max_orders_per_minute: Optional[int] = None
    rolling_drawdown_limit_pct: Optional[float] = None
    rolling_drawdown_window_days: Optional[int] = None
    auto_close_only_on_drawdown: Optional[bool] = None


SUPERVISOR_API_URL = os.environ.get("SUPERVISOR_API_URL", "")


@router.get("/api/llm_status")
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


@router.post("/api/llm_status")
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


@router.post("/api/test_llm_decision")
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


@router.get("/api/risk_limits")
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


@router.post("/api/risk_limits")
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


@router.get("/api/strategy_thresholds")
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


@router.post("/api/strategy_thresholds")
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


@router.post("/api/test_kill_switch")
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


@router.post("/api/agent_healthcheck")
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


@router.get("/api/system_health/status")
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


@router.post("/api/ops/health/run")
def run_ops_healthcheck(request: Request) -> JSONResponse:
    """Run ops healthcheck and return cached status."""
    try:
        if not _ops_health_run_authorized(request):
            return JSONResponse(content={"error": "unauthorized"}, status_code=403)

        from src.healthcheck import run_and_cache_healthcheck, get_health_status_for_api

        run_and_cache_healthcheck(settings)
        return JSONResponse(content=get_health_status_for_api())
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@router.get("/api/ops/health/status")
def get_ops_health_status() -> JSONResponse:
    """Get cached ops health status."""
    try:
        from src.healthcheck import get_cached_health_status, get_health_status_for_api

        if get_cached_health_status() is None:
            return JSONResponse(status_code=404, content={"error": "no_healthcheck_cached"})
        return JSONResponse(content=get_health_status_for_api())
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@router.get("/api/llm_readiness")
def get_llm_readiness_endpoint() -> JSONResponse:
    """Check if LLM is ready for diagnostic tests."""
    try:
        from src.healthcheck import get_llm_readiness
        
        result = get_llm_readiness(settings)
        return JSONResponse(content={"ok": True, **result})
    except Exception as e:
        return JSONResponse(content={"ok": False, "error": str(e)})


@router.get("/api/health/iv_sanity")
def get_iv_sanity_check() -> JSONResponse:
    """
    Run IV sanity check to validate synthetic IV pricing layer.
    
    This runs backtests with different IV multipliers and verifies that
    results differ meaningfully (not stuck/broken). May take several seconds.
    """
    try:
        from scripts.iv_sanity_check import run_iv_sanity_check
        
        result = run_iv_sanity_check()
        return JSONResponse(content={"ok": result.get("status") == "ok", **result})
    except Exception as e:
        return JSONResponse(content={
            "ok": False, 
            "status": "error",
            "error": str(e),
            "selectors": [],
            "summary": f"Error running IV sanity check: {e}",
        })


@router.post("/api/steward/run")
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


@router.get("/api/steward/report")
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


@router.get("/api/greg_sweetspots")
def get_greg_sweetspots() -> JSONResponse:
    """
    Return the latest Greg environment sweet spots, if available.
    Reads backtest/output/greg_heatmap_sweetspots.json and wraps it in {ok, data}.
    """
    try:
        import json as json_lib
        base_dir = Path(__file__).resolve().parent.parent.parent
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


@router.post("/api/greg_sweetspots/run")
def run_greg_sweetspots() -> JSONResponse:
    """
    Trigger a Greg environment sweet spot sweep.

    This is a research-only operation that runs synchronously within a single
    request/response cycle. It analyzes synthetic market data across metric
    pairs and strategies to find optimal trading regions.
    """
    try:
        from src.backtest.greg_sweetspots import run_greg_sweetspot_sweep
        
        base_dir = Path(__file__).resolve().parent.parent.parent
        
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


@router.get("/api/system/runtime-config")
def get_runtime_config() -> JSONResponse:
    """Fetch current runtime configuration settings."""
    from src.ops.heartbeat import get_heartbeat_status
    from src.ops.rate_limiter import get_rate_limit_status
    from src.ops.rolling_drawdown import get_rolling_drawdown_status
    
    heartbeat = get_heartbeat_status(settings.heartbeat_timeout_sec)
    rate_limit = get_rate_limit_status(settings.max_orders_per_minute)
    rolling_dd = get_rolling_drawdown_status(
        settings.rolling_drawdown_window_days,
        settings.rolling_drawdown_limit_pct,
    )
    
    return JSONResponse(content={
        "ok": True,
        "kill_switch_enabled": settings.kill_switch_enabled,
        "daily_drawdown_limit_pct": settings.daily_drawdown_limit_pct,
        "decision_mode": settings.decision_mode,
        "dry_run": settings.dry_run,
        "position_reconcile_action": settings.position_reconcile_action,
        "trade_mode": settings.trade_mode.value if hasattr(settings.trade_mode, 'value') else settings.trade_mode,
        "heartbeat_timeout_sec": settings.heartbeat_timeout_sec,
        "heartbeat_status": heartbeat.to_dict(),
        "max_orders_per_minute": settings.max_orders_per_minute,
        "rate_limit_status": rate_limit.to_dict(),
        "rolling_drawdown_limit_pct": settings.rolling_drawdown_limit_pct,
        "rolling_drawdown_window_days": settings.rolling_drawdown_window_days,
        "rolling_drawdown_status": rolling_dd.to_dict(),
        "auto_close_only_on_drawdown": settings.auto_close_only_on_drawdown,
    })


@router.post("/api/system/runtime-config")
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
    
    if update.trade_mode is not None:
        from src.config import TradingMode
        valid_trade_modes = ["normal", "close_only", "halt"]
        if update.trade_mode not in valid_trade_modes:
            errors.append(f"trade_mode must be one of: {', '.join(valid_trade_modes)}")
        else:
            settings.trade_mode = TradingMode(update.trade_mode)
            updated["trade_mode"] = update.trade_mode
    
    if update.heartbeat_timeout_sec is not None:
        if update.heartbeat_timeout_sec < 0:
            errors.append("heartbeat_timeout_sec must be >= 0")
        else:
            settings.heartbeat_timeout_sec = update.heartbeat_timeout_sec
            updated["heartbeat_timeout_sec"] = update.heartbeat_timeout_sec
    
    if update.max_orders_per_minute is not None:
        if update.max_orders_per_minute < 0:
            errors.append("max_orders_per_minute must be >= 0")
        else:
            settings.max_orders_per_minute = update.max_orders_per_minute
            updated["max_orders_per_minute"] = update.max_orders_per_minute
    
    if update.rolling_drawdown_limit_pct is not None:
        if update.rolling_drawdown_limit_pct < 0:
            errors.append("rolling_drawdown_limit_pct must be >= 0")
        else:
            settings.rolling_drawdown_limit_pct = update.rolling_drawdown_limit_pct
            updated["rolling_drawdown_limit_pct"] = update.rolling_drawdown_limit_pct
    
    if update.auto_close_only_on_drawdown is not None:
        settings.auto_close_only_on_drawdown = update.auto_close_only_on_drawdown
        updated["auto_close_only_on_drawdown"] = update.auto_close_only_on_drawdown
    
    if update.rolling_drawdown_window_days is not None:
        if update.rolling_drawdown_window_days < 0:
            errors.append("rolling_drawdown_window_days must be >= 0")
        else:
            settings.rolling_drawdown_window_days = update.rolling_drawdown_window_days
            updated["rolling_drawdown_window_days"] = update.rolling_drawdown_window_days
    
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
            "trade_mode": settings.trade_mode.value if hasattr(settings.trade_mode, 'value') else settings.trade_mode,
            "heartbeat_timeout_sec": settings.heartbeat_timeout_sec,
            "max_orders_per_minute": settings.max_orders_per_minute,
            "rolling_drawdown_limit_pct": settings.rolling_drawdown_limit_pct,
            "auto_close_only_on_drawdown": settings.auto_close_only_on_drawdown,
        }
    })


@router.get("/api/supervisor/jobs")
async def get_supervisor_jobs():
    """Proxy supervisor jobs list from supervisor service."""
    if not SUPERVISOR_API_URL:
        return JSONResponse(content={"error": "not_configured", "jobs": []})
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{SUPERVISOR_API_URL.rstrip('/')}/jobs")
            resp.raise_for_status()
            return JSONResponse(content=resp.json())
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "jobs": []}
        )


@router.get("/api/supervisor/jobs/{job_id}")
async def get_supervisor_job(job_id: str):
    """Proxy single supervisor job from supervisor service."""
    if not SUPERVISOR_API_URL:
        return JSONResponse(
            status_code=503,
            content={"error": "Supervisor not configured"}
        )
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{SUPERVISOR_API_URL.rstrip('/')}/jobs/{job_id}")
            resp.raise_for_status()
            return JSONResponse(content=resp.json())
    except httpx.HTTPStatusError as e:
        return JSONResponse(
            status_code=e.response.status_code,
            content={"error": str(e)}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@router.get("/api/system/heartbeat-status")
def get_heartbeat_status_endpoint() -> JSONResponse:
    """Get current heartbeat monitor status."""
    from src.ops.heartbeat import get_heartbeat_status
    
    try:
        status = get_heartbeat_status()
        return JSONResponse(content={
            "ok": True,
            **status,
        })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": str(e)}
        )


@router.get("/api/system/rate-limit-status")
def get_rate_limit_status_endpoint() -> JSONResponse:
    """Get current rate limiter status."""
    from src.ops.rate_limiter import get_rate_limit_status
    
    try:
        status = get_rate_limit_status()
        return JSONResponse(content={
            "ok": True,
            **status,
        })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": str(e)}
        )


@router.post("/api/system/reset-rolling-drawdown")
def reset_rolling_drawdown_endpoint() -> JSONResponse:
    """Reset rolling drawdown tracker and close-only trigger."""
    from src.ops.rolling_drawdown import reset_rolling_drawdown
    
    try:
        reset_rolling_drawdown(clear_close_only_trigger=True)
        return JSONResponse(content={
            "ok": True,
            "message": "Rolling drawdown tracker reset successfully",
        })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": str(e)}
        )


@router.get("/api/system/daily-drawdown-state")
def get_daily_drawdown_state() -> JSONResponse:
    """Get current daily drawdown state."""
    from src.ops.drawdown_store import get_daily_drawdown_status
    
    try:
        status = get_daily_drawdown_status()
        return JSONResponse(content={
            "ok": True,
            **status,
        })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": str(e)}
        )


@router.post("/api/system/reset-daily-drawdown")
def reset_daily_drawdown_endpoint() -> JSONResponse:
    """Reset daily drawdown state."""
    from src.risk_engine import reset_daily_drawdown_state
    
    try:
        reset_daily_drawdown_state()
        return JSONResponse(content={
            "ok": True,
            "message": "Daily drawdown state reset successfully",
        })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": str(e)}
        )


@router.get("/api/system/reconciliation-status")
def get_reconciliation_status_endpoint() -> JSONResponse:
    """Get current position reconciliation status."""
    from src.ops.reconciliation_state import get_reconciliation_status
    
    try:
        status = get_reconciliation_status()
        return JSONResponse(content={
            "ok": True,
            **status.to_dict(),
        })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": str(e)}
        )


@router.post("/api/system/reconcile-now")
def reconcile_now_endpoint() -> JSONResponse:
    """
    Trigger manual position reconciliation.
    
    This will:
    1. Fetch current positions from Deribit
    2. Compare with local tracker
    3. If divergent and action=auto_heal, rebuild local state
    4. Update reconciliation status
    5. Clear trading block if positions now match
    """
    from src.deribit_client import DeribitClient
    from src.position_tracker import position_tracker
    from src.reconciliation import run_reconciliation_once
    from src.ops.reconciliation_state import (
        set_reconciliation_status,
        build_mismatches_from_diff,
        clear_reconciliation_block,
        ReconciliationResult,
    )
    
    try:
        client = DeribitClient()
        
        try:
            diff = run_reconciliation_once(
                deribit_client=client,
                position_tracker=position_tracker,
                settings=settings,
            )
            diff_dict = diff.to_dict()
            mismatches = build_mismatches_from_diff({
                "missing_in_local": diff_dict.get("untracked_on_exchange", []),
                "missing_in_exchange": diff_dict.get("missing_on_exchange", []),
                "size_mismatches": diff_dict.get("size_mismatches", []),
            })
            
            if diff.is_clean:
                set_reconciliation_status(
                    result=ReconciliationResult.CLEAN,
                    is_clean=True,
                    mismatches=[],
                    exchange_count=diff.exchange_count,
                    local_count=diff.local_count,
                    action_taken="none",
                    trading_blocked=False,
                )
                clear_reconciliation_block()
                
                return JSONResponse(content={
                    "ok": True,
                    "result": "clean",
                    "message": f"Positions IN SYNC (exchange={diff.exchange_count}, local={diff.local_count})",
                    "exchange_count": diff.exchange_count,
                    "local_count": diff.local_count,
                    "trading_blocked": False,
                })
            else:
                # Divergent - check action
                if settings.position_reconcile_action == "auto_heal":
                    # Fetch all positions and rebuild
                    all_positions = (
                        client.get_positions(currency="BTC", kind="option")
                        + client.get_positions(currency="ETH", kind="option")
                    )
                    rebuilt_count = position_tracker.rebuild_from_exchange(all_positions)
                    
                    set_reconciliation_status(
                        result=ReconciliationResult.CLEAN,
                        is_clean=True,
                        mismatches=[],
                        exchange_count=diff.exchange_count,
                        local_count=rebuilt_count,
                        action_taken="auto_heal",
                        trading_blocked=False,
                    )
                    clear_reconciliation_block()
                    
                    return JSONResponse(content={
                        "ok": True,
                        "result": "healed",
                        "message": f"Auto-healed: rebuilt {rebuilt_count} positions from exchange",
                        "exchange_count": diff.exchange_count,
                        "local_count": rebuilt_count,
                        "mismatches_before": [m.to_dict() for m in mismatches],
                        "trading_blocked": False,
                    })
                else:
                    # Halt mode - report but don't auto-heal
                    set_reconciliation_status(
                        result=ReconciliationResult.DIVERGENT,
                        is_clean=False,
                        mismatches=mismatches,
                        exchange_count=diff.exchange_count,
                        local_count=diff.local_count,
                        action_taken="halt",
                        trading_blocked=True,
                    )
                    
                    return JSONResponse(content={
                        "ok": True,
                        "result": "divergent",
                        "message": "Position mismatch detected. Trading blocked until resolved.",
                        "exchange_count": diff.exchange_count,
                        "local_count": diff.local_count,
                        "mismatches": [m.to_dict() for m in mismatches],
                        "trading_blocked": True,
                        "hint": "Change position_reconcile_action to 'auto_heal' or manually reconcile positions.",
                    })
        finally:
            client.close()
            
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": str(e)}
        )


@router.post("/api/system/clear-reconciliation-block")
def clear_reconciliation_block_endpoint() -> JSONResponse:
    """
    Clear the trading block caused by position reconciliation divergence.
    Use with caution - only after manually verifying positions are correct.
    """
    from src.ops.reconciliation_state import clear_reconciliation_block, get_reconciliation_status
    
    try:
        clear_reconciliation_block()
        status = get_reconciliation_status()
        return JSONResponse(content={
            "ok": True,
            "message": "Reconciliation block cleared. Trading can resume.",
            "current_status": status.to_dict(),
        })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": str(e)}
        )
