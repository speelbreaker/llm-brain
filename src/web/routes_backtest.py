"""
Backtest API routes for the Options Trading Agent.
Provides endpoints for running backtests, managing runs, and analysis tools.
"""
from __future__ import annotations

import csv
import io
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field
from typing import Literal as TypingLiteral, cast

from src.config import settings
from src.rules_summary import build_rules_summary
from src.backtest.config_schema import (
    BacktestConfig,
    BacktestPreset,
)
from src.backtest.config_presets import resolve_backtest_config, get_preset_config

router = APIRouter()


class BacktestType(str, Enum):
    """Type of backtest to run."""
    GENERIC = "generic"
    GREG_SELECTOR = "greg_selector"


class SelectorDataSource(str, Enum):
    """Data source for selector frequency scans."""
    SYNTHETIC = "synthetic"
    HARVESTER = "harvester"
    LIVE = "live"


class GregStrategyStatus(BaseModel):
    """Per-strategy diagnostic status from Greg selector."""
    status: TypingLiteral["PASS", "BLOCKED", "NO_DATA"]
    detail: str = ""


class StrategyBacktestSummary(BaseModel):
    """Per-strategy summary for Greg selector backtests."""
    bot_id: str = "GregBot"
    strategy_code: str
    strategy_name: str
    underlying: str
    selections: int = 0
    pass_count: int = 0
    blocked_count: int = 0
    no_data_count: int = 0
    selection_pct: float = 0.0


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
    sigma_mode: str = "rv_x_multiplier"
    chain_mode: str = "synthetic_grid"


class SkewSourceType(str, Enum):
    """Skew source for synthetic IV calculations."""
    NONE = "none"
    HARVESTED = "harvested"
    LIVE = "live"


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
    # Backtests assume linear USDC-settled option marks unless explicit conversions are implemented.
    margin_type: TypingLiteral["inverse", "linear"] = "linear"
    settlement_ccy: TypingLiteral["ANY", "USDC", "BTC", "ETH"] = "USDC"
    sigma_mode: str = "rv_x_multiplier"
    chain_mode: Optional[str] = Field(
        default=None,
        description="Chain mode for candidates: None (auto based on backtest type), 'synthetic_grid', or 'live_chain'"
    )
    synthetic_iv_multiplier: float = 1.0
    selector_name: str = "generic_covered_call"
    backtest_type: BacktestType = Field(
        default=BacktestType.GENERIC,
        description="Type of backtest: 'generic' for covered calls or 'greg_selector' for Greg strategy selection only"
    )
    greg_underlyings: List[str] = Field(
        default=["BTC", "ETH"],
        description="Underlyings for Greg selector mode (ignored in generic mode)"
    )
    skew_source: Optional[SkewSourceType] = Field(
        default=None,
        description="Skew source for IV: None (auto based on backtest type), 'none', 'harvested', 'live' (blocked for historical)"
    )


class InsightsRequest(BaseModel):
    metrics: Dict[str, Any]
    trades_sample: List[Dict[str, Any]]
    config: Dict[str, Any]


class SelectorScanRequest(BaseModel):
    """Request model for selector frequency scan."""
    selector_id: str = "greg"
    underlyings: List[str] = Field(default=["BTC", "ETH"])
    num_paths: int = 1
    horizon_days: int = 365
    decision_interval_days: float = 1.0
    threshold_overrides: Dict[str, float] = Field(default_factory=dict)
    iv_mode: str = Field(
        default="synthetic",
        description="IV data source: 'synthetic' (estimated), 'live' (current market), or 'hybrid' (live with synthetic fallback)"
    )
    iv_fallback_warning: bool = Field(
        default=True,
        description="If true, emit warnings when falling back from live to synthetic IV"
    )
    data_source: SelectorDataSource = Field(
        default=SelectorDataSource.SYNTHETIC,
        description="Data source for scan: 'synthetic' (universe), 'harvester' (historical), or 'live' (current snapshot)"
    )


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


def _get_iv_mode_description(mode: str) -> str:
    """Get human-readable description for IV mode."""
    descriptions = {
        "synthetic": "Uses synthetic/estimated IV based on historical patterns",
        "live": "Uses current market IV from Deribit",
        "hybrid": "Uses live market IV when available; falls back to synthetic estimates otherwise",
    }
    return descriptions.get(mode, "Unknown IV mode")


from datetime import timedelta


def is_historical_backtest(end_dt: datetime) -> bool:
    """
    Determine if a backtest is historical based on its end date.
    
    Historical = end_dt < (now_utc - 5 minutes).
    This matches the threshold used in existing API validation.
    """
    now_utc = datetime.now(timezone.utc)
    return end_dt < (now_utc - timedelta(minutes=5))


def apply_historical_defaults(
    is_historical: bool,
    chain_mode: Optional[str],
    skew_source: Optional[SkewSourceType],
) -> tuple[str, SkewSourceType]:
    """
    Apply smart defaults for chain_mode and skew_source based on backtest type.
    
    Historical backtests (end_dt > 5 min ago):
      - chain_mode defaults to 'live_chain' (use harvested option chains)
      - skew_source defaults to 'harvested' (historical skew data)
      - LIVE skew is BLOCKED and remapped to HARVESTED to prevent look-ahead bias
    
    Live-ish backtests (end_dt within 5 min of now):
      - chain_mode defaults to 'synthetic_grid' (safe generated candidates)
      - skew_source defaults to 'none' (flat, safe estimates)
    
    Explicit overrides take precedence, except LIVE skew for historical (blocked).
    """
    if is_historical:
        effective_chain_mode = chain_mode if chain_mode is not None else "live_chain"
        if skew_source == SkewSourceType.LIVE:
            effective_skew_source = SkewSourceType.HARVESTED
        else:
            effective_skew_source = skew_source if skew_source is not None else SkewSourceType.HARVESTED
    else:
        effective_chain_mode = chain_mode if chain_mode is not None else "synthetic_grid"
        effective_skew_source = skew_source if skew_source is not None else SkewSourceType.NONE
    
    return effective_chain_mode, effective_skew_source


@router.get("/api/backtest/presets")
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


@router.get("/api/backtest/strategy_caps")
def get_strategy_capabilities(selector: str = "generic_covered_call") -> JSONResponse:
    """
    Get capability metadata for a selector/strategy.
    
    Returns what configuration fields the strategy supports, owns, or ignores.
    The UI uses this to show/hide relevant configuration controls.
    """
    from src.backtest.strategy_caps import get_strategy_caps, list_available_strategies
    
    caps = get_strategy_caps(selector)
    if caps is None:
        return JSONResponse(
            status_code=404,
            content={
                "error": f"Unknown selector: {selector}",
                "available": [s["selector_name"] for s in list_available_strategies()],
            }
        )
    
    return JSONResponse(content=caps.to_dict())


@router.get("/api/backtest/strategy_caps/{selector_name}")
def get_strategy_capabilities_by_name(selector_name: str) -> JSONResponse:
    """Get capability metadata for a specific selector by path."""
    from src.backtest.strategy_caps import get_strategy_caps, list_available_strategies
    
    caps = get_strategy_caps(selector_name)
    if caps is None:
        return JSONResponse(
            status_code=404,
            content={
                "error": f"Unknown selector: {selector_name}",
                "available": [s["selector_name"] for s in list_available_strategies()],
            }
        )
    
    return JSONResponse(content=caps.to_dict())


@router.get("/api/backtest/strategies")
def list_backtest_strategies() -> JSONResponse:
    """List all available backtest strategies with their capabilities."""
    from src.backtest.strategy_caps import list_available_strategies
    return JSONResponse(content={"strategies": list_available_strategies()})


@router.post("/api/backtest/resolve-config")
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


@router.post("/api/backtest/start")
def start_backtest(req: BacktestStartRequest) -> JSONResponse:
    """Start a new backtest in the background."""
    from src.backtest.manager import backtest_manager
    from src.backtest.strategy_caps import apply_strategy_overrides
    
    backtest_type_value = req.backtest_type.value if hasattr(req.backtest_type, 'value') else str(req.backtest_type)
    
    if backtest_type_value == BacktestType.GREG_SELECTOR.value:
        return _run_greg_selector_backtest(req)
    
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

    is_historical = is_historical_backtest(end_dt)
    
    effective_chain_mode, effective_skew_source = apply_historical_defaults(
        is_historical=is_historical,
        chain_mode=req.chain_mode,
        skew_source=req.skew_source,
    )
    
    if is_historical and effective_skew_source == SkewSourceType.LIVE:
        raise HTTPException(
            status_code=400,
            detail=(
                "skew_source='live' is not allowed for historical backtests "
                "(look-ahead bias). Use 'harvested' or 'none'."
            ),
        )
    
    user_config = {
        "exit_style": req.exit_style,
        "target_dte": req.target_dte,
        "target_delta": req.target_delta,
        "min_dte": req.min_dte,
        "max_dte": req.max_dte,
        "delta_min": req.delta_min,
        "delta_max": req.delta_max,
    }
    
    validation = apply_strategy_overrides(req.selector_name, user_config)
    effective = validation.effective_config
    
    effective_exit_style = effective.get("exit_style", req.exit_style)
    
    if effective_exit_style == "gregbot_managed" and req.selector_name != "gregbot":
        raise HTTPException(
            status_code=400,
            detail="exit_style 'gregbot_managed' is only valid for the gregbot selector"
        )
    
    valid_exit_styles = ["hold_to_expiry", "tp_and_roll", "both", "gregbot_managed"]
    if effective_exit_style not in valid_exit_styles:
        raise HTTPException(status_code=400, detail=f"Invalid exit_style. Must be one of: {valid_exit_styles}")
    
    warnings = list(validation.warnings)

    effective_margin_type = req.margin_type
    effective_settlement_ccy = req.settlement_ccy

    # Normalize to linear+USDC to avoid unit-mismatched PnL (inverse marks are typically in underlying units).
    if effective_margin_type != "linear" or effective_settlement_ccy != "USDC":
        effective_margin_type = "linear"
        effective_settlement_ccy = "USDC"
        warnings.append(
            "Normalized margin_type/settlement_ccy to linear/USDC for backtest correctness (unit consistency)."
        )

    started = backtest_manager.start(
        underlying=req.underlying,
        start_date=start_dt,
        end_date=end_dt,
        timeframe=req.timeframe,
        decision_interval_hours=req.decision_interval_hours,
        exit_style=effective_exit_style,
        target_dte=effective.get("target_dte", req.target_dte),
        target_delta=effective.get("target_delta", req.target_delta),
        min_dte=effective.get("min_dte", req.min_dte),
        max_dte=effective.get("max_dte", req.max_dte),
        delta_min=effective.get("delta_min", req.delta_min),
        delta_max=effective.get("delta_max", req.delta_max),
        margin_type=effective_margin_type,
        settlement_ccy=effective_settlement_ccy,
        sigma_mode=req.sigma_mode,
        chain_mode=effective_chain_mode,
        synthetic_iv_multiplier=req.synthetic_iv_multiplier,
        selector_name=req.selector_name,
        skew_source=effective_skew_source.value,
    )
    
    if not started:
        return JSONResponse(
            status_code=409,
            content={"started": False, "error": "Backtest already running"},
        )
    
    return JSONResponse(content={
        "started": True,
        "backtest_type": "generic",
        "warnings": warnings,
        "effective_config": {
            **validation.effective_config,
            "chain_mode": effective_chain_mode,
            "skew_source": effective_skew_source.value,
            "is_historical": is_historical,
        },
    })


def _run_greg_selector_backtest(req: BacktestStartRequest) -> JSONResponse:
    """Run Greg selector-only backtest across specified underlyings.
    
    This mode runs a synchronous selector scan on the synthetic universe and returns
    per-strategy pass/block diagnostics. Unlike generic backtests, this is an immediate
    response that does NOT use the backtest_manager and does NOT block other backtests.
    """
    from src.backtest.selector_scan import SelectorScanConfig, run_selector_scan
    
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

    is_historical = is_historical_backtest(end_dt)
    
    _, effective_skew_source = apply_historical_defaults(
        is_historical=is_historical,
        chain_mode=req.chain_mode,
        skew_source=req.skew_source,
    )
    
    if is_historical and effective_skew_source == SkewSourceType.LIVE:
        raise HTTPException(
            status_code=400,
            detail=(
                "skew_source='live' is not allowed for historical backtests "
                "(look-ahead bias). Use 'harvested' or 'none'."
            ),
        )
    
    horizon_days = (end_dt - start_dt).days
    if horizon_days < 1:
        horizon_days = 1
    
    decision_interval_days = req.decision_interval_hours / 24.0
    
    try:
        config = SelectorScanConfig(
            selector_id="greg",
            underlyings=req.greg_underlyings,
            num_paths=1,
            horizon_days=horizon_days,
            decision_interval_days=decision_interval_days,
            threshold_overrides={},
            iv_mode="synthetic",
            iv_fallback_warning=False,
        )
        result = run_selector_scan(config)
        
        strategy_summaries: List[Dict[str, Any]] = []
        for underlying, strats in result.summary.items():
            total_steps = result.total_steps.get(underlying, 1)
            for strat_key, strat_data in strats.items():
                pass_count = strat_data.get("pass_count", 0)
                blocked_count = strat_data.get("blocked_count", 0)
                no_data_count = strat_data.get("no_data_count", 0)
                
                if blocked_count == 0 and no_data_count == 0:
                    blocked_count = int(total_steps - pass_count)
                
                total = pass_count + blocked_count + no_data_count
                pass_pct = pass_count / total if total > 0 else 0.0
                
                strategy_summaries.append({
                    "bot_id": "GregBot",
                    "strategy_code": strat_key,
                    "strategy_name": strat_key,
                    "underlying": underlying,
                    "selections": int(pass_count),
                    "pass_count": int(pass_count),
                    "blocked_count": int(blocked_count),
                    "no_data_count": int(no_data_count),
                    "selection_pct": round(pass_pct * 100, 2),
                    "status": "PASS" if pass_count > 0 else ("NO_DATA" if no_data_count > 0 else "BLOCKED"),
                })
        
        return JSONResponse(content={
            "started": True,
            "backtest_type": "greg_selector",
            "completed": True,
            "execution_mode": "synchronous",
            "greg_underlyings": req.greg_underlyings,
            "horizon_days": horizon_days,
            "decision_interval_days": decision_interval_days,
            "summary": result.summary,
            "total_steps": result.total_steps,
            "strategy_summaries": strategy_summaries,
        })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "started": False,
                "backtest_type": "greg_selector",
                "error": str(e),
            }
        )


@router.get("/api/backtest/status")
def get_backtest_status() -> JSONResponse:
    """Get the current backtest status."""
    from src.backtest.manager import backtest_manager
    return JSONResponse(content=backtest_manager.get_status())


@router.post("/api/backtest/stop")
def stop_backtest() -> JSONResponse:
    """Stop the currently running backtest."""
    from src.backtest.manager import backtest_manager
    backtest_manager.stop()
    return JSONResponse(content={"stopping": True})


@router.post("/api/backtest/pause")
def pause_backtest() -> JSONResponse:
    """Pause the currently running backtest."""
    from src.backtest.manager import backtest_manager
    backtest_manager.pause()
    return JSONResponse(content={"paused": True})


@router.post("/api/backtest/resume")
def resume_backtest() -> JSONResponse:
    """Resume the paused backtest."""
    from src.backtest.manager import backtest_manager
    backtest_manager.resume()
    return JSONResponse(content={"resumed": True})


@router.post("/api/backtest/run")
def run_backtest(req: BacktestRequest) -> JSONResponse:
    """Run a backtest using the CoveredCallSimulator and save to database."""
    from src.backtest.types import CallSimulationConfig, SigmaMode, ChainMode
    from src.backtest.data_source import Timeframe
    from src.backtest.covered_call_simulator import CoveredCallSimulator, always_trade_policy
    from src.backtest.deribit_data_source import DeribitDataSource
    from src.db import get_db_session
    from src.db.backtest_service import create_backtest_run, complete_run, fail_run
    from src.db.models_backtest import BacktestRun as BacktestRunModel
    
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


@router.post("/api/backtest/insights")
def get_backtest_insights(req: InsightsRequest) -> JSONResponse:
    """Generate LLM insights from backtest results."""
    try:
        from openai import OpenAI
        
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


@router.get("/api/backtests")
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


@router.get("/api/backtests/{run_id}")
def get_backtest_run(run_id: str) -> JSONResponse:
    """Get the full result for a specific backtest run from database."""
    from src.db import get_db_session
    from src.db.backtest_service import get_run_with_details
    
    with get_db_session() as db:
        result = get_run_with_details(db, run_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"Backtest run '{run_id}' not found")
        
        return JSONResponse(content=result)


@router.get("/api/backtests/{run_id}/download")
def download_backtest_run(run_id: str) -> JSONResponse:
    """Download the backtest run data as JSON."""
    from src.db import get_db_session
    from src.db.backtest_service import get_run_with_details
    
    with get_db_session() as db:
        result = get_run_with_details(db, run_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"Backtest run '{run_id}' not found")
        
        return Response(
            content=json.dumps(result, indent=2, default=str),
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="{run_id}_backtest_result.json"'},
        )


@router.delete("/api/backtests/{run_id}")
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


@router.get("/api/backtests/{run_id}/events")
def get_backtest_events(
    run_id: str,
    strategy_key: Optional[str] = None,
    event_type: Optional[str] = None,
) -> JSONResponse:
    """
    Get the event timeline for a backtest run.
    
    Optionally filter by strategy_key or event_type.
    """
    from src.db import get_db_session
    from src.db.models_backtest import BacktestRun, BacktestEvent
    
    with get_db_session() as db:
        run = db.query(BacktestRun).filter(BacktestRun.run_id == run_id).first()
        if run is None:
            raise HTTPException(status_code=404, detail=f"Backtest run '{run_id}' not found")
        
        query = db.query(BacktestEvent).filter(BacktestEvent.run_id == run.id)
        
        if strategy_key:
            query = query.filter(BacktestEvent.strategy_key == strategy_key)
        if event_type:
            query = query.filter(BacktestEvent.event_type == event_type)
        
        events = query.order_by(BacktestEvent.event_time).all()
        
        return JSONResponse(content={
            "run_id": run_id,
            "count": len(events),
            "events": [e.to_dict() for e in events],
        })


@router.get("/api/backtests/{run_id}/strategy_summary")
def get_backtest_strategy_summary(run_id: str) -> JSONResponse:
    """
    Get strategy breakdown summary for a backtest run.
    
    Returns aggregated metrics grouped by strategy_key.
    """
    from src.db import get_db_session
    from src.db.models_backtest import BacktestRun, BacktestEvent
    
    with get_db_session() as db:
        run = db.query(BacktestRun).filter(BacktestRun.run_id == run_id).first()
        if run is None:
            raise HTTPException(status_code=404, detail=f"Backtest run '{run_id}' not found")
        
        events = db.query(BacktestEvent).filter(BacktestEvent.run_id == run.id).all()
        
        if not events:
            return JSONResponse(content={
                "run_id": run_id,
                "strategy_summary": [],
                "total_events": 0,
            })
        
        strategy_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "opens": 0, "closes": 0, "total_pnl": 0.0, "wins": 0,
            "decisions": 0, "skips": 0, "rolls": 0, "take_profits": 0,
        })
        
        for event in events:
            stats = strategy_stats[event.strategy_key]
            if event.event_type == "DECISION":
                stats["decisions"] += 1
            elif event.event_type == "OPEN":
                stats["opens"] += 1
            elif event.event_type == "SKIP":
                stats["skips"] += 1
            elif event.event_type == "ROLL":
                stats["rolls"] += 1
            elif event.event_type == "TAKE_PROFIT":
                stats["take_profits"] += 1
                stats["closes"] += 1
                if event.pnl is not None:
                    stats["total_pnl"] += event.pnl
                    if event.pnl > 0:
                        stats["wins"] += 1
            elif event.event_type in ("CLOSE", "STOP_LOSS", "EXPIRY"):
                stats["closes"] += 1
                if event.pnl is not None:
                    stats["total_pnl"] += event.pnl
                    if event.pnl > 0:
                        stats["wins"] += 1
        
        summaries = []
        for key, stats in strategy_stats.items():
            closes = stats["closes"]
            summaries.append({
                "strategy_key": key,
                "opens": stats["opens"],
                "closes": closes,
                "total_pnl": round(stats["total_pnl"], 2),
                "avg_pnl": round(stats["total_pnl"] / closes, 2) if closes > 0 else 0.0,
                "win_rate": round(stats["wins"] / closes, 4) if closes > 0 else 0.0,
                "decisions": stats["decisions"],
                "skips": stats["skips"],
                "rolls": stats["rolls"],
                "take_profits": stats["take_profits"],
            })
        
        summaries.sort(key=lambda x: x["total_pnl"], reverse=True)
        
        return JSONResponse(content={
            "run_id": run_id,
            "strategy_summary": summaries,
            "total_events": len(events),
        })


@router.get("/api/backtests/{run_id}/events/download")
def download_backtest_events(run_id: str):
    """Download backtest events as CSV."""
    from src.db import get_db_session
    from src.db.models_backtest import BacktestRun, BacktestEvent
    
    with get_db_session() as db:
        run = db.query(BacktestRun).filter(BacktestRun.run_id == run_id).first()
        if run is None:
            raise HTTPException(status_code=404, detail=f"Backtest run '{run_id}' not found")
        
        events = db.query(BacktestEvent).filter(
            BacktestEvent.run_id == run.id
        ).order_by(BacktestEvent.event_time).all()
        
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "event_time", "selector_name", "strategy_key", "event_type",
            "trade_id", "position_id", "pnl"
        ])
        
        for e in events:
            writer.writerow([
                e.event_time.isoformat() if e.event_time else "",
                e.selector_name,
                e.strategy_key,
                e.event_type,
                e.trade_id or "",
                e.position_id or "",
                e.pnl if e.pnl is not None else "",
            ])
        
        return Response(
            content=output.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{run_id}_events.csv"'},
        )


@router.get("/api/backtests/{run_id}/strategy_summary/download")
def download_strategy_summary(run_id: str, format: str = "json"):
    """
    Download strategy summary as JSON or CSV.
    
    Args:
        run_id: The backtest run ID
        format: 'json' or 'csv' (default: json)
    """
    from src.db import get_db_session
    from src.db.models_backtest import BacktestRun, BacktestEvent
    
    with get_db_session() as db:
        run = db.query(BacktestRun).filter(BacktestRun.run_id == run_id).first()
        if run is None:
            raise HTTPException(status_code=404, detail=f"Backtest run '{run_id}' not found")
        
        events = db.query(BacktestEvent).filter(BacktestEvent.run_id == run.id).all()
        
        strategy_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "opens": 0, "closes": 0, "total_pnl": 0.0, "wins": 0,
            "decisions": 0, "skips": 0, "rolls": 0, "take_profits": 0,
        })
        
        for event in events:
            stats = strategy_stats[event.strategy_key]
            if event.event_type == "DECISION":
                stats["decisions"] += 1
            elif event.event_type == "OPEN":
                stats["opens"] += 1
            elif event.event_type == "SKIP":
                stats["skips"] += 1
            elif event.event_type == "ROLL":
                stats["rolls"] += 1
            elif event.event_type == "TAKE_PROFIT":
                stats["take_profits"] += 1
                stats["closes"] += 1
                if event.pnl is not None:
                    stats["total_pnl"] += event.pnl
                    if event.pnl > 0:
                        stats["wins"] += 1
            elif event.event_type in ("CLOSE", "STOP_LOSS", "EXPIRY"):
                stats["closes"] += 1
                if event.pnl is not None:
                    stats["total_pnl"] += event.pnl
                    if event.pnl > 0:
                        stats["wins"] += 1
        
        summaries = []
        for key, stats in strategy_stats.items():
            closes = stats["closes"]
            summaries.append({
                "strategy_key": key,
                "opens": stats["opens"],
                "closes": closes,
                "total_pnl": round(stats["total_pnl"], 2),
                "avg_pnl": round(stats["total_pnl"] / closes, 2) if closes > 0 else 0.0,
                "win_rate": round(stats["wins"] / closes, 4) if closes > 0 else 0.0,
                "decisions": stats["decisions"],
                "skips": stats["skips"],
                "rolls": stats["rolls"],
                "take_profits": stats["take_profits"],
            })
        
        summaries.sort(key=lambda x: x["total_pnl"], reverse=True)
        
        if format.lower() == "csv":
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow([
                "strategy_key", "opens", "closes", "total_pnl", "avg_pnl",
                "win_rate", "decisions", "skips", "rolls", "take_profits"
            ])
            for s in summaries:
                writer.writerow([
                    s["strategy_key"], s["opens"], s["closes"], s["total_pnl"],
                    s["avg_pnl"], s["win_rate"], s["decisions"], s["skips"],
                    s["rolls"], s["take_profits"]
                ])
            return Response(
                content=output.getvalue(),
                media_type="text/csv",
                headers={"Content-Disposition": f'attachment; filename="{run_id}_strategy_summary.csv"'},
            )
        else:
            export_data = {
                "run_id": run_id,
                "export_time": datetime.now(timezone.utc).isoformat(),
                "total_events": len(events),
                "strategy_summary": summaries,
            }
            return Response(
                content=json.dumps(export_data, indent=2),
                media_type="application/json",
                headers={"Content-Disposition": f'attachment; filename="{run_id}_strategy_summary.json"'},
            )


@router.post("/api/backtest/selector_scan")
def selector_scan(req: SelectorScanRequest) -> JSONResponse:
    """
    Run a selector frequency scan and return summary.
    Supports synthetic universe, harvester historical data, or live snapshot.
    Backtest-only; no orders.
    """
    from src.backtest.selector_scan import SelectorScanConfig, run_selector_scan
    
    try:
        data_source_value = req.data_source.value if hasattr(req.data_source, 'value') else str(req.data_source)
        
        if data_source_value == SelectorDataSource.LIVE.value:
            return _run_live_selector_scan(req)
        elif data_source_value == SelectorDataSource.HARVESTER.value:
            return _run_harvester_selector_scan(req)
        
        config = SelectorScanConfig(
            selector_id=req.selector_id,
            underlyings=req.underlyings,
            num_paths=req.num_paths,
            horizon_days=req.horizon_days,
            decision_interval_days=req.decision_interval_days,
            threshold_overrides=req.threshold_overrides,
            iv_mode=req.iv_mode,
            iv_fallback_warning=req.iv_fallback_warning,
        )
        result = run_selector_scan(config)
        
        response_data: Dict[str, Any] = {
            "ok": True,
            "data_source": data_source_value,
            "summary": result.summary,
            "total_steps": result.total_steps,
            "config": {
                "iv_mode": req.iv_mode,
                "iv_mode_description": _get_iv_mode_description(req.iv_mode),
            },
        }
        
        if hasattr(result, 'iv_fallback_count') and result.iv_fallback_count:
            response_data["iv_fallback_count"] = result.iv_fallback_count
            response_data["iv_fallback_warning"] = "Live IV was unavailable for some data points; fell back to synthetic estimates."
        
        return JSONResponse(content=response_data)
    except Exception as e:
        return JSONResponse(content={"ok": False, "error": str(e)})


def _run_live_selector_scan(req: SelectorScanRequest) -> JSONResponse:
    """Run selector scan using current live market snapshot."""
    from src.bots.gregbot import get_gregbot_evaluations_for_underlying
    
    try:
        summary: Dict[str, Dict[str, Dict[str, float]]] = {}
        strategy_diagnostics: Dict[str, Dict[str, Dict[str, str]]] = {}
        
        for underlying in req.underlyings:
            try:
                eval_result = get_gregbot_evaluations_for_underlying(underlying)
                selected = eval_result.get("selected_strategy", "NO_TRADE")
                strategies = eval_result.get("strategies", [])
                
                underlying_summary: Dict[str, Dict[str, float]] = {}
                underlying_diagnostics: Dict[str, Dict[str, str]] = {}
                
                for strat in strategies:
                    key = strat.strategy_key
                    status = strat.status.upper()
                    is_selected = 1.0 if key == selected else 0.0
                    underlying_summary[key] = {
                        "pass_count": 1.0 if status == "PASS" else 0.0,
                        "total_steps": 1,
                        "pass_pct": 1.0 if status == "PASS" else 0.0,
                        "selected": is_selected,
                    }
                    underlying_diagnostics[key] = {
                        "status": status,
                        "detail": strat.summary,
                    }
                
                summary[underlying.upper()] = underlying_summary
                strategy_diagnostics[underlying.upper()] = underlying_diagnostics
            except Exception as e:
                summary[underlying.upper()] = {"error": {"pass_count": 0, "total_steps": 0, "pass_pct": 0}}
        
        return JSONResponse(content={
            "ok": True,
            "data_source": "live",
            "summary": summary,
            "total_steps": {u: 1 for u in req.underlyings},
            "strategy_diagnostics": strategy_diagnostics,
            "config": {"iv_mode": "live", "iv_mode_description": "Current market snapshot"},
        })
    except Exception as e:
        return JSONResponse(content={"ok": False, "data_source": "live", "error": str(e)})


def _run_harvester_selector_scan(req: SelectorScanRequest) -> JSONResponse:
    """Run selector scan using harvester historical data."""
    from pathlib import Path
    import pandas as pd
    from src.strategies.greg_selector import GregSelectorSensors, evaluate_greg_selector
    
    try:
        harvester_base = Path("data/live_deribit")
        if not harvester_base.exists():
            return JSONResponse(content={
                "ok": False,
                "data_source": "harvester",
                "error": "harvester_not_configured",
                "detail": "No harvester data directory found at data/live_deribit",
            })
        
        summary: Dict[str, Dict[str, Dict[str, float]]] = {}
        total_steps: Dict[str, int] = {}
        
        for underlying in req.underlyings:
            underlying_dir = harvester_base / underlying.upper()
            if not underlying_dir.exists():
                summary[underlying.upper()] = {}
                total_steps[underlying.upper()] = 0
                continue
            
            parquet_files = list(underlying_dir.rglob("*.parquet"))
            if not parquet_files:
                summary[underlying.upper()] = {}
                total_steps[underlying.upper()] = 0
                continue
            
            strategy_counts: Dict[str, int] = {}
            step_count = 0
            
            for pf in parquet_files[:min(len(parquet_files), req.horizon_days)]:
                try:
                    df = pd.read_parquet(pf)
                    if df.empty:
                        continue
                    
                    sensors = GregSelectorSensors(
                        vrp_30d=df.get("vrp_30d", pd.Series([None])).iloc[0] if "vrp_30d" in df.columns else None,
                        adx_14d=df.get("adx_14d", pd.Series([None])).iloc[0] if "adx_14d" in df.columns else None,
                        skew_25d=df.get("skew_25d", pd.Series([None])).iloc[0] if "skew_25d" in df.columns else None,
                    )
                    
                    decision = evaluate_greg_selector(sensors)
                    selected = decision.selected_strategy
                    strategy_counts[selected] = strategy_counts.get(selected, 0) + 1
                    step_count += 1
                except Exception:
                    continue
            
            underlying_summary: Dict[str, Dict[str, float]] = {}
            for strat, count in strategy_counts.items():
                underlying_summary[strat] = {
                    "pass_count": float(count),
                    "total_steps": float(step_count),
                    "pass_pct": count / step_count if step_count > 0 else 0.0,
                }
            
            summary[underlying.upper()] = underlying_summary
            total_steps[underlying.upper()] = step_count
        
        return JSONResponse(content={
            "ok": True,
            "data_source": "harvester",
            "summary": summary,
            "total_steps": total_steps,
            "config": {"iv_mode": "harvester", "iv_mode_description": "Historical harvester snapshots"},
        })
    except Exception as e:
        return JSONResponse(content={"ok": False, "data_source": "harvester", "error": str(e)})


@router.post("/api/backtest/selector_heatmap")
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


@router.post("/api/environment_heatmap")
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
