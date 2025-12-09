"""
FastAPI web application for the Options Trading Agent.
Provides live status, chat interface, Live Agent Dashboard, and Backtesting Lab.
"""
from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Body, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import cast

from agent_loop import run_agent_loop_forever
from src.status_store import status_store
from src.decisions_store import decisions_store
from src.chat_with_agent import chat_with_agent_full, get_chat_messages, clear_chat_history
from src.config import settings
from src.position_tracker import position_tracker
from src.calibration import run_calibration
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
    """
    if underlying not in ("BTC", "ETH"):
        return JSONResponse(
            status_code=400,
            content={"error": "underlying must be BTC or ETH"},
        )
    
    try:
        result = run_calibration(
            underlying=underlying,
            min_dte=min_dte,
            max_dte=max_dte,
            iv_multiplier=iv_multiplier,
            default_iv=default_iv,
        )

        payload = {
            "underlying": result.underlying,
            "spot": result.spot,
            "min_dte": result.min_dte,
            "max_dte": result.max_dte,
            "iv_multiplier": result.iv_multiplier,
            "default_iv": result.default_iv,
            "rv_annualized": result.rv_annualized,
            "atm_iv": result.atm_iv,
            "recommended_iv_multiplier": result.recommended_iv_multiplier,
            "count": result.count,
            "mae_pct": result.mae_pct,
            "bias_pct": result.bias_pct,
            "timestamp": result.timestamp.isoformat(),
            "rows": [
                {
                    "instrument": r.instrument,
                    "dte": r.dte,
                    "strike": r.strike,
                    "mark_price": r.mark_price,
                    "syn_price": r.syn_price,
                    "diff": r.diff,
                    "diff_pct": r.diff_pct,
                    "mark_iv": r.mark_iv,
                    "syn_iv": r.syn_iv,
                }
                for r in result.rows
            ],
        }
        return JSONResponse(content=payload)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "calibration_failed", "message": str(e)},
        )


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
        })
    except Exception as e:
        return JSONResponse(content={"ok": False, "error": str(e)})


class RiskLimitsUpdate(BaseModel):
    """Request model for updating risk limits."""
    max_margin_used_pct: Optional[float] = None
    max_net_delta_abs: Optional[float] = None
    daily_drawdown_limit_pct: Optional[float] = None
    kill_switch_enabled: Optional[bool] = None


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
        
        return get_risk_limits()
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
        decision = evaluate_greg_selector(sensors)

        payload = decision.model_dump()
        payload["ok"] = True
        payload["timestamp"] = datetime.now(timezone.utc).isoformat()

        return JSONResponse(content=payload)
    except Exception as e:
        return JSONResponse(content={"ok": False, "error": str(e)})


# =============================================================================
# BOTS API ENDPOINTS
# =============================================================================

@app.get("/api/bots/market_sensors")
def get_bots_market_sensors() -> JSONResponse:
    """
    Return current high-level sensors per underlying for Bots tab.
    Computes Greg Phase 1 sensor bundle for each underlying.
    """
    try:
        from src.bots.gregbot import compute_greg_sensors
        
        underlyings = list(settings.underlyings or ["BTC", "ETH"])
        data = {}
        
        for u in underlyings:
            sensors = compute_greg_sensors(u)
            data[u] = sensors
        
        return JSONResponse(content={"ok": True, "sensors": data})
    except Exception as e:
        return JSONResponse(content={"ok": False, "error": str(e)})


@app.get("/api/bots/strategies")
def get_bots_strategies() -> JSONResponse:
    """
    Aggregate StrategyEvaluation objects for all expert bots.
    For now, only GregBot is implemented.
    """
    try:
        from src.bots.gregbot import get_gregbot_evaluations_for_underlying
        
        underlyings = list(settings.underlyings or ["BTC", "ETH"])
        all_evals = []
        
        for u in underlyings:
            payload = get_gregbot_evaluations_for_underlying(u)
            strat_evals = payload.get("strategies", [])
            all_evals.extend([e.model_dump() for e in strat_evals])
        
        return JSONResponse(content={"ok": True, "strategies": all_evals})
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
    """Run full agent healthcheck and return results."""
    try:
        from src.healthcheck import run_agent_healthcheck
        
        result = run_agent_healthcheck(settings)
        
        return JSONResponse(content={
            "ok": result.get("overall_status") != "FAIL",
            "overall_status": result.get("overall_status", "UNKNOWN"),
            "results": result.get("results", []),
        })
    except Exception as e:
        return JSONResponse(content={"ok": False, "error": str(e)})


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
    <button class="tab active" onclick="showTab('live')">Live Agent</button>
    <button class="tab" onclick="showTab('backtest')">Backtesting Lab</button>
    <button class="tab" onclick="showTab('runs')">Backtest Runs</button>
    <button class="tab" onclick="showTab('calibration')">Calibration</button>
    <button class="tab" onclick="showTab('strategies')">Bots</button>
    <button class="tab" onclick="showTab('health')">System Health</button>
    <button class="tab" onclick="showTab('chat')">Chat</button>
  </div>

  <!-- LIVE AGENT TAB -->
  <div id="tab-live" class="tab-content active">
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
      
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.5rem;">
        <h3 style="margin:0;">Open Positions</h3>
        <div id="positions-pnl-summary" style="font-size:0.9rem;color:#888;"></div>
      </div>
      <div style="overflow-x: auto; max-height: 260px; overflow-y: auto; margin-bottom: 1.5rem;">
        <table class="steps-table">
          <thead>
            <tr>
              <th>Underlying</th>
              <th>Type</th>
              <th>Strategy</th>
              <th>Symbol</th>
              <th>Qty</th>
              <th>Entry</th>
              <th>Mark</th>
              <th>Unreal. PnL</th>
              <th>Unreal. %</th>
              <th>DTE</th>
              <th>Rolls</th>
              <th>Mode</th>
            </tr>
          </thead>
          <tbody id="live-open-positions-body">
            <tr><td colspan="12" style="text-align:center;color:#666;">No open positions</td></tr>
          </tbody>
        </table>
      </div>
      
      <h3 style="margin-bottom: 0.5rem;">Closed Chains</h3>
      <div style="overflow-x: auto; max-height: 260px; overflow-y: auto;">
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

  <!-- BACKTESTING LAB TAB -->
  <div id="tab-backtest" class="tab-content">
    <div class="section">
      <h2>Backtest Configuration</h2>
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
    </div>

    <div class="loading" id="backtest-loading" style="display:none;">
      <div class="spinner"></div>
      <p>Running backtest... This may take a minute.</p>
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
        </div>
        
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
    </div>
  </div>

  <!-- BACKTEST RUNS TAB -->
  <div id="tab-runs" class="tab-content">
    <div class="section">
      <h2>Backtest Run History</h2>
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

  <!-- BOTS TAB -->
  <div id="tab-strategies" class="tab-content">
    <div class="section">
      <h2>Bots</h2>
      <p style="color: #666; margin-bottom: 1.5rem;">A live view of expert bots, their market sensors, and which strategies currently pass.</p>
      
      <!-- Live Market Sensors -->
      <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 8px; border-left: 4px solid #1565c0; margin-bottom: 1.5rem;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
          <h3 style="margin: 0; color: #1565c0; font-size: 1.1rem;">Live Market Sensors</h3>
          <button onclick="refreshBotsSensors()" style="padding: 0.5rem 1rem; background: #1565c0; color: white; border: none; border-radius: 4px; cursor: pointer;">
            Refresh Sensors
          </button>
        </div>
        <div id="bots-live-sensors" style="overflow-x: auto;">
          <p style="color: #666; font-style: italic;">Loading live market sensors...</p>
        </div>
      </div>
      
      <!-- Strategy Matches (All Bots) -->
      <div style="background: #e8f5e9; padding: 1.5rem; border-radius: 8px; border-left: 4px solid #2e7d32; margin-bottom: 1.5rem;">
        <h3 style="margin: 0 0 1rem 0; color: #2e7d32; font-size: 1.1rem;">Strategy Matches (All Bots)</h3>
        <div id="bots-strategy-matches" style="overflow-x: auto;">
          <p style="color: #666; font-style: italic;">Loading strategy matches...</p>
        </div>
      </div>
      
      <!-- Expert Bots -->
      <div style="background: #f3e5f5; padding: 1.5rem; border-radius: 8px; border-left: 4px solid #7c4dff; margin-bottom: 1.5rem;">
        <h3 style="margin: 0 0 1rem 0; color: #7c4dff; font-size: 1.1rem;">Expert Bots</h3>
        
        <div id="bots-expert-tabs" style="margin-bottom: 1rem;">
          <button class="bots-expert-tab active" data-expert-id="greg_mandolini" onclick="selectExpertTab('greg_mandolini')" style="padding: 0.5rem 1rem; background: #7c4dff; color: white; border: none; border-radius: 4px; cursor: pointer; margin-right: 0.5rem;">
            GregBot
          </button>
        </div>
        
        <div id="bots-expert-table" style="overflow-x: auto;">
          <p style="color: #666; font-style: italic;">Loading expert strategies...</p>
        </div>
      </div>
      
      <p style="color: #888; font-size: 0.85rem; margin-top: 1rem;">
        <strong>Phase 1 (Read-Only):</strong> Computes sensors, runs decision tree, displays recommendation. No orders placed.
      </p>
    </div>
  </div>

  <!-- SYSTEM HEALTH TAB -->
  <div id="tab-health" class="tab-content">
    <div class="section">
      <h2>System Controls & Health</h2>
      <p style="color: #666; margin-bottom: 1.5rem;">Run diagnostics and test system components. These are dry-run tests that do not execute trades.</p>
      
      <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1.5rem;">
        
        <!-- LLM & Decision Mode -->
        <div style="background: #f8f9fa; padding: 1.25rem; border-radius: 8px; border-left: 4px solid #1565c0;">
          <h3 style="margin: 0 0 0.75rem 0; color: #1565c0; font-size: 1rem;">LLM & Decision Mode</h3>
          <div id="llm-status-line" style="font-size: 0.85rem; color: #666; margin-bottom: 1rem;">Loading...</div>
          <button onclick="testLlmDecision()" style="width: 100%; margin-bottom: 0.5rem;">Test LLM Decision Pipeline</button>
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
            </div>
            <button onclick="saveRiskLimits()" style="margin-top: 1rem;">Save Risk Limits</button>
            <div id="risk-limits-feedback" style="font-size: 0.8rem; min-height: 1.5rem; margin-top: 0.5rem;"></div>
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
      if (name === 'runs') {{
        fetchBacktestRuns();
      }}
      if (name === 'health') {{
        loadSystemHealthStatus();
      }}
      if (name === 'strategies') {{
        loadBotsTab();
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
    
    async function refreshBotsSensors() {{
      const container = document.getElementById('bots-live-sensors');
      container.innerHTML = '<p style="color: #666; font-style: italic;">Loading...</p>';
      
      try {{
        const res = await fetch('/api/bots/market_sensors');
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
          
          const sensorNames = ['vrp_30d', 'chop_factor_7d', 'iv_rank_6m', 'term_structure_spread', 'skew_25d', 'adx_14d', 'rsi_14d', 'price_vs_ma200'];
          const sensorLabels = {{'vrp_30d': 'VRP 30d', 'chop_factor_7d': 'Chop Factor 7d', 'iv_rank_6m': 'IV Rank 6m', 'term_structure_spread': 'Term Spread', 'skew_25d': 'Skew 25d', 'adx_14d': 'ADX 14d', 'rsi_14d': 'RSI 14d', 'price_vs_ma200': 'Price vs MA200'}};
          
          for (const name of sensorNames) {{
            html += `<tr style="border-bottom: 1px solid #eee;">
              <td style="padding: 0.5rem; color: #666;">${{sensorLabels[name] || name}}</td>
              ${{underlyings.map(u => {{
                const val = sensors[u] ? sensors[u][name] : null;
                const display = formatSensorValue(val, name === 'chop_factor_7d' ? 3 : 2);
                const color = val !== null ? '#333' : '#999';
                return `<td style="text-align: right; padding: 0.5rem; color: ${{color}}; font-weight: ${{val !== null ? '500' : '400'}};">${{display}}</td>`;
              }}).join('')}}
            </tr>`;
          }}
          
          html += '</tbody></table>';
          container.innerHTML = html;
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
          
          // Render Strategy Matches (only passing strategies)
          const passing = botsStrategiesData.filter(s => s.status === 'pass');
          
          if (passing.length === 0) {{
            matchesContainer.innerHTML = '<p style="color: #666; font-style: italic;">No strategies currently passing.</p>';
          }} else {{
            let html = `<table style="width: 100%; border-collapse: collapse; font-size: 0.9rem;">
              <thead>
                <tr style="border-bottom: 2px solid #ddd;">
                  <th style="text-align: left; padding: 0.5rem;">Bot</th>
                  <th style="text-align: left; padding: 0.5rem;">Underlying</th>
                  <th style="text-align: left; padding: 0.5rem;">Strategy</th>
                  <th style="text-align: center; padding: 0.5rem;">Status</th>
                </tr>
              </thead>
              <tbody>`;
            
            for (const s of passing) {{
              const tooltip = s.criteria.map(c => `${{c.metric}}: ${{formatSensorValue(c.value)}} (${{c.note || 'ok'}})`).join('; ');
              html += `<tr style="border-bottom: 1px solid #eee;" title="${{tooltip}}">
                <td style="padding: 0.5rem;">${{s.bot_name}}</td>
                <td style="padding: 0.5rem;">${{s.underlying}}</td>
                <td style="padding: 0.5rem;">${{s.label}}</td>
                <td style="padding: 0.5rem; text-align: center; color: #2e7d32; font-weight: 600;">PASS</td>
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
      const filtered = botsStrategiesData.filter(s => s.expert_id === currentExpertId);
      
      if (filtered.length === 0) {{
        container.innerHTML = '<p style="color: #666; font-style: italic;">No strategies for this expert.</p>';
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
        const statusColors = {{'pass': '#2e7d32', 'blocked': '#c62828', 'no_data': '#666'}};
        const statusLabels = {{'pass': 'PASS', 'blocked': 'BLOCKED', 'no_data': 'NO DATA'}};
        
        const shortDetails = s.criteria.slice(0, 3).map(c => {{
          const mark = c.ok ? 'OK' : (c.note === 'missing_data' ? '?' : 'X');
          return `${{c.metric}} ${{mark}}`;
        }}).join('; ');
        
        const tooltip = s.criteria.map(c => `${{c.metric}}: ${{formatSensorValue(c.value)}} (${{c.note || 'ok'}})`).join('\\n');
        
        html += `<tr style="border-bottom: 1px solid #eee;" title="${{tooltip}}">
          <td style="padding: 0.5rem;">${{s.label}}</td>
          <td style="padding: 0.5rem;">${{s.underlying}}</td>
          <td style="padding: 0.5rem; text-align: center; color: ${{statusColors[s.status]}}; font-weight: 600;">${{statusLabels[s.status]}}</td>
          <td style="padding: 0.5rem; font-size: 0.8rem; color: #666;">${{shortDetails || s.summary.substring(0, 50)}}</td>
        </tr>`;
      }}
      
      html += '</tbody></table>';
      container.innerHTML = html;
    }}
    
    function loadBotsTab() {{
      refreshBotsSensors();
      refreshBotsStrategies();
    }}
    
    // ==============================================
    // SYSTEM HEALTH TAB FUNCTIONS
    // ==============================================
    
    async function loadSystemHealthStatus() {{
      // Load LLM status
      try {{
        const llmRes = await fetch('/api/llm_status');
        const llmData = await llmRes.json();
        if (llmData.ok) {{
          const llmStatus = `Decision: ${{llmData.decision_mode}} | Env: ${{llmData.deribit_env}} | LLM: ${{llmData.llm_enabled ? 'enabled' : 'disabled'}}`;
          document.getElementById('llm-status-line').textContent = llmStatus;
        }} else {{
          document.getElementById('llm-status-line').textContent = 'Error loading LLM status';
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
        }}
      }} catch (e) {{
        console.error('Error loading risk limits:', e);
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
        max_net_delta_abs: parseFloat(document.getElementById('max-net-delta-input').value)
      }};
      
      try {{
        const res = await fetch('/api/risk_limits', {{
          method: 'POST',
          headers: {{'Content-Type': 'application/json'}},
          body: JSON.stringify(payload)
        }});
        const data = await res.json();
        if (data.ok) {{
          feedbackEl.innerHTML = '<span style="color: #2e7d32;">✓ Risk limits updated</span>';
          setTimeout(() => {{ feedbackEl.innerHTML = ''; }}, 3000);
          loadSystemHealthStatus();
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
      el.innerHTML = '<span style="color: #666;">Running healthcheck...</span>';
      detailsEl.style.display = 'none';
      try {{
        const res = await fetch('/api/agent_healthcheck', {{ method: 'POST' }});
        const data = await res.json();
        if (data.ok) {{
          const status = data.overall_status;
          const checksStr = data.results.map(r => `${{r.name}}=${{r.status}}`).join(', ');
          if (status === 'OK') {{
            el.innerHTML = `<span style="color: #2e7d32;">✅ Healthcheck OK – ${{checksStr}}</span>`;
            statusEl.textContent = 'Last check: OK';
          }} else if (status === 'WARN') {{
            el.innerHTML = `<span style="color: #ff9800;">⚠️ Healthcheck WARN – ${{checksStr}}</span>`;
            statusEl.textContent = 'Last check: WARN';
          }} else {{
            el.innerHTML = `<span style="color: #c62828;">❌ Healthcheck FAIL – ${{checksStr}}</span>`;
            statusEl.textContent = 'Last check: FAIL';
          }}
          // Show details
          detailsContent.textContent = data.results.map(r => `${{r.name}}: ${{r.status}} - ${{r.detail}}`).join('\\n');
          detailsEl.style.display = 'block';
        }} else {{
          el.innerHTML = `<span style="color: #c62828;">❌ Healthcheck failed: ${{data.error || data.overall_status}}</span>`;
          statusEl.textContent = 'Last check: ERROR';
        }}
      }} catch (e) {{
        el.innerHTML = `<span style="color: #c62828;">❌ Request error: ${{e.message}}</span>`;
        statusEl.textContent = 'Last check: ERROR';
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
      try {{
        const res = await fetch('/api/positions/open');
        const data = await res.json();
        const tbody = document.getElementById('live-open-positions-body');
        const positions = data.positions || [];
        const totals = data.totals || {{}};
        const summaryEl = document.getElementById('positions-pnl-summary');

        if (positions.length === 0) {{
          tbody.innerHTML = '<tr><td colspan="12" style="text-align:center;color:#666;">No open positions</td></tr>';
          summaryEl.innerHTML = '';
          return;
        }}

        const totalPnl = totals.unrealized_pnl || 0;
        const pnlColor = totalPnl >= 0 ? '#26a69a' : '#ef5350';
        summaryEl.innerHTML = `Total Unrealized: <span style="color:${{pnlColor}};font-weight:600;">${{totalPnl >= 0 ? '+' : ''}}${{totalPnl.toFixed(2)}}</span>`;

        tbody.innerHTML = positions.map(pos => {{
          const typeLabel = (pos.side || 'SHORT') + ' ' + (pos.option_type || 'CALL');
          const stratLabel = (pos.strategy_type || '').replace(/_/g, ' ');
          const pnlClass = pos.unrealized_pnl >= 0 ? 'traded-yes' : 'traded-no';

          return `<tr>
            <td>${{pos.underlying}}</td>
            <td>${{typeLabel}}</td>
            <td>${{stratLabel}}</td>
            <td>${{pos.symbol}}</td>
            <td>${{pos.quantity.toFixed(3)}}</td>
            <td>${{pos.entry_price.toFixed(6)}}</td>
            <td>${{pos.mark_price.toFixed(6)}}</td>
            <td class="${{pnlClass}}">${{pos.unrealized_pnl.toFixed(2)}}</td>
            <td class="${{pnlClass}}">${{pos.unrealized_pnl_pct.toFixed(1)}}%</td>
            <td>${{Math.max(0, pos.dte).toFixed(1)}}</td>
            <td>${{pos.num_rolls}}</td>
            <td>${{pos.mode}}</td>
          </tr>`;
        }}).join('');
      }} catch (err) {{
        console.error('Open positions fetch error:', err);
      }}
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
      }} catch (err) {{
        console.error('Calibration error:', err);
        summaryEl.textContent = 'Calibration failed. Check console/logs.';
        tbody.innerHTML = '<tr><td colspan="6" style="text-align:center;color:#c00;">Error</td></tr>';
      }} finally {{
        btn.disabled = false;
      }}
    }}
    
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

    fetchStatus();
    fetchDecisions();
    refreshBacktestStatus();
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
