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
from src.chat_with_agent import chat_with_agent
from src.config import settings


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
    """Ask the agent a question about its recent behavior."""
    question = payload.get("question", "").strip()
    if not question:
        return JSONResponse(
            status_code=400,
            content={"error": "Missing 'question' field in request body."},
        )

    try:
        answer = chat_with_agent(question, log_limit=20)
        return JSONResponse(content={"question": question, "answer": answer})
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to generate answer: {str(e)}"},
        )


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
    """Run a backtest using the CoveredCallSimulator."""
    from src.backtest.types import CallSimulationConfig
    from src.backtest.data_source import Timeframe
    from src.backtest.covered_call_simulator import CoveredCallSimulator, always_trade_policy
    from src.backtest.deribit_data_source import DeribitDataSource
    
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
    
    return JSONResponse(content={
        "config": {
            "underlying": req.underlying,
            "start": req.start,
            "end": req.end,
            "timeframe": req.timeframe,
            "target_dte": req.target_dte,
            "target_delta": req.target_delta,
        },
        "metrics": {
            "num_trades": result.metrics.get("num_trades", 0),
            "final_pnl": round(result.metrics.get("final_pnl", 0), 4),
            "avg_pnl": round(result.metrics.get("avg_pnl", 0), 4),
            "max_drawdown_pct": round(result.metrics.get("max_drawdown_pct", 0), 2),
            "win_rate": round(result.metrics.get("win_rate", 0) * 100, 1),
        },
        "equity_curve": equity_curve,
        "trades_sample": trades_sample,
    })


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

    <div class="section">
      <h2>Latest Decision</h2>
      <div class="decision-card" id="latest-decision">
        <h3 id="decision-time">Waiting for first decision...</h3>
        <div class="action" id="decision-action">--</div>
        <div class="details" id="decision-details">--</div>
        <div class="reasoning" id="decision-reasoning">Agent is starting up...</div>
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

  <!-- CHAT TAB -->
  <div id="tab-chat" class="tab-content">
    <div class="section">
      <h2>Chat with Agent</h2>
      <div class="form-group">
        <label for="question">Ask a question about the agent's behavior:</label>
        <textarea id="question" placeholder="e.g., Why did you pick the 97k call? What would you do next?"></textarea>
      </div>
      <button id="ask-btn" onclick="sendQuestion()">Ask Agent</button>
      
      <h3>Answer</h3>
      <pre id="answer-box" class="answer-box">Ask a question to get started...</pre>
    </div>
  </div>

  <script>
    let backtestResult = null;
    
    function showTab(name) {{
      document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
      document.querySelector(`[onclick="showTab('${{name}}')"]`).classList.add('active');
      document.getElementById(`tab-${{name}}`).classList.add('active');
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
      }} catch (err) {{
        console.error('Status fetch error:', err);
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
          
          document.getElementById('decision-reasoning').innerText = proposed.reasoning || final.reasoning || 'No reasoning provided';
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

    async function sendQuestion() {{
      const q = document.getElementById('question').value;
      if (!q.trim()) {{
        alert('Please enter a question first.');
        return;
      }}
      
      const btn = document.getElementById('ask-btn');
      btn.disabled = true;
      btn.innerText = 'Thinking...';
      document.getElementById('answer-box').innerText = 'Analyzing logs and generating response...';

      try {{
        const res = await fetch('/chat', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify({{ question: q }})
        }});
        const data = await res.json();
        document.getElementById('answer-box').innerText = data.error || data.answer;
      }} catch (err) {{
        document.getElementById('answer-box').innerText = 'Error: ' + err;
      }} finally {{
        btn.disabled = false;
        btn.innerText = 'Ask Agent';
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
