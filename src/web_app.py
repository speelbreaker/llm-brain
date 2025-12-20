"""
FastAPI web application for the Options Trading Agent.
Provides live status, chat interface, Live Agent Dashboard, and Backtesting Lab.

This is the thin entrypoint that wires together the routers from src/web/ package.
"""
from __future__ import annotations

import os
import threading
from typing import Any, Dict

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from agent_loop import run_agent_loop_forever
from src.status_store import status_store
from src.config import settings

from src.web.dashboard import render_dashboard_html
from src.web.routes_main import router as main_router
from src.web.routes_backtest import router as backtest_router
from src.web.routes_backtest import BacktestStartRequest
from src.web.routes_positions import router as positions_router
from src.web.routes_bots import router as bots_router
from src.web.routes_health import router as health_router
from src.web.routes_deploy import router as deploy_router



app = FastAPI(
    title="Options Trading Agent Dashboard",
    description="Deribit testnet covered-call agent with live status, chat, and backtesting.",
    version="0.2.0",
)

app.include_router(main_router)
app.include_router(backtest_router)
app.include_router(positions_router)
app.include_router(bots_router)
app.include_router(health_router)
if os.environ.get("DEPLOY_WEBHOOK_SECRET"):
    app.include_router(deploy_router)


def _agent_thread_target() -> None:
    """Run the agent loop forever, updating status_store each iteration."""
    def status_callback(snapshot: Dict[str, Any]) -> None:
        status_store.update(snapshot)

    run_agent_loop_forever(status_callback=status_callback)


def _healthcheck_scheduler_target() -> None:
    """Background thread that runs periodic healthchecks.
    
    Always runs an initial healthcheck on startup to populate the cache.
    Then runs periodic checks if interval > 0.
    """
    import time
    from src.healthcheck import run_and_cache_healthcheck
    
    interval = settings.health_recheck_interval_seconds
    
    print("[Healthcheck] Running initial healthcheck on startup...")
    try:
        result = run_and_cache_healthcheck()
        print(f"[Healthcheck] Initial check: {result.overall_status} - {result.summary}")
    except Exception as e:
        print(f"[Healthcheck] Initial check failed: {e}")
    
    if interval <= 0:
        print("[Healthcheck] Periodic checks disabled (interval <= 0)")
        return
    
    print(f"[Healthcheck] Periodic scheduler running (interval={interval}s)")
    
    while True:
        time.sleep(interval)
        try:
            result = run_and_cache_healthcheck()
            print(f"[Healthcheck] Periodic check: {result.overall_status} - {result.summary}")
        except Exception as e:
            print(f"[Healthcheck] Periodic check failed: {e}")


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
    
    healthcheck_thread = threading.Thread(target=_healthcheck_scheduler_target, daemon=True)
    healthcheck_thread.start()
    print("Healthcheck scheduler started in background thread")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """Full HTML dashboard with Live Agent view and Backtesting Lab."""
    return render_dashboard_html()
