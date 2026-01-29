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
from src.web.routes_fidelity import router as fidelity_router
from src.web.routes_bots import router as bots_router
from src.web.routes_health import router as health_router
from src.web.routes_deploy import router as deploy_router
from src.web.routes_supervisor import router as supervisor_router

from src.supervisor.config import get_settings as get_supervisor_settings
from src.supervisor.store import JobStore
from src.supervisor.app import job_worker as supervisor_job_worker
from src.supervisor.github import GitHubClient
from src.supervisor.telegram_bot import TelegramBotManager
import httpx
import asyncio

app = FastAPI(
    title="Options Trading Agent Dashboard",
    description="Deribit testnet covered-call agent with live status, chat, and backtesting.",
    version="0.2.0",
)

app.include_router(main_router)
app.include_router(backtest_router)
app.include_router(positions_router)
app.include_router(fidelity_router)
app.include_router(bots_router)
app.include_router(health_router)
app.include_router(supervisor_router)
# Back-compat for GitHub webhooks already configured at /github/webhook
try:
    from src.web.routes_supervisor import legacy_router as supervisor_legacy_router
    app.include_router(supervisor_legacy_router)
except ImportError:
    import logging
    logging.getLogger(__name__).info("[Supervisor] legacy_router not available (ImportError)")
except Exception:
    import logging
    logging.getLogger(__name__).exception("[Supervisor] Failed to include legacy_router")

if os.environ.get("DEPLOY_WEBHOOK_SECRET"):
    app.include_router(deploy_router)


def _background_threads_enabled() -> bool:
    # Tests frequently construct a TestClient(app), which triggers FastAPI startup.
    # Starting the live agent loop in unit tests is both noisy and can require
    # local market data fixtures that aren't present.
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return False
    if (os.environ.get("DISABLE_BACKGROUND_THREADS") or "").strip().lower() in {"1", "true", "yes"}:
        return False
    return True


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


def _init_supervisor_state(app_instance: FastAPI) -> None:
    """Initialize Supervisor state on the main app instance."""
    print("[Supervisor] Initializing configuration...")
    sup_settings = get_supervisor_settings()
    # Do NOT overwrite app.state.settings (used by the main app in many codebases).
    # Store supervisor config under a dedicated namespace.
    app_instance.state.supervisor_settings = sup_settings
    
    # Initialize common components
    app_instance.state.job_queue = asyncio.Queue()
    app_instance.state.store = JobStore(f"{sup_settings.base_jobs_dir}/job_history.jsonl")
    app_instance.state.github_client = None
    app_instance.state.telegram_http = None
    app_instance.state.telegram_bot = None
    app_instance.state.supervisor_worker_task = None
    app_instance.state.telegram_bot_task = None
    app_instance.state.startup_errors = []
    app_instance.state.ready = False

    if sup_settings.enabled:
        missing_github = []
        if not sup_settings.github_webhook_secret:
            missing_github.append("GITHUB_WEBHOOK_SECRET")
        if not sup_settings.github_token:
            missing_github.append("GITHUB_TOKEN")
            
        if missing_github:
            print(f"[Supervisor] Warning: Missing GitHub config {missing_github}. GitHub features will be disabled.")
            app_instance.state.startup_errors.extend(missing_github)
        else:
            print("[Supervisor] GitHub Config OK. Initializing client...")
            app_instance.state.github_client = GitHubClient(sup_settings.github_token)
            
        if sup_settings.telegram_enabled and sup_settings.telegram_bot_token:
            print("[Supervisor] Telegram Config OK. Initializing bot...")
            app_instance.state.telegram_http = httpx.AsyncClient(timeout=httpx.Timeout(20.0))
            app_instance.state.telegram_bot = TelegramBotManager(sup_settings, app_instance.state.store)
            app_instance.state.telegram_bot.ready = True
            
        app_instance.state.ready = True
        
        # Start the async worker task if background threads are enabled
        if _background_threads_enabled():
            try:
                loop = asyncio.get_running_loop()
                
                # Only start worker if GitHub client is available
                if app_instance.state.github_client:
                    app_instance.state.supervisor_worker_task = loop.create_task(supervisor_job_worker(app_instance))
                    print("[Supervisor] Job worker started")
                else:
                    print("[Supervisor] Job worker SKIPPED (missing GitHub config)")
                
                if app_instance.state.telegram_bot:
                    app_instance.state.telegram_bot_task = loop.create_task(app_instance.state.telegram_bot.start())
                    print("[Supervisor] Telegram bot (polling) started")
            except RuntimeError:
                print("[Supervisor] Warning: No running event loop to start worker (expected in uvicorn)")
            else:
                print("[Supervisor] Job worker SKIPPED (background threads disabled)")
    else:
        print("[Supervisor] Disabled (SUPERVISOR_ENABLED=0)")


@app.on_event("startup")
async def start_background_services() -> None:
    """Start all background services: Agent, Healthcheck, Supervisor."""
    
    # Always initialize Supervisor State (needed for endpoints)
    _init_supervisor_state(app)

    if not _background_threads_enabled():
        return

    # 1. Trading Agent (Thread)
    try:
        from src.db import init_db
        init_db()
    except Exception as e:
        print(f"[DB] Warning: Could not initialize database: {e}")
    
    thread = threading.Thread(target=_agent_thread_target, daemon=True)
    thread.start()
    print("Agent loop started in background thread")
    
    # 2. Healthcheck (Thread)
    healthcheck_thread = threading.Thread(target=_healthcheck_scheduler_target, daemon=True)
    healthcheck_thread.start()
    print("Healthcheck scheduler started in background thread")



@app.on_event("shutdown")
async def shutdown_background_services():
    """Cleanup async resources."""
    # Supervisor Cleanup
    if hasattr(app.state, "supervisor_worker_task") and app.state.supervisor_worker_task:
        print("[Supervisor] Cancelling worker...")
        app.state.supervisor_worker_task.cancel()
        try:
            await app.state.supervisor_worker_task
        except asyncio.CancelledError:
            pass
            
    if hasattr(app.state, "telegram_bot") and app.state.telegram_bot:
        print("[Supervisor] Stopping Telegram bot...")
        await app.state.telegram_bot.stop()
        if app.state.telegram_bot_task:
            app.state.telegram_bot_task.cancel()
            try:
                await app.state.telegram_bot_task
            except asyncio.CancelledError:
                pass
        
    if hasattr(app.state, "telegram_http") and app.state.telegram_http:
        await app.state.telegram_http.aclose()
        
    if hasattr(app.state, "github_client") and app.state.github_client:
        if hasattr(app.state.github_client, "close"):
             # If it has an async close, await it; otherwise call it
             # GitHubClient implementation in src/supervisor/github.py uses httpx, likely needs aclose or similar.
             # We'll just pass for now as the process is dying.
             pass


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """Full HTML dashboard with Live Agent view and Backtesting Lab."""
    return render_dashboard_html()
