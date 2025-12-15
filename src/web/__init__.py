"""Web layer package for Options Trading Agent FastAPI application.

This package contains:
- dashboard.py: Main HTML dashboard rendering
- routes_main.py: Core status/chat/training endpoints
- routes_backtest.py: Backtest APIs
- routes_positions.py: Position and calibration endpoints
- routes_bots.py: Bot/strategy endpoints
- routes_health.py: Health, risk limits, strategy thresholds, and system configuration endpoints
  - LLM status and configuration (GET/POST /api/llm_status)
  - LLM decision testing (POST /api/test_llm_decision)
  - Risk limits configuration (GET/POST /api/risk_limits)
  - Strategy thresholds (GET/POST /api/strategy_thresholds)
  - Kill switch testing (POST /api/test_kill_switch)
  - Agent healthcheck (POST /api/agent_healthcheck)
  - System health status (GET /api/system_health/status)
  - LLM readiness (GET /api/llm_readiness)
  - IV sanity check (GET /api/health/iv_sanity)
  - AI Steward (POST /api/steward/run, GET /api/steward/report)
  - Greg sweetspots (GET/POST /api/greg_sweetspots)
  - Runtime config (GET/POST /api/system/runtime-config)
  - Supervisor jobs (GET /api/supervisor/jobs, GET /api/supervisor/jobs/{job_id})
"""
