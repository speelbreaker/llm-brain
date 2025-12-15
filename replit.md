# Options Trading Agent - Deribit Testnet

## Overview
This project is a modular Python framework for automated BTC/ETH covered call trading on the Deribit testnet. It aims to develop and test trading strategies, generate training data, and analyze performance via a web dashboard and backtesting suite. The system supports both rule-based and LLM-powered decision-making, with a focus on exploration-based learning and ambitions for eventual mainnet deployment.

## User Preferences
- Python 3.11
- Type hints everywhere
- Pydantic for configs and models
- httpx for HTTP
- Clarity over cleverness

## System Architecture
The agent features a clear separation of concerns, with modules for configuration, data modeling, API interaction, market context generation, risk management, policy decisions (rule-based and LLM), execution, and logging. It supports "research" and "production" modes, with a FastAPI web application providing a real-time dashboard for monitoring, interaction, and backtesting.

### UI/UX Decisions
The web dashboard offers a user-friendly interface with sections for "Live Agent" status, "Backtesting", "Calibration", "System Health", "Chat" interface, "Bots" for expert trading bot analysis, "Greg Lab" for position management, "Supervisor" for PR verification monitoring, and an "AI Steward" panel for project insights.

### Web Layer Structure
The FastAPI app entrypoint is `src/web_app.py`, which owns lifespan/app.state initialization and includes routers from `src/web/`.

| Module | Description |
|--------|-------------|
| `src/web/dashboard.py` | UI changes - Main Options Agent dashboard HTML + inline JS |
| `src/web/routes_main.py` | API routes - Core endpoints (status, chat, training, strategy-status) |
| `src/web/routes_backtest.py` | API routes - Backtest Lab APIs |
| `src/web/routes_positions.py` | API routes - Position and calibration endpoints |
| `src/web/routes_bots.py` | API routes - Bot/strategy endpoints |
| `src/web/routes_health.py` | API routes - Health, risk, supervisor, steward endpoints |

**Guidelines:**
- UI changes → `src/web/dashboard.py`
- API routes → `src/web/routes_*.py`
- Entrypoint → `src/web_app.py` owns lifespan/app.state init

# Replit Agent Rules (Non-Negotiable)

## Engineering Constitution (Non-Negotiable Quality Gates)

These rules are mandatory for ALL changes. If any rule fails, the work is rejected.

### 1) Regression Gate (Hard Stop)
- MUST run the full suite: `python -m pytest -q`
- If any existing test fails, FIX THE CODE (not the test) unless the behavior change is intentional and documented.
- PROHIBITED: deleting/skipping tests to make the build pass.

### 2) Endpoint Testing Rule (Hard Stop)
- For EVERY new endpoint, MUST add at least ONE endpoint-level test (FastAPI TestClient) covering:
  - success path (expected status + minimal response shape)
  - at least one failure/guard case (400/401/403/404/422/503 as appropriate)
- For modified endpoints, MUST add/extend endpoint-level tests proving the bugfix and preventing regressions.
- Any prompt that adds an endpoint without an endpoint-level test is REJECTED.

### 3) Static Integrity Gate (“Does it compile/import?”)
Before running logic tests, code must pass a static integrity check:
- No SyntaxError / ImportError in modified modules.
- MUST run at least one:
  - `python -m compileall -q src`
  - AND/OR an import smoke test that imports the modified modules (and critical entrypoints).

### 4) Route Parity Gate (for refactors / router splits)
If routes are moved/refactored:
- MUST run a route-parity test comparing `[(path, methods, name)]` to a committed baseline.
- Only update the baseline if changes are intentional and documented.
- MUST preserve paths/methods/tags unless explicitly requested.

### 5) Shared Dependency Constraint (Stop breaking callers)
If you change any shared utility/model/schema/helper:
- MUST search all usages (e.g., `rg "FunctionName\\(" -g'*.py'`).
- If signature/return shape changes, MUST update ALL callers in the same change.
- PROHIBITED: “fix one place, break five others.”

### 6) Frontend/Backend Contract Gate (if HTML/JS touched)
If API response shape changes OR frontend uses an endpoint:
- MUST search frontend code for the endpoint/path and updated keys.
- MUST update frontend to match the backend change.
- SHOULD add a small contract test (or endpoint-level test asserting key names used by frontend).

### 7) Security Gate (especially Supervisor)
- Webhook endpoints MUST 503 when required secrets are missing/empty.
- Signature verification MUST be enforced (no empty-secret fallback).
- Debug endpoints MUST be disabled unless `SUPERVISOR_DEBUG=1` AND protected (token/header).
- MUST redact secrets from PR comments, Telegram messages, and API payloads.
- PROHIBITED: returning raw exception strings to clients.

### 8) External API Resilience Gate
All outbound calls (GitHub/Telegram/LLMs) MUST:
- use explicit httpx timeouts
- use retries w/ backoff for 429/5xx (bounded attempts)
- avoid leaking sensitive payloads in logs
- properly close AsyncClient (context manager or shared lifecycle client)

### 9) Concurrency & Persistence Gate
If writing shared state (job store, approvals, JSONL/SQLite):
- MUST be atomic and safe under concurrency (lock + atomic replace where applicable)
- PROHIBITED: swallowing write errors silently
- MUST have tests for concurrent writes or corruption recovery if using files.

### 10) Definition of Done Checklist (must be stated in completion message)
Before marking work complete, report:
- ✅ Full test suite passed: `python -m pytest -q`
- ✅ Static integrity passed (compile/import smoke test)
- ✅ List of new/changed endpoints + each has endpoint-level tests
- ✅ Impact analysis done for any shared dependency changes (all callers updated)
- ✅ Route parity baseline updated (only if intentional) and test passed
- ✅ No secrets leaked (redaction applied where relevant)

### Technical Implementations
- **Configuration**: Pydantic settings manage application configuration.
- **API Wrapper**: An `httpx`-based wrapper for the Deribit testnet API.
- **State Management**: Aggregates market data and manages thread-safe status updates.
- **Structured Logging**: Uses JSONL for logging decisions and actions.
- **Health Guard System**: Runtime health guard with severity-based decision making for intelligent recovery.
- **IV Sanity Check**: Validates synthetic IV pricing layer response to parameter changes.
- **Decision Policies**: Supports rule-based strategies with scoring and epsilon-greedy exploration, and an LLM-powered decision mode validated by a risk engine. Decision modes include `rule_only`, `llm_only`, and `hybrid_shadow`.
- **Risk Management**: A `risk_engine` performs pre-trade validation, checking margin, delta, exposure limits, liquidity guards, and hard safety rails.
- **Backtesting Framework**: Includes `CoveredCallSimulator` for historical analysis, supporting various exit styles and training data generation. Persistent backtest runs are stored in PostgreSQL.
- **Training Mode**: Allows multi-profile data collection for ML/RL.
- **LLM Fine-Tuning Data**: Scripts transform candidate CSVs into chat-style JSONL corpora.
- **Runtime Controls**: System Health tab offers interactive controls for adjusting safety and operational settings.
- **AI Steward (Project Brain)**: A project planning and QA helper that summarizes project state and suggests next tasks using an LLM.
- **State-Aware Chat Assistant**: A multi-turn assistant that understands current trading state, answers questions, and provides project information.
- **Position Reconciliation**: Compares local position tracker against exchange positions.
- **Synthetic Universe v2**: Incorporates `RegimeParams`, KMeans clustering, AR(1) IV dynamics, and Greg-sensor clusters for realistic IV evolution and regime modeling.
- **Extended Calibration System (v2)**: Enhanced calibration with liquidity filtering, multi-DTE bands, bucket metrics, skew fitting, recommended `vol_surface` generation, and vega-weighted MAE.
- **Auto IV Calibration Pipeline**: Persists time series of auto-calculated IV multipliers in a `calibration_history` table.
- **Calibration Update Policy System**: Policy layer with smoothing, thresholds, and file-based history storage for applying updates.
- **Harvester Data Quality & Reproducibility**: Includes schema validation and quality assessment of harvested Parquet snapshots.
- **Bots System**: Provides a comprehensive view of expert trading bots, market sensors, and strategy evaluations, including Greg Mandolini VRP Harvester (GregBot), Greg Position Management, and a Delta Hedging Engine.
- **Strategy Layer**: A pluggable architecture allowing multiple trading strategies to run concurrently.
- **Sandbox Position System**: Creates isolated test positions for Greg strategies.
- **Real-Trading Mode (Phase 2)**: Execution support with strong safety gates and tiny-size guardrails.
- **Decision Logging System**: Comprehensive audit trail for Greg decisions stored in a `greg_decision_log` database table.
- **Greg Lab UI**: Dedicated dashboard tab for viewing and managing Greg strategy positions.
- **Strategy Capabilities System**: Strategy-aware backtest configuration with metadata, parameter management, config override logic, field hints for UX, backtest event logging, and API endpoints for analysis.
- **Selector Frequency Scan with Live IV**: Extended scan configuration with `iv_mode` toggle supporting 'synthetic', 'live', or 'hybrid' IV.
- **Telegram Code Review Agent**: A Telegram bot for automated code review and Repo Q&A with LLM integration.
- **PR Supervisor Service**: Automated PR verification and auto-fix service that listens to GitHub PR webhooks, runs verification commands, posts structured PR comments, and can invoke auto-fixes if approved. It includes multi-provider LLM abstraction and Telegram Status Card UX. This service is disabled by default.

## External Dependencies
- **Deribit API**: Used for real-time market data (testnet) and historical data.
- **OpenAI**: Integrated for LLM-powered decision-making and generating insights.
- **PostgreSQL**: Used for persistent storage of backtest runs and decision logs.
- **GitHub**: Integrated for PR Supervisor webhooks and interaction.
- **Telegram**: Used for the Code Review Agent and PR Supervisor notifications.
  