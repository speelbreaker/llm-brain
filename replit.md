# Options Trading Agent - Deribit Testnet

## Overview
This project is a modular Python framework for automated BTC/ETH covered call trading on the Deribit testnet. It serves as a research and experimentation system for developing and testing trading strategies, generating training data, and analyzing performance via a web dashboard and backtesting suite. The system supports both rule-based and LLM-powered decision-making, with a focus on exploration-based learning and ambitions for eventual mainnet deployment.

## User Preferences
- Python 3.11
- Type hints everywhere
- Pydantic for configs and models
- httpx for HTTP
- Clarity over cleverness

## System Architecture
The agent features a clear separation of concerns, with modules for configuration, data modeling, API interaction, market context generation, risk management, policy decisions (rule-based and LLM), execution, and logging. It supports "research" and "production" modes, with a FastAPI web application providing a real-time dashboard for monitoring, interaction, and backtesting.

### UI/UX Decisions
The web dashboard offers a user-friendly interface with sections for "Live Agent" status, unified "Backtesting" tab, "Calibration", "System Health", "Chat" interface, "Bots" tab for expert trading bot analysis, "Greg Lab" for position management, "Supervisor" tab for PR verification monitoring, and an "AI Steward" panel for project insights.

### Technical Implementations
- **Configuration**: Pydantic settings manage application configuration.
- **API Wrapper**: An `httpx`-based wrapper for the Deribit testnet API.
- **State Management**: Aggregates market data and manages thread-safe status updates.
- **Structured Logging**: Uses JSONL for logging decisions and actions.
- **Health Guard System**: Runtime health guard with severity-based decision making (TRANSIENT, DEGRADED, FATAL) for intelligent recovery, including startup checks and runtime re-checks.
- **IV Sanity Check**: Validates that the synthetic IV pricing layer responds correctly to parameter changes. Runs backtests with different IV multipliers (generic: 0.8/1.2, gregbot: 0.9/1.1) and verifies results differ meaningfully. Available via `/api/health/iv_sanity` endpoint and integrated into the healthcheck system (`scripts/iv_sanity_check.py`).
- **Decision Policies**: Supports rule-based strategies with scoring and epsilon-greedy exploration, and an LLM-powered decision mode validated by a risk engine. Decision modes include `rule_only`, `llm_only`, and `hybrid_shadow`.
- **Risk Management**: A `risk_engine` performs pre-trade validation, checking margin, delta, exposure limits, liquidity guards, and hard safety rails.
- **Backtesting Framework**: Includes `CoveredCallSimulator` for historical analysis, supporting various exit styles, training data generation, and TradingView-style metrics. Persistent backtest runs are stored in PostgreSQL.
- **Training Mode**: Allows multi-profile data collection for ML/RL.
- **LLM Fine-Tuning Data**: Scripts transform candidate CSVs into chat-style JSONL corpora.
- **Runtime Controls**: System Health tab offers interactive controls for adjusting safety and operational settings (e.g., Global Kill Switch, Daily Drawdown Limit, Decision Mode, Dry Run Mode).
- **AI Steward (Project Brain)**: A project planning and QA helper that summarizes project state and suggests next tasks using an LLM.
- **State-Aware Chat Assistant**: A multi-turn assistant that understands current trading state, answers questions, and provides project information.
- **Position Reconciliation**: Compares local position tracker against exchange positions.
- **Synthetic Universe v2**: Incorporates `RegimeParams`, KMeans clustering, AR(1) IV dynamics, and Greg-sensor clusters for realistic IV evolution and regime modeling.
- **Extended Calibration System (v2)**: Enhanced calibration with liquidity filtering, multi-DTE bands, bucket metrics, skew fitting, recommended `vol_surface` generation, and vega-weighted MAE.
- **Auto IV Calibration Pipeline**: Persists time series of auto-calculated IV multipliers in a `calibration_history` table, with runtime in-memory override store and API endpoints/UI for management.
- **Calibration Update Policy System**: Policy layer with smoothing (EWMA), thresholds, and file-based history storage for applying updates.
- **Harvester Data Quality & Reproducibility**: Includes schema validation and quality assessment of harvested Parquet snapshots.
- **Bots System**: Provides a comprehensive view of expert trading bots, market sensors, and strategy evaluations. This includes:
    - **Greg Mandolini VRP Harvester (GregBot)**: A quantitative VRP strategy selector based on volatility sensors and a decision waterfall.
    - **Greg Position Management**: Advisory-only position management module for open Greg positions.
    - **Delta Hedging Engine**: Advisory-only delta-neutral hedging module for short-vol strategies.
- **Strategy Layer**: A pluggable architecture allowing multiple trading strategies to run concurrently.
- **Sandbox Position System**: Creates isolated test positions for Greg strategies.
- **Real-Trading Mode (Phase 2)**: Execution support with strong safety gates (Global mode, master switch, per-strategy flags, dry_run cross-check) and tiny-size guardrails.
- **Decision Logging System**: Comprehensive audit trail for Greg decisions stored in a `greg_decision_log` database table.
- **Greg Lab UI**: Dedicated dashboard tab for viewing and managing Greg strategy positions with mode banner, sandbox summary, filters, positions table, PnL tracking, suggested actions, and log timelines.
- **Strategy Capabilities System**: Strategy-aware backtest configuration with:
    - **StrategyCapabilities metadata**: Each strategy declares which config fields it owns vs. user-configurable
    - **GregBot parameter management**: GregBot owns exit_style, DTE, and delta targeting; user controls underlying, date range, IV multiplier
    - **Config override logic**: `apply_strategy_overrides()` returns warnings when user inputs conflict with strategy requirements
    - **Backtest event logging**: `BacktestEvent` table tracks strategy decisions, opens, closes, rolls, take-profits for analytics
    - **Event emitter**: `BacktestEventEmitter` provides standard interface for strategies to emit analytics events
    - **Strategy breakdown endpoints**: `/api/backtests/{run_id}/events` and `/api/backtests/{run_id}/strategy_summary` for post-run analysis
    - **API endpoints**: `GET /api/backtest/strategy_caps`, `GET /api/backtest/strategy_caps/{selector}`, `GET /api/backtest/strategies`
- **Telegram Code Review Agent**: A Telegram bot for automated code review and Repo Q&A with modular design. It offers commands for `/review`, `/diff`, `/risks`, `/ask`, `/search`, `/open`, and natural language chat, utilizing LLM integration with automatic model fallback and secret redaction.
- **PR Supervisor Service**: Automated PR verification and auto-fix service that:
    - Listens to GitHub PR webhooks (opened/synchronize/reopened)
    - Checks out PR branch in isolated workspace using git worktree
    - Runs verification commands (pytest/lint)
    - Posts structured PR comments with results
    - Runs 3-agent debate (Optimist/Skeptic/Arbiter) to decide on auto-fix
    - Invokes Codex CLI for minimal auto-fixes if approved
    - **Multi-provider LLM abstraction**: Supports OpenAI (primary) and Gemini (fallback) with automatic failover
    - **Telegram Status Card UX**: Single updateable message per PR with phase-based updates (STARTING → CHECKS → DEBATE → CODEX_FIX → DONE/NEEDS_HUMAN), with HTML escaping and plaintext fallback
    - **Dashboard Integration**: Supervisor tab in web dashboard with job listing, filtering, and detail views (requires `SUPERVISOR_API_URL`)
    - **Production Hardening (v0.2.0)**:
        - In-process asyncio.Queue job worker (replaces BackgroundTasks for reliability)
        - Fail-fast startup validation (503 on missing GITHUB_TOKEN/GITHUB_WEBHOOK_SECRET)
        - Secret redaction in PR comments, Telegram, and API responses (`src/supervisor/redact.py`)
        - Retry helper with exponential backoff for external APIs (`src/supervisor/retry.py`)
        - Job store write safety with asyncio.Lock and atomic file writes
        - API response truncation to prevent payload bloat
        - HTML escaping for Telegram messages with automatic plaintext fallback
    - **Disabled by default** (SUPERVISOR_ENABLED=0)

### PR Supervisor Configuration
The PR Supervisor is located in `src/supervisor/` and is disabled by default. To enable:

**Required Environment Variables (when enabled):**
- `SUPERVISOR_ENABLED=1` - Enable the supervisor
- `GITHUB_WEBHOOK_SECRET` - Secret for webhook signature verification
- `GITHUB_TOKEN` - Fine-grained PAT with PR comments + push access

**Autofix Policy (Opt-in Codex Gating):**
- `SUPERVISOR_ENABLE_CODEX` (default: 0) - Enable Codex auto-fix (must be 1 to allow fixes)
- `SUPERVISOR_AUTOFIX_POLICY` (default: label) - Policy: `label`, `telegram`, or `both`
- `SUPERVISOR_AUTOFIX_LABEL` (default: autofix-ok) - Required label for `label` policy
- `SUPERVISOR_REQUIRE_HUMAN_FOR_HIGH_RISK` (default: 1) - Block high-risk auto-fixes

**Telegram Commands:**
- `TELEGRAM_ALLOWED_USER_IDS` - Comma-separated user IDs allowed to use commands
- `TELEGRAM_ADMIN_CHAT_ID` - Optional admin chat for alerts

| Command | Description |
|---------|-------------|
| `/supervisor last` | Show last 5 jobs |
| `/supervisor pr <n>` | Show PR status |
| `/rerun <n>` | Queue rerun for PR |
| `/autofix <n>` | Approve autofix |
| `/pause <n>` | Pause PR processing |
| `/resume <n>` | Resume PR |
| `/revoke <n>` | Revoke autofix approval |

**Debug & Deployment:**
- `SUPERVISOR_DEBUG` (default: 0) - Enable debug endpoints (conditionally registers routes)
- `SUPERVISOR_DEBUG_TOKEN` - Optional token for debug endpoint authentication (header: X-Debug-Token)
- `SUPERVISOR_WORKSPACE_TTL_HOURS` (default: 24) - Workspace cleanup TTL
- See `docs/SUPERVISOR_VPS_SETUP.md` for VPS deployment guide
- Docker files: `docker/supervisor.Dockerfile`, `docker/docker-compose.supervisor.yml`

**Safety Features (v0.3.1):**
- Webhook secret validation returns 503 with details when misconfigured
- Debug endpoints only registered when SUPERVISOR_DEBUG=1
- Approval state uses threading.Lock for concurrent write safety
- Workspace cleanup uses .supervisor_active sentinel files to protect active jobs
- All timestamps are timezone-aware UTC
- Retry helper with exponential backoff for external API calls (429, 500-504)

**Other Settings:**
- `SUPERVISOR_MAX_LOOPS` (default: 3) - Max fix attempts
- `SUPERVISOR_MAX_FILES_CHANGED` (default: 10) - Max files in fix diff
- `SUPERVISOR_MAX_LOC_CHANGED` (default: 300) - Max LOC in fix diff
- `SUPERVISOR_ALLOW_FORKS` (default: 0) - Allow PRs from forks
- `MODEL_OPTIMIST` / `MODEL_SKEPTIC` / `MODEL_ARBITER` - LLM models
- `CHECK_CMD_1` / `CHECK_CMD_2` / `CHECK_CMD_3` - Verification commands

**Running the Supervisor:**
```bash
python -m src.supervisor  # Runs on port 8001
```

## External Dependencies
- **Deribit API**: Used for real-time market data (testnet) and historical data.
- **OpenAI**: Integrated for LLM-powered decision-making and generating insights.
- **PostgreSQL**: Used for persistent storage of backtest runs and decision logs.