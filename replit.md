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
The web dashboard offers a user-friendly interface with sections for "Live Agent" status, unified "Backtesting" tab (with collapsible Lab and Runs sections), "Calibration", "System Health", "Chat" interface, "Bots" tab for expert trading bot analysis, "Greg Lab" for position management, and an "AI Steward" panel for project insights.

### Technical Implementations
- **Configuration**: Pydantic settings manage application configuration.
- **API Wrapper**: An `httpx`-based wrapper (`deribit_client.py`) for the Deribit testnet API.
- **State Management**: Aggregates market data, manages thread-safe status updates, and unifies state-building logic.
- **Order Execution**: Handles order translation and dry-run simulations.
- **Structured Logging**: Uses JSONL for logging decisions and actions.
- **Position Persistence**: Bot-managed positions are saved to `data/positions.json` and restored on restart.
- **Agent Healthcheck Module**: Self-contained system for critical pipeline validation with expanded config validation (risk settings, LLM config), auto-trigger on tab open (60s throttle), status badge (OK/WARN/FAIL), and LLM diagnostic gating.
- **Health Guard System**: Runtime health guard with severity-based decision making:
    - **HealthSeverity Enum**: Classifies errors as TRANSIENT (retry), DEGRADED (pause on testnet), or FATAL (halt on mainnet).
    - **CachedHealthStatus**: Thread-safe cached health results with `worst_severity` for intelligent recovery.
    - **Startup Guard**: Severity-aware startup checks - FATAL aborts on mainnet, TRANSIENT proceeds with warning.
    - **Runtime Re-check**: Periodic health checks during trading loop with severity-based pause/resume.
    - **Agent Pause Indicator**: UI display of health guard status and agent pause state.
    - **Config Settings**: `health_check_on_startup`, `auto_kill_on_health_fail`, `health_recheck_interval_seconds`.
- **Decision Policies**: Supports rule-based strategies with scoring and epsilon-greedy exploration, and an LLM-powered decision mode validated by a risk engine. Decision modes include `rule_only`, `llm_only`, and `hybrid_shadow`.
- **LLM Validation**: Comprehensive validation of LLM decisions including symbol, action type, position, and size clamping.
- **Market Context**: Integrates `MarketContext` for regime detection, returns, and realized volatility.
- **Risk Management**: A `risk_engine` performs pre-trade validation, checking margin, delta, exposure limits, liquidity guards, and hard safety rails.
- **Backtesting Framework**: Includes `CoveredCallSimulator` for historical analysis, supporting various exit styles, training data generation, synthetic pricing, and TradingView-style metrics.
- **Selector Analysis**: Includes "Selector Frequency Scan" for rule pass rates and "Selector Heatmap" for 2D parameter sweeps.
- **Environment Heatmap**: Provides a selector-free 2D occupancy heatmap for market time spent across metric pairs, with UI automation for analysis.
- **Training Mode**: Allows multi-profile data collection for ML/RL, with aggressive policies and configurable per-expiry limits.
- **LLM Fine-Tuning Data**: Scripts transform candidate CSVs into chat-style JSONL corpora for LLM fine-tuning.
- **Runtime Controls**: System Health tab offers interactive controls for adjusting safety and operational settings (e.g., Global Kill Switch, Daily Drawdown Limit, Decision Mode, Dry Run Mode, Liquidity Guards, Position Reconciliation Config), and LLM/strategy tuning.
- **Persistent Backtest Runs**: Stores backtest results in PostgreSQL using SQLAlchemy ORM.
- **Comparison & Health Check Scripts**: Tools for comparing synthetic vs. live backtests and performing strategy health checks.
- **AI Steward (Project Brain)**: A project planning and QA helper (`src/system_steward.py`) that summarizes project state and suggests next tasks using an LLM.
- **State-Aware Chat Assistant**: A multi-turn assistant that understands current trading state, answers questions, and provides project information.
- **Position Reconciliation**: Compares local position tracker against exchange positions, with configurable actions.
- **Calibration vs Deribit**: Compares synthetic Black-Scholes prices against live Deribit marks.
- **Synthetic Universe v2**: Incorporates `RegimeParams`, KMeans clustering, AR(1) IV dynamics, and Greg-sensor clusters for realistic IV evolution and regime modeling.
- **Extended Calibration System (v2)**: Enhanced `run_calibration_extended()` with liquidity filtering, multi-DTE bands, bucket metrics, skew fitting, recommended `vol_surface` generation, and vega-weighted MAE. Includes UI panels for calibration coverage and skew fit analysis.
- **Auto IV Calibration Pipeline**: Persists time series of auto-calculated IV multipliers in a `calibration_history` table. Includes a CLI script (`scripts/auto_calibrate_iv.py`) that uses the extended calibration engine, a runtime in-memory override store, and API endpoints/UI for management. Guardrails assess calibration realism (multiplier bounds, MAE thresholds) and data provenance is tracked.
- **Calibration Update Policy System**: Policy layer with smoothing (EWMA), thresholds, and file-based history storage. Includes configurable thresholds, decision logic for applying updates, CLI integration, and API endpoints/UI for policy management.
- **Harvester Data Quality & Reproducibility**: Includes `harvester/health.py` for schema validation and quality assessment of harvested Parquet snapshots, integrated into historical calibration and realism checks, and displayed in a UI panel.
- **Bots System**: Provides a comprehensive view of expert trading bots, market sensors, and strategy evaluations.
    - **Greg Mandolini VRP Harvester (GregBot) ENTRY_ENGINE v8.0**: A quantitative VRP strategy selector based on 11 volatility sensors and a decision waterfall, with advisory functionality and 8 evaluated strategies per underlying. Supports dynamic calibration of strategy thresholds.
    - **Greg Position Management v1.0 (POSITION_ENGINE)**: Advisory-only position management module for open Greg positions, evaluating for hedge triggers, profit targets, stop-loss, roll, and assignment rules.
    - **Delta Hedging Engine v1.0 (HEDGE_ENGINE)**: Advisory-only delta-neutral hedging module for short-vol strategies. Supports various strategy-specific hedging modes and uses perpetual futures as hedge instruments.
- **Strategy Layer**: A pluggable architecture allowing multiple trading strategies to run concurrently, with `strategy_id` for attribution.
- **Smoke Test Harness**: `scripts/smoke_greg_strategies.py` provides comprehensive testing for environmental matrix, strategy execution, position management, and hedge engine integration.
- **Sandbox Position System**: Creates isolated test positions for Greg strategies with `scripts/open_greg_sandbox_positions.py`. Sandbox positions are tagged with sandbox=True, origin="GREG_SANDBOX", and run_id for batch tracking. Main bots ignore sandbox positions by default via filtering in PositionTracker.
- **Real-Trading Mode (Phase 2)**: Execution support with strong safety gates (Global mode, master switch, per-strategy flags, environment verification, dry_run cross-check) and tiny-size guardrails (`greg_live_max_notional_usd_per_position`, `greg_live_max_notional_usd_per_underlying`). Includes API for executing suggestions and managing trading modes, along with UI modals and execute buttons.
- **Decision Logging System**: Comprehensive audit trail for Greg decisions stored in a `greg_decision_log` database table, with API for history and statistics, and a "What-If Report Script" for analysis.
- **UI Enhancements**: Priority coloring, demo vs live badges, mode badges, and view hedge links.
- **Greg Lab UI**: Dedicated dashboard tab for viewing and managing Greg strategy positions. Features mode banner (ADVICE ONLY / LIVE EXECUTION), sandbox summary, underlying/sandbox filters, positions table with badges (SANDBOX, DEMO, LIVE), PnL tracking, DTE display, suggested actions, and per-position log timelines. Includes Observer Notes stub for future LLM summaries. API endpoints: `/api/greg/positions`, `/api/greg/positions/{position_id}/logs`.
- **Telegram Code Review Agent** (`agent/` module): A Telegram bot for automated code review and Repo Q&A:
    - **Architecture**: Modular design with config, storage (SQLite), change detection (git-based with snapshot fallback), diff analysis, LLM-powered review, chat tools, and Telegram bot interface.
    - **Code Review Commands**: `/start`, `/help` (onboarding), `/status` (system health), `/review` (analyze latest changes), `/diff` (changed files summary), `/risks` (detailed issues with severity), `/next` (recommended actions).
    - **Repo Q&A Commands**: `/ask <question>` (ask about codebase), `/search <query>` (search code patterns), `/open <path>:<lines>` (view file excerpt), `/clear` (clear chat session).
    - **Natural Language Chat**: Type any question to get answers about the codebase. The LLM routes queries to appropriate tools.
    - **Chat Tools** (`agent/chat_tools.py`):
        - `search_repo`: Case-insensitive grep with context, directory prioritization (src, agent, app, etc.), exclusions (.git, node_modules, __pycache__), and limit controls.
        - `open_file`: Read file sections (max 200 lines) with line numbers, secret redaction, path traversal protection, and sensitive file blocking.
        - `list_files`: Directory exploration with optional glob pattern and file size display.
        - `get_project_map`: High-level project structure overview.
        - `get_status`, `get_latest_diff`, `run_smoke_tests`, `run_security_scans`, `tail_logs`: Supporting tools.
    - **Secret Redaction**: Automatic redaction of API keys (sk-*, sk_live_*, sk_test_*), passwords, tokens, authorization headers, PEM blocks, and long base64 strings.
    - **LLM Integration**: Uses configurable OpenAI models with automatic fallback chain. Default: `gpt-5.2-pro` (smartest) with fallback to `gpt-5.2`, `o3`, `o1`, `gpt-4.1`, `gpt-4o`. Maximum reasoning effort enabled for deep analysis.
    - **Model Configuration** (via environment variables):
        - `OPENAI_MODEL_REVIEW`: Primary model for reviews (default: `gpt-5.2-pro`)
        - `OPENAI_MODEL_FAST`: Fast model for quick summaries and chat (default: `gpt-5.2`)
        - `OPENAI_REASONING_EFFORT`: Reasoning effort level - `low`, `medium`, `high` (default: `high`)
    - **Severity Levels**: CRITICAL (must fix), HIGH (strongly recommended), MEDIUM (fix soon), LOW (nice to have), INFO (observations).
    - **Security**: User authorization via `TELEGRAM_ALLOWED_USER_IDS` secret. Bot token via `TELEGRAM_BOT_TOKEN` secret.
    - **Workflow**: "Telegram Review Bot" runs `python -m agent.telegram_bot`.

## Daily Auto-Calibration (Cron Example)

The script `scripts/daily_auto_calibrate.sh` can be run once per day to add entries to **Calibration History (Auto-Calibrate)** for BTC and ETH using harvested Parquet data.

### Safety Note
This is **safe** because auto-calibration only writes to the `calibration_history` table. It does **not** modify the currently applied IV multipliers used by the synthetic vol surface. Applied multipliers are still controlled exclusively by live calibration runs + the update policy system.

### Usage

Run manually from the repo root:
```bash
scripts/daily_auto_calibrate.sh
```

Override underlyings via environment variable:
```bash
AUTO_CALIBRATE_UNDERLYINGS="BTC,ETH" scripts/daily_auto_calibrate.sh
```

### Cron Example (Linux/Unix)

To run daily at 03:10 server time:
```cron
# Run daily auto-calibration at 03:10 server time
10 3 * * * /path/to/repo/scripts/daily_auto_calibrate.sh >> /path/to/repo/logs/auto_calibrate_cron.log 2>&1
```

With custom underlyings:
```cron
10 3 * * * AUTO_CALIBRATE_UNDERLYINGS="BTC,ETH" /path/to/repo/scripts/daily_auto_calibrate.sh >> /path/to/repo/logs/auto_calibrate_cron.log 2>&1
```

**Notes:**
- Adjust `/path/to/repo` to your actual repository path.
- Ensure the `logs/` directory exists: `mkdir -p /path/to/repo/logs`
- Results appear in the UI under **Calibration → Calibration History (Auto-Calibrate)**.

## External Dependencies
- **Deribit API**: Used for real-time market data (testnet) and historical data (mainnet public API for backtesting and data harvesting).
- **OpenAI**: Integrated for LLM-powered decision-making and generating insights.
- **PostgreSQL**: Used for persistent storage of backtest runs.