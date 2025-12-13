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
The web dashboard offers a user-friendly interface with sections for "Live Agent" status, unified "Backtesting" tab, "Calibration", "System Health", "Chat" interface, "Bots" tab for expert trading bot analysis, "Greg Lab" for position management, and an "AI Steward" panel for project insights.

### Technical Implementations
- **Configuration**: Pydantic settings manage application configuration.
- **API Wrapper**: An `httpx`-based wrapper for the Deribit testnet API.
- **State Management**: Aggregates market data and manages thread-safe status updates.
- **Structured Logging**: Uses JSONL for logging decisions and actions.
- **Health Guard System**: Runtime health guard with severity-based decision making (TRANSIENT, DEGRADED, FATAL) for intelligent recovery, including startup checks and runtime re-checks.
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
- **Telegram Code Review Agent**: A Telegram bot for automated code review and Repo Q&A with modular design. It offers commands for `/review`, `/diff`, `/risks`, `/ask`, `/search`, `/open`, and natural language chat, utilizing LLM integration with automatic model fallback and secret redaction.

## External Dependencies
- **Deribit API**: Used for real-time market data (testnet) and historical data.
- **OpenAI**: Integrated for LLM-powered decision-making and generating insights.
- **PostgreSQL**: Used for persistent storage of backtest runs and decision logs.