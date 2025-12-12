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
The web dashboard offers a user-friendly interface with sections for "Live Agent" status, "Backtesting Lab", "Backtest Runs", "Calibration", "System Health", "Chat" interface, "Bots" tab for expert trading bot analysis, and an "AI Steward" panel for project insights.

### Technical Implementations
- **Configuration**: Pydantic settings manage application configuration.
- **API Wrapper**: An `httpx`-based wrapper (`deribit_client.py`) for the Deribit testnet API.
- **State Management**: Aggregates market data, manages thread-safe status updates, and unifies state-building logic.
- **Order Execution**: Handles order translation and dry-run simulations.
- **Structured Logging**: Uses JSONL for logging decisions and actions.
- **Position Persistence**: Bot-managed positions are saved to `data/positions.json` and restored on restart.
- **Agent Healthcheck Module**: Self-contained system for critical pipeline validation with expanded config validation (risk settings, LLM config), auto-trigger on tab open (60s throttle), status badge (OK/WARN/FAIL), and LLM diagnostic gating.
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
- **Real-Trading Mode (Phase 2)**: Execution support with strong safety gates (Global mode, master switch, per-strategy flags, environment verification, dry_run cross-check) and tiny-size guardrails (`greg_live_max_notional_usd_per_position`, `greg_live_max_notional_usd_per_underlying`). Includes API for executing suggestions and managing trading modes, along with UI modals and execute buttons.
- **Decision Logging System**: Comprehensive audit trail for Greg decisions stored in a `greg_decision_log` database table, with API for history and statistics, and a "What-If Report Script" for analysis.
- **UI Enhancements**: Priority coloring, demo vs live badges, mode badges, and view hedge links.

## External Dependencies
- **Deribit API**: Used for real-time market data (testnet) and historical data (mainnet public API for backtesting and data harvesting).
- **OpenAI**: Integrated for LLM-powered decision-making and generating insights.
- **PostgreSQL**: Used for persistent storage of backtest runs.