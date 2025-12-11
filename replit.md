# Options Trading Agent - Deribit Testnet

## Overview
This project is a modular Python framework for automated BTC/ETH covered call trading on the Deribit testnet. It functions as a research and experimentation system for testing trading strategies, generating training data, and analyzing performance via a web dashboard and backtesting suite. The system supports both rule-based and LLM-powered decision-making, with a focus on exploration-based learning and ambitions for eventual mainnet deployment.

## User Preferences
- Python 3.11
- Type hints everywhere
- Pydantic for configs and models
- httpx for HTTP
- Clarity over cleverness

## System Architecture
The agent features a clear separation of concerns, with modules for configuration, data modeling, API interaction, market context generation, risk management, policy decisions (rule-based and LLM), execution, and logging. It supports "research" and "production" modes, with a FastAPI web application providing a real-time dashboard for monitoring, interaction, and backtesting.

### UI/UX Decisions
The web dashboard provides a user-friendly interface with sections for "Live Agent" status, "Backtesting Lab", "Backtest Runs", "Calibration", "System Health", "Chat" interface, "Bots" tab for expert trading bot analysis, and an "AI Steward" panel for project insights.

### Technical Implementations
- **Configuration**: Pydantic settings manage application configuration.
- **API Wrapper**: `deribit_client.py` provides an `httpx`-based wrapper for the Deribit testnet API.
- **State Management**: `state_builder.py` aggregates market data; `status_store.py` manages thread-safe status updates. `state_core.py` provides unified state-building logic.
- **Order Execution**: `execution.py` handles order translation and dry-run simulations.
- **Structured Logging**: Uses JSONL for logging decisions and actions.
- **Position Persistence**: Bot-managed positions are saved to `data/positions.json` and restored on restart.
- **Agent Healthcheck Module**: Self-contained healthcheck system for critical pipeline validation.

### Feature Specifications
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
- **Synthetic Universe v2 (Greg-sensor cluster regimes)**: Incorporates `RegimeParams`, KMeans clustering, AR(1) IV dynamics, and Greg-sensor clusters for realistic IV evolution and regime modeling.
- **Extended Calibration System (v2)**: Enhanced `run_calibration_extended()` with liquidity filtering, multi-DTE bands, bucket metrics, skew fitting, recommended `vol_surface` generation, and vega-weighted MAE. Includes UI panels for calibration coverage and skew fit analysis.
- **Auto IV Calibration Pipeline**: Persists time series of auto-calculated IV multipliers in a `calibration_history` table. Includes a CLI script (`scripts/auto_calibrate_iv.py`) that uses the extended calibration engine (`run_historical_calibration_from_harvest`), a runtime in-memory override store, and API endpoints/UI for management.
    - **Guardrails & Realism Assessment**: The `assess_calibration_realism()` function validates calibrations with thresholds for multiplier bounds (0.7-1.6), max MAE (400%), and max vega-weighted MAE (200%). Returns status as OK/DEGRADED/FAILED with reasons.
    - **Enhanced History Table**: Calibration history now includes `status`, `reason`, `vega_weighted_mae_pct`, `bias_pct`, and `source` fields. UI displays status badges (green/orange/red) and failure reasons.
- **Calibration Update Policy System**: Policy layer with smoothing (EWMA), thresholds (min_delta, min_sample_size, min_vega_sum), and file-based history storage. Includes configurable thresholds, decision logic for applying updates, CLI integration, and API endpoints/UI for policy management.
- **Harvester Data Quality & Reproducibility**: Includes `harvester/health.py` for schema validation and quality assessment of harvested Parquet snapshots. Provides `DataQualityStatus` (OK/DEGRADED/FAILED), integrates into historical calibration and realism checks, and displays a "Data Health & Reproducibility" UI panel.
- **Bots System**: Provides a comprehensive view of expert trading bots, market sensors, and strategy evaluations.
    - **Greg Mandolini VRP Harvester (GregBot) v6.0 "Diamond-Grade"**: A quantitative VRP strategy selector based on 11 volatility sensors and a decision tree, with advisory functionality and 8 evaluated strategies per underlying.
    - **Greg v1 Calibration & Tests**: Decision tree thresholds are dynamically loaded from JSON. Includes invariant and scenario tests, an API endpoint for calibration spec, and a UI panel.
- **Strategy Layer**: A pluggable architecture allowing multiple trading strategies to run concurrently, with `strategy_id` for attribution.

## External Dependencies
- **Deribit API**: Used for real-time market data (testnet) and historical data (mainnet public API for backtesting and data harvesting).
- **OpenAI**: Integrated for LLM-powered decision-making and generating insights.
- **PostgreSQL**: Used for persistent storage of backtest runs.