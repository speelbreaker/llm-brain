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