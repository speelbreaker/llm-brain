# Options Trading Agent - Deribit Testnet

## Overview
This project is a modular Python framework for automated BTC/ETH covered call trading on the Deribit testnet. It serves as a research and experimentation system for testing trading strategies, generating training data, and analyzing performance through a web dashboard and backtesting suite. The system supports both rule-based and LLM-powered decision-making, with a focus on exploration-based learning and ambitions for mainnet deployment.

## User Preferences
- Python 3.11
- Type hints everywhere
- Pydantic for configs and models
- httpx for HTTP
- Clarity over cleverness

## System Architecture
The agent features a clear separation of concerns, with modules for configuration, data modeling, API interaction, market context generation, risk management, policy decisions (rule-based and LLM), execution, and logging. It supports "research" and "production" modes, with a FastAPI web application providing a real-time dashboard for monitoring, interaction, and backtesting.

### UI/UX Decisions
The web dashboard provides a user-friendly interface with sections for "Live Agent" status, "Backtesting Lab" with interactive charts, "Backtest Runs" management, "Calibration" for price comparison, "System Health" with runtime controls, and a "Chat" interface for natural language interaction. It also includes "Bots" tab for expert trading bot analysis and an "AI Steward" panel for project insights.

### Technical Implementations
- **Configuration**: Pydantic settings manage application configuration.
- **API Wrapper**: `deribit_client.py` provides an `httpx`-based wrapper for the Deribit testnet API.
- **State Management**: `state_builder.py` aggregates market data; `status_store.py` manages thread-safe status updates. `state_core.py` provides unified state-building logic for live and backtest agents.
- **Order Execution**: `execution.py` handles order translation and dry-run simulations.
- **Structured Logging**: Uses JSONL for logging decisions and actions.
- **Position Persistence**: Bot-managed positions are saved to `data/positions.json` and restored on restart.
- **Agent Healthcheck Module**: Self-contained healthcheck system that exercises the critical pipeline (config → Deribit → state builder).

### Feature Specifications
- **Decision Policies**: Supports rule-based strategies with scoring and epsilon-greedy exploration, and an LLM-powered decision mode validated by a risk engine. Decision modes include `rule_only`, `llm_only`, and `hybrid_shadow`.
- **LLM Validation**: `validate_llm_decision()` function provides comprehensive validation including symbol verification, action type checking, position verification, and size clamping.
- **Market Context**: Integrates `MarketContext` for regime detection, returns, and realized volatility.
- **Risk Management**: A `risk_engine` performs pre-trade validation, checking margin, delta, exposure limits, liquidity guards, and hard safety rails (global kill switch, daily drawdown guard).
- **Backtesting Framework**: Includes `CoveredCallSimulator` for historical analysis, supporting various exit styles and training data generation. Features a synthetic pricing mode and provides TradingView-style metrics and equity curve visualization.
- **Selector Analysis**: Includes "Selector Frequency Scan" for analyzing rule pass rates and "Selector Heatmap" for 2D parameter sweep visualization.
- **Environment Heatmap**: Provides a selector-free 2D occupancy heatmap to show market time spent across metric pairs. A CLI script (`scripts/run_greg_environment_heatmaps.py`) and UI panel automate environment sweet spot analysis.
- **Training Mode**: Allows multi-profile data collection for ML/RL, with aggressive policies and configurable per-expiry limits. Exports training data in chain-level and candidate-level formats.
- **LLM Fine-Tuning Data**: Scripts transform candidate CSVs into chat-style JSONL corpora for LLM fine-tuning.
- **Runtime Controls**: System Health tab offers interactive controls for adjusting safety and operational settings (e.g., Global Kill Switch, Daily Drawdown Limit, Decision Mode, Dry Run Mode, Liquidity Guards, Position Reconciliation Config).
- **LLM & Strategy Tuning**: System Health tab includes a panel for adjusting LLM and strategy configuration at runtime (e.g., LLM Enabled toggle, Explore Probability, Training Profile, Strategy Thresholds, Risk Limits).
- **Persistent Backtest Runs**: Stores backtest results in PostgreSQL using SQLAlchemy ORM.
- **Comparison & Health Check Scripts**: Tools for comparing synthetic vs. live backtests and performing strategy health checks.
- **AI Steward (Project Brain)**: A project planning and QA helper (`src/system_steward.py`) that summarizes project state and suggests next tasks using an LLM, exposed via API endpoints and a UI panel.
- **State-Aware Chat Assistant**: A multi-turn assistant that understands current trading state, answers questions, and provides project information.
- **Position Reconciliation**: Compares local position tracker against exchange positions, with configurable actions.
- **Calibration vs Deribit**: Compares synthetic Black-Scholes prices against live Deribit marks.
- **Bots System**: Provides a comprehensive view of expert trading bots, market sensors, and strategy evaluations, including debug mode for sensor computations.
    - **Greg Mandolini VRP Harvester (GregBot) v6.0 "Diamond-Grade"**: A quantitative VRP strategy selector based on 11 volatility sensors and a decision tree. Currently advisory (read-only) with 8 evaluated strategies per underlying. Sensor mapping and calibration variables are defined.
- **Strategy Layer**: A pluggable architecture allowing multiple trading strategies to run concurrently (`src/strategies/`). Strategies are built from settings via `build_default_registry()` and decisions include `strategy_id` for attribution.

## External Dependencies
- **Deribit API**: Used for real-time market data (testnet) and historical data (mainnet public API for backtesting and data harvesting).
- **OpenAI**: Integrated for LLM-powered decision-making and generating insights.
- **PostgreSQL**: Used for persistent storage of backtest runs.