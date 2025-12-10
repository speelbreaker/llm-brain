# Options Trading Agent - Deribit Testnet

## Overview
This project is a modular Python framework for automated BTC/ETH covered call trading on the Deribit testnet. It functions as a research and experimentation system, supporting both rule-based and LLM-powered decision-making with a focus on exploration-based learning. The system provides a robust platform for testing trading strategies, generating training data, and analyzing performance through a web dashboard and backtesting suite, with ambitions to explore mainnet deployment.

## User Preferences
- Python 3.11
- Type hints everywhere
- Pydantic for configs and models
- httpx for HTTP
- Clarity over cleverness

## System Architecture
The agent features a clear separation of concerns, with modules for configuration, data modeling, API interaction, market context generation, risk management, policy decisions (rule-based and LLM), execution, and logging. It supports "research" and "production" modes. A FastAPI web application provides a real-time dashboard for monitoring, interaction, and backtesting.

### Key Features
- **Decision Policies**: Supports rule-based strategies with scoring and epsilon-greedy exploration, and an LLM-powered decision mode validated by a risk engine. Supports three decision modes: `rule_only` (safe baseline), `llm_only` (validated LLM with fallback), and `hybrid_shadow` (rules execute, LLM logs for comparison).
- **LLM Validation**: The `validate_llm_decision()` function provides comprehensive validation including symbol verification against candidates, action type checking, position verification for closes/rolls, and size clamping to configured limits.
- **Market Context**: Integrates `MarketContext` for regime detection, returns, and realized volatility.
- **Risk Management**: A `risk_engine` performs pre-trade validation, checking margin, delta, exposure limits, liquidity guards (bid/ask spread, open interest), and hard safety rails (global kill switch, daily drawdown guard).
- **Backtesting Framework**: Includes `CoveredCallSimulator` for historical analysis, supporting various exit styles and training data generation. Features a synthetic pricing mode using Black-Scholes and realized volatility for self-consistent historical backtests. Provides TradingView-style metrics and equity curve visualization.
- **Selector Frequency Scan**: Analyzes how often a selector's rules (e.g., GregBot Phase 1) would allow trading in a synthetic universe. Supports threshold overrides for backtesting parameter sensitivity without affecting live trading. Uses synthetic sensor generation for historical analysis across arbitrary time ranges.
- **Selector Heatmap**: 2D parameter sweep visualization showing trade frequency across two threshold metrics. Generates a color-coded heatmap (green = higher pass%) to help identify optimal parameter combinations. Configurable grid axes for metrics like VRP min, ADX max, IV Rank min, etc.
- **Training Mode**: Allows for multi-profile data collection to generate diverse datasets for ML/RL, with aggressive policies and configurable per-expiry limits. Exports training data in chain-level and candidate-level formats.
- **LLM Fine-Tuning Data**: Scripts transform candidate CSVs into chat-style JSONL corpora for LLM fine-tuning, supporting per-candidate classification and per-decision ranking tasks.
- **Web Dashboard**: A FastAPI application offering "Live Agent" status, "Backtesting Lab" with interactive charts, "Backtest Runs" management, "Calibration" for price comparison, "System Health" with runtime controls, and a "Chat" interface for natural language interaction.
- **Intraday Scraper Status**: The Backtesting Lab includes a status widget showing the Deribit data harvester's health: rows collected, days covered, data size, timestamps, and whether the scraper is actively running (RUNNING vs STALE/STOPPED).
- **Runtime Controls**: The System Health tab includes interactive controls for adjusting safety and operational settings without editing environment variables. Controls include: Global Kill Switch toggle, Daily Drawdown Limit input, Decision Mode selector, Dry Run Mode toggle, Liquidity Guards (Max Spread %, Min Open Interest), and Position Reconciliation Config (action, startup behavior, loop behavior, tolerance). Changes apply immediately but do not persist across restarts.
- **LLM & Strategy Tuning**: The System Health tab includes an "LLM & Strategy Tuning" panel for adjusting LLM and strategy configuration at runtime:
  - **LLM Enabled toggle**: Turn LLM-powered decision making on/off
  - **Explore Probability slider**: Adjust epsilon-greedy exploration percentage (0-100%)
  - **Training Profile dropdown**: Switch between "Single" and "Ladder" training modes
  - **Strategy Thresholds**: Edit Min IV/RV, Delta Min/Max, DTE Min/Max (effective values for current mode)
  - **Risk Limits**: Edit Max Margin Used %, Max Net Delta abs, Max Spread %, Min Open Interest
  All changes are runtime-only and will reset on restart.
- **Persistent Backtest Runs**: Stores backtest results in PostgreSQL using SQLAlchemy ORM for status tracking, viewing, and downloading. Supports `synthetic`, `live_deribit`, and `real_scraper` data sources.
- **Comparison & Health Check Scripts**: Tools for comparing synthetic vs. live backtests, generating diff reports, and performing strategy health checks.
- **Agent Healthcheck Module**: Self-contained healthcheck system (`src/healthcheck.py`) that exercises the critical pipeline (config → Deribit → state builder). Includes CLI script (`scripts/agent_healthcheck.py`) and automatic startup integration with PASS/WARN/FAIL status display.
- **State-Aware Chat Assistant**: A multi-turn assistant that understands current trading state, answers questions, and provides project information.
- **Position Reconciliation**: Compares local position tracker against exchange positions, with configurable actions (`halt`, `auto_heal`) and a manual recovery script.
- **Calibration vs Deribit**: Compares synthetic Black-Scholes prices against live Deribit marks, using an RV-based IV model and deriving skew from live data.
- **Structured Logging**: Uses JSONL for logging decisions and actions.
- **Position Persistence**: Bot-managed positions are saved to `data/positions.json` and restored on restart.

### Bots System
The "Bots" tab provides a comprehensive view of expert trading bots, their market sensors, and strategy evaluations.

**Features:**
- **Live Market Sensors**: Per-underlying signal table showing computed sensors for BTC and ETH
- **Strategy Matches**: Aggregated view of strategies that currently pass market filters
- **Expert Bots Panel**: Per-bot breakdown showing all strategies with pass/blocked/no_data status

**API Endpoints:**
- `GET /api/bots/market_sensors` - Returns sensor values for all underlyings
- `GET /api/bots/market_sensors?debug=1` - Returns sensors with debug inputs (formulas, raw inputs, intermediate calculations)
- `GET /api/bots/strategies` - Returns StrategyEvaluation objects for all bots

**Debug Mode:**
- "Show debug inputs" checkbox exposes raw computation values for each sensor
- Displays formula, input values, and intermediate calculations (e.g., VRP = IV - RV with all numeric values shown)
- Useful for verifying sensor computations and diagnosing unexpected values

**Types (`src/bots/types.py`):**
- `StrategyCriterion` - Single metric check with value, min/max thresholds, and pass/fail status
- `StrategyEvaluation` - Complete strategy evaluation with bot_name, status, summary, and criteria list

### Greg Mandolini VRP Harvester (GregBot)
A quantitative volatility risk premium (VRP) strategy selector based on market sensors.

**Phase 1 (Current - Read-Only):**
- Computes 8 volatility sensors for each underlying
- Runs 9-rule decision tree to select optimal strategy
- Displays recommendation with pass/blocked/no_data status per strategy
- No orders placed - purely advisory

**Sensor Mapping (computed from OHLC + real options chain data):**
- `vrp_30d`: IV - RV spread (computed from ATM options mark_iv)
- `chop_factor_7d`: RV_7d / IV_30d ratio
- `iv_rank_6m`: 6-month IV percentile rank (computed from historical IV range)
- `term_structure_spread`: IV_7d - IV_30d (computed from term structure options)
- `skew_25d`: 25-delta put-call skew (computed from OTM put/call mark_iv)
- `adx_14d`: 14-day Average Directional Index (computed from OHLC)
- `rsi_14d`: 14-day Relative Strength Index (computed from OHLC)
- `price_vs_ma200`: % distance from 200-day moving average (computed from OHLC)

**Strategies Evaluated (8 per underlying):**
1. STRATEGY_A_STRADDLE - ATM Straddle (High VRP, Calm)
2. STRATEGY_A_STRANGLE - OTM Strangle (Moderate VRP, Ranging)
3. STRATEGY_B_CALENDAR - Calendar Spread (Term Structure)
4. STRATEGY_C_SHORT_PUT - Short Put (Bullish)
5. STRATEGY_D_IRON_BUTTERFLY - Iron Butterfly (Low Vol, Pinned)
6. STRATEGY_F_BULL_PUT_SPREAD - Bull Put Spread (Bullish Bias)
7. STRATEGY_F_BEAR_CALL_SPREAD - Bear Call Spread (Bearish Bias)
8. NO_TRADE - Default fallback (displayed as orange caution badge)

**Spec Location:** `docs/greg_mandolini/GREG_SELECTOR_RULES_FINAL.json`

### Strategy Layer
A pluggable architecture (`src/strategies/`) allows multiple trading strategies to run concurrently, sharing market data and risk controls.

**Key Types:**
- `Strategy` – Base class with `strategy_id` property and `propose_actions(state: AgentState)` method
- `StrategyConfig` – Configuration dataclass for strategy parameters
- `CandidateAction` – Typed action with scoring metadata (delta, DTE, premium, IVRV score)
- `StrategyDecision` – Final decision for execution with strategy attribution
- `StrategyRegistry` – Manages active strategies, built from settings via `build_default_registry()`

**Current Implementation:**
- `CoveredCallStrategy` – Implements covered call logic (rule-based/LLM/training modes)
- All decisions include `strategy_id` for multi-strategy logging and attribution

**How to add a new strategy:**
1. Create `src/strategies/my_strategy.py` extending `Strategy`
2. Implement `propose_actions(state: AgentState) -> List[dict]`
3. Register in `build_default_registry()` in `src/strategies/registry.py`

### Technical Implementations
- **Configuration**: Pydantic settings manage application configuration.
- **API Wrapper**: `deribit_client.py` provides an `httpx`-based wrapper for the Deribit testnet API.
- **State Management**: `state_builder.py` aggregates market data; `status_store.py` manages thread-safe status updates. `state_core.py` provides unified state-building logic for live and backtest agents.
- **Order Execution**: `execution.py` handles order translation and dry-run simulations.
- **UI/UX**: The web dashboard provides a user-friendly interface.

### Real Scraper Data Source
Supports external Real Scraper datasets for backtesting, with import scripts and a data loader for canonical schema.

### Data Harvester
A standalone process (`scripts/data_harvester.py`) continuously collects real Deribit options data (Inverse and Linear USDC options) from the mainnet public API, saving snapshots to Parquet files at configurable intervals.

## External Dependencies
- **Deribit API**: Used for real-time market data (testnet), historical data (mainnet public API for backtesting and data harvesting).
- **OpenAI**: Integrated for LLM-powered decision-making and generating insights.

## Related Documents
- **ARCHITECTURE_OVERVIEW.md** – Technical details on how the system works
- **HEALTHCHECK.md** – Smoke tests, unit tests, and technical debt status
- **ROADMAP.md** – Project roadmap, phases, and next steps
- **ROADMAP_BACKLOG.md** – Unprioritized ideas and future work items