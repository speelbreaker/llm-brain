# Options Trading Agent - Deribit Testnet

## Overview
This project is a modular Python framework for automated BTC/ETH covered call trading on the Deribit testnet. It serves as a research and experimentation system, supporting both rule-based and LLM-powered decision-making, with a focus on exploration-based learning. The system aims to provide a robust platform for testing trading strategies, generating training data, and analyzing performance through a comprehensive web dashboard and backtesting suite.

## User Preferences
- Python 3.11
- Type hints everywhere
- Pydantic for configs and models
- httpx for HTTP
- Clarity over cleverness

## System Architecture

### Core Design
The agent is built with a clear separation of concerns, featuring modules for configuration, data modeling, API interaction, market context generation, risk management, policy decisions (rule-based and LLM), execution, and logging. It supports distinct "research" and "production" modes, allowing for broader exploration on the testnet and stricter, more conservative parameters for potential mainnet deployment. A FastAPI web application provides a real-time dashboard for monitoring, interaction, and backtesting.

### Key Features
- **Decision Policies**: Supports rule-based strategies with a scoring function and epsilon-greedy exploration, and an LLM-powered decision mode (using OpenAI) that proposes actions validated by a risk engine.
- **Market Context**: Integrates `MarketContext` to provide regime detection (bull/sideways/bear), returns, and realized volatility for informed decision-making.
- **Risk Management**: A `risk_engine` module performs pre-trade validation, checking margin, delta, and exposure limits. Environment-aware training mode bypasses risk checks only when training on testnet (training_mode=True AND deribit_env="testnet"); mainnet always enforces full risk controls.
- **Backtesting Framework**: Includes a `CoveredCallSimulator` for historical analysis, supporting various exit styles (hold-to-expiry, take-profit and roll), a scoring function for candidate options, multi-leg chain visualization, and training data generation.
- **Synthetic Pricing Mode**: Black-Scholes based option pricing for self-consistent historical backtests without requiring live Deribit option data. Uses realized volatility computed from spot history to avoid look-ahead bias. Generates synthetic strikes around spot price with synthetic expiries (target_dte days from decision time) to ensure candidates are always available regardless of historical date range.
- **TradingView-style Metrics**: Enhanced backtest summary with net profit, max drawdown, profit factor, Sharpe ratio, Sortino ratio, win rate, and gross profit/loss calculations.
- **Equity Curve Visualization**: Interactive chart comparing Strategy returns vs HODL benchmark with dual-line display.
- **Training Mode**: Allows for multi-profile data collection (conservative, moderate, aggressive strategies) to generate diverse datasets for ML/RL. In training mode on testnet:
  - The policy layer allows multiple covered calls per underlying (up to `max_calls_per_underlying_training`, default 6), excluding already-open symbols to build delta ladders
  - Both rule-based and LLM policies are aware of training mode and behave more aggressively
  - The rb_v1_explore policy does NOT block new positions due to existing positions
  - Training profiles use wide DTE (1-21 days) and delta ranges to maximize candidate matching
  - A fallback mechanism picks the best premium candidate if no profile matches, ensuring trades are executed when candidates exist
  - DO_NOTHING only occurs when truly no valid candidates are available or max positions reached
  - **Per-Expiry Limits**: Configurable via `training_max_calls_per_expiry` (default 3), tracks positions by actual expiry date (not DTE) to prevent over-concentration on a single expiry
  - **Profile Modes**: Configurable via `TRAINING_PROFILE_MODE`:
    - `single`: Traditional one-action-per-profile-per-iteration (conservative)
    - `ladder`: Multi-candidate per profile selection with per-expiry limits. Each profile has `max_legs` (default 2) and `enabled` flag. Candidates are sorted by premium and selected respecting per-expiry caps.
  - **Diagnostics**: Each training action includes a `diagnostics` field with delta, dte, premium_usd, and ivrv for analysis. Agent loop logs display these metrics for visibility.
- **Training Data Export**: Captures (state, action, reward, strategy) tuples and exports to CSV/JSONL. Supports two export formats:
  - **Chain-level**: One row per trade decision (`training_dataset_*.csv`) with the chosen candidate, outcome, and strategy profile used
  - **Candidate-level**: One row per candidate per decision step (`training_candidates_*.csv`) with binary labels for chosen/not-chosen, strategy profile, including SKIP examples for all rejected candidates and no-trade decisions. Useful for training LLM policies that learn decision boundaries.
  Note: Historical backtests can use synthetic pricing for self-consistent option prices or live option data from Deribit's public API.
- **LLM Fine-Tuning Data**: The `scripts/build_llm_training_from_candidates.py` script transforms candidate CSVs into chat-style JSONL corpora for LLM fine-tuning. Produces two flavors:
  - **Per-candidate classification** (`*_per_candidate.jsonl`): One record per candidate with task SELL_CALL or SKIP
  - **Per-decision ranking** (`*_per_decision_ranking.jsonl`): One record per decision_time with task to pick best candidate index or NO_TRADE
  Usage: `python scripts/build_llm_training_from_candidates.py --input <csv> --exit-style <style> --underlying <asset>`
- **Web Dashboard**: A FastAPI application offers a "Live Agent" view with real-time status and recent decisions, a "Backtesting Lab" with TradingView-style summary panel, equity curve charts, a "Backtest Runs" panel for viewing/downloading historical backtest results, a "Calibration" tab for comparing synthetic BS prices vs live Deribit marks, and a "Chat" interface for natural language interaction with the agent.
- **Persistent Backtest Runs (PostgreSQL)**: Backtest results are stored in PostgreSQL using SQLAlchemy ORM:
  - **Database Tables**: `backtest_runs` (one row per run), `backtest_metrics` (per exit style), `backtest_chains` (trade chains)
  - Status tracking: queued, running, finished, failed
  - API endpoints: `GET /api/backtests` (list with optional filters), `GET /api/backtests/{run_id}` (view with metrics and chains), `GET /api/backtests/{run_id}/download` (download JSON), `DELETE /api/backtests/{run_id}` (delete)
  - UI panel showing all runs with Net PnL %, Max DD, Sharpe, and View/Download/Delete actions
  - **Data Source Types**: `synthetic` (Black-Scholes), `live_deribit` (captured harvester data), `real_scraper` (external normalized data)
- **LIVE_DERIBIT Comparison Script**: `scripts/compare_synthetic_vs_live.py` runs side-by-side backtests comparing SYNTHETIC vs LIVE_DERIBIT data sources over the same period, saving both runs to PostgreSQL and printing a detailed metrics comparison table.
- **Backtest Diff Report**: `scripts/diff_backtest_runs.py` generates a human-readable diff report comparing any two existing backtest runs from PostgreSQL. Shows A vs B vs Diff for all key metrics (net profit, max drawdown, Sharpe, Sortino, win rate, profit factor, etc.). Usage: `python scripts/diff_backtest_runs.py --run-a <run_id_A> --run-b <run_id_B> [--exit-style <style>]`
- **Reusable Exam Builder**: `src/data/live_deribit_exam.py` provides `build_live_deribit_exam_dataset()` function that can be called programmatically from the backtester or CLI scripts.
- **State-Aware Chat Assistant**: The Chat tab is a multi-turn, state-aware assistant that:
  - Knows current trading state (positions, unrealized PnL, training vs live mode, spot prices)
  - Maintains conversation history across messages (up to 20 turns)
  - Can explain what the bot is doing right now based on live data
  - Answers questions about trading rules (when it opens, rolls, or closes positions)
  - Provides architecture and safety information from project docs (ARCHITECTURE_OVERVIEW.md, HEALTHCHECK.md, ROADMAP.md)
  - Uses thread-safe context assembly to avoid race conditions with the background agent
- **Position Reconciliation**: At each agent loop iteration, the system compares local position tracker against Deribit exchange positions. Configurable via `POSITION_RECONCILE_ACTION`:
  - `halt`: Stops trading when positions diverge (safe mode)
  - `auto_heal`: Automatically rebuilds local tracker from exchange data
  - Manual recovery script: `scripts/reconcile_positions_once.py --heal`
- **Calibration vs Deribit**: Compares synthetic Black-Scholes option prices against live Deribit mark prices. Uses RV-based IV model matching the synthetic backtester (sigma = RV(7d) * iv_multiplier * skew_factor), fetches real-time option chains from Deribit public API, and reports Mean Absolute Error (MAE) and bias. Displays computed realized volatility and allows tuning via iv_multiplier slider. Handles both inverse (BTC/ETH-settled) and linear (USDC-settled) contracts correctly.
- **Synthetic Skew Engine**: Derives IV skew factors from live Deribit smile data. Computes skew anchors at deltas [0.15, 0.25, 0.35, 0.50] by comparing IV to ATM IV, then interpolates linearly for any delta. Skew anchors are cached per (underlying, option_type) to minimize API calls. Skew is applied to both calibration and synthetic backtester pricing.
- **Structured Logging**: Uses JSONL for structured logging of all decisions and actions, facilitating future analysis and ML/RL training.
- **Position Persistence**: Bot-managed positions are automatically saved to `data/positions.json` and restored on restart, ensuring position tracking survives workflow restarts.

### Strategy Layer
- **Pluggable Architecture**: The agent uses a strategy layer (`src/strategies/`) that allows multiple trading strategies to run side by side while sharing market data and risk controls.
- **Core Components**:
  - `StrategyConfig`: Dataclass holding all configuration for a strategy (underlyings, delta range, DTE range, mode)
  - `Strategy`: Base interface with `propose_actions(state)` method that returns proposed trades
  - `StrategyRegistry`: Container holding all registered strategies with methods to get active strategies
  - `CoveredCallStrategy`: The current (and only) strategy, wrapping existing rule-based, LLM, and training logic
- **Future Strategies**: Designed to support WheelStrategy, CrashHedgeStrategy, SpreadStrategy, etc. Each new strategy implements the Strategy interface and is registered alongside existing strategies.

### Technical Implementations
- **Configuration**: Pydantic settings are used for managing application configuration and switching between research/production modes.
- **API Wrapper**: `deribit_client.py` provides an `httpx`-based wrapper for the Deribit testnet API. Extends `DeribitBaseClient` from `src/deribit/base_client.py` which contains shared HTTP/JSON-RPC logic used by both trading and backtest clients.
- **State Management**: `state_builder.py` aggregates market data and `status_store.py` manages thread-safe status updates.
- **Unified State Core**: `state_core.py` provides shared state-building logic for both live and backtest agents. Contains `RawMarketSnapshot`, `RawOption`, `RawPortfolio` dataclasses and `build_agent_state_from_raw()` function. Only data sources differ between live (Deribit API) and backtest (historical/synthetic); construction rules are unified.
- **Order Execution**: `execution.py` handles order translation, supporting dry-run simulations.
- **UI/UX**: The web dashboard provides a user-friendly interface for monitoring and interaction, with specific tabs for live agent status, backtesting, and chat.

### Real Scraper Data Source
- **External Dataset Support**: Imports Real Scraper datasets (e.g., Kaggle Deribit options data) for backtesting with historical data.
- **Import Script**: `scripts/import_real_scraper_deribit.py` converts external CSV/Parquet files to canonical schema.
- **Data Loader**: `src/backtest/real_scraper_data_source.py` implements `MarketDataSource` protocol for Real Scraper data.
- **Storage Format**: `data/real_scraper/<UNDERLYING>/<YYYY-MM-DD>/<UNDERLYING>_<YYYY-MM-DD>.parquet`
- **Test Script**: `scripts/run_backtest_real_scraper_example.py` runs backtests with REAL_SCRAPER data source.
- **Comparison**: Enables comparing REAL_SCRAPER vs SYNTHETIC backtests over the same date ranges.

### Data Harvester
- **Standalone Process**: `scripts/data_harvester.py` runs independently from the trading bot, continuously collecting real Deribit options data.
- **Multi-Asset Support**: Fetches option chains from Deribit mainnet public API:
  - **Inverse options** (BTC, ETH): Settled in the underlying asset
  - **Linear USDC options** (SOL_USDC, XRP_USDC, AVAX_USDC, TRX_USDC, PAXG_USDC, BTC_USDC, ETH_USDC): Settled in USDC
- **Parquet Storage**: Saves snapshots to `data/live_deribit/<ASSET>/<YYYY>/<MM>/<DD>/<ASSET>_<YYYY-MM-DD_HHMM>.parquet`.
- **Data Fields**: Includes instrument info, pricing (mark, bid, ask), volume, open interest, IV, and Greeks (delta, gamma, theta, vega).
- **Configurable Interval**: Default 15-minute polling, adjustable via `HARVESTER_INTERVAL_MINUTES` env var.
- **Usage**: Run `python -m scripts.data_harvester` in a separate process/terminal.
- **Documentation**: See `docs/DATA_HARVESTER.md` for full details.

## External Dependencies
- **Deribit API**: Used for fetching real-time market data (testnet) and historical data (mainnet public API for backtesting and data harvesting).
- **OpenAI**: Integrated via Replit AI Integrations for the LLM-powered decision mode and for generating insights from backtest results.