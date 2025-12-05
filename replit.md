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
- **Training Mode**: Allows for multi-profile data collection (conservative, moderate, aggressive strategies) to generate diverse datasets for ML/RL. In training mode on testnet, the policy layer allows multiple covered calls per underlying (up to `max_calls_per_underlying_training`), excluding already-open symbols to build delta ladders. Both rule-based and LLM policies are aware of training mode and behave more aggressively.
- **Training Data Export**: Captures (state, action, reward) tuples and exports to CSV/JSONL. Supports two export formats:
  - **Chain-level**: One row per trade decision (`training_dataset_*.csv`) with the chosen candidate and outcome
  - **Candidate-level**: One row per candidate per decision step (`training_candidates_*.csv`) with binary labels for chosen/not-chosen, including SKIP examples for all rejected candidates and no-trade decisions. Useful for training LLM policies that learn decision boundaries.
  Note: Historical backtests can use synthetic pricing for self-consistent option prices or live option data from Deribit's public API.
- **Web Dashboard**: A FastAPI application offers a "Live Agent" view with real-time status and recent decisions, a "Backtesting Lab" with TradingView-style summary panel, equity curve charts, a "Calibration" tab for comparing synthetic BS prices vs live Deribit marks, and a "Chat" interface for natural language interaction with the agent.
- **Calibration vs Deribit**: Compares synthetic Black-Scholes option prices against live Deribit mark prices. Uses RV-based IV model matching the synthetic backtester (sigma = RV(7d) * iv_multiplier * skew_factor), fetches real-time option chains from Deribit public API, and reports Mean Absolute Error (MAE) and bias. Displays computed realized volatility and allows tuning via iv_multiplier slider. Handles both inverse (BTC/ETH-settled) and linear (USDC-settled) contracts correctly.
- **Synthetic Skew Engine**: Derives IV skew factors from live Deribit smile data. Computes skew anchors at deltas [0.15, 0.25, 0.35, 0.50] by comparing IV to ATM IV, then interpolates linearly for any delta. Skew anchors are cached per (underlying, option_type) to minimize API calls. Skew is applied to both calibration and synthetic backtester pricing.
- **Structured Logging**: Uses JSONL for structured logging of all decisions and actions, facilitating future analysis and ML/RL training.

### Technical Implementations
- **Configuration**: Pydantic settings are used for managing application configuration and switching between research/production modes.
- **API Wrapper**: `deribit_client.py` provides an `httpx`-based wrapper for the Deribit testnet API.
- **State Management**: `state_builder.py` aggregates market data and `status_store.py` manages thread-safe status updates.
- **Order Execution**: `execution.py` handles order translation, supporting dry-run simulations.
- **UI/UX**: The web dashboard provides a user-friendly interface for monitoring and interaction, with specific tabs for live agent status, backtesting, and chat.

## External Dependencies
- **Deribit API**: Used for fetching real-time market data (testnet) and historical data (mainnet public API for backtesting).
- **OpenAI**: Integrated via Replit AI Integrations for the LLM-powered decision mode and for generating insights from backtest results.