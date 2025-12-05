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
- **Risk Management**: A `risk_engine` module performs pre-trade validation, checking margin, delta, and exposure limits.
- **Backtesting Framework**: Includes a `CoveredCallSimulator` for historical analysis, supporting various exit styles (hold-to-expiry, take-profit and roll), a scoring function for candidate options, multi-leg chain visualization, and training data generation.
- **Training Mode**: Allows for multi-profile data collection (conservative, moderate, aggressive strategies) to generate diverse datasets for ML/RL.
- **Training Data Export**: Captures (state, action, reward) tuples and exports to CSV/JSONL. Note: Historical backtests use live option data from Deribit's public API; for meaningful training data, use live agent trading or integrate a paid historical options data source.
- **Web Dashboard**: A FastAPI application offers a "Live Agent" view with real-time status and recent decisions, a "Backtesting Lab" for configuring and analyzing simulations, and a "Chat" interface for natural language interaction with the agent.
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