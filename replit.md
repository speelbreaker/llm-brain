# Options Trading Agent - Deribit Testnet

## Overview
A modular Python framework for automated BTC/ETH covered call trading on Deribit testnet. This is a research/experimentation system with both rule-based and LLM-powered decision modes.

## Project Status
- Core framework: Complete
- Rule-based policy: Implemented
- LLM decision mode: Implemented (uses Replit AI Integrations for OpenAI)
- Backtesting scaffold: Stub ready for future expansion

## Recent Changes
- 2024-12: Initial implementation of all core modules
- 2024-12: Added OpenAI integration via Replit AI Integrations

## Architecture

### Core Modules (src/)
- `config.py` - Pydantic settings with risk limits and strategy parameters
- `models.py` - Type-safe data models for instruments, positions, state
- `deribit_client.py` - httpx-based API wrapper for Deribit testnet
- `state_builder.py` - Market data aggregation and candidate filtering
- `risk_engine.py` - Pre-trade validation (margin, delta, exposure)
- `policy_rule_based.py` - Deterministic decision logic
- `agent_brain_llm.py` - LLM-based decisions via OpenAI Chat API
- `execution.py` - Order translation with dry-run support
- `logging_utils.py` - Structured JSONL logging

### Entry Point
- `agent_loop.py` - Main CLI orchestration script

### Future Development
- `backtest/env_simulator.py` - RL environment stub

## Configuration

### Environment Variables
Required for live trading (testnet):
- `DERIBIT_CLIENT_ID` - Deribit testnet API client ID
- `DERIBIT_CLIENT_SECRET` - Deribit testnet API secret

Optional settings:
- `DRY_RUN=true` - Simulate orders without placing them
- `LLM_ENABLED=false` - Toggle LLM decision mode
- `LLM_MODEL_NAME=gpt-4.1-mini` - OpenAI model for LLM mode
- `LOOP_INTERVAL_SEC=300` - Sleep between iterations

### Risk Parameters
- `MAX_MARGIN_USED_PCT=80` - Maximum margin usage
- `MAX_NET_DELTA_ABS=5.0` - Maximum absolute delta
- `IVRV_MIN=1.0` - Minimum IV/RV ratio
- `DELTA_MIN=0.10`, `DELTA_MAX=0.35` - Delta range
- `DTE_MIN=1`, `DTE_MAX=14` - Days to expiry range
- `PREMIUM_MIN_USD=50` - Minimum premium in USD

## User Preferences
- Python 3.11
- Type hints everywhere
- Pydantic for configs and models
- httpx for HTTP
- Clarity over cleverness

## Running the Agent
```bash
python agent_loop.py
```

The agent runs in dry-run mode by default (no real orders placed).

## Key Decisions
- Uses Replit AI Integrations for OpenAI access (no API key needed)
- All trades are testnet-only for safety
- Structured JSONL logging for future ML/RL training
- Risk engine validates all decisions before execution
