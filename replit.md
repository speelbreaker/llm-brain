# Options Trading Agent - Deribit Testnet

## Overview
A modular Python framework for automated BTC/ETH covered call trading on Deribit testnet. This is a research/experimentation system with both rule-based and LLM-powered decision modes.

## Project Status
- Core framework: Complete
- Rule-based policy: Implemented
- LLM decision mode: Implemented (uses Replit AI Integrations for OpenAI)
- FastAPI web dashboard: Complete with live status and chat
- Backtesting scaffold: Stub ready for future expansion

## Recent Changes
- 2024-12: Initial implementation of all core modules
- 2024-12: Added OpenAI integration via Replit AI Integrations
- 2024-12: Enhanced risk_engine.py with critical safety checks
- 2024-12: Refactored to FastAPI web dashboard with:
  - `/` - Live dashboard with status cards and chat interface
  - `/status` - JSON endpoint for current agent state
  - `/chat` - POST endpoint for natural language queries
  - `/health` - Health check for deployments
  - Background agent loop running in separate thread
  - In-memory status store for real-time updates

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
- `chat_with_agent.py` - Natural language query interface
- `status_store.py` - Thread-safe status storage
- `web_app.py` - FastAPI web application

### Entry Points
- `agent_loop.py` - Standalone CLI orchestration script
- `src/web_app.py` - FastAPI web app with background agent

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

## Running the Web Dashboard
```bash
uvicorn src.web_app:app --host 0.0.0.0 --port 5000
```

The dashboard shows:
- Live BTC/ETH prices
- Portfolio value and positions
- Last action taken
- Full status JSON (expandable)
- Chat interface for querying agent decisions

## Running the Agent (CLI)
```bash
python agent_loop.py
```

The agent runs in dry-run mode by default (no real orders placed).

## Chatting with the Agent

You can ask the agent why it took certain actions using the chat interface on the web dashboard, or via CLI:

```bash
python -m src.chat_with_agent "Why do you keep choosing the 97k call?"
python -m src.chat_with_agent "Summarize your last 10 decisions" --limit 10
python -m src.chat_with_agent "What would you likely do right now?"
```

## API Endpoints

### GET /
HTML dashboard with live status updates and chat interface.

### GET /status
Returns the latest agent status snapshot as JSON:
```json
{
  "log_timestamp": "...",
  "state": {
    "spot": {"BTC": 92310, "ETH": 3135},
    "portfolio": {...},
    "top_candidates": [...]
  },
  "final_action": {...},
  "execution": {...}
}
```

### POST /chat
Send a question about agent behavior:
```json
{"question": "Why did you pick the 97k call?"}
```
Returns:
```json
{"question": "...", "answer": "..."}
```

### GET /health
Health check endpoint for deployment monitoring.

## Key Decisions
- Uses Replit AI Integrations for OpenAI access (no API key needed)
- All trades are testnet-only for safety
- Structured JSONL logging for future ML/RL training
- Risk engine validates all decisions before execution
- FastAPI with background thread for non-blocking web + agent
