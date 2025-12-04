# Options Trading Agent - Deribit Testnet

## Overview
A modular Python framework for automated BTC/ETH covered call trading on Deribit testnet. This is a research/experimentation system with both rule-based and LLM-powered decision modes, featuring exploration-based learning for testnet experimentation.

## Project Status
- Core framework: Complete
- Rule-based policy: Implemented with scoring and exploration
- LLM decision mode: Implemented (uses Replit AI Integrations for OpenAI)
- FastAPI web dashboard: Complete with live status and chat
- Research/Production mode: Implemented with mode-specific parameters
- Backtesting scaffold: Stub ready for future expansion

## Recent Changes
- 2024-12: Initial implementation of all core modules
- 2024-12: Added OpenAI integration via Replit AI Integrations
- 2024-12: Enhanced risk_engine.py with critical safety checks
- 2024-12: Refactored to FastAPI web dashboard
- 2024-12: Added research vs production mode system:
  - Research mode: Wider delta/DTE/IVRV ranges for testnet exploration
  - Exploration: 25% probability of picking non-best candidate (epsilon-greedy)
  - Scoring function for candidate ranking
  - Mode and policy_version tracked in all decisions
- 2024-12: LLM decision mode enabled:
  - LLM brain uses OpenAI via Replit AI Integrations
  - Sandboxed to only allow valid actions (DO_NOTHING, OPEN/ROLL/CLOSE)
  - Uses effective_* parameters for mode-aware decision making
  - decision_source field tracks "llm" vs "rule_based" in all outputs
  - Risk engine still validates all LLM proposals before execution
- 2024-12: Backtesting framework implemented:
  - CoveredCallSimulator for historical "what if" analysis
  - MarketDataSource abstraction for pluggable data sources
  - DeribitDataSource using mainnet public API for historical data
  - Training data generation for ML/RL (state, action, reward tuples)
  - CSV/JSONL export and grid search utilities

## Architecture

### Core Modules (src/)
- `config.py` - Pydantic settings with mode selection and effective parameters
- `models.py` - Type-safe data models for instruments, positions, state
- `deribit_client.py` - httpx-based API wrapper for Deribit testnet
- `state_builder.py` - Market data aggregation and candidate filtering (uses effective params)
- `risk_engine.py` - Pre-trade validation (margin, delta, exposure)
- `policy_rule_based.py` - Decision logic with scoring and exploration
- `agent_brain_llm.py` - LLM-based decisions via OpenAI Chat API
- `execution.py` - Order translation with dry-run support
- `logging_utils.py` - Structured JSONL logging
- `chat_with_agent.py` - Natural language query interface
- `status_store.py` - Thread-safe status storage
- `web_app.py` - FastAPI web application

### Entry Points
- `agent_loop.py` - Standalone CLI orchestration script
- `src/web_app.py` - FastAPI web app with background agent

### Backtesting Module (src/backtest/)
- `data_source.py` - Generic MarketDataSource protocol interface
- `deribit_client.py` - Deribit mainnet public API wrapper for historical data
- `deribit_data_source.py` - MarketDataSource implementation using Deribit API
- `types.py` - CallSimulationConfig, SimulatedTrade, SimulationResult, TrainingExample
- `covered_call_simulator.py` - Core simulation engine with:
  - `simulate_single_call()` - "What if I sold this call here?"
  - `simulate_policy()` - Run policy across multiple decision times
  - `generate_training_data()` - Emit (state, action, reward) tuples
- `training_dataset.py` - Export training data to CSV/JSONL, grid search
- `backtest_example.py` - Example usage script

### Future Development
- RL environment wrapper around CoveredCallSimulator

## Configuration

### Mode Selection
- `MODE=research` - Research mode (default, wider ranges, exploration enabled)
- `MODE=production` - Production mode (stricter ranges, no exploration)

### Environment Variables
Required for live trading (testnet):
- `DERIBIT_CLIENT_ID` - Deribit testnet API client ID
- `DERIBIT_CLIENT_SECRET` - Deribit testnet API secret

Optional settings:
- `DRY_RUN=true` - Simulate orders without placing them (default: true for safety)
- `LLM_ENABLED=true` - Toggle LLM decision mode (currently enabled)
- `LLM_MODEL_NAME=gpt-4.1-mini` - OpenAI model for LLM mode
- `LOOP_INTERVAL_SEC=300` - Sleep between iterations

### Research Mode Parameters (wider ranges for testnet)
- `RESEARCH_DELTA_MIN=0.10`, `RESEARCH_DELTA_MAX=0.40` - Delta range
- `RESEARCH_DTE_MIN=1`, `RESEARCH_DTE_MAX=21` - Days to expiry range
- `RESEARCH_IVRV_MIN=1.0` - Minimum IV/RV ratio
- `RESEARCH_MAX_EXPIRY_EXPOSURE=1.0` - Higher per-expiry limit
- `EXPLORE_PROB=0.25` - 25% chance of exploration
- `EXPLORE_TOP_K=3` - Explore among top 3 candidates

### Production Mode Parameters (stricter for mainnet)
- `DELTA_MIN=0.20`, `DELTA_MAX=0.30` - Delta range
- `DTE_MIN=5`, `DTE_MAX=10` - Days to expiry range
- `IVRV_MIN=1.2` - Minimum IV/RV ratio
- `MAX_EXPIRY_EXPOSURE=0.3` - Conservative per-expiry limit

### Risk Parameters
- `MAX_MARGIN_USED_PCT=80` - Maximum margin usage
- `MAX_NET_DELTA_ABS=5.0` - Maximum absolute delta
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
- Current mode (Research/Production) with exploration percentage
- Last action taken
- Full status JSON (expandable)
- Chat interface for querying agent decisions

## Running the Agent (CLI)
```bash
python agent_loop.py
```

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
  "final_action": {
    "action": "...",
    "mode": "research",
    "policy_version": "rb_v1_explore"
  },
  "config_snapshot": {
    "mode": "research",
    "explore_prob": 0.25,
    "effective_delta_range": [0.1, 0.4],
    "effective_dte_range": [1, 21]
  }
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
- Research mode with epsilon-greedy exploration for data collection
