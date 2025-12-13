# Options Trading Agent - Deribit Testnet

A modular Python framework for automated BTC/ETH covered call trading on Deribit testnet. This is a **RESEARCH/EXPERIMENTATION** system for learning and testing options strategies.

## Disclaimer

**IMPORTANT**: This is NOT financial advice. This system is designed for:
- Research and experimentation only
- Deribit testnet (not live trading)
- Educational purposes

Never use this for live trading without extensive testing and professional advice.

## Features

- **Dual Decision Modes**: Rule-based policy or LLM-powered decisions (OpenAI)
- **Risk Engine**: Pre-trade validation with margin, delta, and exposure limits
- **State Builder**: Real-time market data aggregation and candidate filtering
- **Execution Module**: Order placement with dry-run support
- **Structured Logging**: JSON logs for future ML/RL training
- **Backtesting Scaffold**: RL-compatible environment for future training

## Requirements

- Python 3.11+
- Deribit testnet account with API credentials
- OpenAI API access (optional, for LLM mode via Replit AI Integrations)

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the project root:

```env
# Deribit Testnet Credentials
DERIBIT_CLIENT_ID=your_client_id
DERIBIT_CLIENT_SECRET=your_client_secret

# Optional: Override defaults
DRY_RUN=true
LLM_ENABLED=false
LOOP_INTERVAL_SEC=300
MAX_MARGIN_USED_PCT=80
```

### Configuration Options

| Variable | Default | Description |
|----------|---------|-------------|
| `DERIBIT_BASE_URL` | `https://test.deribit.com` | Deribit API URL |
| `DERIBIT_CLIENT_ID` | - | Your testnet API client ID |
| `DERIBIT_CLIENT_SECRET` | - | Your testnet API secret |
| `DRY_RUN` | `true` | If true, no real orders are placed |
| `LLM_ENABLED` | `false` | Enable LLM-based decisions |
| `LLM_MODEL_NAME` | `gpt-4.1-mini` | OpenAI model for LLM mode |
| `LOOP_INTERVAL_SEC` | `300` | Sleep between iterations (seconds) |
| `MAX_MARGIN_USED_PCT` | `80` | Maximum margin usage % |
| `MAX_NET_DELTA_ABS` | `5.0` | Maximum absolute net delta |
| `IVRV_MIN` | `1.0` | Minimum IV/RV ratio |
| `DELTA_MIN` | `0.10` | Minimum option delta |
| `DELTA_MAX` | `0.35` | Maximum option delta |
| `DTE_MIN` | `1` | Minimum days to expiry |
| `DTE_MAX` | `14` | Maximum days to expiry |
| `PREMIUM_MIN_USD` | `50` | Minimum premium in USD |

## Usage

### Running the Agent

```bash
# Run in dry-run mode (default, recommended for testing)
python agent_loop.py

# The agent will:
# 1. Fetch market data and positions from Deribit testnet
# 2. Filter candidate options based on configured criteria
# 3. Make trading decisions (rule-based or LLM)
# 4. Validate against risk limits
# 5. Execute orders (or simulate in dry-run mode)
# 6. Log everything for later analysis
```

### Testing the Backtest Environment

```bash
python -m backtest.env_simulator
```

## Project Structure

```
├── agent_loop.py          # Main entry point
├── src/
│   ├── config.py          # Pydantic settings configuration
│   ├── models.py          # Data models (Pydantic)
│   ├── deribit_client.py  # Deribit API wrapper
│   ├── state_builder.py   # Market data aggregation
│   ├── risk_engine.py     # Pre-trade risk validation
│   ├── policy_rule_based.py  # Deterministic decision logic
│   ├── agent_brain_llm.py # LLM-based decisions (OpenAI)
│   ├── execution.py       # Order execution
│   └── logging_utils.py   # Structured JSON logging
├── backtest/
│   └── env_simulator.py   # RL environment stub
├── logs/                  # Decision logs (JSONL)
├── data/                  # Market data (placeholder)
├── requirements.txt
└── README.md
```

## Decision Flow

```
1. Build State
   └── Fetch spot prices, positions, instruments
   └── Filter candidate options
   └── Calculate risk metrics

2. Make Decision
   ├── Rule-based: policy_rule_based.decide_action()
   └── LLM-based: agent_brain_llm.choose_action_with_llm()

3. Risk Check
   └── check_action_allowed() validates:
       - Margin limits
       - Delta exposure
       - Per-expiry caps

4. Execute
   └── execute_action() places orders (or simulates)

5. Log
   └── log_decision() saves state, decision, result
```

## Available Actions

| Action | Description |
|--------|-------------|
| `DO_NOTHING` | No action taken this iteration |
| `OPEN_COVERED_CALL` | Sell a new call option |
| `ROLL_COVERED_CALL` | Close existing, open new call |
| `CLOSE_COVERED_CALL` | Buy back existing short call |

## Logs

Logs are saved as JSONL files in `logs/agent_decisions_YYYYMMDD.jsonl`.

Each entry contains:
- Timestamp
- Agent state snapshot
- Proposed action
- Risk check result
- Execution result
- LLM reasoning (if applicable)

## LLM Mode

When `LLM_ENABLED=true`, the agent uses OpenAI's Chat Completions API to make decisions. The LLM receives:
- Portfolio state (balances, margin, positions)
- Volatility snapshot (IV/RV/skew)
- Risk limits
- Candidate options

The LLM returns a structured JSON decision that is then validated by the risk engine.

Note: LLM mode uses Replit AI Integrations for OpenAI access.

### Troubleshooting LLM Mode

If LLM mode returns `DO_NOTHING` with a configuration error message, check:
1. The Replit AI Integrations OpenAI extension is properly installed
2. The `AI_INTEGRATIONS_OPENAI_API_KEY` and `AI_INTEGRATIONS_OPENAI_BASE_URL` environment variables are set
3. If integration is unavailable, disable LLM mode with `LLM_ENABLED=false`

The agent gracefully falls back to `DO_NOTHING` when LLM mode encounters errors, preventing crashes while logging the issue.

## Future Development

- [ ] Implement proper IV/RV calculations
- [ ] Add Greeks calculation (Black-Scholes)
- [ ] Build complete backtesting with historical data
- [ ] Train RL agents on logged decisions
- [ ] Add monitoring dashboard
- [ ] WebSocket support for real-time data

## Getting Deribit Testnet Credentials

1. Go to https://test.deribit.com
2. Create a testnet account
3. Navigate to Account > API > Add new key
4. Enable trading permissions
5. Copy the client ID and secret to your `.env` file

## IV Calibration & Synthetic Universe Policy

### Why This Matters

The synthetic universe uses an **IV multiplier** (`synthetic_iv_multiplier`) to scale implied volatility when pricing options. This single parameter cascades through the entire decision pipeline:

```
IV Multiplier → Synthetic Option Prices → Premium Estimates → Greg VRP Signals → Trade Decisions
```

**Getting this wrong means your backtests are meaningless.**

### The Calibration Problem

- **Too low (e.g., 0.8)**: Synthetic premiums undervalue options → fewer trades trigger → backtests look overly conservative
- **Too high (e.g., 1.3)**: Synthetic premiums overvalue options → too many trades trigger → backtests show false profits
- **Just right (~1.0-1.05)**: Matches live Deribit marks → backtest results approximate real trading

### How to Calibrate

1. **Run Extended Calibration** (UI: Calibration tab → Run Extended Calibration)
   - Compares Black-Scholes synthetic prices vs. live Deribit marks
   - Returns recommended `vol_surface` multipliers per DTE band

2. **Use Auto-Calibration** (daily cron or manual)
   ```bash
   python -m scripts.auto_calibrate_iv --underlying BTC
   ```
   - Writes to `calibration_history` table
   - Does NOT auto-apply—you must review and enable via Update Policy

3. **Apply via Update Policy** (UI: Calibration → Update Policy)
   - EWMA smoothing prevents sudden jumps
   - Threshold gates prevent applying bad calibrations
   - File-based history for audit trail

### Regression Test

The test `TestGregVRPIVSensitivity.test_greg_vrp_is_sensitive_to_iv_multiplier` in `tests/test_synthetic_modes.py` proves that Greg's VRP decisions respond correctly to IV multiplier changes:

| IV Multiplier | Expected Behavior |
|---------------|-------------------|
| 0.9 (low)     | Fewer trades, lower/negative profit |
| 1.1 (high)    | More trades, higher profit |

If this test fails, the calibration → synthetic pricing → Greg decisions pipeline is broken.

### Best Practices

1. **Never run backtests without calibrating first** — uncalibrated results are fiction
2. **Re-calibrate weekly** — vol surface dynamics shift over time
3. **Compare synthetic vs. live** before any major backtest campaign
4. **Document your multiplier** when sharing backtest results

## License

This project is for educational and research purposes only.
