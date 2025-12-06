# Architecture Overview

This document provides a bird's-eye view of the Options Trading Agent project. It's written for the project owner, not engineers, so I'll avoid technical jargon and focus on what each part does and how to use it.

---

## What This Project Does

This is an automated options trading system that sells "covered calls" on Bitcoin and Ethereum using the Deribit testnet (a practice exchange with fake money). It can make decisions using either simple rules or by asking an AI (OpenAI's GPT). The system also includes tools for testing strategies on historical data and generating training datasets for machine learning.

---

## Main Modules / Packages

| Location | What It Does |
|----------|--------------|
| `src/web_app.py` | The main web dashboard. Shows live status, recent decisions, open positions, a chat interface to ask the agent questions, and the Backtesting Lab for testing strategies on historical data. |
| `agent_loop.py` | The "heartbeat" of the live agent. Runs continuously, fetching market data, making trading decisions, and executing trades every few minutes. |
| `src/config.py` | Central settings hub. Controls everything from which cryptocurrencies to trade, to risk limits, to whether the AI is enabled. All settings can be changed via environment variables. |
| `src/state_builder.py` | Gathers all current market information (prices, account balances, available options) into a single snapshot that the decision-making code uses. |
| `src/policy_rule_based.py` | The "rule-based brain." Makes trading decisions using handwritten rules like "if the option premium is juicy and risk is low, sell the call." |
| `src/agent_brain_llm.py` | The "AI brain." Sends the current market snapshot to OpenAI's GPT and asks it what trade to make. The AI's response is validated before execution. |
| `src/risk_engine.py` | The safety guard. Before any trade goes through, this checks margin limits, position limits, and other safeguards. Can block trades that are too risky. |
| `src/execution.py` | Actually sends orders to Deribit. Has a "dry run" mode where it pretends to trade without actually doing anything. |
| `src/training_policy.py` | Special mode for collecting training data. Opens multiple positions with different strategies (conservative, moderate, aggressive) to see what works. |
| `src/training_profiles.py` | Defines the three training strategy profiles and how to score option candidates for each. |
| `src/market_context.py` | Calculates "big picture" market metrics like whether we're in a bull/bear market, recent volatility, and distance from moving averages. |
| `src/deribit_client.py` | Talks to the Deribit exchange API. Handles authentication, fetching prices, getting account info, and placing orders. |
| `src/position_tracker.py` | Remembers which positions the bot has opened. Saves to disk so it survives restarts. |
| `src/calibration.py` | Compares our synthetic (calculated) option prices against real Deribit prices to check accuracy. |
| `src/synthetic_skew.py` | Fetches real volatility "smile" data from Deribit to make our synthetic prices more realistic. |
| `src/chat_with_agent.py` | Powers the chat feature. Sends your question plus recent decision logs to GPT and returns an explanation. |
| `src/backtest/` | The backtesting engine (see detailed breakdown below). |
| `scripts/` | Utility scripts for one-off tasks like converting training data. |
| `data/` | Output files: training datasets, decision logs, etc. |
| `logs/` | Daily JSONL logs of all agent decisions. |

### Inside the Backtest Package (`src/backtest/`)

| File | What It Does |
|------|--------------|
| `covered_call_simulator.py` | The simulation engine. Simulates what would have happened if you sold a specific call option at a specific time. |
| `data_source.py` + `deribit_data_source.py` | Fetches historical price data from Deribit's public API. |
| `state_builder.py` | Builds historical market snapshots for backtesting (similar to the live state builder but for past dates). |
| `pricing.py` | Black-Scholes option pricing formulas for synthetic/calculated option prices. |
| `training_dataset.py` | Exports backtest results as training data (CSV/JSONL) for machine learning. |
| `manager.py` | Coordinates running backtests in the background and tracking their status. |
| `config_schema.py` + `config_presets.py` | Backtest configuration system with presets (ULTRA_SAFE, BALANCED, AGGRESSIVE). |
| `types.py` | Data structures used throughout backtesting. |

---

## Data Flow Diagram

```
                           ┌─────────────────────────────────────────┐
                           │         DERIBIT EXCHANGE                │
                           │  (testnet for live, mainnet for data)   │
                           └────────────────┬────────────────────────┘
                                            │
                    ┌───────────────────────┼───────────────────────┐
                    │                       │                       │
                    ▼                       ▼                       ▼
            ┌───────────────┐      ┌───────────────┐      ┌───────────────┐
            │ Live Market   │      │ Historical    │      │ Synthetic     │
            │ Data (testnet)│      │ Price Data    │      │ Pricing (BS)  │
            └───────┬───────┘      └───────┬───────┘      └───────┬───────┘
                    │                      │                      │
                    ▼                      └──────────┬───────────┘
            ┌───────────────┐                         │
            │ State Builder │                         ▼
            │ (live)        │              ┌─────────────────────┐
            └───────┬───────┘              │ Backtest State      │
                    │                      │ Builder (historical)│
                    ▼                      └──────────┬──────────┘
            ┌───────────────┐                         │
            │ Agent Loop    │                         ▼
            │ (runs every   │              ┌─────────────────────┐
            │  5 minutes)   │              │ Covered Call        │
            └───────┬───────┘              │ Simulator           │
                    │                      └──────────┬──────────┘
         ┌──────────┴──────────┐                      │
         ▼                     ▼                      ▼
┌─────────────────┐  ┌─────────────────┐   ┌─────────────────────┐
│ Rule-Based      │  │ LLM Brain       │   │ Training Data       │
│ Policy          │  │ (OpenAI GPT)    │   │ Exporter (CSV/JSONL)│
└────────┬────────┘  └────────┬────────┘   └──────────┬──────────┘
         │                    │                       │
         └──────────┬─────────┘                       ▼
                    ▼                       ┌─────────────────────┐
            ┌───────────────┐               │ LLM Training Script │
            │ Risk Engine   │               │ (chat-style JSONL)  │
            └───────┬───────┘               └─────────────────────┘
                    │
                    ▼
            ┌───────────────┐
            │ Executor      │
            │ (sends orders)│
            └───────┬───────┘
                    │
                    ▼
            ┌───────────────┐
            │ Deribit       │
            │ (testnet)     │
            └───────────────┘
```

### Where Data Comes From

1. **Live Data**: The agent fetches real-time prices, account balances, and option chains from Deribit testnet every iteration.

2. **Historical Data**: For backtesting, we fetch historical 1-hour candlestick data from Deribit's public mainnet API (no authentication needed).

3. **Synthetic Pricing**: For backtests before 2024 (when Deribit option history is limited), we calculate option prices ourselves using the Black-Scholes formula, calibrated against real Deribit prices.

### Where Data Goes

- **Live trades** → Deribit testnet
- **Decision logs** → `logs/agent_decisions_YYYYMMDD.jsonl`
- **Training datasets** → `data/training_*.csv` and `data/training_*.jsonl`
- **LLM training files** → `data/*_per_candidate.jsonl` and `data/*_per_decision_ranking.jsonl`

---

## Major Features and How to Use Them

### 1. Live Trading Agent

**What it does**: Monitors BTC and ETH options on Deribit testnet, decides when to sell covered calls, and executes trades automatically.

**Entrypoints**:
- `agent_loop.py` → `run_agent_loop_forever()` function
- `src/web_app.py` starts the agent automatically on launch

**How to run it**: 
- The agent starts automatically when you run the web app
- Visit the web dashboard at the root URL `/` to see live status
- Check `/status` for a JSON snapshot

---

### 2. Web Dashboard

**What it does**: Provides a visual interface to monitor the agent, view positions, ask questions, run backtests, and calibrate pricing.

**Entrypoints**:
- `src/web_app.py` → FastAPI `app` object

**How to run it**: 
- Already running! Click the webview panel or visit the URL
- **Tabs available**:
  - "Live Agent" - Real-time status, decisions, positions
  - "Backtesting Lab" - Run historical simulations
  - "Calibration" - Compare synthetic vs real prices
  - "Chat" - Ask questions about agent behavior

---

### 3. Backtesting Engine

**What it does**: Simulates what would have happened if you ran the covered call strategy on historical data. Calculates profit/loss, drawdowns, and performance metrics.

**Entrypoints**:
- `src/backtest/covered_call_simulator.py` → `CoveredCallSimulator` class
- `src/backtest/manager.py` → `BacktestManager` for background runs
- Web UI: `/backtest` tab

**How to run it**:
- **Web UI**: Go to "Backtesting Lab" tab, set parameters, click "Run Backtest"
- **Console**: `python -m src.backtest.backtest_example`

---

### 4. Training Data Generation

**What it does**: Creates datasets for training machine learning models. Records what options were available at each decision point, which one was chosen, and what the outcome was.

**Entrypoints**:
- `src/backtest/training_dataset.py` → `build_candidate_level_examples()` and `export_training_candidates_csv()`
- Web UI: Enable "Export Training Data" checkbox in Backtesting Lab

**How to run it**:
- In the web UI, check "Export Training Data" before running a backtest
- Output goes to `data/training_candidates_*.csv`

---

### 5. LLM Training Data Transformer

**What it does**: Converts the CSV training data into chat-style JSONL files that can be used to fine-tune language models.

**Entrypoints**:
- `scripts/build_llm_training_from_candidates.py` → `main()` function

**How to run it**:
```bash
python scripts/build_llm_training_from_candidates.py \
    --input data/training_candidates_BTC_....csv \
    --exit-style tp_and_roll \
    --underlying BTC
```
- Produces `*_per_candidate.jsonl` (one record per option) and `*_per_decision_ranking.jsonl` (one record per decision)

---

### 6. Training Mode

**What it does**: Special agent mode that opens multiple positions with different risk profiles (conservative, moderate, aggressive) to gather diverse data for ML training.

**Entrypoints**:
- `src/training_policy.py` → `build_training_actions()`
- `src/training_profiles.py` → profile definitions

**How to enable it**:
- Set environment variable `TRAINING_MODE=True`
- Only works on testnet (mainnet always has full risk checks)
- Configure profiles via `TRAINING_STRATEGIES=conservative,moderate,aggressive`

---

### 7. Calibration Tool

**What it does**: Compares our synthetic Black-Scholes option prices against real Deribit prices to see how accurate they are. Helps tune the IV multiplier.

**Entrypoints**:
- `src/calibration.py` → `run_calibration()` function
- Web UI: `/calibration` tab

**How to run it**:
- Go to "Calibration" tab in web dashboard
- Select underlying (BTC/ETH) and adjust IV multiplier
- Click "Run Calibration"
- Look for low MAE (Mean Absolute Error)

---

### 8. Chat with Agent

**What it does**: Ask natural language questions about why the agent made certain decisions. Uses GPT to analyze recent logs and explain reasoning.

**Entrypoints**:
- `src/chat_with_agent.py` → `chat_with_agent()` function
- Web UI: "Chat" tab

**How to run it**:
- Go to "Chat" tab
- Type a question like "Why did you sell the 100k call yesterday?"
- The AI reads recent logs and explains

---

### 9. Strategy & Safeguards Status Panel

**What it does**: Shows which trading rules and safety limits are currently active, and what values they have.

**Entrypoints**:
- `src/strategy_status.py` → `build_strategy_status()`
- `src/rules_summary.py` → `build_rules_summary()`
- Web API: `/api/strategy-status`

**How to see it**:
- In the "Live Agent" tab, look for the "Strategy & Safeguards" panel
- Shows different rules for Training vs Live mode

---

## Configuration Presets (for Backtesting)

| Preset | Philosophy |
|--------|------------|
| ULTRA_SAFE | Very conservative. High margin limits, strict delta caps, requires high IV/RV ratio. |
| BALANCED | Middle ground. Good for most situations. |
| AGGRESSIVE | Riskier. Wider delta ranges, lower thresholds, more trades. |
| CUSTOM | Start from BALANCED and tweak individual settings. |

---

## Known Rough Edges / TODOs

See `HEALTHCHECK.md` for the full prioritized list with risk assessments. Below is a summary.

---

## Technical Debt Priorities

### Priority Legend

| Priority | Meaning |
|----------|---------|
| **P0** | Must fix soon – bug risk or correctness issue |
| **P1** | Worth fixing, but not urgent – code clarity or duplication |
| **P2** | Nice-to-have / cosmetic – style, structure, readability |

---

### P0 – Must Fix Soon

| Item | Risk if Unfixed | Risk of Refactoring | Status |
|------|-----------------|---------------------|--------|
| **Duplicate scoring functions** (`policy_rule_based.py` vs `covered_call_simulator.py`) | Live agent and backtester may score candidates differently, leading to strategy drift where backtest results don't match live behavior. | Could break both backtests and live trading if the unified scorer has bugs or different default parameters. | **FIXED** |
| **IVRV calculated in multiple places** (state_builder, simulator, training_profiles) | Inconsistent IVRV values could cause the agent to make different decisions than the backtester, making training data unreliable. | Minor risk if done carefully; mainly need to update all call sites to use the centralized version. | **FIXED** |

---

### P1 – Worth Fixing (Not Urgent)

| Item | Risk if Unfixed | Risk of Refactoring |
|------|-----------------|---------------------|
| **Duplicate state builders** (live vs backtest) | Makes it harder to add new features—changes must be made in two places, increasing the chance one gets forgotten. | Moderate risk; the live and backtest builders have different data sources, so a hasty merge could break either mode. |
| **Duplicate Deribit clients** (trading vs public data) | Maintenance burden; any bug fix or API change must be applied twice. | Low-to-moderate risk; the clients serve different purposes (auth vs no-auth), so they can share base code without full merge. |
| **Duplicate expiry parsing** | Minor—mostly code clarity. Same logic in two places means potential for subtle date parsing bugs. | Very low risk; simple utility functions that can be extracted without touching core logic. |
| **No unit tests** | Regressions go undetected until they cause visible problems in production or backtests. | Time investment rather than code risk; tests should be added incrementally without modifying existing code. |

---

### P2 – Nice-to-Have / Cosmetic

| Item | Risk if Unfixed | Risk of Refactoring |
|------|-----------------|---------------------|
| **Dead code: `main.py`, `server.py`, `backtest/`** | Confuses new developers. Zero functional risk. | No risk—just delete after confirming nothing imports them. |
| **Large files** (`web_app.py`, `covered_call_simulator.py`) | Harder to navigate and maintain. No correctness issue. | Moderate effort; splitting requires careful testing. |
| **Configuration sprawl** (`config.py`) | Minor—still manageable. Could benefit from grouping. | Low risk; sub-models can be introduced without changing behavior. |

---

### Summary: Recommended Order

1. **P0 first**: Fix scoring and IVRV duplication to ensure backtest-live consistency.
2. **P1 next**: Unify state builders and clients when adding new features.
3. **P2 whenever convenient**: Delete dead code (easy wins), split large files during related work.

---

### Detailed Reference

For the full tables with specific file paths and suggestions, see the "Technical Debt" section in `HEALTHCHECK.md`.

---

### Missing Features / Improvements

1. **No automated testing**: No unit tests or integration tests exist. Adding them would catch regressions.

2. **No position reconciliation**: If the bot crashes and restarts, it relies on saved positions file. Should cross-check with Deribit.

3. **Training data doesn't capture all features**: Market context (regime, volatility) is captured but not all potential signals.

4. **Backtest doesn't simulate fees accurately**: Uses a fixed fee rate rather than Deribit's actual fee schedule.

5. **No multi-leg strategy support**: Only single covered calls, not spreads or condors.

---

## Quick Reference: File Locations

| Need to... | Look in... |
|------------|-----------|
| Change trading parameters | `src/config.py` or environment variables |
| Modify decision logic | `src/policy_rule_based.py` or `src/agent_brain_llm.py` |
| Adjust risk limits | `src/risk_engine.py` |
| Add a new training profile | `src/training_profiles.py` |
| Change the web dashboard UI | `src/web_app.py` (HTML is at the bottom) |
| Modify backtest logic | `src/backtest/covered_call_simulator.py` |
| Change how options are scored | `src/backtest/covered_call_simulator.py` → `_score_candidate()` |
| Export training data differently | `src/backtest/training_dataset.py` |
| Convert training data to JSONL | `scripts/build_llm_training_from_candidates.py` |

---

*Last updated: December 2024*
