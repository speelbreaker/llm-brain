# Options Trading Agent – Roadmap

This document outlines where the project is today and where it's headed. Written for the project owner, not engineers.

---

## 1. Current Status

We have a working covered call trading bot on Deribit testnet. Here's what's in place:

### Core Features
- **Live trading agent** – Monitors BTC and ETH, sells covered calls automatically
- **Two decision modes** – Rule-based (simple rules) or LLM-powered (asks GPT for advice)
- **Risk engine** – Checks every trade against margin and position limits before execution
- **Position tracking** – Remembers open positions even after restarts

### Web Dashboard
- **Live Agent tab** – Real-time status, recent decisions, open positions
- **Backtesting Lab** – Test strategies on historical data with TradingView-style metrics
- **Calibration tab** – Compare our synthetic prices against real Deribit prices
- **Chat tab** – Ask questions about agent behavior in plain English

### Training & Data
- **Training mode** – Collects data using multiple risk profiles (conservative, moderate, aggressive)
- **Training data export** – Generates CSV files suitable for machine learning
- **LLM fine-tuning scripts** – Converts training data to chat-style format for model training

### Quality Assurance
- **4 smoke tests** – Quick checks for live agent, backtesting, training export, and web API
- **30 unit tests** – Automated tests for core calculations (IVRV, scoring, expiry parsing)
- **Technical debt tracking** – All P0 items fixed, most P1 items addressed

---

## 2. Phase 1 – Stabilize "One Good Bot" ✅

**Status: Mostly complete**

This phase focused on getting one reliable bot working end-to-end. Key accomplishments:

- Agent runs continuously on testnet without crashes
- Risk checks prevent dangerous trades
- Backtesting produces consistent, reproducible results
- Training data pipeline works from backtest → CSV → LLM format
- Core calculations are centralized (no duplicates that could cause bugs)

### Optional Polish Items
These are nice-to-have, not blocking:
- Better documentation for training mode settings
- More descriptive error messages in the dashboard
- Clean up unused files (main.py, server.py, old backtest folder)

---

## 3. Phase 2 – Multi-Strategy Architecture

**Status: Not started**

### The Big Idea
Instead of one trading strategy, we want to run several "mini-bots" that each have their own personality but share the same market data and risk controls.

### How It Would Work

1. **Strategy Registry** – A list of named strategies, for example:
   - `covered_call_conservative` – Only sells calls with very low delta, high premium
   - `covered_call_aggressive` – Willing to take more risk for higher potential returns
   - `puts_experimental` – Testing cash-secured puts (new territory)

2. **Each Strategy Defines**:
   - A name (so we can track it separately)
   - Its risk profile (how aggressive it is)
   - Which instruments it can trade (BTC only? ETH too? Calls? Puts?)
   - How it makes decisions (rules or LLM)

3. **The Agent Loop Changes**:
   - Fetch market data once (same as now)
   - Ask each strategy: "What would you like to do?"
   - Collect all proposals
   - Risk engine reviews everything and decides what actually gets executed

4. **Benefits**:
   - Compare strategies side-by-side in real-time
   - One bad strategy can't take down the whole account (risk engine protects)
   - Easy to add new strategies without touching existing ones
   - Backtesting Lab can test strategies individually by name

### Dashboard Changes
- New "Strategy Status" table showing each strategy's positions and performance
- Filter decisions by strategy name
- Run backtests for specific strategies

---

## 4. Phase 3 – LLM "Researcher Mode"

**Status: Future idea**

### The Big Idea
Let the AI propose new strategies or parameter tweaks, then test them automatically before any real trading.

### How It Would Work

1. **Periodic Research Job**:
   - Every few hours, the LLM reviews recent backtest results and training metrics
   - It proposes new strategy configurations or parameter changes
   - Example: "I noticed high-delta calls are underperforming. Try reducing max_delta from 0.35 to 0.25"

2. **Experiment Queue**:
   - Proposals go into a queue for automated testing
   - The backtesting engine runs each experiment against historical data
   - Results are scored automatically (profit, drawdown, win rate)

3. **Promotion or Rejection**:
   - If an experiment beats the baseline, it can be promoted to a real strategy
   - If it underperforms, it's logged as a failed experiment
   - All experiments are reviewed by the risk engine before any live trading

4. **Key Safety Rule**:
   - The LLM can only *propose* changes
   - Everything goes through backtests and risk checks first
   - No experiment trades real money without explicit approval

---

## 5. Phase 4 – Real Historical Data (Tardis) and Advanced Training

**Status: R&D phase (requires data subscription)**

### The Big Idea
Right now, we use synthetic (calculated) option prices for backtesting. Real historical option data would make our models more accurate.

### What This Enables
- **Tardis.dev integration** – Access to actual Deribit option prices going back years
- **Better training data** – Real volatility smiles and skew patterns
- **Fine-tuned models** – Train small models on actual market behavior, not simulated data

### Why Wait
- Requires a paid data subscription (~$100-500/month depending on depth)
- Only valuable after multi-strategy architecture is in place
- Synthetic pricing works well enough for now (MAE is acceptable)

---

## 6. Priorities & Next Steps

### Near-Term (Next 2-4 Weeks)
- [ ] Implement multi-strategy support
  - Add `strategy_id` field to all decisions and positions
  - Create a strategy registry (config-based, not code changes)
  - Update risk engine to track limits per strategy
- [ ] Add strategy status table to Live Agent dashboard
- [ ] Allow backtests to filter by strategy name

### Medium-Term (1-2 Months)
- [ ] Add a simple "researcher job" that proposes experiments
- [ ] Build an experiment queue with automatic backtest evaluation
- [ ] Create a results dashboard showing experiment outcomes

### Long-Term (3+ Months)
- [ ] Integrate Tardis historical data (requires subscription decision)
- [ ] Experiment with more complex structures (puts, spreads, butterflies)
- [ ] Consider mainnet deployment (with full risk controls)

---

## Related Documents

- **ARCHITECTURE_OVERVIEW.md** – Technical details on how the system works
- **HEALTHCHECK.md** – Smoke tests, unit tests, and technical debt status
- **replit.md** – Quick reference for developers

---

*Last updated: December 2024*
