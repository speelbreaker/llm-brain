# Options Trading Agent – Backlog & Roadmap

_Last updated: 2025-12-08_

This document tracks the **open backlog** for the options trading agent, combining:
- AI Builder's healthcheck/architecture notes,
- Deep Research / external model review,
- Manual design discussions across chats.

It does **not** list every implementation detail – only things explicitly marked as "later / not implemented yet".

Priorities:
- **P0** – Must fix before mainnet / real capital.
- **P1** – Important for robustness and clarity; ideal for Phase 2.
- **P2** – Nice-to-have / Phase 3+ or polish.

Phases:
- **Phase 1** – One good covered-call bot on testnet (mostly done).
- **Phase 2** – One smart bot with LLM co-pilot + stronger safety.
- **Phase 3** – Multi-bot, supervisor AI, real historical data, production-grade ops.

---

## 0. Recently Completed (for context)

These are **implemented** and here only as reference so we don't re-plan them:

- Centralized IVRV calculation in `src/metrics/volatility.py` and removed duplicates.  
- Centralized expiry parsing in `src/utils/expiry.py` and removed duplicates.  
- Basic unit test suite for helpers (IVRV, expiry, scoring) + smoke tests:
  - `scripts/smoke_live_agent.sh`
  - `scripts/smoke_backtest.sh`
  - `scripts/smoke_training_export.sh`
  - `scripts/smoke_web_api.sh`
- **Synthetic universe & backtesting**:
  - `CoveredCallSimulator` with scoring and training export.
  - Training data export to CSV/JSONL and LLM training script.
- **Live vs synthetic data sources**:
  - Fixed REAL_SCRAPER data source to correctly return calls & puts with proper `option_type`.
  - Implemented `LiveDeribitDataSource` in `src/backtest/live_deribit_data_source.py`:
    - Implements `MarketDataSource` for backtester.
    - Proper IV percentage → decimal conversion.
    - Spot OHLC synthesis + caching.
  - Reusable live exam builder in `src/data/live_deribit_exam.py`:
    - `build_live_deribit_exam_dataset()` for backtester/CLI.
  - Refactored `scripts/build_exam_dataset_from_live_deribit.py` to call shared builder.
  - Comparison script `scripts/compare_synthetic_vs_live.py`:
    - Side-by-side SYNTHETIC vs LIVE_DERIBIT backtests.
    - Writes results to PostgreSQL.
    - Prints metrics (Net PnL, Max DD, Sharpe, Sortino, etc.).
  - Backtest diff report `scripts/diff_backtest_runs.py`:
    - Compares any two backtest runs from PostgreSQL.
    - Shows A vs B vs Diff for all key metrics.
  - Strategy health check `scripts/strategy_health_check.py`:
    - Runs standard battery of synthetic vs live_deribit comparisons.
    - Computes optimism factor (synthetic/live profit ratio).
    - JSON export to `data/health_checks/`.
  - Reusable modules:
    - `src/backtest/compare.py` with `run_synthetic_vs_live_pair()`.
    - `src/backtest/diff.py` with `compute_diff_for_runs()`.
  - Verified workflow:
    - Backtest runs persisted to PostgreSQL.
    - Data source type (`synthetic`, `live_deribit`, `real_scraper`) tracked in DB.
- Web app:
  - Live Agent, Backtesting Lab, Chat, Calibration tabs.
  - Chat agent can answer "why did the bot do X?" by reading decision logs.
- **Strategy abstraction (Phase 2 #1)**:
  - `Strategy` base class with `strategy_id` property and `propose_actions()` method
  - `CandidateAction` and `StrategyDecision` typed action types
  - `CoveredCallStrategy` implementing the interface
  - `StrategyRegistry` for managing active strategies
  - `strategy_id` propagates through agent_loop, decisions_store, and JSONL logs
- **Shared StateBuilder core (Phase 2 #1)**:
  - `src/state_core.py` with `RawMarketSnapshot` → `AgentState` pure functions
  - Both live and backtest builders use same `build_agent_state_from_raw()` logic
  - `build_historical_agent_state()` for typed backtest state construction
  - Tests ensuring deterministic output and live/backtest parity

Everything below is **TODO / backlog**.

---

## A. Architecture & Design

### [A1] Strategy abstraction & multi-strategy readiness  
**Priority:** P1 (Phase 2)  
**Status:** IMPLEMENTED (2025-12-08)  

**Implemented:**
- `Strategy` base class in `src/strategies/types.py` with `strategy_id` property
- `StrategyConfig` for strategy configuration
- `CandidateAction` and `StrategyDecision` typed action representations
- `CoveredCallStrategy` implementation in `src/strategies/covered_call.py`
- `StrategyRegistry` in `src/strategies/registry.py` for managing strategies
- Agent loop now uses `strategy.propose_actions(state)` via registry
- `strategy_id` included in all decision logs, JSONL flight recorder, and status snapshots

---

### [A2] Shared StateBuilder core (live + backtest)  
**Priority:** P1 (Phase 2)  
**Status:** IMPLEMENTED (2025-12-08)  

**Implemented:**
- `src/state_core.py` with:
  - `RawMarketSnapshot`, `RawOption`, `RawPosition`, `RawPortfolio` types
  - `build_agent_state_from_raw()` pure function
- Live `src/state_builder.py` uses state_core for AgentState construction
- Backtest `src/backtest/state_builder.py` has:
  - `build_historical_agent_state()` using same state_core logic
  - `create_agent_state_builder()` factory for strategy integration
- Tests in `tests/test_state_core.py` verify deterministic output and live/backtest consistency

---

### [A3] Shared Deribit base client  
**Priority:** P1 (Phase 2)  
**Status:** Not implemented  

- There are two Deribit clients:
  - `src/deribit_client.py` (trading/authenticated).
  - `src/backtest/deribit_client.py` (public data).
- Currently they duplicate some parsing and error handling.

**Goal:**
- Introduce `DeribitBaseClient` with shared HTTP/JSON-RPC/error handling + response parsing.
- Make trading & public clients extend this base, so behaviour is consistent.

---

### [A4] Dependency injection / config injection  
**Priority:** P2  
**Status:** Not standardized  

- Many modules directly import a global `settings` object.
- Makes testing harder and tightly couples config to implementation.

**Goal:**
- Gradually move toward passing `config` / `StrategyConfig` into constructors or function arguments.
- Keep globals for now, but aim for cleaner DI in new code.

---

### [A5] Web app modularization (web_app "God module")  
**Priority:** P2  
**Status:** Not done  

- `src/web_app.py` holds:
  - All FastAPI routes,
  - HTML templates (inline strings),
  - Agent background task wiring.

**Goal:**
- Split into smaller routers:
  - `/live`, `/backtest`, `/chat`, `/calibration`, `/health`.
- Move HTML into proper template files.
- Make it easier to evolve UI without touching core agent logic.

---

### [A6] Generic backtest runner & metrics  
**Priority:** P2  
**Status:** Partially present  

- Backtesting is currently centred on `CoveredCallSimulator`.
- Metric support is improving (Sharpe/Sortino in comparison script), but not a fully generic multi-strategy runner.

**Goal:**
- Abstract a generic backtest runner capable of:
  - Running any `Strategy` over historical data.
  - Tracking consistent metrics (PnL, DD, Sharpe, Sortino, etc.).
  - Storing results in PostgreSQL for any strategy ID / config.

---

## B. Persistence & Infrastructure

### [B1] Database for trades/decisions (SQLite/PostgreSQL core)  
**Priority:** P1 (pre-mainnet), can be Phase 2–3  
**Status:** Partially done  

- Trade/backtest metrics are already persisted to PostgreSQL for some runs.
- Live decisions/positions are still largely JSONL + simple file tracking.

**Goal:**
- Migrate decision logs and position tracking to a DB (SQLite or Postgres core).
- Keep JSONL as an append-only "flight recorder", but make DB the source of truth for analytics & reconciliation.

---

### [B2] Watchdogs & process health (kill switch, heartbeat, rate limits)  
**Priority:** P0 before real money, can be Phase 2 on testnet  
**Status:** Not implemented  

Missing pieces:

- Global **kill switch** (config + UI) to stop new orders and optionally close positions.
- **Heartbeat monitor** to detect if the loop stalls (no state update for X minutes).
- **Max orders per minute** to prevent order spam (buggy loop or API errors).
- **Close-only mode** where the bot may only reduce risk (no new positions).

---

## C. Risk & Safety

### [C1] Position reconciliation vs exchange  
**Priority:** P0 (pre-mainnet), P1 (Phase 2 on testnet)  
**Status:** IMPLEMENTED (2025-12-08)  

**Implementation:**
- `src/reconciliation.py` with `PositionReconciliationDiff`, `PositionSizeMismatch` dataclasses
- `diff_positions()` pure function compares local tracker vs exchange positions
- `run_reconciliation_once()` helper for CLI and agent loop
- Agent loop runs reconciliation on startup (`position_reconcile_on_startup`) and per-loop (`position_reconcile_on_each_loop`)
- Configurable tolerance (`position_reconcile_tolerance_usd`) to avoid panicking over tiny rounding differences
- On mismatch: `halt` (block new openings) or `auto_heal` (rebuild local state from exchange)
- CLI script: `scripts/reconcile_positions_once.py` for manual diagnosis
- 17 unit tests in `tests/test_reconciliation.py`

---

### [C2] Global PnL / drawdown limits  
**Priority:** P1  
**Status:** Not implemented  

- Risk engine currently monitors:
  - margin usage,
  - net delta,
  - per-expiry exposure.
- No daily or rolling drawdown limits.

**Goal:**
- Track daily (and rolling) NAV changes.
- If NAV drawdown exceeds threshold (e.g. -X% in 24h):
  - Close positions and/or switch to close-only mode.
  - Require manual reset.

---

### [C3] Per-bot & cross-bot exposure limits (for future multi-bot)  
**Priority:** P2 (Phase 3)  
**Status:** Not implemented  

- Needed for multi-strategy world:
  - Per-bot max margin, per-underlying short exposure.
  - Cross-bot caps per underlying.

**Goal:**
- Design a global risk engine that understands multiple `strategy_id`s and enforces global caps.

---

### [C4] Liquidity & slippage safeguards  
**Priority:** P1–P2  
**Status:** Not implemented  

- No explicit filters yet for:
  - wide bid–ask spreads,
  - zero/low open interest.

**Goal:**
- In candidate selection / risk engine, add optional filters:
  - spread ≤ threshold.
  - min volume / OI.
- Later, refine with real historical data (Tardis).

---

## D. Strategies & Edge

### [D1] Cash-secured puts & "Wheel" strategy  
**Priority:** P1–P2 (Phase 2–3)  
**Status:** Not implemented  

- Only covered calls are modelled.
- No support for:
  - short puts,
  - Wheel entry/exit cycle.

**Goal:**
- Introduce Wheel strategy:
  - Sell cash-secured puts to enter underlying.
  - Sell covered calls to exit / generate yield.
- Integrate with shared Strategy abstraction and risk engine.

---

### [D2] Regime-aware overlays (trend & volatility filters)  
**Priority:** P1  
**Status:** Not implemented in rule policy  

- `market_context` exists but rule policy doesn't enforce regime rules.

**Goal:**
- Use trend/volatility context to:
  - Enable/disable strategies,
  - Adjust delta & DTE ranges,
  - Avoid selling calls/puts in unfavourable regimes.

Examples:
- Avoid selling calls in aggressive bull breakouts.
- Require IV percentile above a threshold before selling options.

---

### [D3] Dynamic position sizing  
**Priority:** P1  
**Status:** Not implemented  

- Orders use `default_order_size` instead of dynamic sizing.
- No "1% of equity per position" style rule.

**Goal:**
- Add position sizing formula based on:
  - equity,
  - volatility/IV,
  - regime.
- Allow config like:
  - "max 1%–2% of portfolio per short option line".

---

### [D4] Multi-year, regime-sliced backtests  
**Priority:** P2  
**Status:** Not implemented  

- Backtests mainly cover:
  - single covered-call simulations over limited windows.

**Goal:**
- Use LIVE_DERIBIT and other data sets to:
  - Run multi-year backtests,
  - Slice by regime: e.g. 2021 bull vs 2022 bear,
  - Compare strategies (Covered Call vs Wheel vs others).

---

### [D5] Multi-leg strategies (spreads, butterflies, etc.)  
**Priority:** P2  
**Status:** Not implemented  

- UI sketches exist for multi-leg chains.
- No backtesting or execution support yet.

**Goal:**
- After single-leg + Wheel strategies are stable, add support for defined-risk structures:
  - vertical spreads,
  - butterflies, etc.
- Extend risk engine to handle their margin & payoff profiles.

---

## E. Data & Backtesting

### [E1] Real historical options data (Tardis or similar)  
**Priority:** P1 for serious research, Phase 3  
**Status:** Not integrated  

- Deribit public API lacks full historical greeks/IV.
- Current backtests use synthetic BS pricing + captured snapshots.

**Goal:**
- Integrate a historical data provider (e.g. Tardis):
  - BTC/ETH options order books and greeks for last N years.
- Use it to:
  - validate synthetic universe,
  - run realistic backtests,
  - generate higher-quality training data.

---

### [E2] More granular decision intervals (1h/4h vs daily)  
**Priority:** P2  
**Status:** Backtester supports different intervals but not fully explored  

**Goal:**
- Experiment with:
  - 1h / 4h decision intervals,
  - different bar sizes (not only daily).
- Evaluate trade-off between:
  - more data points vs overtrading vs noise.

---

### [E3] Overfitting Mitigation & Robustness  
**Priority:** P1 (Phase 2–3)  
**Status:** Not implemented  

When optimizing strategies, we must guard against "curve-fitting" parameters that only work on the training period. This section outlines the planned safeguards:

**Cross-asset testing:**
- Strategies should be tested on multiple underlyings (BTC, ETH, and later SOL/DOGE where data allows).
- A genuine edge should not be asset-specific; if a config only works on BTC 2021, it's likely overfit.

**Time-based splits (train / validation / test):**
- Define explicit periods:
  - **Training:** used to search/tune parameters.
  - **Validation:** used to judge parameter sets and detect overfitting.
  - **Test/holdout:** kept untouched until final sanity-check.
- Example: 2019–2022 training, 2023 validation, 2024 holdout.

**Regime-sliced evaluation:**
- Backtests must be sliced into different market regimes: bull, bear, sideways, high-vol, low-vol.
- A "good strategy" should perform reasonably across very different conditions (e.g. 2021 bull vs 2022 bear).
- Reject configs that only shine in one hand-picked period.

**Walk-forward / rolling window tests:**
- Mature version should support walk-forward testing:
  - Fit parameters on window 1, test on window 2, roll forward, repeat.
  - This simulates real deployment where future data is unknown.

**Monte Carlo style perturbations:**
- Random perturbations of parameters around the "best" configuration to check stability.
- If small param changes cause large performance swings, the config is fragile.
- Randomized slippage/fee assumptions to stress-test robustness (not just optimistic fee rates).

**Simplicity bias:**
- When comparing two parameter sets with similar performance, favour the simpler one:
  - Narrower parameter ranges.
  - Fewer toggles and special cases.
  - More interpretable rules.
- Occam's razor: simpler models generalize better.

This section directly supports the Phase 2–3 tuning and optimization work (see [I1] Strategy Factory below).

---

## F. LLM Layer & Intelligence

> **Important design principle:** The LLM is a **research assistant and analyst**, not a random parameter generator. Its role is to reason about backtest results, compare runs across regimes, propose small structured adjustments, and highlight overfitting risks. Numeric search (grid/random/Bayesian) remains the primary engine for exploring parameter space; the LLM guides and interprets, not brute-force samples.

### [F1] Stronger LLM output validation  
**Priority:** P1 (Phase 2)  
**Status:** Basic JSON parsing, no deep validation  

- Currently:
  - LLM output is parsed as JSON and action name validated.
- Missing:
  - symbol validity checks,
  - bounds on size/DTE/delta,
  - automatic fallback to rule-based on validation failure.

**Goal:**
- Add a strict validator:
  - If LLM output is invalid or out-of-bounds → reject & fall back to rule-based decision.
- Log reason for fallback for later analysis.

---

### [F2] Shadow testing & hybrid policy mode  
**Priority:** P1 (Phase 2)  
**Status:** Not implemented  

- No systematic "run rule and LLM in parallel and compare" mechanism.

**Goal:**
- On each decision:
  - Compute both `rule_action` and `llm_action`.
  - Decide what to execute based on `policy_mode`:
    - `rule_only`, `llm_only`, `hybrid` with `hybrid_llm_prob`.
- Tag each decision with:
  - `decision_source_executed`,
  - `rule_action`, `llm_action`.
- Use logs for performance comparison & training.

---

### [F3] LLM role shift: param-setter / regime detector  
**Priority:** P2 (design direction)  
**Status:** Not fully implemented  

- Long-term goal:
  - Use LLM more for:
    - regime classification,
    - parameter selection (delta/DTE ranges, aggressiveness),
  - and less for picking precise strikes/sizes directly.

**Goal:**
- Define an LLM output schema like:
  - `{"regime": "choppy", "target_delta": [0.20, 0.25], "target_dte": [7, 14], "aggressiveness": "medium"}`.
- Deterministic code then chooses actual strikes within that window.

**Role clarification:**
- The LLM **must not** be treated as a random parameter generator.
- Its intended role is to:
  - **Reason** about backtest results and market conditions.
  - **Compare** different runs across regimes (bull vs bear, high-vol vs low-vol).
  - Propose **small, structured adjustments** to parameter ranges and risk settings.
  - **Highlight overfitting risks** (e.g. "this config only works in one bull market segment").
- Numeric search (grid/random/Bayesian) remains the primary engine for exploring parameter space; the LLM is a researcher/analyst on top, not a brute-force optimizer.

---

### [F4] Supervisor LLM (multi-bot orchestrator)  
**Priority:** P2–P3  
**Status:** Concept only  

- Future overseer of multiple bots:
  - reads metrics per strategy,
  - proposes:
    - which bots to enable/disable,
    - capital allocation,
    - parameter ranges,
    - new backtests to run.

**Goal:**
- Once multi-bot architecture exists, design a Supervisor module that:
  - reads from DB,
  - outputs a human-reviewable config diff,
  - does **not** directly send trades.

---

## G. UI / UX & Chat

### [G1] Dashboard cleanup & clarity  
**Priority:** P1  
**Status:** Partially rough  

- Home page feels noisy and confusing.
- Some PnL / summary fields not wired clearly.

**Goal:**
- Clear sections:
  - Portfolio summary (NAV, margin, DD).
  - Open positions (grouped by expiry).
  - Recent decisions (timeline).
  - Training vs live vs research flags.
- Fix any missing/wrong PnL wiring.

---

### [G2] Backtest Lab enhancements  
**Priority:** P2  
**Status:** Basic UI present  

**Goal:**
- Add:
  - equity curve visualization,
  - key metrics (CAGR, max DD, win rate, Sharpe/Sortino),
  - side-by-side comparisons (SYNTHETIC vs LIVE_DERIBIT vs REAL_SCRAPER).
- Approach a TradingView-style "strategy report" page.

---

### [G3] Chat UX & persona polish  
**Priority:** P2  
**Status:** Works but basic  

- Chat answers one question at a time; previous answers may disappear.
- Persona can confuse "bot" vs "explainer".

**Goal:**
- Make chat threaded and conversational:
  - retain last N turns in UI,
  - maintain context when referring to "that 91k call".
- Clarify persona:
  - "I am the voice of the system; trading decisions are made by rule/LLM policy modules and risk engine."

---

## H. Testing & QA

### [H1] Integration tests for agent loop  
**Priority:** P1  
**Status:** Not implemented  

**Goal:**
- Create tests that:
  - Mock Deribit client,
  - Run a single agent step,
  - Assert that the expected order(s) would be sent for given state.

---

### [H2] Risk engine boundary tests  
**Priority:** P1  
**Status:** Not implemented  

**Goal:**
- Test exact boundaries:
  - exactly at max margin,
  - exactly at net delta caps,
  - per-expiry limits.
- Ensure behaviour is predictable and logged when limits are hit.

---

### [H3] Crash / restart recovery tests  
**Priority:** P1–P2  
**Status:** Not implemented  

**Goal:**
- Simulate:
  - bot running,
  - crash,
  - restart.
- Confirm `position_tracker` + reconciliation logic restore consistent state and do not duplicate or lose positions.

---

## I. Strategy Factory & Experiment Lab

This section describes the long-term vision for a "strategy factory" – an AI-assisted experiment lab that helps create, tune, and evaluate trading strategies systematically.

### [I1] Strategy Definition as First-Class Object  
**Priority:** P1 (Phase 2–3)  
**Status:** Not implemented  

**Goal:**
- Store strategy configurations as first-class objects in PostgreSQL:
  - `id`, `name`, `description`
  - `base_strategy_type`: "covered_call" | "wheel" | "other"
  - `risk_profile`: "conservative" | "balanced" | "aggressive" | "custom"
  - Parameter ranges: `delta_range`, `dte_range`, `ivrv_min`, `exit_style`, `position_size_rule`
  - LLM settings: `policy_mode`, `hybrid_llm_prob`, `llm_allowed_to_edit_params`
  - Lifecycle status: "draft" | "backtesting" | "testnet_live" | "mainnet_live" | "paused"
- UI wizard: "Create New Strategy" with template selection, profile, parameters, and LLM settings.
- Attach backtest runs and LLM analyses to strategy definitions for traceability.

---

### [I2] Backtest Runs Tied to Strategies  
**Priority:** P1 (Phase 2–3)  
**Status:** Partially present  

**Goal:**
- Formalize `BacktestRun` as "experiments for a given strategy definition":
  - `strategy_id` (FK → StrategyDefinition)
  - `data_source`: "synthetic" | "live_deribit" | "real_scraper"
  - `time_range`: [start_date, end_date]
  - `decision_interval`: "1h" | "4h" | "1d"
  - Metrics: net PnL, PnL %, max drawdown, Sharpe, Sortino, win rate, num_trades
  - Run metadata: timestamp, code version, config snapshot
- UI: strategy detail page with table of past runs, sorted by metrics.

---

### [I3] LLM Analysis & Research Notes  
**Priority:** P2 (Phase 3)  
**Status:** Not implemented  

**Goal:**
- Attach AI "research reports" to strategies:
  - `id`, `strategy_id`
  - `runs_considered`: list of run IDs
  - `summary_text`: human-readable explanation
  - `suggested_changes`: structured JSON (e.g. tighter delta range, higher IVRV min)
  - `accepted`: whether changes were applied to create a new strategy version
- The LLM reviews backtest results and proposes **small, interpretable changes**, not random parameter sampling.
- Example output:
  ```json
  {
    "delta_range": [0.20, 0.30],
    "dte_range": [7, 14],
    "ivrv_min": 1.3,
    "notes": [
      "2022 bear regime shows too much drawdown; tighten delta",
      "IVRV often under 1.1 in quiet regimes; skip those"
    ]
  }
  ```

---

### [I4] Optimization Loop: Three Levels  
**Priority:** P2 (Phase 2–3)  
**Status:** Not implemented  

The system supports three levels of automation, rolled out incrementally:

**Level 1 – Manual + AI Advisor (short-term, low-risk):**
- Run 3–10 backtests manually.
- Click "Ask AI for suggestions" in UI.
- LLM reviews metrics, proposes small parameter changes with explanation.
- User decides whether to apply changes.
- Human remains fully in the loop.

**Level 2 – Semi-auto "Auto-Tune up to N runs":**
- Define parameter search space (delta bounds, DTE bounds, IVRV range).
- Use numeric search (random/Bayesian) to pick next parameter set.
- Run backtest, store metrics.
- LLM provides commentary after each run: "Are we overfitting? Which region looks promising?"
- Stop when: N runs reached, no improvement after X iterations, or user clicks "Stop".
- Mark best config as "recommended" with LLM analysis attached.

**Level 3 – Mostly-auto (bounded sandbox):**
- System automatically runs initial sweep, LLM refines, stops when metrics plateau.
- Presents: "Here is the best config found; here is performance across train vs validation; do you want to start this on testnet?"
- Hard limits:
  - Max experiments per strategy.
  - Time-based train/validation split enforced.
  - No auto-promotion to live without manual approval.

---

### [I5] Supervisor LLM for Multi-Strategy  
**Priority:** P3  
**Status:** Concept only  

**Goal:**
- Future overseer of multiple strategies:
  - Reads metrics per strategy from DB.
  - Proposes which strategies to enable/disable.
  - Suggests capital allocation across strategies.
  - Schedules tuning runs (e.g. overnight batch optimization).
  - Outputs human-reviewable config diff; does **not** directly send trades.
- Autonomy is bounded: all changes require human approval before execution.

---

## Design Principles for Strategy Optimization

1. **LLM as analyst, not optimizer:** The LLM reasons about results, compares regimes, and proposes structured changes. It does not randomly sample parameters.

2. **Numeric search for exploration:** Grid search, random search, or Bayesian optimization handle the heavy lifting of parameter space exploration.

3. **Guard against overfitting:** All optimization work must respect [E3] Overfitting Mitigation guidelines – train/validation splits, regime slicing, simplicity bias.

4. **Incremental autonomy:** Start with manual + AI advisor, graduate to semi-auto with limits, and only later consider mostly-auto in a sandboxed environment.

5. **Audit trail:** Every run, every LLM suggestion, and every parameter change is logged and traceable.

---
