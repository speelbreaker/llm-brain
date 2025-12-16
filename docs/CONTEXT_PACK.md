# Options Trading Agent — Context Pack (Canonical)

**Purpose:** Single source of truth for humans + AI Builder about what exists, how it’s wired, and what “done” means.  
**Rule:** Prefer *facts about the repo* over opinions. If something is not verified in code/config, label it **Planned**.

**Last updated:** 2025-12-16

---

## 0) How to Use This File

When you (or an AI Builder) are implementing changes:

1) **Repo-first:** read the relevant files before proposing edits.  
2) **Keep invariants sacred:** risk rails are not “nice to have.”  
3) **No “Done” claims without evidence:** if you claim something is implemented, reference a file path + behavior.

---

## 1) What the App Does Today (Current Reality)

- Automates **BTC/ETH covered-call trading** on Deribit **testnet** with **two decision paths**:
  - **Rule-based policy** (deterministic)
  - **LLM co-pilot** that returns structured JSON actions
- Runs a continuous **agent loop** that:
  - builds market/portfolio state
  - selects an action
  - enforces risk limits
  - executes (dry-run by default unless enabled)
  - logs outcomes for audit/training
- Provides a **FastAPI web dashboard** (tabs typically include Live Agent, Backtesting Lab, Calibration, and Chat/Explain).
- Includes a **backtesting engine**, synthetic pricing tools, and training data exporters (for RL/ML experiments).

---

## 2) Architecture Map (Modules & Key Flows)

### Entry Points
- `agent_loop.py` — orchestrates the live loop
- `src/web_app.py` — serves the dashboard (and may boot/host the agent loop)

### Core Wiring (high level)
- **Config:** `src/config.py` (Pydantic settings)
- **State construction:**
  - Live: `src/state_builder.py` + shared `src/state_core.py` (`AgentState`)
  - Backtests: `src/backtest/state_builder.py`
- **Decision (“brains”):**
  - Rule-based: `src/policy_rule_based.py`
  - LLM brain: `src/agent_brain_llm.py` (OpenAI decision JSON)
- **Strategy layer:**
  - Strategy interface + current covered-call implementation: `src/strategies/covered_call.py`
  - Registry: `src/strategies/registry.py`
- **Risk & execution:**
  - `src/risk_engine.py` enforces invariants
  - `src/execution.py` places/simulates Deribit orders
- **Market data:**
  - Live authenticated: `src/deribit_client.py`
  - Backtest/public: `src/backtest/deribit_client.py`
  - Context aggregation: `src/market_context.py`

### Backtest Stack (covered calls)
- `src/backtest/covered_call_simulator.py`
- `src/backtest/pricing.py`
- `src/backtest/manager.py`
- `src/backtest/config_presets.py`
- `src/backtest/types.py`

### Training / Utilities / Ops
- `src/training_policy.py`, `src/training_profiles.py`
- `src/calibration.py`
- `src/synthetic_skew.py`
- `src/chat_with_agent.py`
- `scripts/*` helpers

### Data + Logs
- Decision logs: `logs/` (JSONL append-only)
- Datasets/model artifacts: `data/`

> **Planned (preferred direction):** modularize `src/web_app.py` into a `src/web/` package with routers + dashboard templates (see backlog).

---

## 3) Strategy Catalog

### 3.1 CoveredCallStrategy (current default)
- A “wrapper strategy” that can run **rule-based** or **LLM** mode depending on settings.
- Training profiles exist (conservative/moderate/aggressive) in `src/training_profiles.py`.

### 3.2 GregBot (Magadini VRP Harvester) — **Bundle, not a single strategy**
**Key point:** “GregBot” should be treated as a **selector + management framework** that can fire *different strategies* based on sensors.  
This is exactly why your UI/backtest results must record **which sub-strategy fired, when, and P&L attribution**.

**Example strategy codes (bundle):**
- `STRATEGY_A_STRADDLE` — ATM straddle (VRP + regime dependent)
- `STRATEGY_A_STRANGLE` — OTM strangle
- `STRATEGY_B_CALENDAR` — calendar spread (term structure signal)
- `STRATEGY_C_SHORT_PUT` — accumulation / short put (bullish + VRP)
- `STRATEGY_D_IRON_BUTTERFLY` — defined risk for extreme IV rank
- `STRATEGY_F_BULL_PUT_SPREAD` — defined risk bullish
- `STRATEGY_F_BEAR_CALL_SPREAD` — defined risk bearish
- `NO_TRADE` — safety filter triggered / no valid setup

**Key sensors (must be computed + logged):**
- `vrp_30d`, `vrp_7d` (Implied − Realized volatility premium)
- `chop_factor_7d` (range-bound vs trending proxy)
- `adx_14d` (trend strength)
- `iv_rank_6m` (IV percentile)
- `skew_25d` (put/call skew)
- `term_structure_spread` (tenor IV spread)

> **Important:** Numeric thresholds (e.g., VRP > 15) are *policy knobs*, not guaranteed truth. If thresholds exist, they must live in settings/config and be visible in the UI.

### 3.3 Planned plug-ins (architecture supports; not necessarily implemented)
Wheel, CrashHedge, and Spread strategies added via the `Strategy` interface + registry.

---

## 4) Data Sources & Modes (Synthetic vs Live)

### Primary Sources
- **Live trading data (testnet):** via `src/deribit_client.py` (authenticated)  
  balances, positions, option chains, execution, etc.
- **Historical / public for backtests:** via `src/backtest/deribit_client.py` (mainnet candles + public option data when available)
- **Synthetic pricing & smiles:** Black–Scholes in `src/backtest/pricing.py` + skew/regime helpers like `src/synthetic_skew.py`
- **Calibration:** `src/calibration.py` + scripts like `scripts/compare_synthetic_vs_live.py` to compare synthetic vs live IV/smiles

### Backtest / Scan modes (conceptual contract)
- `BacktestType.GENERIC` — full P&L simulation (positions, rolls, exits)
- `BacktestType.GREG_SELECTOR` — selector-only analysis (which sub-strategy would fire; fast)

### Selector Data Sources (contract the UI should expose)
- `SelectorDataSource.SYNTHETIC` — synthetic universe
- `SelectorDataSource.HARVESTER` — historical harvested snapshots (if present)
- `SelectorDataSource.LIVE` — current live snapshot

### IV sourcing modes (contract the selector should support)
- `iv_mode="synthetic"` — synthetic IV surface
- `iv_mode="live"` — Deribit live IV
- `iv_mode="hybrid"` — live IV with synthetic fallback

---

## 5) Risk Invariants (Hard Rules)

These are “stop-the-line” constraints. If violated, the system must choose `DO_NOTHING` and log why.

- **Kill switch:** global halt blocks all non-`DO_NOTHING` actions
- **Portfolio validity:** no trading when equity is missing/zero (guards missing private API / unfunded accounts)
- **Margin discipline:** block at/above `max_margin_used_pct`; warn before limit (e.g., 90% of max)
- **Delta guardrail:** `abs(net_delta) <= max_net_delta_abs`
- **Daily drawdown cap:** optional daily stop if breached (UTC day)
- **Per-expiry exposure:** projected short call size per expiry must remain covered + under `max_expiry_exposure`
- **Liquidity gates:** min open interest + max spread requirements for candidates
- **Training bypass:** if `is_training_on_testnet` is enabled, risk checks may be bypassed (must be loudly logged)

> **Security requirements (policy):** never log secrets; explicit timeouts on external calls; webhook signature verification where used; return 503 when required secrets/config are missing.

---

## 6) Observability & Artifacts

Minimum viable observability:
- Every decision log must include: **timestamp, market context, chosen action, rationale, risk pass/fail reasons, execution result**
- For GregBot: log **sub-strategy code**, sensor values, and why alternatives were rejected
- Keep **append-only JSONL** for traceability; DB can be added later, but don’t lose the recorder

---

## 7) Current Backlog (Top Priorities)

1) **Selector/UI config correctness:** when GregBot is selected, hide/disable generic exit knobs (hold-to-expiry/take-profit/roll) that don’t apply; show Greg-specific controls instead.  
2) **GregBot attribution:** backtests must report which sub-strategy fired, when, count, and P&L per sub-strategy.  
3) **Selector scan sources:** extend selector scans beyond synthetic to **LIVE** and **HARVESTER** where available, including `iv_mode` handling.  
4) **Database-first position/decision store:** SQLite/Postgres with JSONL as recorder (dual-write or migration path).  
5) **Watchdogs & process health:** heartbeat, rate-limit guards, safe-mode, kill-switch enforcement, restart policy.  
6) **Position reconciliation & roll rules:** stronger tracker with audit trail; fewer silent mismatches.  
7) **LLM safety rails:** strict schema validation, action whitelist, better fallbacks.  
8) **Synthetic vs live realism:** tighten optimism factor, extend calibration, improve surface fit.  
9) **Reduce coupling:** config injection, fewer globals, easier testing.  
10) **Web app modularization:** split `src/web_app.py` into routers/templates to reduce “God module” coupling.

---

## 8) Definition of Done (Non-Negotiable)

A change is “done” only if all are true:

- **Functionality:** matches the described behavior; strategy/registry wiring updated where relevant.
- **Risk:** all actions still pass `risk_engine` invariants; kill-switch remains effective.
- **UI:** dashboard endpoints/templates updated so the user can see and operate the feature (no “backend-only” delivery).
- **Observability:** logs show action + rationale + risk reasons + execution result; errors are operator-readable.
- **Docs:** update this Context Pack if behavior/interfaces/paths changed.

### Tests (Mandatory Gates)
- `python -m pytest -q` passes
- **Every new endpoint ships with at least one endpoint-level test** (FastAPI `TestClient`)  
  - include success path + at least one failure case (e.g., 400/401/403/404/422/503)
- If an endpoint changes behavior/shape, update tests to prove it.

---

## 9) Builder Prompt Standard (Required)

Every Builder prompt must include:
- **Read-first steps** (list tree, open key modules, summarize before editing)
- **Scope + files to touch**
- **Acceptance criteria** (including UI changes + logging)
- **Tests required**
- **Rollback plan** (how to revert safely)

