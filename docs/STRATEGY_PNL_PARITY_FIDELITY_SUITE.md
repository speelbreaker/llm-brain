# Strategy PnL Parity + Synthetic Fidelity Suite (North Star)

## Goal
Create a **Strategy PnL Parity Suite** that runs canonical options strategies on:

- **Live-replay historical** (harvested Deribit snapshots; as close to reality as possible)
- **Synthetic-replay historical** (synthetic universe)

…and produces a **Synthetic Fidelity Score (0–100)** that:

- quantifies how close synthetic is to live on the dimensions that matter for options PnL, and
- gates whether backtests are **TRUSTED** or **UNTRUSTED**, and optionally gates live trading size/permission.

## Definition of Done (Hard Acceptance Criteria)
Running:

- `python -m scripts.run_fidelity_suite --start ... --end ...`

produces:

- `fidelity_report.json` (machine readable)
- `fidelity_report.md` (human readable)
- (optional) `fidelity_report.html`

Outputs include:

- component scores + overall score
- per-strategy parity metrics (distributions + tail risk + drawdowns)

The app exposes at least one endpoint returning the latest fidelity result, with an endpoint-level test.

---

## Architecture
### Core Design Principle
**One backtest engine, two market adapters.**

Everything else is noise.

We will implement a single **strategy runner** that accepts a `MarketReplay` interface:

- `LiveReplayMarket` (replays harvested Deribit snapshots)
- `SyntheticReplayMarket` (replays synthetic-generated snapshots for the same timestamps)

Both markets must expose the same fields:

- underlying spot price at time `t`
- option chain snapshot at time `t` (marks, IV, greeks if available)
- fill/slippage model (even if minimal first pass)

This ensures parity testing is apples-to-apples.

---

## Part A — Strategy PnL Parity Suite
### A1) Canonical Strategy Set (Minimum Viable)
Pick **6** strategies that cover Greeks + tail behavior:

1. **Covered Call** (short call vs spot)
2. **Cash-Secured Put** (short put with cash collateral)
3. **Short Strangle** (short put + short call)
4. **Put Spread (credit)** (short put + long lower put)
5. **Call Spread (debit)** (long call + short higher call)
6. **Calendar** (long farther-dated, short nearer-dated, same strike/delta bucket)

Each strategy is frozen as a measurement instrument (not optimized) and defined by `StrategySpec`:

- entry schedule (e.g. daily 00:00 UTC)
- selection rule (delta target + DTE target + liquidity filter)
- exit rule (hold-to-expiry / TP / SL / roll)
- sizing rule (fixed notional, % collateral, or risk budget)

### A2) Standardize the Fill Model
PnL parity will be garbage if fills differ.

We standardize fills and apply the **same** model to live + synthetic.

- Phase 1: fill at mark price ± fixed bps (config)
- Phase 2: spread model based on moneyness + tenor + regime
- Phase 3: stress slippage multiplier when spot move > X sigma

### A3) Suite Outputs
For each strategy:

**Trade-level metrics**
- win rate
- avg win, avg loss
- profit factor
- mean/median trade return
- average DTE held
- number of trades

**Equity curve / risk**
- CAGR (or average monthly return)
- volatility of returns
- max drawdown
- time-to-recovery
- worst day / worst week
- tail risk: 1% VaR and 1% Expected Shortfall

**Distribution comparisons (live vs synthetic)**
- difference in key quantiles: 5%, 25%, 50%, 75%, 95%, 99%
- KS statistic (or Wasserstein distance) on trade returns
- drawdown distribution distance

---

## Part B — Synthetic Fidelity Score
### B1) Score Components (What Matters)
Overall score is a weighted composite:

1. **Underlying Returns Fidelity** (~20%)
   - log return distribution (kurtosis + tail quantiles)
   - jump frequency (|return| > N sigma)
   - volatility clustering (autocorr of |returns| or RV)

2. **Realized Vol Fidelity** (~10%)
   - rolling RV (7d/14d/30d): level + change dynamics

3. **IV Surface Level Error** (~20%)
   - bucket by tenor (7D/14D/30D) × delta (10d/25d/50d)
   - MAE/RMSE of IV per bucket

4. **IV Surface Dynamics Error** (~15%)
   - compare ∆IV per bucket (day-over-day or snapshot-to-snapshot)
   - compare skew changes

5. **Spot–IV Coupling** (~15%)
   - corr(spot return, ∆IV)
   - conditional: spot drop > X% ⇒ avg ∆IV + skew change

6. **Strategy PnL Parity Score** (~20%)
   - distribution distances for each canonical strategy
   - tail-loss parity (ES, worst-day)
   - drawdown parity

### B2) Mapping Metrics → 0–100
Per metric:

- `error_ratio = metric_error / tolerance`
- `score_metric = clamp(100 * exp(-k * error_ratio), 0, 100)`

Then:

- `score_component = weighted_avg(metric_scores)`
- `overall_score = weighted_avg(component_scores)`

### Gates
- **TRUSTED**: overall ≥ 80 AND StrategyParity ≥ 75 AND TailParity ≥ 70
- **WARNING**: 65–80 (allow backtests but label “UNCERTAIN”)
- **UNTRUSTED**: < 65 (block research conclusions; optionally reduce live risk)

---

## Part C — Calibration Integration (Make It Real)
### C1) Versioned Calibration Bundles
Every calibration run produces an immutable bundle:

- timestamp
- parameters (IV multipliers, regime params, skew params, etc.)
- diagnostics summary
- fidelity score snapshot

Stored as:

- `calibration_runs/{YYYYMMDD_HHMMSS}/bundle.json`
- `calibration_runs/{...}/fidelity_report.json`
- (optional) `calibration_runs/{...}/parity_results.parquet`

### C2) Promotion Logic
Only promote calibration to active if gates pass.

Otherwise keep last good calibration active.

### C3) Drift Detection
Even between calibrations:

- if live IV buckets diverge from synthetic expected by > threshold → degrade score/warn
- if live spot–IV coupling shifts → warn

---

## Part D — UI + API (MVP)
### Endpoints
- `GET /calibration/fidelity/latest`
  - returns overall score + component scores + timestamp + TRUSTED/WARNING/UNTRUSTED
- `GET /calibration/fidelity/history?limit=30`
  - returns last N runs summary
- (optional) `GET /calibration/fidelity/report/{run_id}`
  - returns full report

### UI Panel
- latest score badge
- component bar chart/table
- per-strategy parity table (live vs synthetic: return, maxDD, ES1%)

---

## Part E — Testing (Non-negotiable)
### Unit Tests (Fast)
- metric calculators (VaR/ES, drawdown, KS/Wasserstein wrapper)
- bucketing logic (tenor × delta)
- scoring mapping function
- aggregation handling (weights sum to 1, missing buckets handled)

### Integration Tests
- run suite on tiny known dataset (fixture snapshots)
- assert deterministic output ranges

### Endpoint Tests (Required)
- `GET /calibration/fidelity/latest` returns 200 and schema fields exist
- `GET /calibration/fidelity/history` returns expected list shape
- failure mode when no runs exist (404 or empty) — choose and test

---

## Execution Plan (TOC Style)
### Step 1 — Identify Constraint
Synthetic fidelity.

### Step 2 — Exploit Constraint (Build Parity Suite First)
**P0 (must do first)**

- Define `MarketReplay` interface
- Implement `LiveReplayMarket` + `SyntheticReplayMarket`
- Implement canonical `StrategySpec`s (6)
- Implement parity runner + metrics + report JSON
- Implement scoring + gates

### Step 3 — Subordinate Everything Else
- Backtest UI displays TRUSTED/WARNING/UNTRUSTED
- Backtest results include fidelity run id
- “Expert bot” backtest refuses to claim success if UNTRUSTED

### Step 4 — Elevate Constraint
- Add IV surface dynamics scoring
- Add spot–IV coupling scoring
- Improve fill/slippage model
- Add drift detection
- Add nightly automation + promotion logic

### Step 5 — Repeat
New strategy logic must be measured via the suite.

---

## Deliverables Checklist
### P0 Deliverables
- `src/fidelity/` module:
  - `market_replay.py` (interfaces + two adapters)
  - `strategies_canonical.py` (6 StrategySpecs)
  - `parity_runner.py`
  - `metrics.py`
  - `scoring.py`
  - `reporting.py`
- `scripts/run_fidelity_suite.py`
- API endpoints + endpoint tests
- Stored artifacts per run

### P1 Deliverables
- UI panel
- promotion logic for calibration bundles
- drift detection warnings
