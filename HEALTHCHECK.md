# Health Check & Smoke Tests

This document lists quick commands to verify that core parts of the system are working. Run these after making changes or when troubleshooting.

---

## Quick Smoke Tests

### 1. Live Agent Dry-Run Test

**What it does**: Connects to Deribit testnet, fetches current market data, makes a rule-based decision, and checks with the risk engine. No actual trades are placed.

**When to run**: After code changes to the agent loop, policy, or risk engine.

**Command**:
```bash
bash scripts/smoke_live_agent.sh
```

**What success looks like**:
```
=== Smoke Test: Live Agent (1 iteration, dry-run) ===

Connecting to Deribit testnet...
Building agent state...
  Underlyings: ['BTC', 'ETH']
  Spot prices: {'BTC': 89474.36, 'ETH': 3042.06}
  Candidates: 5
  Equity: $11989422.89

Making rule-based decision...
  Action: DO_NOTHING
  Reasoning: [TRAINING] No new candidates...

Checking risk engine...
  Allowed: True

=== SMOKE TEST PASSED ===
```

---

### 2. Backtesting Lab Minimal Test

**What it does**: Runs a 7-day backtest using synthetic Black-Scholes pricing. Tests the core simulation engine.

**When to run**: After changes to the backtest engine, simulator, or scoring logic.

**Command**:
```bash
bash scripts/smoke_backtest.sh
```

**What success looks like**:
```
=== Smoke Test: Backtesting Engine (7-day synthetic) ===

Running backtest: 2024-09-24 to 2024-10-01
Underlying: BTC, Pricing: synthetic_bs

Decision points: 8

=== Results ===
  Trades executed: 8
  Final PnL: $-12495.56
  Win rate: 12.5%
  Max drawdown: 1018.19%

=== SMOKE TEST PASSED ===
```
(Note: PnL and metrics will vary based on the date range tested)

---

### 3. Training Data Export Test

**What it does**: Runs a tiny 3-day backtest and exports candidate-level training data to CSV. Verifies the training data pipeline.

**When to run**: After changes to the training dataset module or candidate export logic.

**Command**:
```bash
bash scripts/smoke_training_export.sh
```

**What success looks like**:
```
=== Smoke Test: Training Data Export ===

Running mini-backtest: 2024-09-28 to 2024-10-01
Decision points: 4
Decision steps collected: 4
Candidate examples: 28

=== Output file: data/smoke_test_training.csv ===

decision_time,underlying,spot,instrument,strike,dte,delta,score...
2024-09-28T00:00:00+00:00,BTC,65924.0,BTC-05OCT24-68000-C,68000...
...

=== SMOKE TEST PASSED ===
(Temp file cleaned up)
```

---

### 4. LLM Decision Test (Research Mode)

**What it does**: Asks the LLM brain to make a trading decision based on current market state. Requires OpenAI API key.

**When to run**: After changes to the LLM brain or prompt engineering. (Optional - requires OpenAI API)

**Command**:
```bash
python -c "
from src.config import settings
from src.deribit_client import DeribitClient
from src.state_builder import build_agent_state
from src.agent_brain_llm import choose_action_with_llm

print('Building state...')
client = DeribitClient()
state = build_agent_state(client)

print(f'Candidates: {len(state.candidate_options)}')
print('')
print('Asking LLM for decision...')
action = choose_action_with_llm(state, state.candidate_options)
print(f'Action: {action.get(\"action\")}')
print(f'Reasoning: {action.get(\"reasoning\", \"N/A\")[:200]}')
print('')
print('=== LLM TEST PASSED ===')
"
```

**What success looks like**:
- Shows "Building state..." and candidate count
- Shows "Asking LLM for decision..."
- Displays the action (DO_NOTHING, OPEN_COVERED_CALL, etc.)
- Shows reasoning from the LLM
- Ends with "LLM TEST PASSED"

---

### 5. Web App + API Status Check

**What it does**: Verifies the FastAPI web app is running and the API endpoints respond correctly.

**When to run**: After starting the web app, or if the dashboard seems unresponsive.

**Commands** (run these in the Shell tab while the web app is running):

```bash
# Check basic health
curl -s http://localhost:5000/health

# Check agent status
curl -s http://localhost:5000/status

# Check recent decisions (may be empty if no trades yet)
curl -s http://localhost:5000/api/agent/decisions
```

**What success looks like**:

1. `/health` returns HTTP 200 with:
   ```json
   {"status": "healthy", "service": "options-trading-agent"}
   ```

2. `/status` returns HTTP 200 with JSON containing:
   - `running` (true/false)
   - `iterations` (number)
   - `mode` (research/production)
   - `dry_run` (true/false)

3. `/api/agent/decisions` returns HTTP 200 with JSON containing:
   - `decisions` (list of recent decisions, may be empty)
   - `count` (number of decisions)

**Or use the smoke script**:
```bash
bash scripts/smoke_web_api.sh
```

---

## Technical Debt

The following items have been identified as needing cleanup. They are marked with `TODO` comments in the code.

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

| Item | Risk if Unfixed | Risk of Refactoring |
|------|-----------------|---------------------|
| **Duplicate scoring functions** (`policy_rule_based.py` vs `covered_call_simulator.py`) | Live agent and backtester may score candidates differently, leading to strategy drift where backtest results don't match live behavior. | Could break both backtests and live trading if the unified scorer has bugs or different default parameters. |
| **IVRV calculated in multiple places** (state_builder, simulator, training_profiles) | Inconsistent IVRV values could cause the agent to make different decisions than the backtester, making training data unreliable. | Minor risk if done carefully; mainly need to update all call sites to use the centralized version. |

---

### P1 – Worth Fixing (Not Urgent)

| Item | Risk if Unfixed | Risk of Refactoring |
|------|-----------------|---------------------|
| **Duplicate state builders** (live vs backtest) | Makes it harder to add new features—changes must be made in two places, increasing the chance one gets forgotten. | Moderate risk; the live and backtest builders have different data sources, so a hasty merge could break either mode. |
| **Duplicate Deribit clients** (trading vs public data) | Maintenance burden; any bug fix or API change must be applied twice. | Low-to-moderate risk; the clients serve different purposes (auth vs no-auth), so they can share base code without full merge. |
| **Duplicate expiry parsing** (`_parse_expiry` vs `parse_deribit_expiry`) | Minor—mostly code clarity. Same logic in two places means potential for subtle date parsing bugs. | Very low risk; simple utility functions that can be extracted without touching core logic. |
| **No unit tests** | Regressions go undetected until they cause visible problems in production or backtests. | Time investment rather than code risk; tests should be added incrementally without modifying existing code. |

---

### P2 – Nice-to-Have / Cosmetic

| Item | Risk if Unfixed | Risk of Refactoring |
|------|-----------------|---------------------|
| **Dead code: `main.py`** | Confuses new developers who might think it's the entry point. Zero functional risk. | No risk—just delete it. |
| **Dead code: `server.py`** | Same confusion issue; someone might try to run it instead of the FastAPI app. | No risk—just delete it after confirming nothing imports it. |
| **Dead code: `backtest/env_simulator.py`** | Misleading file in the wrong folder. No functional risk. | No risk—delete the entire `backtest/` folder (not `src/backtest/`). |
| **Large file: `web_app.py` (2500+ lines)** | Harder to navigate and maintain. No correctness issue. | Moderate effort; extracting templates and routes requires careful testing of all UI features. |
| **Large file: `covered_call_simulator.py` (1000+ lines)** | Harder to understand; encourages monolithic changes. | Some risk; splitting could introduce import issues or break the simulation if done incorrectly. |
| **Large file: `config.py` (260 lines)** | Minor—still manageable. Could benefit from grouping. | Low risk; sub-models can be introduced without changing behavior. |

---

### Summary: Recommended Order

1. **P0 first**: Fix scoring and IVRV duplication to ensure backtest-live consistency.
2. **P1 next**: Unify state builders and clients when adding new features.
3. **P2 whenever convenient**: Delete dead code (easy wins), split large files during related work.

---

### Dead Code (Reference Table)

| File | Issue |
|------|-------|
| `main.py` | Placeholder file that just prints "Hello". Not used by the app. |
| `server.py` | Old Flask server superseded by `src/web_app.py` (FastAPI). |
| `backtest/env_simulator.py` | RL-environment stub superseded by `src/backtest/covered_call_simulator.py`. |
| `backtest/__init__.py` | Root backtest folder is legacy; real code is in `src/backtest/`. |

### Duplicated Logic (Reference Table)

| Duplication | Files Involved | Suggestion |
|-------------|----------------|------------|
| **Candidate scoring** | `src/policy_rule_based.py:score_candidate()` vs `src/backtest/covered_call_simulator.py:_score_candidate()` | Create a shared `src/scoring.py` module with configurable scoring logic. |
| **State building** | `src/state_builder.py:build_agent_state()` vs `src/backtest/state_builder.py:build_historical_state()` | Unify with a shared interface that handles both live and historical data sources. |
| **Deribit clients** | `src/deribit_client.py:DeribitClient` vs `src/backtest/deribit_client.py:DeribitPublicClient` | Add public endpoints to main client, or create shared base class. |
| **Expiry parsing** | `src/state_builder.py:_parse_expiry()` vs `src/backtest/state_builder.py:parse_deribit_expiry()` | Extract to a shared utility module (e.g., `src/utils/deribit_utils.py`). |
| **IVRV calculation** | Computed in `src/state_builder.py`, `src/backtest/covered_call_simulator.py`, `src/training_profiles.py` | Centralize IVRV calculation in one place. |

### Large Files That Could Be Split

| File | Lines | Suggestion |
|------|-------|------------|
| `src/web_app.py` | ~2,500 | Move HTML templates to separate files in `templates/`. Extract API routes into `src/api/` modules. |
| `src/backtest/covered_call_simulator.py` | ~1,000 | Split into: `simulator.py` (core), `scoring.py` (candidate scoring), `features.py` (feature extraction). |
| `src/config.py` | ~260 | Consider grouping related settings into sub-models (RiskConfig, TrainingConfig, BacktestConfig). |

### Missing Test Coverage

- No unit tests exist for any module
- No integration tests for the agent loop
- No regression tests for the backtest engine

Consider adding tests in a `tests/` directory using pytest.

---

## Running All Smoke Tests

To run all smoke tests in sequence:

```bash
echo "=== Running all smoke tests ==="
bash scripts/smoke_live_agent.sh && \
bash scripts/smoke_backtest.sh && \
bash scripts/smoke_training_export.sh && \
bash scripts/smoke_web_api.sh && \
echo "" && \
echo "=== ALL SMOKE TESTS PASSED ==="
```

**Note**: The web API test (`smoke_web_api.sh`) requires the web app to be running.

---

*Last updated: December 2024*
