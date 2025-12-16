# Options Trading Agent - Context Pack

> Canonical reference for developers and AI agents. Keep this under 4 pages.

---

## 1. What the App Does Today (Current Reality)

**Purpose:** Automated BTC/ETH covered call trading framework for Deribit testnet.

**Current State:**
- **Research/Training Mode:** Active. Generates training data, runs backtests, explores strategies with epsilon-greedy exploration.
- **Live Trading:** Testnet only. Strong safety gates with tiny-size guardrails.
- **Production/Mainnet:** Not deployed. Planned for future.

**Core Capabilities:**
- Rule-based and LLM-powered decision-making
- Real-time market monitoring via Deribit API
- Web dashboard for monitoring, backtesting, and interaction
- Telegram bot for code review and PR notifications
- GregBot VRP Harvester strategy evaluation

---

## 2. Architecture Map

```
┌─────────────────────────────────────────────────────────────────┐
│                         Web Layer                                │
│  src/web_app.py (entrypoint)                                    │
│  src/web/dashboard.py (UI)                                      │
│  src/web/routes_*.py (API endpoints)                            │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐   ┌──────────────────┐   ┌───────────────┐
│   Strategies  │   │   Backtesting    │   │  Data Layer   │
│  greg_selector│   │  CoveredCallSim  │   │  harvester/   │
│  policy_rule  │   │  manager.py      │   │  deribit/     │
└───────────────┘   │  selector_scan   │   │  synthetic/   │
        │           └──────────────────┘   └───────────────┘
        │                     │
        ▼                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Core Modules                               │
│  config.py         risk_engine.py       position_tracker.py    │
│  deribit_client.py execution.py         market_context.py      │
│  calibration*.py   state_builder.py     logging_utils.py       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     External Services                           │
│  Deribit API (testnet)   OpenAI   PostgreSQL   GitHub  Telegram│
└─────────────────────────────────────────────────────────────────┘
```

**Key Modules:**

| Directory | Purpose |
|-----------|---------|
| `src/strategies/` | Strategy selectors (GregBot, rule-based policies) |
| `src/backtest/` | CoveredCallSimulator, selector scans, manager |
| `src/synthetic/` | Synthetic universe generation, regime modeling |
| `src/harvester/` | Live data collection from Deribit |
| `src/bots/` | Bot analysis and strategy evaluations |
| `src/supervisor/` | PR verification and auto-fix service |
| `src/web/` | Dashboard UI and API routes |

---

## 3. Strategy Catalog

### GregBot (Magadini VRP Harvester)

GregBot is a **bundle of 7+ strategies**, not a single strategy. It uses a decision waterfall to select the optimal strategy based on market sensors.

| Strategy Code | Name | Trigger Condition |
|---------------|------|-------------------|
| `STRATEGY_A_STRADDLE` | ATM Straddle | High VRP (>15%), low chop (<0.6), low ADX (<20), neutral skew |
| `STRATEGY_A_STRANGLE` | OTM Strangle | Good VRP (>10%), low chop (<0.8), low ADX (<30), neutral skew |
| `STRATEGY_B_CALENDAR` | Calendar Spread | Term structure spread >5%, favorable front RV/IV ratio |
| `STRATEGY_C_SHORT_PUT` | Accumulation (Short Put) | Bullish trend, expensive puts, positive VRP |
| `STRATEGY_D_IRON_BUTTERFLY` | Iron Butterfly | Extreme IV rank (>80%), high VRP. Defined risk only. |
| `STRATEGY_F_BULL_PUT_SPREAD` | Bull Put Spread | Oversold + fear skew + positive VRP |
| `STRATEGY_F_BEAR_CALL_SPREAD` | Bear Call Spread | Overbought + FOMO skew + positive VRP |
| `NO_TRADE` | No Trade | Safety filter triggered or no valid setup |

**Key Sensors:**
- `vrp_30d`, `vrp_7d` (Implied - Realized Volatility)
- `chop_factor_7d` (RV/IV ratio)
- `adx_14d` (trend strength)
- `iv_rank_6m` (IV percentile)
- `skew_25d` (put/call skew)
- `term_structure_spread` (IV term structure)

---

## 4. Data Sources

| Source | Description | Usage |
|--------|-------------|-------|
| **Synthetic Universe** | Generated price paths with AR(1) IV dynamics, regime modeling, and Greg-sensor clusters | Backtesting, selector frequency scans |
| **Live Chain (Deribit)** | Real-time option chains from Deribit testnet | Live trading, position monitoring |
| **Live IV** | Current market implied volatility from Deribit | IV mode='live' in selector scans |
| **Harvester Historical** | Parquet snapshots from data harvester (`data/live_deribit/`) | Historical analysis, harvester mode backtests |
| **Hybrid IV** | Live IV with synthetic fallback when unavailable | IV mode='hybrid' |

**Backtest Modes:**
- `BacktestType.GENERIC` - Full covered call simulation with P&L tracking
- `BacktestType.GREG_SELECTOR` - Synchronous selector-only analysis (immediate results)

**Selector Data Sources:**
- `SelectorDataSource.SYNTHETIC` - Synthetic universe
- `SelectorDataSource.HARVESTER` - Historical harvester data
- `SelectorDataSource.LIVE` - Current market snapshot

---

## 5. Risk Invariants (Hard Rules - Never Violated)

These safety rails are enforced by `risk_engine.py` and cannot be bypassed:

| Invariant | Default | Description |
|-----------|---------|-------------|
| `max_margin_used_pct` | 80% | Maximum margin utilization |
| `max_net_delta_abs` | 5.0 | Maximum absolute net delta exposure |
| `max_expiry_exposure` | 0.3 BTC/ETH | Maximum exposure per expiry |
| `daily_drawdown_limit_pct` | 0% (disabled) | Daily peak-to-trough loss limit |
| `kill_switch_enabled` | False | Global trading halt (blocks all actions when True) |

**Additional Safety Gates:**
- Webhook endpoints return 503 when required secrets are missing
- Signature verification enforced on all webhooks
- Secrets never logged or returned to clients
- All external API calls use explicit timeouts and retry logic

---

## 6. Current Backlog (Top 10)

| # | Item | Status |
|---|------|--------|
| 1 | Mainnet deployment preparation | Planned |
| 2 | Position management UI for GregBot | In Progress |
| 3 | Delta hedging engine integration | Implemented |
| 4 | Multi-strategy concurrent execution | Implemented |
| 5 | Improved IV calibration pipeline | Implemented |
| 6 | Training data generation for ML/RL | Active |
| 7 | Backtest Lab per-strategy breakdown | Done |
| 8 | Telegram PR Supervisor auto-fix | Implemented |
| 9 | AI Steward project brain | Implemented |
| 10 | Production database safety gates | Implemented |

---

## 7. Definition of Done Checklist

Before marking any work complete, verify:

### Code Quality
- [ ] Full test suite passed: `python -m pytest -q`
- [ ] Static integrity passed: `python -m compileall -q src`
- [ ] No SyntaxError/ImportError in modified modules

### Endpoint Testing (Mandatory)
- [ ] Every new endpoint has at least ONE endpoint-level test (FastAPI TestClient)
- [ ] Tests cover success path AND at least one failure case (400/401/403/404/422/503)
- [ ] Modified endpoints have extended tests proving the fix

### Route Parity
- [ ] Route parity baseline updated (only if intentional)
- [ ] Route parity test passed

### Shared Dependencies
- [ ] Impact analysis done for shared utility/model changes
- [ ] All callers updated in same change

### Frontend/Backend Contract
- [ ] Frontend updated to match backend response changes
- [ ] Contract tests added for key field names

### Security
- [ ] No secrets leaked in logs/responses
- [ ] Webhook endpoints 503 when secrets missing
- [ ] Debug endpoints protected

### Documentation
- [ ] replit.md updated with significant changes
- [ ] CONTEXT_PACK.md updated if architecture changes

---

*Last updated: December 2024*
