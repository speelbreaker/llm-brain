# Architecture Decision Log

> Major design choices with rationale and tradeoffs. Newest entries first.

---

## ADR-010: Greg Selector Synchronous Execution Mode

**Date:** December 2024

**What:** Greg Selector backtests run synchronously and return immediately, bypassing the async backtest manager.

**Why:**
- Selector-only analysis doesn't need full P&L simulation
- Users want instant feedback on strategy pass/block status
- No need for job queuing for lightweight evaluations

**Tradeoffs:**
- (+) Immediate results, no polling required
- (+) Simpler frontend code path
- (-) Can't run very long selector scans without timeout risk
- (-) Two different execution paths (sync vs async) to maintain

---

## ADR-009: Dual Backtest Mode Architecture

**Date:** December 2024

**What:** Two distinct backtest types: `GENERIC` (full simulation) and `GREG_SELECTOR` (selector analysis).

**Why:**
- Generic mode provides traditional covered call P&L metrics
- Greg Selector mode focuses on per-strategy diagnostics
- Users need different views: aggregate P&L vs strategy breakdown

**Tradeoffs:**
- (+) Clean separation of concerns
- (+) Each mode optimized for its use case
- (-) Additional UI complexity (mode selector, conditional displays)
- (-) Two code paths to maintain and test

---

## ADR-008: Strategy Capabilities System

**Date:** November 2024

**What:** Each strategy declares its own capabilities, parameters, and config overrides via a capabilities registry.

**Why:**
- Not all strategies support all backtest parameters
- UI needs to know which fields to show/hide per strategy
- Backtest runner needs strategy-specific defaults

**Tradeoffs:**
- (+) Self-documenting strategies
- (+) UI can auto-generate forms from metadata
- (-) More boilerplate per strategy
- (-) Capabilities must stay in sync with actual implementation

---

## ADR-007: Modular Web Router Architecture

**Date:** November 2024

**What:** Split FastAPI routes into separate modules: `routes_main.py`, `routes_backtest.py`, `routes_positions.py`, `routes_bots.py`, `routes_health.py`.

**Why:**
- Single router file became too large (>1000 lines)
- Easier to find and modify related endpoints
- Better separation of concerns

**Tradeoffs:**
- (+) Each file has a clear purpose
- (+) Easier code review and testing
- (-) Must maintain route parity tests
- (-) Import complexity in `web_app.py`

---

## ADR-006: GregBot as Strategy Bundle (Decision Waterfall)

**Date:** October 2024

**What:** GregBot is not a single strategy but a bundle of 7+ strategies selected via a decision waterfall defined in JSON.

**Why:**
- Market conditions vary; one strategy doesn't fit all
- Greg Mandolini's approach adapts to VRP, trend, and volatility regime
- JSON-driven rules allow calibration without code changes

**Tradeoffs:**
- (+) Adaptive to market conditions
- (+) Calibration is data-driven, not code-driven
- (-) More complex debugging (which rule triggered?)
- (-) JSON spec must stay in sync with code

---

## ADR-005: Three Data Source Options

**Date:** October 2024

**What:** Backtests support three data sources: SYNTHETIC, HARVESTER, LIVE.

**Why:**
- Synthetic provides reproducible, large-scale testing
- Harvester gives historical real-world data
- Live enables current market snapshot analysis

**Tradeoffs:**
- (+) Flexibility for different testing needs
- (+) Can validate strategies across data types
- (-) Each source has different data shapes and quirks
- (-) Must maintain adapters for each source

---

## ADR-004: Risk Engine as Final Gate

**Date:** September 2024

**What:** All trade decisions pass through `risk_engine.py` before execution. No bypass possible.

**Why:**
- Safety is non-negotiable for live trading
- Single enforcement point prevents bugs from bypassing limits
- Clear audit trail of what was blocked and why

**Tradeoffs:**
- (+) Guaranteed safety enforcement
- (+) Centralized risk logic
- (-) Additional latency on every decision
- (-) Risk engine must be highly reliable (single point of failure)

---

## ADR-003: Pydantic for All Configuration

**Date:** August 2024

**What:** Use Pydantic `BaseSettings` for configuration and Pydantic models for all data structures.

**Why:**
- Type safety and validation out of the box
- Environment variable parsing built-in
- Self-documenting schemas

**Tradeoffs:**
- (+) Catch config errors at startup
- (+) IDE autocomplete and type checking
- (-) More verbose than plain dicts
- (-) Learning curve for contributors

---

## ADR-002: HTTPX Over Requests

**Date:** August 2024

**What:** Use `httpx` for all HTTP calls instead of `requests`.

**Why:**
- Async support for FastAPI integration
- Connection pooling and timeout handling
- Modern API with better defaults

**Tradeoffs:**
- (+) Native async/await support
- (+) Better performance for concurrent calls
- (-) Slightly different API from requests
- (-) Some third-party libs assume requests

---

## ADR-001: Testnet-First Development

**Date:** July 2024

**What:** All development and testing happens on Deribit testnet. Mainnet deployment is a separate, future milestone.

**Why:**
- No financial risk during development
- Can experiment freely with strategies
- Testnet mimics production API

**Tradeoffs:**
- (+) Safe experimentation
- (+) Full API compatibility with mainnet
- (-) Testnet liquidity differs from mainnet
- (-) Some edge cases may not surface until mainnet

---

## Template for New Entries

```markdown
## ADR-XXX: [Title]

**Date:** [Month Year]

**What:** [One-line description of the decision]

**Why:**
- [Reason 1]
- [Reason 2]

**Tradeoffs:**
- (+) [Benefit]
- (-) [Drawback]
```

---

*Last updated: December 2024*
