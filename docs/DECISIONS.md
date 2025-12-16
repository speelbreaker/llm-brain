# Architecture Decision Log (ADR) — Canonical

> Major design choices with rationale and tradeoffs. **Newest first.**  
> Rule: If an entry is not confirmed in code, it’s still a valid decision — but mark it as **Design decision** (vs **Implemented**).

**Last updated:** 2025-12-16

---

## ADR-012: Dual Brain Architecture (Rule-Based + LLM Co‑Pilot)

**Date:** Undated (recorded 2025-12-16)

**What:** Maintain both a deterministic policy path (rule-based) and an LLM JSON decision-maker, selectable via settings.

**Why:**
- Deterministic policy gives reliability and testability (baseline you can trust)
- LLM adds adaptability and richer rationale for exploration and operator insight

**Tradeoffs:**
- (+) Robust fallback when LLM is unavailable or wrong
- (+) Safer experimentation (LLM can be bounded by schema + whitelist)
- (-) Two implementations to maintain
- (-) LLM adds latency/cost and requires strict guardrails


---

## ADR-011: Strategy Abstraction via a Shared `Strategy` Interface

**Date:** Undated (recorded 2025-12-16)

**What:** All tradable behaviors implement a shared `Strategy` interface with a registry to activate one or more strategies in the agent loop.

**Why:**
- Keeps covered calls, training profiles, and future strategies (Wheel/CrashHedge/Spreads) pluggable without rewriting the loop

**Tradeoffs:**
- (+) Extensible architecture; adding strategies doesn’t rewrite the loop
- (+) Cleaner testing boundaries per strategy
- (-) Boilerplate + registry management overhead
- (-) Interface evolution must be disciplined to avoid breaking strategies


---

## ADR-010: Greg Selector Synchronous Execution Mode

**Date:** December 2024

**What:** Greg Selector backtests run synchronously and return immediately, bypassing the async backtest manager.

**Why:**
- Selector-only analysis doesn’t need full P&L simulation
- Users want instant feedback on strategy pass/block status
- No need for job queuing for lightweight evaluations

**Tradeoffs:**
- (+) Immediate results, no polling required
- (+) Simpler frontend code path
- (-) Long selector scans risk timeouts
- (-) Two execution paths (sync vs async) to maintain


---

## ADR-009: Dual Backtest Mode Architecture

**Date:** December 2024

**What:** Two distinct backtest types: `GENERIC` (full simulation) and `GREG_SELECTOR` (selector analysis).

**Why:**
- Generic mode provides covered call P&L metrics
- Greg Selector mode focuses on per-strategy diagnostics
- Users need both views: aggregate P&L vs strategy breakdown

**Tradeoffs:**
- (+) Separation of concerns; each mode optimized
- (-) More UI complexity and more code paths to test


---

## ADR-008: Strategy Capabilities System

**Date:** November 2024

**What:** Each strategy declares its own capabilities, parameters, and config overrides via a capabilities registry.

**Why:**
- Not all strategies support all backtest parameters
- UI must know which fields to show/hide per strategy
- Backtest runner needs strategy-specific defaults

**Tradeoffs:**
- (+) Self-documenting strategies; forms can be generated from metadata
- (-) More boilerplate per strategy; metadata can drift from implementation


---

## ADR-007: Modular Web Router Architecture

**Date:** November 2024

**What:** Split FastAPI routes into separate modules (e.g., `routes_main`, `routes_backtest`, `routes_positions`, `routes_bots`, `routes_health`).

**Why:**
- Single router file became too large
- Easier to find/modify related endpoints
- Better separation of concerns

**Tradeoffs:**
- (+) Clearer code organization; easier review/testing
- (-) Must maintain route parity tests; more imports/wiring


---

## ADR-006: GregBot as a Strategy Bundle (Decision Waterfall)

**Date:** October 2024

**What:** GregBot is not a single strategy — it’s a bundle selected via a decision waterfall (JSON-driven rules).

**Why:**
- Market conditions vary; one strategy doesn’t fit all
- Approach adapts to VRP, trend, and volatility regime
- JSON-driven rules allow calibration without code changes

**Tradeoffs:**
- (+) Adaptive and tunable without redeploys (if JSON is externalized)
- (-) Debugging is harder (“which rule triggered?”); spec must remain in sync with code


---

## ADR-005: Three Data Source Options

**Date:** October 2024

**What:** Backtests support three data sources: `SYNTHETIC`, `HARVESTER`, `LIVE`.

**Why:**
- Synthetic provides reproducible, scalable testing
- Harvester provides historical real-world snapshots
- Live enables current market snapshot analysis

**Tradeoffs:**
- (+) Flexibility and cross-validation
- (-) Each source has quirks; adapters must be maintained


---

## ADR-004: Risk Engine as Final Gate (No Bypass)

**Date:** September 2024

**What:** All trade decisions pass through `risk_engine.py` before execution; no bypass in production mode.

**Why:**
- Safety is non-negotiable for live trading
- Single enforcement point prevents bugs from bypassing limits
- Clear audit trail of what was blocked and why

**Tradeoffs:**
- (+) Guaranteed safety enforcement; centralized risk logic
- (-) Risk engine becomes a reliability bottleneck; adds latency


---

## ADR-003: Pydantic for All Configuration

**Date:** August 2024

**What:** Use Pydantic `BaseSettings` for configuration and Pydantic models for core data structures.

**Why:**
- Validation and type safety
- Environment variable parsing built-in
- Self-documenting schemas

**Tradeoffs:**
- (+) Catch config errors at startup; better IDE support
- (-) More verbose than plain dicts; learning curve


---

## ADR-002: HTTPX Over Requests

**Date:** August 2024

**What:** Use `httpx` for HTTP calls instead of `requests`.

**Why:**
- Async support for FastAPI integration
- Connection pooling and timeouts
- Better fit for concurrent calls

**Tradeoffs:**
- (+) Native async/await; better concurrency
- (-) API differences vs requests; some third-party libs assume requests


---

## ADR-001: Testnet‑First Development (Deribit)

**Date:** July 2024

**What:** All development and testing happens on Deribit testnet. Mainnet deployment is a separate milestone.

**Why:**
- No financial risk during development
- Experiment freely with strategies
- Testnet mimics production API surface

**Tradeoffs:**
- (+) Safe iteration
- (-) Testnet liquidity differs; some edge cases appear only on mainnet


---

## Appendix A: Additional System-Level Decisions (Not yet dated)

These entries were captured in a separate ADR draft and should be promoted to dated ADRs once verified against the repo and commit history.

### A1: Deribit as Primary Venue (Testnet-First)

**Status:** Covered by ADR-001 (keep here only if adding portability/adapter plan later).

### A2: FastAPI Dashboard + Unified Process

**Date:** Undated (recorded 2025-12-16)

**What:** FastAPI dashboard can bootstrap the agent loop in the same process.

**Why:**
- Simplifies deployment and shared logs/config
- One process to operate early on

**Tradeoffs:**
- (+) Simple deployment
- (-) Resource contention; scaling UI independently requires process split


### A3: JSONL Logging for Decisions and Actions

**Date:** Undated (recorded 2025-12-16)

**What:** Persist decisions, rationales, and execution results as append-only JSONL logs.

**Why:**
- Auditable traces for debugging/compliance
- Easy dataset generation for future ML

**Tradeoffs:**
- (+) Simple, robust, append-only recorder
- (-) Harder querying/retention; long-term needs DB/indexing


### A4: Backtesting Stack Mirrors Live Types

**Date:** Undated (recorded 2025-12-16)

**What:** Backtest modules mirror live types/state builders to reduce simulation drift.

**Why:**
- Keeps sims faithful to production flows
- Lower cognitive load switching between live/backtest

**Tradeoffs:**
- (+) Less sim drift
- (-) Coupling between live and sim code increases; shared types must remain stable


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
