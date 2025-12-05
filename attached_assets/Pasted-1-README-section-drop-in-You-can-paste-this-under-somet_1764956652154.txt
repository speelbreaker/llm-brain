1) README section (drop-in)

You can paste this under something like docs/README_AGENT.md or inside your main README under “Trading Agent”.

## Trading Agent – Behavior & Architecture

### High-Level Summary

The trading agent is a **net options seller** on Deribit (BTC/ETH), built around a conservative premium-selling framework:

- Instruments: **Options only** (no spot / futures trading).
- Underlyings: **BTC** and **ETH**.
- Side: **Net short options** (we sell options, we don’t run net-long option books).
- Strategies (current & near-term):
  - **Covered calls** on existing BTC/ETH.
  - **Cash-secured puts** in USDC (optional, via config).
- Execution modes:
  - **Live mode** (real orders).
  - **Dry-run mode** (paper trading, no live orders).

The agent is designed to be extended later to **multi-leg structures** (spreads, strangles, butterflies) using the same state, scoring, and simulation pipeline.

---

### Execution Mode: DRY RUN vs LIVE

Execution mode is controlled via configuration:

- `settings.DRY_RUN: bool`

Behavior:

- When `DRY_RUN = true`:
  - **No live orders** are sent to Deribit.
  - The agent still **generates actions** as usual (e.g. “sell call”, “roll call”, “sell put”).
  - Actions are recorded in a local **paper-trading ledger** (e.g. `data/paper_trades.json` or a DB table).
  - Fills are simulated using current **mid price** (or a simple slippage model).
  - The UI shows **P&L and positions** for this simulated book.

- When `DRY_RUN = false`:
  - Orders are submitted to Deribit (testnet or mainnet, depending on config).
  - All orders from the bot are tagged with a **client order id prefix**:
    - `settings.MANAGED_CLIENT_ID_PREFIX`, e.g. `CCBOT_`.
  - Bot-managed positions and P&L are computed from actual Deribit positions & trade history filtered by that prefix.

---

### Bot vs Manual Positions

The state builder fetches **all open positions** from Deribit, but the agent keeps a clear distinction between:

- **Bot-managed positions** – opened with `client_order_id` starting with `settings.MANAGED_CLIENT_ID_PREFIX` (e.g. `CCBOT_...`).
- **External/manual positions** – anything else in the account.

Configuration:

- `settings.MANAGE_EXTERNAL_POSITIONS: bool`

Behavior:

- If `MANAGE_EXTERNAL_POSITIONS = false` (recommended default):
  - The bot **only manages** positions it originally opened (tagged with the prefix).
  - Manual positions are visible in state (for risk context) but **never rolled/closed** by the bot.

- If `MANAGE_EXTERNAL_POSITIONS = true`:
  - The bot is allowed to **adopt and manage** any qualifying short calls/puts in the account, subject to the risk engine rules.

This separation makes it safe to experiment with the agent without risking manual discretionary trades.

---

### Position & PnL Tracking (UI-Friendly)

To avoid digging through logs, the agent tracks **all bot-managed trades and positions** and exposes them via the API/UI.

#### Data model (conceptual)

Each **bot-managed position** (call or put) is represented as a “chain”:

- `id` (internal UUID)
- `underlying` (BTC or ETH)
- `option_type` (CALL or PUT)
- `strategy_type` (`COVERED_CALL` or `CASH_SECURED_PUT`)
- `status` (`OPEN` or `CLOSED`)
- `legs`:
  - For each open/closed leg: side (short), strike, expiry, quantity, entry price, exit price, timestamps, Deribit instrument name, client_order_id.
- `realized_pnl`, `unrealized_pnl`
- `created_at`, `closed_at`

The agent keeps an append-only **trade ledger** plus **position aggregates** so we can reconstruct chains, rolls, and PnL.

#### API / UI

The backend should expose endpoints like:

- `GET /api/positions/bot/open`
  - Returns all **open bot-managed positions**, with:
    - underlying, type (short call / short put), strike, expiry, qty
    - entry price (or weighted average), current mark, **unrealized P&L** (abs + %)
    - strategy type and execution mode (live / dry-run)

- `GET /api/positions/bot/closed`
  - Returns all **fully closed bot-managed positions**, with:
    - underlying, type, strike/expiry of final leg
    - chain history summary (number of rolls, total days in trade)
    - **realized P&L** and max drawdown during life (if available)

The UI should display two tables:

1. **Open Positions (Bot-managed)**
   - Columns: Underlying, Type (Short Call/Put), Strike, Expiry, Qty, Entry, Mark, Unrealized P&L, % P&L, Mode (Live/Dry).
2. **Closed Positions (Bot-managed)**
   - Columns: Underlying, Type, Entry Date, Exit Date, Net Premium, Realized P&L, % P&L, Rolls Count.

Optional: show totals (sum of realized P&L, sum of unrealized) at the bottom for quick performance overview.

---

### Current Strategy Logic

#### 1. Covered Calls (implemented)

- Assumption: user holds some BTC/ETH spot.
- Agent sells **OTM calls** on that inventory.
- One “chain” per underlying:
  - At most **one active covered call chain per underlying** (BTC, ETH).
- Actions:
  - `OPEN_COVERED_CALL`
  - `ROLL_CALL` (take profit roll, or defensive roll when spot approaches strike)
  - `CLOSE_CALL` (exit without re-opening)

Risk constraints:
- The **short call notional** must be backed by BTC/ETH holdings.
- IV/RV filters and delta/OTM ranges are applied by the scoring engine and risk engine.

#### 2. Cash-Secured Puts (to be implemented)

We extend the action space to include **short puts**, but still in a conservative way:

- New strategy type: **CASH_SECURED_PUT**.
- New actions:
  - `OPEN_CASH_SECURED_PUT`
  - `ROLL_PUT`
  - `CLOSE_PUT`

Collateral & risk:

- Short puts are **cash-secured**:
  - Required USDC is reserved in the risk engine (simulator + live).
  - No naked short puts – position size is capped by `available_usdc` and config:
    - e.g. `settings.MAX_PUT_NOTIONAL_PER_UNDERLYING` or a % of equity.

We allow **at most one put chain per underlying** to keep things simple:
- 1 covered call chain + 1 cash-secured put chain per underlying (max).

---

### Call vs Put Decision Logic (High-Level)

When the agent decides to **open new risk**, it can now choose between:

- Opening / rolling a **covered call**.
- Opening / rolling a **cash-secured put**.
- Doing nothing.

High-level rules:

1. **Pre-conditions**:
   - Only consider covered calls if there is **sufficient BTC/ETH inventory**.
   - Only consider cash-secured puts if there is **sufficient free USDC**.
   - Respect per-underlying caps on total short notional (calls + puts).

2. **Candidate generation**:
   - Build candidate **call** options (OTM, within configured delta & DTE ranges).
   - Build candidate **put** options (OTM, within configured delta & DTE ranges).
   - Score each with the existing scoring function (IV/RV, premium, OTM %, etc.).

3. **Risk-aware scoring**:
   - For calls: penalize candidates that are too close to spot or clash with pump-risk.
   - For puts: penalize candidates that push total downside exposure beyond:
     - Max acceptable drawdown, and/or
     - Target allocation to that underlying.

4. **Action selection**:
   - Compare the **best scored call** vs **best scored put**, after risk penalties.
   - If neither exceeds a minimum score or risk budget → **no trade**.
   - If one clearly wins on risk-adjusted score → choose that side:
     - e.g. if market is stressed but you’re happy to accumulate more BTC, a conservative short put may be favored.
     - If you’re already heavy BTC and crash risk is elevated, covered call might be favored over more downside risk.

Over time, this logic can be further refined and/or learned via training on historical data.

---

### Extensibility: Spreads, Strangles, Butterflies

Architecturally, the system is ready to extend beyond single-leg calls/puts:

- The state builder already exposes option chains (calls & puts).
- The scoring system can be applied to **bundles** (multi-leg trades).
- The simulator/backtester can be extended to handle **trade bundles** (e.g. vertical spreads, short strangles).

To add these:

- Introduce new actions like:
  - `OPEN_CALL_SPREAD`, `OPEN_PUT_SPREAD`, `OPEN_STRANGLE`, etc.
- Treat a “trade” as a **bundle of legs**, tracked and managed as a single position with its own P&L and margin tracking.
- Reuse the same P&L and position UI: each multi-leg trade is one row with internal leg details.

For now, the live agent focuses on **covered calls and cash-secured puts** with clear, conservative risk controls.

2) Prompt block for your Builder/LLM agent

You can drop this into your spec file as a system/behavior block. It tells the Builder exactly how to implement features and what invariants to respect.

You are implementing and modifying a BTC/ETH options trading agent that runs on Deribit.

Core invariants:

1. The agent is a **net seller of options** (no net-long option books).
2. Instruments: **options only**, on BTC and ETH underlyings.
3. Strategies:
   - Covered calls on existing BTC/ETH.
   - Cash-secured puts in USDC (no naked puts).
4. The agent must always respect:
   - Available BTC/ETH spot as cover for short calls.
   - Available USDC as collateral for short puts.
   - Per-underlying risk caps on total short notional.

Execution modes:

- Use the config flag `settings.DRY_RUN: bool`:
  - If `DRY_RUN = true`:
    - Do NOT send orders to Deribit.
    - Instead, record simulated trades in a local paper ledger and simulate fills using mid prices.
  - If `DRY_RUN = false`:
    - Send live orders to Deribit (testnet or mainnet depending on config).

Bot vs manual positions:

- All orders sent by the bot MUST have a `client_order_id` starting with:
  - `settings.MANAGED_CLIENT_ID_PREFIX` (e.g. "CCBOT_").
- Use `settings.MANAGE_EXTERNAL_POSITIONS: bool`:
  - If false:
    - The agent may SEE all account positions in state, but it only MANAGES positions whose client_order_id starts with the managed prefix.
  - If true:
    - The agent may adopt and manage external positions if they match strategy rules.

Action space (extend or modify code accordingly):

- Covered calls (already supported):
  - `OPEN_COVERED_CALL`
  - `ROLL_CALL`
  - `CLOSE_CALL`

- Cash-secured puts (to implement):
  - `OPEN_CASH_SECURED_PUT`
  - `ROLL_PUT`
  - `CLOSE_PUT`

- Each action must be implemented both in:
  - The live execution layer (respecting DRY_RUN).
  - The simulator/backtester.

Position and P&L tracking (critical requirement):

- Implement a registry of **bot-managed positions**, each identified by an internal `position_id`.
- A position is a “chain” of legs (rolls) for a single strategy:
  - Fields: position_id, underlying, option_type (CALL/PUT), strategy_type (COVERED_CALL / CASH_SECURED_PUT), status (OPEN/CLOSED), list of legs, created_at, closed_at, realized_pnl, unrealized_pnl.
- Legs correspond to individual orders/fills, linked to Deribit trades when live, or to simulated fills when DRY_RUN = true.
- All P&L calculations should be centralized in a utility module so we can reuse it in the simulator and in the live agent.

UI / API requirements:

- Expose two HTTP endpoints for the frontend:

  1) `GET /api/positions/bot/open`
     - Returns all OPEN bot-managed positions, with:
       - underlying, option_type, strategy_type, strike, expiry, quantity
       - entry_price or average entry, current_mark, unrealized_pnl_abs, unrealized_pnl_pct
       - execution_mode (LIVE or DRY_RUN).

  2) `GET /api/positions/bot/closed`
     - Returns all CLOSED bot-managed positions, with:
       - underlying, option_type, strategy_type, created_at, closed_at
       - net_premium, realized_pnl_abs, realized_pnl_pct
       - number_of_rolls in the chain.

- The frontend will render:
  - A table for OPEN positions.
  - A table for CLOSED positions.
  - You do NOT need to design the HTML in this step, but the JSON shape must be stable and well-documented in code.

Call vs Put decision logic:

- The decision loop must consider both:
  - A candidate covered call (if BTC/ETH inventory allows).
  - A candidate cash-secured put (if USDC collateral allows).
- Use the existing scoring pipeline (IV/RV, delta, OTM %, premium, etc.) to score each candidate.
- Apply additional risk penalties:
  - Penalize calls if pump-risk is high or if call capacity is already heavily used.
  - Penalize puts if crash-risk is high or if total downside exposure would breach configured limits.
- Compare the best scored call vs the best scored put:
  - If neither passes a minimum score threshold or risk check → NO ACTION.
  - Otherwise execute the higher-scoring side’s action:
    - e.g. OPEN_COVERED_CALL vs OPEN_CASH_SECURED_PUT, or a roll if a chain is already open.

Risk and capacity constraints:

- Maintain per-underlying caps such as:
  - `settings.MAX_CALL_NOTIONAL_PER_UNDERLYING`
  - `settings.MAX_PUT_NOTIONAL_PER_UNDERLYING`
  - and/or percentages of total equity.
- At most:
  - 1 covered call chain per underlying.
  - 1 cash-secured put chain per underlying.
- The agent must never exceed these caps when opening or rolling positions, both in the simulator and in live trading.

Backtester/simulator:

- Mirror the live logic:
  - Support both call and put actions.
  - Use the same scoring and risk checks.
  - Track P&L using the same position/chain data model as the live agent.
- Ensure backtest output includes:
  - Summary P&L.
  - Per-position P&L (so we can compare to the UI’s live view conceptually).

When you generate or modify code:

- Respect the config flags (`DRY_RUN`, `MANAGED_CLIENT_ID_PREFIX`, `MANAGE_EXTERNAL_POSITIONS`, notional caps).
- Never place live orders in DRY_RUN mode.
- Always tag bot orders correctly so P&L and positions can be attributed to the agent.
- Keep functions and types well-documented so future prompts can easily extend the system with spreads, strangles, or other multi-leg structures.
