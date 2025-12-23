# Northstar: Automated Fidelity & Safety

## Vision
A fully autonomous options trading system that prioritizes **fidelity** (reality-matching backtests) and **safety** (risk gates) above all else.

## Core Pillars
1.  **Fidelity First**: If backtest != live, stop trading.
2.  **Safety Gates**: Risk limits are hard laws. No exceptions.
3.  **Observability**: Every decision, trade, and state change is logged and auditable.
4.  **Autonomy Loop**: The supervisor loop runs the show, guided by strict policy.

## Operational Modes
-   **Close-Only**: Default safety state.
-   **Dry-Run**: Simulation mode for validation.
-   **Live**: Requires explicit, multi-factor approval.
