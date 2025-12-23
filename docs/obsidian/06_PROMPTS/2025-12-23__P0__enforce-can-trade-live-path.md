# 2025-12-23 — P0 Enforce `can_trade` in Live Path

## Objective
Guarantee that the live trading path always runs the canonical `can_trade` guard (with any close-only logic) before touching the exchange, so the trading stack can never bypass permission gates coming from feature flags or risk policies.

## Scope
- Permission APIs, risk checks, and order writers that exist between `agent_loop.py` (or its orchestration cousins) and the Deribit order placement client.
- Config sources for `can_trade` and any `close_only_mode` overrides.
- Logs and alarms that report when permission gates block an order.

## Non-goals
- Rebuilding the permission APIs from scratch.
- Changing how the UI signals the current permission state.

## Acceptance Criteria
- The canonical `can_trade` guard runs as close to the exchange as possible, after any transformations but before the live order call.
- The guard knows about `close_only_mode` so it can return the same blocking outcome when the mode is active.
- Permission failures are logged with a consistent message and surface in the health dashboards.
- There is no code path that calls `place_order`/`exchange.enter` without first checking `can_trade`.

## Tests required
- Unit tests for the guard that show permission block reasons when `close_only_mode` is true and allow orders when it is false.
- The one-tick integration test described in `2025-12-23__P1__one-tick-integration-test.md` also captures the live path behavior by exercising both the blocked and allowed scenarios.

## Rollback plan
- Revert the guard placement to its previous location and confirm the logging and test suite fall back to the old behavior.

## Done means
- [ ] Live trading uses the canonical `can_trade` guard before every exchange call.
- [ ] Guard integrates `close_only_mode` semantics and logs the rationale on block.
- [ ] Health dashboards report permission gate outcomes.
- [ ] Queue/changelog/locks reflect the completion and the prompt moves to `99_ARCHIVE` if further work shifts away.
