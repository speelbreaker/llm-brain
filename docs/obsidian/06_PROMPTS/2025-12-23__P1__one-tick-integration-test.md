# 2025-12-23 — P1 One-Tick Integration Test

## Objective
Design the deterministic integration-style test that runs one agent loop tick (the orchestration level closest to the live path) with a mocked Deribit client so we can assert both blocked and allowed outcomes for the new permission guard.

## Scope
- The agent loop or orchestrator that triggers a single tick, the mocked Deribit client, and the permission guard that evaluates `can_trade` and `close_only_mode`.
- Fixtures, mocks, and configuration that keep timestamps deterministic.
- Smoke verification that order placement calls occur only when gates allow trading.

## Non-goals
- Running a full backtest or spinning up the real Deribit network.
- Testing every trading strategy—only the one-tick permission story matters.

## Acceptance Criteria
- The test executes exactly one tick and evaluates both scenarios:
  1. `can_trade` returns false (or `close_only_mode=true`), ensuring no order placement methods are invoked.
  2. `can_trade` returns true (and `close_only_mode=false`), ensuring the expected order placement calls are issued.
- The Deribit client and any dependency that would reach the network are mocked or stubbed, so no external calls happen.
- Timestamps, random seeds, and environment settings are deterministic so the test passes consistently.

## Tests required
- `pytest tests/test_one_tick_permission.py` (or equivalent) covering the blocked/allowed scenarios described above.

## Rollback plan
- Remove the new test or restore the previous agent loop fixture if the deterministic behavior regresses.

## Done means
- [ ] The one-tick test runs once per CI invocation and covers both gate states.
- [ ] All mocks stay offline; no real Deribit calls occur.
- [ ] The test surfaces the permission guard outcomes so future engineers can see why orders were blocked.
- [ ] Queue and changelog updates reference this test prompt, and the prompt moves through the workflow lanes before archiving.
