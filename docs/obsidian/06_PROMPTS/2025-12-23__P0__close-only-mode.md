# 2025-12-23 — P0 Close-Only Mode

## Objective
Add a runtime-configurable close-only mode so the trading stack can be put into reduce-only state and new position opens are blocked consistently.

## Scope
- Runtime flags and configs that drive `close_only_mode` (e.g., `src/config.py`, any environment-specific overrides).
- Permission layers that run before risk computations and just before the order writer touches Deribit.
- Documentation that lets operators toggle and audit the mode.

## Non-goals
- User interface polish beyond noting the flag exists.
- Reworking risk math unrelated to blocking opens.

## Acceptance Criteria
- `close_only_mode: bool` is surfaced in the runtime config and can be flipped without restarting unrelated services.
- A centralized permission guard reads `close_only_mode` and blocks any attempt to increase risk, even if downstream components still call `place_order` or equivalent.
- The guard runs at the last responsible moment (right before the live order API/stack is invoked), not just at the UI level.
- Operator-facing documentation and logging clearly state when close-only mode is active and why opens were blocked.

## Tests required
- Run a one-tick integration test where `close_only_mode=true`, an open-order path is attempted (must fail with the canonical error), and a reduce/close path still succeeds. Reference `docs/obsidian/06_PROMPTS/2025-12-23__P1__one-tick-integration-test.md` for the exact test story.

## Rollback plan
- Revert the config and permission guard changes, then verify that prior functionality is restored and the one-tick test fails back to the old behavior.

## Done means
- [ ] `close_only_mode` flag exists in the runtime config and is documented for operators.
- [ ] The centralized permission guard enforces reduce-only across the live trading path and logs why openings are blocked.
- [ ] The one-tick close-only integration test described in the related prompt runs cleanly.
- [ ] Queue, changelog, and locking records were updated before merging, and the prompt file is archived if the workflow moves on.
