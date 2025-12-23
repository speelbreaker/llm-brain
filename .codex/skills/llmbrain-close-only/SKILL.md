---
name: llmbrain-close-only
description: Implement Close-Only (reduce-only) mode and enforce it end-to-end in live trading path.
metadata:
  short-description: Add reduce-only + enforcement + tests
---

## Goal
- Add a Close-Only mode that blocks all position-increasing opens, allows only reduce/close actions, and enforces the policy in the live trading path (not just the UI).

## Required design
- Expose a runtime-configurable flag such as `close_only_mode: bool`.
- Centralize enforcement in a single permission function that includes both `can_trade` gating and a `close_only` decision, preventing duplication and drift.
- Apply the permission guard at the last responsible moment—right before an order would be placed on the exchange.

## Test requirements
- Provide one deterministic “one tick” integration-style test using the real trading stack.
  - Start with an existing position and `close_only_mode=true`.
  - Attempt to open a new position; this must be blocked with the canonical error message.
  - Attempt to close or reduce the position; this must be allowed.
