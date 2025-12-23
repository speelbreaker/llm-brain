---
name: llmbrain-guardrails
description: Enforce llm-brain engineering rules (tests, gating, no risky merges) for any change.
metadata:
  short-description: Enforce repo guardrails + acceptance checks
---

## Always-on rules
- If adding or changing an endpoint, add at least one endpoint-level pytest that covers it.
- If modifying trading permission logic, add or extend an integration-style test proving the “cannot trade” path blocks openings.
- Never commit generated artifacts such as `docs/*_latest.*` or contents of `docs/_context_pack_out/`, unless explicitly requested.
- Never work directly on VPS `main`; treat it as deployment/generation only.

## Required checks before claiming done
- Run `pytest -q` (or run a targeted suite and explain why it is sufficient).
- Confirm `can_trade` is fail-closed: every FAIL path sets `can_trade=false`.
- Confirm the trade code path enforces permissions and supports close-only mode at every order gate.
