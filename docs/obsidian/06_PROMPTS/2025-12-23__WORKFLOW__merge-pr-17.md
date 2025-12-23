# 2025-12-23 — Merge PR #17 (ops-health contract)

## Objective
Resolve `vps-salvage/ops-health-contract` (PR #17) into `main`, including the conflict in `scripts/gen_ops_health_latest.py`, so the OPS_HEALTH contract passes everwhere and the repo split-brain is closed.

## Scope
- The conflicting sections of `scripts/gen_ops_health_latest.py` and any supporting tests or scripts mentioned in the salvage branch.
- Documentation that tracks the contract requirements (overall_status/can_trade/worst_severity/summary) after merging.
- Queue + changelog updates, lock handling, and referencing the new skills/validator work that will follow.

## Non-goals
- Implementing the downstream repo skills or validator itself (that is Step 2).
- Touching collection artifacts in the VPS context or rerunning the ops health generator on that machine until the merge is in place.

## Acceptance Criteria
- Conflict in `scripts/gen_ops_health_latest.py` is resolved with the agreed OPS_HEALTH contract semantics (overall_status ∈ {OK,WARN,FAIL}, boolean can_trade, non-empty worst_severity/summary, gate_overall present, fail-closed on generator errors).
- Any tests or contracts touched by the salvage branch are updated or verified to align with the merged logic.
- The queue now lists this prompt as IN_PROGRESS, and the changelog reloads an entry documenting the merge plan (tests still pending). Lock entry recorded before edits.

## Tests required
- `python3 -m pytest tests/test_ops_health_contract.py` to ensure `overall_status/can_trade/worst_severity` are consistently populated.

## Rollback plan
- Revert the merge commit, restore both branch versions if the contract enforcement regresses, and re-open PR #17 for manual resolution.

## Done means
- [x] `scripts/gen_ops_health_latest.py` conflict resolved with the contract semantics and merges cleanly.
- [x] Ops health contract tests/validations pass and log results in the changelog entry.
- [x] Queue/changelog reflect the completed merge, the prompt archives to `99_ARCHIVE`, and downstream skills/validator work can begin.
