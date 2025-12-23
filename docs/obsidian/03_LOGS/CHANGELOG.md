# CHANGELOG (newest first)

- Date: 2025-12-23
  - What changed: Completed the Step 2 repo skills and validator work, added the new skills in `.codex/skills/` and the validator script/tests, and moved the queue prompt to DONE.
  - Why: Guard the obsidian workflow with automation so future agents have explicit skills and validators.
  - Tests run + results: `python3 -m pytest tests/test_validate_vault_workflow.py` (pass).
  - Links to PR/commit: n/a

- Date: 2025-12-23
  - What changed: Created the Step 2 prompt for repo skills & validator, recorded the new lock entry, and moved the queue item into IN_PROGRESS.
  - Why: Prepare the workflow for adding the named skills and validator automation without racing other prompts.
  - Tests run + results: Not run (planning stage).
  - Links to PR/commit: n/a

- Date: 2025-12-23
  - What changed: Completed the Step 1 merge prompt for PR #17, resolved `scripts/gen_ops_health_latest.py` to enforce the OPS_HEALTH contract, and polished the lock/queue entries.
  - Why: The repo needed a single source of truth for OPS_HEALTH generation before any downstream skills/validator work.
  - Tests run + results: `python3 -m pytest tests/test_ops_health_contract.py` (pass).
  - Links to PR/commit: n/a

- Date: 2025-12-23
  - What changed: Created the Step 1 prompt for merging PR #17 (ops-health contract), locked the relevant files, and marked the queue item as IN_PROGRESS.
  - Why: Align the vault workflow with the ordered plan so the merge conflict work is tracked and prevents other agents from jumping ahead.
  - Tests run + results: Not run (planning/lock stage).
  - Links to PR/commit: n/a

- Date: 2025-12-23
  - What changed: Rebuilt the obsidian vault structure (00_README, 01_NOW, NORTHSTAR/OPS/BUILD folders), added three P0/P1 prompt specs, refreshed the queue and changelog entries, and documented the workflow rules and publishing expectations.
  - Why: Align the vault with the “Queue is law” workflow so prompts, logs, and docs stay syncable and reviewable.
  - Tests run + results: Not run (vault-only).
  - Links to PR/commit: n/a

- Date: 2025-12-23
  - What changed: Added `2025-12-23__WORKFLOW__review-template.md` as the new review template prompt and marked the prior workflow item as completed in the queue.
  - Why: Proving the agent can create and manage vault files per the workflow prompt requirements.
  - Tests run + results: Not run (workflow-only).
  - Links to PR/commit: n/a
