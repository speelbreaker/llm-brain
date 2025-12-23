# PROMPT_002: Workflow Infrastructure

## Objective
Implement Phase 2 (Workflow Infrastructure) in the APP REPO so the VPS supervisor + local agents all follow the same rules.

## Acceptance Criteria
- [ ] Vault structure created in `docs/obsidian/`.
- [ ] `docs/obsidian/02_QUEUE/QUEUE.md` format is valid and machine-parsable.
- [ ] `scripts/validate_vault_workflow.py` script created and passing.
- [ ] `tests/test_vault_workflow_validator.py` created and passing.
- [ ] Skills created in `.codex/skills/`.

## Tests / Verification
-   Run `python scripts/validate_vault_workflow.py` -> returns 0.
-   Run `pytest tests/test_vault_workflow_validator.py` -> passes.
