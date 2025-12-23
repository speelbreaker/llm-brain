# 2025-12-23 — Repo skills & vault validator

## Objective
Add the requested `.codex/skills/` guides (vault-ops, pr-discipline, fidelity-northstar, ops-health) and the validator automation plus tests that keep the obsidian workflow honest.

## Scope
- Build SKILL.md files under `.codex/skills/<name>` for each discipline, describing when to invoke them and what artifacts they touch.
- Implement `scripts/validate_vault_workflow.py` plus pytest cases to enforce queue/prompt references and changelog freshness.
- Update `docs/obsidian/02_QUEUE/QUEUE.md`, `docs/obsidian/02_QUEUE/LOCKS.md`, and `docs/obsidian/03_LOGS/CHANGELOG.md` to reflect this new workflow step and the validator’s presence.

## Non-goals
- Touching trading code, ops health contracts, or Docker/systemd automation beyond the validator mention.
- Rewriting existing skills; focus on the four named guides.

## Acceptance Criteria
- Each new skill includes clear triggers, instructions, and references to the vault rules so any agent knows how to behave in its domain.
- The validator script identifies (a) missing prompts referenced in the queue, (b) unreferenced newest prompts, and (c) stale changelog entries, failing loudly if any rule breaks.
- Tests cover normal and failure scenarios for the validator logic.
- Queue/changelog record the start/completion of this step, and locks capture the files touched.

## Tests required
- `python3 -m pytest tests/test_validate_vault_workflow.py`

## Rollback plan
- Delete the new skills and validator artifacts, then revert the queue/changelog updates if the validator causes regressions.

## Done means
- [x] All four skills exist with actionable guidance.
- [x] Validator script and tests run cleanly and gate the queue/changelog rules.
- [x] Queue, changelog, and lock entries mention this prompt and its completion before archiving.
