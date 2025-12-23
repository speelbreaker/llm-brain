# WORKFLOW — Agent Writes Prompt File (Proof)

## Objective
Prove the agent can create and update vault files inside the repo without manual copy/paste into chats.

## Scope
- Create a new prompt file (below)
- Update QUEUE.md (move item to Done)
- Append a changelog entry

## Task
Create:
docs/obsidian/06_PROMPTS/2025-12-23__WORKFLOW__review-template.md

Content must be a "Review Template" with:
- Summary
- Risks
- What to verify
- Files touched
- Tests run
- Verdict (PASS/WARN/FAIL)

Then update:
- docs/obsidian/02_QUEUE/QUEUE.md (move item #1 to Done, add a Done timestamp)
- docs/obsidian/03_LOGS/CHANGELOG.md (one entry describing what happened)

## Acceptance Criteria
- New prompt file exists with correct content and path.
- Queue updated correctly.
- Changelog entry appended.
- No other files changed.

## Tests Required
None (workflow-only).

## Rollback
Delete the created file and revert QUEUE/CHANGELOG changes.

## Done means
- All acceptance criteria met.
