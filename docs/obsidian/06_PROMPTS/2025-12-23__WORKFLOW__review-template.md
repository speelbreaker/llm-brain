# WORKFLOW — Review Template (Proof)

## Objective
Define a reusable review template so future reviewers know exactly what to capture when assessing a workflow change.

## Scope
- docs/obsidian/06_PROMPTS/2025-12-23__WORKFLOW__review-template.md
- Any follow-up reviews that reference this template as a guide

## Non-goals
- Running the review itself
- Updating code beyond this review template reference

## Acceptance Criteria
- Template includes all required sections
- Each section contains guidance so reviewers know what to write
- File lives under docs/obsidian/06_PROMPTS

## Tests required
- None (workflow-only)

## Rollback plan
- Delete this file and revert any queue or changelog updates if the template is incorrect.

## Done means
- Template committed with the sections below, and referenced workflows can rely on it.

## Review Template
### Summary
- Capture the high-level purpose and impact of the change.
- Mention whether the work impacted critical paths or docs.

### Risks
- Note any remaining unknowns, dependencies, or potential regressions.
- Call out anything that needs follow-up verification.

### What to verify
- List concrete verification steps (manual checks, smoke tests, data points).
- Include success criteria for each verification item.

### Files touched
- Enumerate files involved in the change so readers know what to inspect.

### Tests run
- Document test suites executed or mark `Not run (not applicable)` if no tests.

### Verdict (PASS/WARN/FAIL)
- PASS: Review finds no blockers.
- WARN: Minor issues noted that should be tracked.
- FAIL: Blocker discovered that must be addressed before merging.
