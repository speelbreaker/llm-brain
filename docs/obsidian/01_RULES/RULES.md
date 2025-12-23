# RULES (must-follow)

## Single Source of Truth
- All prompts live in: docs/obsidian/06_PROMPTS/
- All reviews live in: docs/obsidian/04_REVIEWS/
- Queue is authoritative: docs/obsidian/02_QUEUE/QUEUE.md

## Prompt Contract (required)
Every prompt file MUST include:
- Objective
- Scope (files)
- Non-goals
- Acceptance Criteria (bulletproof)
- Tests required (endpoint-level when endpoints change)
- Rollback plan
- "Done means" checklist

## Change Logging
Every completed task MUST append to:
docs/obsidian/03_LOGS/CHANGELOG.md

Format:
- Date
- What changed
- Why
- Tests run + results
- Links to PR/commit

## No parallel edits
- One agent at a time.
- If a file is being worked on: create a lock entry in docs/obsidian/02_QUEUE/LOCKS.md
