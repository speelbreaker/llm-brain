# How we work

This vault is the single source of truth for our workflow, trading guardrails, and prompt provenance. Treat every change here as serious, because it reflects how the team collaborates when Codex is doing the heavy lifting.

## Vault layout
- **00_README.md**: This file. Rules, definitions, and links for keeping the vault sane.
- **01_NOW.md**: Living constraint log. Name the current constraint, the current focus, and anything blocked right now—one concise statement.
- **02_QUEUE/QUEUE.md**: The backlog expressed as a queue with four lanes (READY → IN_PROGRESS → IN_REVIEW → DONE). Every task lives here while it executes; no agent acts without pulling the freshest queue.
- **03_NORTHSTAR/**: Immutable-ish product specs—Fidelity NorthStar, Truth/Trust/Trade, and companions. These documents should change rarely and only after consensus.
- **04_OPS/**: Operations and incident truth. System health notes and incident investigations belong here.
- **05_BUILD/**: Architectural maps and release checklists that describe how the product ships.
- **06_PROMPTS/**: Every prompt file must live here, follow the contract in `docs/obsidian/01_RULES/RULES.md`, and include objective, scope, non-goals, acceptance criteria, tests required, rollback plan, and done-means checklist.
- **99_ARCHIVE/**: Prompts that have graduated to DONE move here so the queue only references active work.

## Workflow enforcement
1. **Queue is law**: Always pull `02_QUEUE/QUEUE.md`, pick the top READY item, mark it `IN_PROGRESS`, finish the prompt, mark it `IN_REVIEW` with a PR/description, then let the reviewer move it to `DONE`. Do not skip any lane. Use `02_QUEUE/LOCKS.md` to lock the files before editing shared vault assets.
2. **Publish for alignment**: Extend the existing context-pack automation so it also publishes snapshots named `OBSIDEAN_QUEUE_latest.md` (copy of `02_QUEUE/QUEUE.md`), `OBSIDEAN_NOW_latest.md` (copy of `01_NOW.md`), and, when there is an active prompt, `OBSIDEAN_PROMPT_latest.md` (the prompt under revision). That way I can always answer “what’s next?” by reading the same queue you see.
3. **No pile-ups**: Only one prompt can live in `IN_PROGRESS` at a time, and only one prompt can live in `IN_REVIEW`. Everything else stays in `READY` until one lane clears.
4. **Lock, queue, changelog**: Before touching `02_QUEUE/QUEUE.md` or `03_LOGS/CHANGELOG.md`, add a lock entry in `02_QUEUE/LOCKS.md`. Every completed task must transition the queue, log the work (with date/what changed/why/tests/links), and, if applicable, archive prompts in `99_ARCHIVE` after they finish.

## Context references
- Use `docs/obsidian/06_PROMPTS/2025-12-23__WORKFLOW__review-template.md` as the canonical review structure. When you review a prompt, repeat those sections verbatim.
- Treat the rules in `docs/obsidian/01_RULES/RULES.md` as the binding contract for prompts, queues, changelog entries, and lock usage.
