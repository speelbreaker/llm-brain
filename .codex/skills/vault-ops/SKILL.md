---
name: vault-ops
description: "Vault operations guardrail: use when any agent needs to create or update prompts, queues, or changelog entries inside docs/obsidian."
---

# Vault Ops Skill

## Use when
- A new prompt file is being created or updated under `docs/obsidian/06_PROMPTS/`.
- Shared vault files (`02_QUEUE/QUEUE.md`, `03_LOGS/CHANGELOG.md`, locks) need editing because of that prompt.

## Instructions
1. Lock the vault files you will touch in `docs/obsidian/02_QUEUE/LOCKS.md` before editing.
2. Keep all prompt changes under `docs/obsidian/06_PROMPTS/` and follow the prompt contract in `docs/obsidian/01_RULES/RULES.md`.
3. After finishing the prompt, move the queue item through READY → IN_PROGRESS → IN_REVIEW → DONE and append a dated entry to `docs/obsidian/03_LOGS/CHANGELOG.md`.
4. Only one prompt may be IN_PROGRESS or IN_REVIEW at a time; all others stay READY.
5. When reviewing prompts, mirror the structure of `docs/obsidian/06_PROMPTS/2025-12-23__WORKFLOW__review-template.md`.
