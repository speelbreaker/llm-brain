---
name: vault-workflow
description: "Vault workflow guardrails for writing prompts and reviews: use when modifying docs/obsidian assets, enforce lock entries, queue/changelog updates, and avoid checking in IDE artifacts."
---

# Vault Workflow Skill

## Purpose
Enforce the vault workflow so that every prompt/review change happens inside `docs/obsidian`, the queue and changelog stay authoritative, and IDE artifacts never sneak into commits.

## Shared-file strategy
1. **Lock first**: Before touching any shared vault file (prompt, queue, changelog, locks, or other `docs/obsidian` assets), append a lock record to `docs/obsidian/02_QUEUE/LOCKS.md` documenting the files you will edit, your agent name, start time, and how you will know you are done.
2. **Single directory for prompts**: Only create or edit prompt files under `docs/obsidian/06_PROMPTS/`. Do not scatter prompts elsewhere or copy/paste content into non-vault locations.
3. **Queue and changelog updates**: Every completed task must update `docs/obsidian/02_QUEUE/QUEUE.md` (moving the appropriate item to Done with a timestamp) and append an entry to `docs/obsidian/03_LOGS/CHANGELOG.md` describing the date, changes, rationale, tests, and links.
4. **Respect ignores**: Never commit `.obsidian/` or `.vscode/` artifacts. Confirm those directories are ignored via `.gitignore` (there is a pytest guarding this). If you notice the patterns missing, inform the user before making changes.
5. **Documentation consistency**: When referencing prompt/review templates, point people to `docs/obsidian/06_PROMPTS/2025-12-23__WORKFLOW__review-template.md` for structured examples.

## Prompt Template
- Start each prompt file on its own line with a heading such as `# WORKFLOW — ...` so readers know the intent.
- Include the required sections from the vault rules: `Objective`, `Scope`, `Non-goals`, `Acceptance Criteria` (with bulletproof criteria), `Tests required`, `Rollback plan`, and `Done means` (a checklist). Add any additional context subsections only after satisfying these core sections.
- Keep the prompt focused on actions specific to `docs/obsidian/06_PROMPTS/`. Mention any touched files under `Scope` (restrict to the vault file path) and keep descriptions terse.
- When finishing the prompt change, confirm the queue entry moves to Done and log the work; do not leave prompts hanging without the associated queue/changelog update.

## Review Template
- Use `docs/obsidian/06_PROMPTS/2025-12-23__WORKFLOW__review-template.md` as the canonical sample for `Summary`, `Risks`, `What to verify`, `Files touched`, `Tests run`, and `Verdict (PASS/WARN/FAIL)` sections.
- Every review output referencing this skill should repeat that structure, capturing the high-level impact (Summary), outstanding concerns (Risks), concrete checks (What to verify), touched files, test outcomes, and an explicit verdict.
- If you notice deviations from the template or missing sections when reviewing a vault change, update the review prompt/review output so the structure is complete before saying the task is done.
