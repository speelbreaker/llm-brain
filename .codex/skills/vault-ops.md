# Skill: Vault Operations

## Context
We use an Obsidian vault (`docs/obsidian/`) as the single source of truth for workflow state.

## Rules
1.  **Read First**: Always read `docs/obsidian/02_QUEUE/QUEUE.md` before starting work.
2.  **Pickup Protocol**:
    -   Identify top READY item.
    -   Move to IN_PROGRESS (ensure max 1 item).
    -   Assign `branch: <name>` and `prompt: <path>`.
    -   Update `docs/obsidian/06_PROMPTS/_ACTIVE.md` with the prompt path.
3.  **Validation**: Run `python scripts/validate_vault_workflow.py` before any commit.
4.  **Commit Messages**: Use prefixes `queue: start <id>`, `queue: review <id>`, `queue: done <id>`.
