# Queue Discipline

## The Golden Rule
**The Queue is Law.** No work happens unless it is tracked in `docs/obsidian/02_QUEUE/QUEUE.md`.

## Workflow States
1.  **READY**: Spec is complete, prompts are written, tests defined. Ready to be picked up.
2.  **IN_PROGRESS**: Active development. **Max 1 item.**
    -   Must have a `branch: <name>` assigned.
    -   Must have `_ACTIVE.md` pointing to its prompt file.
3.  **IN_REVIEW**: Code is pushed, PR is open. **Max 1 item.**
    -   Supervisor loop is verifying/fixing.
4.  **DONE**: Merged to main.

## Transition Rules
-   **Pickup**: Move top READY -> IN_PROGRESS. Create branch. Update _ACTIVE.md.
-   **Review**: Move IN_PROGRESS -> IN_REVIEW. Open PR.
-   **Complete**: Move IN_REVIEW -> DONE. Close/Merge PR. Archive prompt.

## Violation Handling
If the validator (`scripts/validate_vault_workflow.py`) fails, **STOP**. Fix the queue state immediately.
