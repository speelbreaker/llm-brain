# CLI Agent Context

## Project: llm-brain-fix
Modular framework for automated options trading and PR supervision.

## Current Loop Stage
The Supervisor uses a hardened loop with the following stages:
- **DEBATE**: LLM analyzes changes and decides if a fix is allowed.
- **FIX_LINT**: LLM-based fix for complex lint/logic errors.
- **FIX_IMPORT**: Deterministic fix for import sorting (`ruff check -I --fix`).
- **FIX_FORMAT**: Deterministic fix for formatting (`ruff format`).
- **FIX_TESTS**: Deterministic cleanup (`ruff format`) and re-verification.
- **VERIFY**: Runs project-specific checks.

## Safety Rules
1.  **No Secrets**: Never print or commit secrets. Use `redact_secrets` logic.
2.  **Clean Worktree**: Always work from a clean tree/branch.
3.  **Deterministic First**: Prefer deterministic fixers over LLM.
4.  **Loop Limits**: Respect per-stage attempt limits and total runtime.
5.  **Dry Run**: Respect `SUPERVISOR_DRY_RUN` setting.
