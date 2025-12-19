# Push Mode Runbook

Use this guide to enable and operate supervisor push-mode safely.

## Defaults (safe)
- `SUPERVISOR_AUTOFIX_PUSH=0` (push **off**)
- `SUPERVISOR_AUTOFIX_DRY_RUN=1` (dry-run on)
- `SUPERVISOR_ENABLE_CODEX=0` unless explicitly enabled
- Debug endpoints off by default

## Enabling push-mode (label-gated)
1. Set env vars in the supervisor container:
   - `SUPERVISOR_AUTOFIX_PUSH=1`
   - `SUPERVISOR_AUTOFIX_DRY_RUN=0`
   - `SUPERVISOR_ENABLE_CODEX=1`
2. Ensure the PR has label `autofix-ok` (required by policy).
3. Restart the supervisor container.

## What gets pushed
- Scope: lint-only fixes (e.g., ruff-unused-import) approved by policy/arbiter.
- Commit author: `PR Supervisor`.
- Audit trail: PR comments reference the run and pushed commit SHA.

## Rollback / disable
- Set `SUPERVISOR_AUTOFIX_PUSH=0` (optionally re-enable dry-run with `SUPERVISOR_AUTOFIX_DRY_RUN=1`).
- Restart the supervisor container.
- Removes further auto-push; manual git reset if a pushed commit must be reverted.

## Guardrails
- Push-mode requires both env + label gating.
- Debug endpoints stay off by default; when enabled they require token + localhost.
- Production profile should keep: push off, debug off, dry-run on.
