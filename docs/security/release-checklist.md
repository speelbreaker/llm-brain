# Release Checklist

## Pre-merge
- Confirm required checks are enabled and green: GitGuardian, Secrets Scan (tripwire + secret-scan), Pytest.
- Verify ruleset/branch protection targets `main` and required checks match the workflow names.
- Ensure no new endpoints were added without tests (endpoint-level coverage required).

## Pre-deploy
- Confirm supervisor safe defaults:
  - SUPERVISOR_DEBUG is 0
  - SUPERVISOR_AUTOFIX_PUSH is 0
  - SUPERVISOR_AUTOFIX_DRY_RUN is 1
  - SUPERVISOR_ENABLE_CODEX is 0
- Confirm debug endpoint is unavailable (expected 404).
- Run `scripts/security/audit_vps.sh` on the VPS.

## Post-deploy
- `curl http://127.0.0.1:8080/health` returns 200.
- `curl http://127.0.0.1:8080/api/diag` returns 200 and contains no secret-like substrings.
- Tail supervisor logs for errors (no secrets or tokens in output).
- Reconfirm ruleset required checks are still enabled for `main`.

## Rollback
- Roll back to the prior release tag or commit.
- Re-run `scripts/security/audit_vps.sh` to confirm safe state.
- Document rollback reason and follow up with a corrective PR.
