# Leak Incident Drill

## What counts as a leak
- Any token, API key, secret, or credential exposed in logs, PR comments, issues, commits, artifacts, or CI output.
- Examples: "Bearer ..." headers, "sk-..." keys, GitHub PATs, webhook secrets, or plaintext database passwords.

## Immediate actions (first 15 minutes)
- Stop further exposure: halt related jobs and disable affected integrations.
- Rotate credentials: invalidate the leaked key/token at the provider and issue a new one.
- Revoke GitHub tokens and webhooks if involved (Settings -> Developer settings -> Tokens/Webhooks).
- Notify the on-call owner and open an incident record with timestamp and scope.

## Deep history scan and follow-up
- Run a deep scan locally or in CI:
  - `./scripts/security/scan_git_history_secrets.sh --deep`
- If hits are found:
  - Identify the commit range and affected files.
  - Rotate all referenced credentials immediately (assume exposure).
  - Open a follow-up task to scrub history (see below) and re-run deep scan.

## Scrub git history (high-level)
- Use a history-rewrite tool (git filter-repo or BFG) to remove the secret values.
- Re-run `./scripts/security/scan_git_history_secrets.sh --deep` to confirm clean history.
- Force-push the rewritten history and coordinate with maintainers on a repo-wide reset.

## Verify redaction + CI gates after rotation
- Confirm redaction in supervisor output:
  - `./.venv-security/bin/python -m pytest -q tests/security`
- Confirm CI gates:
  - `./scripts/security/run_security_checks.sh`
  - Ensure the "Secrets Scan / secret-scan" and "Secrets Scan / tripwire" checks are green on the PR.

## GitGuardian false-positive triage
- Capture metadata only: file path, detector name, and commit SHA.
- Do not paste any token strings, payload snippets, or full lines from the finding.
- Use internal scans (gitleaks/secret tripwire) to corroborate without exposing values.
- If it is a false positive, adjust tests or fixtures to avoid static token literals.

## If it is a real leak
- Revoke and rotate credentials immediately.
- Purge the secret from history (use a dedicated history rewrite task).
- Re-run deep history scan and document clean results.
- Add or extend a regression test to prevent recurrence.
