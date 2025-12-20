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
