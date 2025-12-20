Security tooling
================

Pre-commit hooks
----------------
1. Install pre-commit: `pip install pre-commit`
2. Install hooks: `scripts/security/install_hooks.sh`
3. Hooks run gitleaks with redaction and block commits containing secrets.

Manual scans
------------
- Working tree: `scripts/security/scan_worktree_secrets.sh`
- Staged changes: `scripts/security/scan_staged_secrets.sh`
- Git history: `scripts/security/scan_git_history_secrets.sh [deep|shallow]` (deep scans full history; can be slow)
- Scripts prefer local gitleaks, then Docker (`ghcr.io/gitleaks/gitleaks:latest`), then a temp download fallback.
- Override the Docker image with `GITLEAKS_IMAGE=ghcr.io/gitleaks/gitleaks:latest`.
- Tool-missing exit code is `2` (leaks return `1`); in CI it is treated as a failure.

Recommended VPS flow
--------------------
1. `chmod +x scripts/security/*.sh`
2. `scripts/security/run_security_checks.sh`
   - Runs worktree/staged/shallow scans.
   - Runs redaction + tripwire tests inside the `pr-supervisor` container when available,
     otherwise uses a local `.venv-security` with only pytest installed.
   - Host fallback runs only `tests/security/test_secret_tripwire.py` and
     `tests/security/test_redact_minimal.py` with `PYTHONPATH` set to the repo root.
   - Mode is printed as `Mode: container` or `Mode: host-minimal`.

Minimal host requirements
-------------------------
- `python3`, `python3 -m venv`, `pip`
- `git` (for the tripwire scan)
- Docker is optional (used for container mode or gitleaks fallback)

Sample mode output
------------------
```
Mode: container
```
or
```
Mode: host-minimal
```

Incident response quick steps
-----------------------------
1. Rotate the exposed key (use `scripts/security/rotate_supervisor_secrets.sh`).
2. Invalidate the leaked token at the provider (GitHub/OpenAI/Telegram/etc.).
3. Re-run `scripts/security/run_security_checks.sh` to confirm scans are clean.

Rotation helper
---------------
- Interactive env rotation: `scripts/security/rotate_supervisor_secrets.sh [--restart]`
  - Backs up the env file, prompts for new keys without echoing, sets chmod 600.
  - Optionally restarts via `docker/run_pr_supervisor.sh` when `--restart` is passed.
