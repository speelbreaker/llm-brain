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
- Scripts use local gitleaks if present, fall back to docker if available, or download a temp static gitleaks binary (no install).

Rotation helper
---------------
- Interactive env rotation: `scripts/security/rotate_supervisor_secrets.sh [--restart]`
  - Backs up the env file, prompts for new keys without echoing, sets chmod 600.
  - Optionally restarts via `docker/run_pr_supervisor.sh` when `--restart` is passed.
