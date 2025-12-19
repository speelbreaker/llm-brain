# Secret Rotation Runbook

This repo uses a PR Supervisor container with several secrets. Follow these steps to rotate keys safely.

## Secrets and env vars
- OpenAI: `OPENAI_API_KEY`
- GitHub: `GITHUB_TOKEN`
- Telegram: `TELEGRAM_BOT_TOKEN`
- Gemini (optional): `GEMINI_API_KEY`
- Webhook secret: `GITHUB_WEBHOOK_SECRET` (unchanged unless rotating)

Env file location (default): `docker/pr-supervisor.env` (or use `SUPERVISOR_ENV_FILE` when running rotation script).

## Rotation steps
1) Prepare
   - Stop the supervisor container.
   - Backup env: `cp docker/pr-supervisor.env docker/pr-supervisor.env.$(date +%Y%m%d-%H%M%S).bak`
2) Rotate
   - Run: `scripts/security/rotate_supervisor_secrets.sh`
   - Enter new values (leave blank to skip any).
3) Restart supervisor
   - Apply new env: `docker/run_pr_supervisor.sh` (or your deployment restart command).
4) Validate
   - `curl http://127.0.0.1:8080/health` should be ok.
   - `curl http://127.0.0.1:8080/api/diag` should show `enabled=true` and no errors (tokens are redacted).

## Incident response (secret leaked in git)
1) Immediately rotate the affected secret(s) using the steps above.
2) Rewrite history to purge the secret (e.g., `git filter-repo` or GitHub’s secret scanning remediation).
3) Force-push cleaned history if this repo allows it; notify collaborators.
4) Invalidate tokens at the provider (GitHub PAT, OpenAI key, Telegram bot token, Gemini).
5) Re-run scans:
   - `scripts/security/scan_worktree_secrets.sh`
   - `scripts/security/scan_git_history_secrets.sh deep`
6) Restart supervisor with the rotated secrets.

## Policy
- Keep `SUPERVISOR_AUTOFIX_PUSH` off by default in production unless explicitly needed.
- Keep `SUPERVISOR_DEBUG` off and `SUPERVISOR_DEBUG_TOKEN` set when debug is enabled.
- Never commit env files or key material (.env*, docker/*.env, *.pem, *.key, key.md*).
