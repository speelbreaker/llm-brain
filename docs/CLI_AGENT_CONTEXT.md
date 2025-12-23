# LLM-Brain — CLI Agent Context

## What this repo is
We are building an automated crypto-options trading platform (Deribit-focused). The current focus is the **PR Supervisor** system: a service that receives PR/webhook events, runs checks, debates fixes (Optimist/Skeptic/Arbiter), and applies fixes via deterministic tools and/or Codex, with safe defaults and strong auditability.

## Current stage
We are finishing and hardening the **closed loop**:
1. Detect PR → enqueue job.
2. Run checks (pytest/ruff/etc.).
3. If failures:
   - “Optimist” proposes a fix approach.
   - “Skeptic” tries to break it / finds risks.
   - “Arbiter” decides approve/deny + sets fix objectives.
4. Fixer executes (Codex or deterministic fallback).
5. Verify checks.
6. Post an idempotent PR comment with results.
7. Enforce loop limits (attempt caps, runtime caps, backoff) + stage history.

Key expectation: **full observability** about why it acted, what changed, what checks failed/passed, which safety gates triggered, and what stage the job ended in (and why).

## Non-negotiables
1. **Never leak secrets.** Do not print tokens/keys/env file contents or add sensitive files like `.env*`, `docker/*.env`, or `*.pem`. Always confirm `git diff --cached --name-only` before committing and redacted job outputs/logs.
2. **Keep changes minimal and scoped.** Fix the bottleneck first, keep CI green, and avoid unrelated refactors or mass reformatting.
3. **Endpoint-level test requirement.** Every endpoint change must come with at least one endpoint-level test.
4. **Safe defaults.** Debug features off by default, push mode off unless needed, dry-run as default unless policy allows otherwise.

## Operating constraints
- Tests may lack host dependencies (FastAPI/Pydantic/httpx); prefer running `docker exec -i -w /app pr-supervisor python3 -m pytest -q tests/supervisor` when possible.
- Keep the worktree clean; if unknown dirty changes exist, stop and clone fresh rather than mixing them.
- Always respect stage/order invariants for loops, attempt counters, and PR comments.

## Evidence format (always include in the final report)
1. `git diff --stat origin/main..HEAD`
2. Commands run + short results
3. Any relevant job id/status (if using `/jobs`)
4. PR link(s)
5. “No sensitive files staged/committed.”
