# Coding Loop Gaps and Next Steps

## 1. Constraints
- **Missing LLM Context:** Currently, `codex_available` is `false`. This blocks LLM-driven fixes for complex logic, leaving only deterministic fixers active.
- **Environment Isolation:** Testing currently relies on a shared `pr-supervisor` container. Parallel jobs might conflict if not properly isolated at the workspace level.
- **Webhook Latency:** PR file fetching via GitHub API can be slow for large PRs.

## 2. Top 5 Risks
| Risk | Severity | Mitigation |
| :--- | :--- | :--- |
| **LLM Hallucination** | High | Arbiter gating + Verification runner (tests must pass). |
| **Token Exhaustion** | Medium | Loop limits + deterministic-first priority. |
| **Secret Leak in Logs** | High | `redact.py` scrubbing + CI-level secret scanning. |
| **Force-Push Conflict** | Medium | Git branch state checking in `workspace.py`. |
| **Dependency Flakiness** | Low | Targeted test reruns in `FixMode.TESTS`. |

## 3. Next 10 Concrete Steps

| Step | Owner | Effort | Proof of Done |
| :--- | :--- | :--- | :--- |
| 1. Enable LLM Providers | Human | S | `/api/diag` shows `codex_available: true` |
| 2. Implement "Fix Too Large" Check | Codex | S | Job fails with `reason_code: FIX_TOO_LARGE` |
| 3. Add `FixMode.LINT_ONLY` Test Coverage | Codex | M | New test in `tests/supervisor/loop/` |
| 4. Standardize Attempt Counters | Codex | S | All stages use `job.attempt_counters` |
| 5. Harden Workspace Cleanup | Human | S | No orphan directories in `/tmp/supervisor/` |
| 6. Implement Webhook Signature Validation | Codex | M | `verify_signature` returns true for valid payloads |
| 7. Add Job Timeout Middleware | Codex | S | Jobs killed after `MAX_TOTAL_RUNTIME` |
| 8. Enhance Telegram Diff Previews | Codex | M | Notification includes `diff_stats` summary |
| 9. E2E Drill on Live Repo (Dry Run) | Human | M | `run_loop_drill.sh` passes end-to-end |
| 10. Document "Push Mode" Onboarding | Human | S | New section in `docs/supervisor-loop.md` |