# Coding Loop Gaps and Next Steps

## Constraints to Stable Autopilot
The primary constraint blocking "stable autopilot" is the **unavailability of the Codex (LLM) backend** in the current environment (`codex_available=false`), despite the system being configured for `push_enabled=true`. This means the loop acts as a "Deterministic Only" fixer, falling back to human for anything requiring reasoning (Lint/Logic errors).

## Top 5 Risks

| Severity | Risk | Mitigation |
| :--- | :--- | :--- |
| **High** | **Codex Unavailability:** `FIX_LINT` stage will fail or degrade to comment-only, creating noise without fixes. | **Enable Codex** or fallback to `dry_run` if LLM is down. Ensure `LLMFailure` is handled gracefully (verified in code). |
| **Medium** | **State Confusion:** Overlap between `JobStage` and `JobStatus` logic in `app.py` makes it hard to trace exact lifecycle in logs. | Refactor `SupervisorJob` to have a single, clear State Machine with sub-states if needed. |
| **Medium** | **Drill Fragility:** End-to-end drills rely on `gh` CLI and modifying remote branches, making them hard to run in CI/CD or restricted envs. | Create a "Local Drill" mode that simulates GH events without external API calls (using `/debug/simulate_pr_event`). |
| **Low** | **Legacy Code:** Presence of standalone fixer functions alongside `DeterministicFixer` class suggests incomplete refactor. | Deprecate and remove legacy functions in `fixers.py`. |
| **Low** | **Endpoint Security:** `/api/diag` exposes build info and internal config without obvious auth (checked via curl). | Verify upstream auth (nginx/VPS) protects these endpoints. |

## Next 10 Concrete Steps

| Owner | Task | Effort | Proof of Done |
| :--- | :--- | :--- | :--- |
| **Human** | **Fix Codex Config:** Investigate why `codex_available` is false. Check credentials/env vars. | S | `/api/diag` returns `"codex_available": true`. |
| **Codex** | **Refactor State Machine:** Consolidate `JobStatus` and `JobStage` usage in `app.py` and `models.py`. | M | PR merged; logs show clear transitions without ambiguity. |
| **Codex** | **Local Drill Script:** Create `scripts/drills/run_local_drill.sh` using `/debug/simulate_pr_event`. | M | Drill runs successfully without `gh` CLI or internet. |
| **Codex** | **Cleanup Fixers:** Remove legacy standalone functions in `src/supervisor/loop/fixers.py`. | S | Codebase cleaner; no broken imports; tests pass. |
| **Codex** | **Add Endpoint Auth Tests:** Verify if `/api/diag` is protected or intended to be public. Add tests if needed. | S | Security test report. |
| **Codex** | **Expand Deterministic Fixers:** Add `FIX_TYPE_HINT` mode to `DeterministicFixer` (e.g., using `monkeytype` or similar if feasible). | L | New mode in `FixMode`; tests passing. |
| **Codex** | **Standardize Limits:** Move hardcoded limits in `app.py` to `SupervisorSettings`. | S | Configurable limits via env vars. |
| **Codex** | **Enhance Redaction:** Ensure `redact_secrets` covers all new fields in `ArbiterDecision`. | S | Tests confirming redaction in all JSON outputs. |
| **Human** | **Rotate Secrets:** If Codex/GH tokens were exposed or invalid, rotate them. | S | New secrets active; old revoked. |
| **Codex** | **Docs Update:** Update `docs/supervisor-loop.md` to reflect the solidified "Deterministic + Arbiter" architecture. | S | Docs match code exactly. |
