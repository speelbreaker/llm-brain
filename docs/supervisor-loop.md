Supervisor Loop

Overview
The supervisor runs a deterministic loop with explicit roles:
Optimist proposes a fix plan, Skeptic evaluates guardrails and risk, Arbiter decides deny/dry_run/push, and the Executor applies fixes (deterministic first, Codex second if enabled).

Built-in Categories
- lint_only: Ruff-only failures (unused import F401) auto-fixed with ruff --fix.
- single_test_env_leak: Single pytest failure tied to os.environ or patch.dict; fixes patch.dict(..., clear=False) to clear=True.
- format_only: Reserved for future formatter-only fixes.

Scenarios
A. Lint-only (dry-run)
- Pytest passes, ruff fails.
- Optimist classifies lint_only.
- Skeptic approves low risk.
- Arbiter returns dry_run by default.
- Deterministic fixer runs ruff --fix, re-runs checks, marks FIXED without pushing.

B. Single-test env leak (dry-run)
- One pytest test fails due to leaked environment variables.
- Optimist classifies single_test_env_leak.
- Skeptic approves low risk.
- Arbiter returns dry_run by default.
- Deterministic fixer sets patch.dict(..., clear=True), re-runs pytest, marks FIXED without pushing.

Push Gating
Push is only allowed when:
- SUPERVISOR_AUTOFIX_PUSH=1
- PR has the autofix-ok label
Otherwise, fixes are validated in dry-run mode.

Loop Stages
- RECEIVED: Job accepted and stored.
- ANALYZING: Workspace setup and initial checks running.
- DEBATING: Optimist/Skeptic/Arbiter decision phase.
- BYPASSED: Debate skipped (checks already passed).
- FIXING: Deterministic or Codex fixes in progress.
- SKIPPED: Fixing skipped (auto-fix denied or not applicable).
- VERIFYING: Final verification recorded (pass/fail).
- COMMENTING: PR comment updated with results.
- DONE: Terminal state for the run.

Limits and Backoff
- SUPERVISOR_MAX_FIX_ATTEMPTS: Hard cap on total fix attempts.
- SUPERVISOR_MAX_TOTAL_RUNTIME_SECONDS: Max wall-clock runtime per job.
- SUPERVISOR_FIX_BACKOFF_BASE_SECONDS / FACTOR / MAX_SECONDS: Backoff before retrying fixes.

Reason Codes
- LOOP_LIMIT: A stop condition triggered; final_message states the specific limit hit.

Extending Categories
1) Add the category to src/supervisor/loop/policy_defaults.json.
2) Teach Optimist to recognize it in src/supervisor/loop/optimist.py.
3) Add a deterministic fixer in src/supervisor/loop/fixers.py.
4) Add/adjust Skeptic guardrails in src/supervisor/loop/skeptic.py.
5) Add a focused test under tests/supervisor.
