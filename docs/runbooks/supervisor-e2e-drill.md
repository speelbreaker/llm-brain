# Supervisor E2E Drill Runbook

This runbook describes how to execute the End-to-End (E2E) drill for the PR Supervisor.
This drill validates the full loop, from PR creation to fix verification, in both DRY_RUN and PUSH modes.

## Prerequisites

-   `gh` (GitHub CLI) installed and authenticated.
-   `jq` installed.
-   Supervisor running locally (e.g., in Docker or local process).
-   `SUPERVISOR_ENABLED=1`
-   `SUPERVISOR_DEBUG=1` (required for simulation trigger)
-   `GITHUB_TOKEN` and `GITHUB_WEBHOOK_SECRET` configured.

## Execution

To run the automated E2E drill:

```bash
# Set BASE_URL to your supervisor instance (default: http://127.0.0.1:8080)
# Ensure SUPERVISOR_DEBUG=1 is set in the running instance.
export BASE_URL=http://127.0.0.1:8080

# Run the drill script
./scripts/drills/run_loop_drill.sh
```

## Drill Flow

1.  **Preflight Checks**:
    -   Verifies `gh` and `jq` are present.
    -   Checks Supervisor `/health`.
    -   Checks `/api/diag` for configuration state.

2.  **Lint Drill (Deterministic)**:
    -   Creates a temporary branch `drill/lint-<timestamp>`.
    -   Commits a Python file with a known lint error (unused imports).
    -   Opens a PR.
    -   **Phase 1: Dry Run**:
        -   Triggers Supervisor via `/debug/simulate_pr_event`.
        -   Polls `/api/jobs` for the PR job.
        -   Asserts that the fix was found but **NOT** pushed (committed=false).
    -   **Phase 2: Push Mode (Simulation)**:
        -   (Note: The script currently validates Dry Run behavior primarily. Full Push mode validation requires re-configuring the running instance or manually adding the autofix label if policy requires it).
    -   **Cleanup**:
        -   Closes the PR.
        -   Deletes the remote and local branch.

## Expected Outcomes

-   **Success**: Script exits with code 0. Output confirms job status reached `checks_passed` (simulated fix) or `needs_human` (if pushed disabled) and correctly identified the deterministic fix.
-   **Failure**: Script exits with code 1. Check supervisor logs for details.

## Troubleshooting

-   **Job not found**: Ensure webhook secrets match or use the debug trigger correctly.
-   **Simulation failed**: Ensure `SUPERVISOR_DEBUG=1` is enabled on the server.