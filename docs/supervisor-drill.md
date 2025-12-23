# Supervisor Loop Drill Runbook

This runbook describes how to execute a drill to verify the Supervisor's PR loop logic, including deterministic fixers and safety limits.

## Prerequisites

- `gh` (GitHub CLI) installed and authenticated.
- `jq` installed.
- Supervisor running locally or accessible via network.
- `SUPERVISOR_ENABLED=1`
- `GITHUB_TOKEN` and `GITHUB_WEBHOOK_SECRET` configured.

## One-Command Drill

To run the automated drill:

```bash
# Set BASE_URL to your supervisor instance (default: http://127.0.0.1:8080)
export BASE_URL=http://127.0.0.1:8080

# Run the drill
./scripts/drills/run_loop_drill.sh
```

## Drill Stages

1.  **Setup**: Checks health and dependencies. Creates a temporary git branch `drill/<timestamp>`.
2.  **Trigger**: Pushes a new file `src/_drill_lint.py` with intentional lint errors (unused imports) and opens a PR.
3.  **Observation**: polls the Supervisor API (`/api/jobs`) to track the job status.
4.  **Verification**: Asserts the final status is terminal (`fixed`, `checks_passed`, `needs_human`). Checks `dry_run` status via `/api/diag`.
5.  **Cleanup**: Closes the PR and deletes the local/remote branch.

## Expected Outcomes

-   **Dry Run Mode**: Supervisor detects the PR, identifies lint errors, attempts deterministic fix (Import/Format), but *does not* push changes. Status settles at `checks_passed` (simulated) or `needs_human`.
-   **Live Mode**: Supervisor detects PR, fixes imports/formatting, pushes a commit. Status becomes `fixed` or `checks_passed`.

## Troubleshooting

-   If job is not found: Check webhook delivery settings in GitHub or ngrok.
-   If `gh` fails: Ensure `gh auth login` is done.
