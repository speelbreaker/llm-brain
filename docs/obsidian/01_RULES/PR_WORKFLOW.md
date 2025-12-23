# PR Workflow

## Supervisor Loop Integration
All PRs are subject to the **Supervisor Loop**:
1.  **Detection**: Webhook triggers supervisor.
2.  **Verification**: Tests, Lint, Security checks run.
3.  **Fixing**: Deterministic fixers -> Codex fixers.
4.  **Debate**: Optimist/Skeptic debate the changes.
5.  **Policy**: Arbiter decides (Push/Dry-Run/Human).

## Human Role
-   **Reviewer**: Only intervenes if Supervisor escalates (NEEDS_HUMAN).
-   **Approver**: Merges PRs once Supervisor gives the green light (or manual override).

## Branching Strategy
-   `main`: Protected, deployable truth.
-   `feat/<name>`: Feature work (matches queue item).
-   `fix/<name>`: Bug fixes.
-   `drill/<name>`: Verification drills.
