# Feature Specification: Coding Loop Hardening

**Feature Branch**: `feature/001-coding-loop-hardening`  
**Created**: 2025-12-27  
**Status**: Draft  
**Input**: Audit Report (docs/CODING_LOOP_STATUS_REPORT.md)

## User Scenarios & Testing

### User Story 1 - Enable LLM-Powered Fixes (Priority: P1)

As a developer, I want the Supervisor to utilize LLM providers (Codex) for complex linting and logic fixes so that issues not resolvable by deterministic fixers can be automatically addressed.

**Why this priority**: Currently, `codex_available` is false, limiting the loop to simple formatting fixes. Unlocking LLM capabilities is the core value proposition of the "Coding Loop".

**Independent Test**:
- Mock/Configure LLM provider.
- Trigger a job with a lint error that `ruff` cannot fix automatically (e.g., variable renaming or logic adjustment).
- Verify `FIX_LINT` stage uses the LLM and successfully proposes a fix.

**Acceptance Scenarios**:

1. **Given** valid LLM credentials, **When** `/api/diag` is called, **Then** `codex_available` should be `true`.
2. **Given** a non-deterministic lint error, **When** the loop enters `FIX_LINT`, **Then** it should query the LLM and apply the suggestion.
3. **Given** an LLM failure/timeout, **When** debating, **Then** the system should gracefully fallback to deterministic-only or fail-closed (without crashing).

---

### User Story 2 - Loop Safety Guards (Priority: P2)

As an operator, I want the loop to abort if fixes are too large, runtimes are excessive, or webhooks are invalid, so that the system remains safe and does not consume excessive resources or merge risky changes.

**Why this priority**: Prevents runaway costs (tokens/compute) and security risks (unverified webhooks, massive diffs).

**Independent Test**:
- **Fix Too Large**: Mock a fix returning >50 files changed. Verify job halts with `reason_code: FIX_TOO_LARGE`.
- **Timeout**: Set `MAX_TOTAL_RUNTIME` to 1s. Verify job halts immediately.
- **Signature**: Send webhook with invalid signature. Verify 401/403 rejection.

**Acceptance Scenarios**:

1. **Given** a proposed fix exceeding `max_files_changed`, **When** applied, **Then** the job transitions to `NEEDS_HUMAN` with correct reason code.
2. **Given** a running job exceeding `MAX_TOTAL_RUNTIME`, **When** the check runs, **Then** the job is terminated.
3. **Given** a GitHub webhook payload, **When** received, **Then** the `X-Hub-Signature-256` header must be validated against the secret.

---

### User Story 3 - Observability & Cleanup (Priority: P3)

As an operator, I want clearer diff summaries in notifications and reliable workspace cleanup so that I can trust the system state and easily understand what changed.

**Why this priority**: Operational stability and better human-in-the-loop decision making.

**Independent Test**:
- **Cleanup**: Run 10 jobs. Verify `/tmp/supervisor` has no stale directories.
- **Notification**: Trigger a fix. Verify Telegram message includes "+X/-Y lines, Z files changed".

**Acceptance Scenarios**:

1. **Given** a completed job, **When** finalized, **Then** its temporary workspace is fully deleted.
2. **Given** a fix applied, **When** notifying via Telegram, **Then** `diff_stats` are included in the message.

## Requirements

### Functional Requirements

- **FR-001**: System MUST validate GitHub Webhook Signatures.
- **FR-002**: System MUST abort jobs if `files_changed` or `loc_changed` exceed configured thresholds.
- **FR-003**: System MUST terminate jobs exceeding `MAX_TOTAL_RUNTIME`.
- **FR-004**: System MUST enable/disable LLM features based on provider availability checks.
- **FR-005**: System MUST clean up workspace directories after job completion (success or failure).
- **FR-006**: System MUST report `diff_stats` in notifications.
- **FR-007**: System MUST use standardized attempt counters for all stages.

### Key Entities

- **SupervisorJob**: Extended with `attempt_counters` normalization and `diff_stats`.
- **VerificationReport**: Extended to support `FixMode.LINT_ONLY` targeting.
- **WorkspaceManager**: Enhanced with rigorous cleanup logic.

## Success Criteria

### Measurable Outcomes

- **SC-001**: `/api/diag` returns `codex_available: true`.
- **SC-002**: 100% of invalid webhooks are rejected.
- **SC-003**: 0% of workspaces remain on disk after 1 hour (orphaned).
- **SC-004**: `run_loop_drill.sh` passes end-to-end in Dry Run mode on the live repo.
