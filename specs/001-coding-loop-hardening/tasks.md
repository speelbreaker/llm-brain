---
description: "Task list for Coding Loop Hardening"
---

# Tasks: Coding Loop Hardening

**Input**: Feature Spec from `specs/001-coding-loop-hardening/spec.md`
**Prerequisites**: plan.md

## Phase 1: User Story 1 - Enable LLM-Powered Fixes (Priority: P1)

**Goal**: Activate LLM providers to allow complex lint/logic fixes.

**Independent Test**: `/api/diag` returns `codex_available: true`; `FIX_LINT` stage uses LLM.

### Implementation for User Story 1

- [ ] T001 [US1] Verify/Update LLM Provider Config in `src/supervisor/config.py` (ensure API keys read correctly)
- [ ] T002 [US1] Update `src/supervisor/llm/router.py` to correctly instantiate providers based on config
- [ ] T003 [US1] Implement "check availability" logic in `src/supervisor/app.py` for `/api/diag` endpoint
- [ ] T004 [US1] Verify `DebateSystem` in `src/supervisor/debate.py` correctly handles LLM failures (fallback tests)

---

## Phase 2: User Story 2 - Loop Safety Guards (Priority: P2)

**Goal**: Prevent excessive resource usage and insecure webhook handling.

**Independent Test**: Invalid signatures rejected; "Fix Too Large" jobs halted; Timeouts enforced.

### Tests for User Story 2

- [ ] T005 [P] [US2] Unit test for `verify_signature` in `tests/supervisor/test_github.py`
- [ ] T006 [P] [US2] Unit test for "Fix Too Large" logic (mock `DiffStats`) in `tests/supervisor/test_loop.py`

### Implementation for User Story 2

- [ ] T007 [US2] Implement `verify_signature` in `src/supervisor/github.py`
- [ ] T008 [US2] Enforce signature check in `webhook_handler` in `src/supervisor/app.py`
- [ ] T009 [US2] Implement "Fix Too Large" check (files/loc thresholds) in `src/supervisor/app.py`
- [ ] T010 [US2] Implement `MAX_TOTAL_RUNTIME` timeout check in main loop `src/supervisor/app.py`

---

## Phase 3: User Story 3 - Reliability & Observability (Priority: P3)

**Goal**: Clean workspaces, clear notifications, and standardized counters.

**Independent Test**: Workspaces deleted after use; Telegram messages show diff stats.

### Implementation for User Story 3

- [ ] T011 [US3] Refactor `attempt_counters` to be consistent across all stages in `src/supervisor/models.py` & `app.py`
- [ ] T012 [US3] Harden `cleanup_workspace` in `src/supervisor/workspace.py` (retry logic / force delete)
- [ ] T013 [US3] Update `TelegramNotifier` in `src/supervisor/telegram_notify.py` to include `DiffStats`
- [ ] T014 [US3] Add `FixMode.LINT_ONLY` test coverage in `tests/supervisor/loop/`

---

## Phase 4: Polish & Verification

**Goal**: End-to-end verification and documentation.

- [ ] T015 Run `scripts/drills/run_loop_drill.sh` (Dry Run) on live repo to verify E2E
- [ ] T016 Update `docs/supervisor-loop.md` with new configuration options (Push Mode, Limits)
- [ ] T017 Finalize `docs/CODING_LOOP_STATUS_REPORT.md` with post-hardening results
