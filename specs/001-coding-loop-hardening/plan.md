# Implementation Plan: Coding Loop Hardening

**Branch**: `feature/001-coding-loop-hardening` | **Date**: 2025-12-27 | **Spec**: [specs/001-coding-loop-hardening/spec.md]
**Input**: Feature specification from `specs/001-coding-loop-hardening/spec.md`

## Summary

This feature hardens the Supervisor Coding Loop by enabling LLM providers (currently disabled), adding critical safety guards (webhook validation, size limits, timeouts), and improving operational reliability (cleanup, observability). It builds upon the existing "Optimist/Skeptic/Arbiter" architecture.

## Technical Context

**Language/Version**: Python 3.12+
**Primary Dependencies**: FastAPI, Pydantic, Httpx, Ruff (for deterministic fixes), OpenAI/Gemini SDKs.
**Storage**: In-memory `JobStore` (persisted via simple file dumps/snapshots).
**Testing**: `pytest` with `pytest-asyncio`.
**Target Platform**: Dockerized Linux environment (VPS).
**Performance Goals**: Job overhead < 5s (excluding LLM/Network latency).
**Constraints**: strictly no secrets in logs; deterministic fixers preferred over LLM.

## Constitution Check

*GATE: Passed. Regression Gates and Security Gates are primary drivers for this hardening work.*

## Project Structure

### Documentation (this feature)

```text
specs/001-coding-loop-hardening/
├── plan.md              # This file
├── spec.md              # Requirements & Stories
└── tasks.md             # Implementation Tasks
```

### Source Code (repository root)

```text
src/supervisor/
├── app.py              # Entrypoint & Loop Logic
├── config.py           # Settings
├── models.py           # Data Models
├── github.py           # Webhook & API handling
├── workspace.py        # Git operations & Cleanup
├── telegram_notify.py  # Notifications
├── llm/                # LLM Providers
│   ├── base.py
│   ├── router.py
│   └── ...
└── loop/               # Core Loop Logic
    ├── fixers.py       # Deterministic Fixers
    └── ...
```

**Structure Decision**: Enhancing existing `src/supervisor` structure. No new top-level directories required.

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| N/A | N/A | N/A |
