---
name: ops-health
description: "Ops health discipline: use when addressing OPS_HEALTH artifacts, health dashboards, and publishing reliability."
---

# Ops Health Skill

## Use when
- Changes touch `src/healthcheck`, `scripts/gen_ops_health_latest.py`, or any gadget that drives `OPS_HEALTH_latest.json`.
- You are improving the health dashboard, contract enforcement, or context-pack publishing pipeline.

## Instructions
1. Enforce the contract that `overall_status` is OK/WARN/FAIL, `can_trade` is boolean, `worst_severity` is never null, and `summaries` stay non-empty (fail-closed if generation errors occur).
2. Keep the health generator deterministic (use `CONTEXT_PACK_FAKE_OPS_HEALTH` for tests) and avoid touching network services.
3. Before merging, run the ops health contract tests (e.g., `tests/test_ops_health_contract.py`) and document results in `docs/obsidian/03_LOGS/CHANGELOG.md`.
4. Update the context-pack publishing automation to include `OPS_HEALTH_latest.json` snapshots on Drive as described in the vault `00_README`.
