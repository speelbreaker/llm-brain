# Skill: Ops Health

## Context
System health is monitored via `docs/OPS_HEALTH_latest.json`.

## Rules
1.  **Contract**: The JSON schema is strict (see `src/healthcheck.py` and `scripts/gen_ops_health_latest.py`).
2.  **Fail Closed**: If health generation fails, we assume CRITICAL state and stop trading.
3.  **Verification**: Always verify `python scripts/gen_ops_health_latest.py` output after changes.
