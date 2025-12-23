# System Health

## Snapshot process
- Record the status of data feeds, Deribit gateway, and any backtest/agent loops each shift.
- Include metrics for latency, queue depth, and permission gate hits.

## Alerts
- Every alert must reference a monitoring runbook; do not silence without follow-up.
- Health assessments are shared during publishing (see `OBSIDEAN_QUEUE_latest.md`).
