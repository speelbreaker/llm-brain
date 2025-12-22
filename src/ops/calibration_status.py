from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def get_calibration_facts(
    *,
    underlying: str,
    base_dir: str | Path | None = None,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Cheap calibration facts lookup.

    Uses the same sources as the existing healthcheck:
    - src.calibration_update_policy.load_recent_calibration_history
    - src.calibration_store.get_applied_multiplier

    Does not run calibration.
    """
    u = (underlying or "").upper().strip()
    now_dt = now or datetime.now(timezone.utc)

    try:
        from src.calibration_update_policy import get_calibration_runs_dir, load_recent_calibration_history
        from src.calibration_store import get_applied_multiplier
    except Exception as e:
        return {
            "underlying": u,
            "available": False,
            "last_calibration_at": None,
            "age_hours": None,
            "last_status": "unknown",
            "iv_multiplier_current": None,
            "source": "missing",
            "path": None,
            "base_dir": str(Path(base_dir)) if base_dir is not None else None,
            "reason": f"calibration_modules_unavailable: {e}",
        }

    runs_dir = get_calibration_runs_dir(base_dir)

    record = None
    try:
        history = load_recent_calibration_history(u, limit=1, base_dir=runs_dir)
        if history:
            record = history[0]
    except Exception:
        record = None

    if record is None:
        return {
            "underlying": u,
            "available": False,
            "last_calibration_at": None,
            "age_hours": None,
            "last_status": "unknown",
            "iv_multiplier_current": None,
            "source": "missing",
            "path": str(runs_dir),
            "base_dir": str(runs_dir),
            "reason": "no_calibration_records",
        }

    record_data = record.model_dump()
    status_field = str(record_data.get("status") or "").lower()
    applied = bool(record_data.get("applied", False))
    last_at = record.timestamp
    age_hours = max(0.0, (now_dt - last_at).total_seconds() / 3600.0)

    if status_field in ("failed", "error"):
        last_status = "failed"
    elif applied:
        last_status = "applied"
    else:
        last_status = "blocked"

    applied_state = get_applied_multiplier(u)
    iv_current = applied_state.global_multiplier if applied_state else None

    last_applied_at = (
        applied_state.last_updated.isoformat() if applied_state and applied_state.last_updated else None
    )

    return {
        "underlying": u,
        "available": True,
        "last_calibration_at": last_at.isoformat(),
        "age_hours": age_hours,
        "last_status": last_status,
        "iv_multiplier_current": iv_current,
        "last_applied_at": last_applied_at,
        "source": "calibration_history",
        "path": str(runs_dir),
        "base_dir": str(runs_dir),
        "reason": record_data.get("applied_reason"),
    }
