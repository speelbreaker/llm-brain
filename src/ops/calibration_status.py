"""Simple calibration status stubs for healthcheck."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def get_calibration_facts(
    *,
    underlying: str,
    base_dir: str | Path | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Return deterministic, stubbed calibration facts."""
    timestamp = (now or datetime.now(timezone.utc)).isoformat()
    return {
        "underlying": underlying,
        "available": True,
        "last_status": "ok",
        "age_hours": 0,
        "last_run": timestamp,
        "path": str(base_dir or Path("/tmp/calibration").resolve()),
    }
