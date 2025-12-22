"""Stubs for harvest snapshot status checks."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def get_harvest_root(base_dir: str | Path | None = None) -> str:
    """Return the harvest data base directory."""
    return str(base_dir or "/tmp/harvest")


def harvest_freshness_for_underlying(
    *,
    underlying: str,
    base_dir: str | Path | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Return dummy freshness data for an underlying."""
    timestamp = now or datetime.now(timezone.utc)
    return {
        "status": "OK",
        "age_minutes": 0,
        "last_snapshot_at": timestamp.isoformat(),
        "latest_file": None,
        "harvest_dir": str(base_dir or get_harvest_root(base_dir)),
        "dirs_checked": [],
    }
