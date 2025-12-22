from __future__ import annotations

from datetime import datetime, timezone, date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import os


_DEFAULT_HARVEST_DIR = Path("data/live_deribit")


def get_harvest_root(base_dir: str | Path | None = None) -> Path:
    if base_dir is not None:
        return Path(base_dir)
    # Preferred canonical env (used by ops facts resolver)
    override = os.environ.get("LIVE_DERIBIT_DATA_DIR")
    if override:
        return Path(override)
    # Back-compat
    override2 = os.environ.get("HARVEST_DATA_DIR")
    if override2:
        return Path(override2)
    return _DEFAULT_HARVEST_DIR


def candidate_underlying_dirs(underlying: str, *, prefer_usdc: bool = True) -> List[str]:
    u = (underlying or "").upper().strip()
    if not u:
        return []
    # Prefer USDC-settled directory if present (matches LiveDeribitDataSource convention).
    candidates = [f"{u}_USDC", u] if prefer_usdc else [u, f"{u}_USDC"]
    # De-dup while preserving order.
    seen = set()
    out: List[str] = []
    for c in candidates:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def _scan_latest_mtime(base_root: Path, dir_name: str) -> Tuple[Optional[Path], Optional[float]]:
    u_dir = base_root / dir_name
    if not u_dir.exists():
        return None, None

    latest_path: Optional[Path] = None
    latest_mtime: Optional[float] = None

    for path in u_dir.rglob("*.parquet"):
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        if latest_mtime is None or mtime > latest_mtime:
            latest_mtime = mtime
            latest_path = path

    return latest_path, latest_mtime


def harvest_freshness_for_underlying(
    *,
    underlying: str,
    base_dir: str | Path | None = None,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    base_root = get_harvest_root(base_dir)

    checked = candidate_underlying_dirs(underlying, prefer_usdc=True)
    latest_path: Optional[Path] = None
    latest_mtime: Optional[float] = None
    used_dir: Optional[str] = None

    for d in checked:
        lp, lm = _scan_latest_mtime(base_root, d)
        if lm is not None:
            latest_path, latest_mtime, used_dir = lp, lm, d
            break

    if latest_mtime is None:
        return {
            "underlying": (underlying or "").upper().strip(),
            "harvest_dir": used_dir,
            "dirs_checked": checked,
            "status": "FAIL",
            "age_minutes": None,
            "last_snapshot_at": None,
            "latest_file": None,
            "base_dir": str(base_root),
        }

    last_dt = datetime.fromtimestamp(latest_mtime, tz=timezone.utc)
    age_minutes = max(0.0, (now - last_dt).total_seconds() / 60.0)

    if age_minutes <= 60:
        status = "OK"
    elif age_minutes <= 180:
        status = "WARN"
    else:
        status = "FAIL"

    return {
        "underlying": (underlying or "").upper().strip(),
        "harvest_dir": used_dir,
        "dirs_checked": checked,
        "status": status,
        "age_minutes": age_minutes,
        "last_snapshot_at": last_dt.isoformat(),
        "latest_file": str(latest_path) if latest_path else None,
        "base_dir": str(base_root),
    }


def get_harvest_facts(
    *,
    underlying: str,
    base_dir: str | Path | None = None,
    range_start: date | None = None,
    range_end: date | None = None,
    now: Optional[datetime] = None,
    prefer_usdc: bool = True,
) -> Dict[str, Any]:
    """Return raw harvest facts (no policy).

    Facts are designed to be shared across backtest preflight and ops health.
    """
    now = now or datetime.now(timezone.utc)
    base_root = get_harvest_root(base_dir)
    u = (underlying or "").upper().strip()

    checked = candidate_underlying_dirs(u, prefer_usdc=prefer_usdc)
    selected_key: Optional[str] = None

    # Pick the first directory that has any parquet files.
    file_count = 0
    latest_path: Optional[Path] = None
    latest_mtime: Optional[float] = None
    earliest_mtime: Optional[float] = None

    for d in checked:
        u_dir = base_root / d
        if not u_dir.exists():
            continue
        parquets = list(u_dir.rglob("*.parquet"))
        if not parquets:
            continue
        selected_key = d
        file_count = len(parquets)
        for p in parquets:
            try:
                mtime = p.stat().st_mtime
            except OSError:
                continue
            if latest_mtime is None or mtime > latest_mtime:
                latest_mtime = mtime
                latest_path = p
            if earliest_mtime is None or mtime < earliest_mtime:
                earliest_mtime = mtime
        break

    available = selected_key is not None
    latest_snapshot_at = (
        datetime.fromtimestamp(latest_mtime, tz=timezone.utc).isoformat()
        if latest_mtime is not None
        else None
    )
    earliest_snapshot_at = (
        datetime.fromtimestamp(earliest_mtime, tz=timezone.utc).isoformat()
        if earliest_mtime is not None
        else None
    )
    age_minutes = (
        max(0.0, (now - datetime.fromtimestamp(latest_mtime, tz=timezone.utc)).total_seconds() / 60.0)
        if latest_mtime is not None
        else None
    )

    facts: Dict[str, Any] = {
        "underlying": u,
        "available": available,
        "expected_dir": f"{u}_USDC",
        "selected_key": selected_key,
        "available_keys": checked,
        "prefer_usdc": prefer_usdc,
        "latest_snapshot_at": latest_snapshot_at,
        "earliest_snapshot_at": earliest_snapshot_at,
        "age_minutes": age_minutes,
        "file_count": file_count,
        "base_dir": str(base_root),
        "latest_file": str(latest_path) if latest_path else None,
    }

    if available and range_start is not None and range_end is not None:
        # Range scan is more expensive; only do it if asked.
        files = harvest_files_in_range(
            underlying_dir=selected_key or u,
            start_date=range_start,
            end_date=range_end,
            base_dir=base_root,
        )
        facts.update(
            {
                "range_start": range_start.isoformat(),
                "range_end": range_end.isoformat(),
                "range_file_count": len(files),
                "range_earliest_snapshot_at": files[0][1].isoformat() if files else None,
                "range_latest_snapshot_at": files[-1][1].isoformat() if files else None,
            }
        )

    return facts


def harvest_files_in_range(
    *,
    underlying_dir: str,
    start_date: date,
    end_date: date,
    base_dir: str | Path | None = None,
) -> List[Tuple[Path, datetime]]:
    base_root = get_harvest_root(base_dir)
    try:
        from src.data.live_deribit_exam import discover_files

        return discover_files(base_root, underlying_dir, start_date, end_date)
    except Exception:
        return []


def harvest_range_status(
    *,
    underlying: str,
    start_date: date,
    end_date: date,
    base_dir: str | Path | None = None,
) -> Dict[str, Any]:
    base_root = get_harvest_root(base_dir)
    checked = candidate_underlying_dirs(underlying)

    best: Dict[str, Any] = {
        "underlying": (underlying or "").upper().strip(),
        "base_dir": str(base_root),
        "dirs_checked": checked,
        "harvest_dir": None,
        "num_files": 0,
        "time_min": None,
        "time_max": None,
        "status": "FAIL",
        "reason": "no_files",
    }

    for d in checked:
        files = harvest_files_in_range(
            underlying_dir=d,
            start_date=start_date,
            end_date=end_date,
            base_dir=base_root,
        )
        if not files:
            continue
        best["harvest_dir"] = d
        best["num_files"] = len(files)
        best["time_min"] = files[0][1].isoformat() if files else None
        best["time_max"] = files[-1][1].isoformat() if files else None
        best["status"] = "OK"
        best["reason"] = None
        return best

    return best
