from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
import os


def _active_underlyings_from_cfg(cfg: Any) -> list[str]:
    underlyings = getattr(cfg, "underlyings", None)
    if isinstance(underlyings, Iterable) and not isinstance(underlyings, (str, bytes)):
        out = []
        for u in underlyings:
            s = (str(u) or "").upper().strip()
            if s:
                out.append(s)
        if out:
            return out
    return ["BTC", "ETH"]


def _resolve_live_deribit_dir() -> Path:
    # New canonical env name (preferred)
    override = os.environ.get("LIVE_DERIBIT_DATA_DIR")
    if override:
        return Path(override)
    # Back-compat
    override2 = os.environ.get("HARVEST_DATA_DIR")
    if override2:
        return Path(override2)
    return Path("data/live_deribit")


def _resolve_calibration_dir() -> Path:
    override = os.environ.get("CALIBRATION_DIR")
    if override:
        return Path(override)
    # Existing default in codebase
    return Path("data/calibration_runs")


def _resolve_fidelity_dir() -> Path:
    override = os.environ.get("FIDELITY_DIR")
    if override:
        return Path(override)
    # Back-compat
    override2 = os.environ.get("FIDELITY_RUNS_DIR")
    if override2:
        return Path(override2)
    return Path("data/fidelity_runs")


def resolve_ops_facts(cfg: Any, *, now: Optional[datetime] = None) -> Dict[str, Any]:
    """Resolve raw ops facts for all active underlyings.

    Single source of truth for base dirs. This resolver returns facts only (no PASS/WARN/FAIL policy).
    """
    from src.harvest_status import get_harvest_facts
    from src.ops.calibration_status import get_calibration_facts
    from src.ops.fidelity_status import get_fidelity_facts

    now_dt = now or datetime.now(timezone.utc)
    underlyings_active = _active_underlyings_from_cfg(cfg)

    live_dir = _resolve_live_deribit_dir()
    calibration_dir = _resolve_calibration_dir()
    fidelity_dir = _resolve_fidelity_dir()

    harvest: Dict[str, Any] = {}
    calibration: Dict[str, Any] = {}
    fidelity: Dict[str, Any] = {}

    for u in underlyings_active:
        harvest[u] = get_harvest_facts(underlying=u, base_dir=live_dir, now=now_dt)
        calibration[u] = get_calibration_facts(underlying=u, base_dir=calibration_dir, now=now_dt)
        fidelity[u] = get_fidelity_facts(underlying=u, base_dir=fidelity_dir)

    return {
        "now": now_dt.isoformat(),
        "underlyings_active": list(underlyings_active),
        "paths": {
            "live_deribit_data_dir": str(live_dir),
            "calibration_dir": str(calibration_dir),
            "fidelity_dir": str(fidelity_dir),
        },
        "harvest": harvest,
        "calibration": calibration,
        "fidelity": fidelity,
    }
