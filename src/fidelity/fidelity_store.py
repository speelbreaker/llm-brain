from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


# File-based store for fidelity reports.
#
# Layout:
#   data/fidelity_runs/{UNDERLYING}/latest/fidelity_report.json
#   data/fidelity_runs/{UNDERLYING}/history/{RUN_ID}/fidelity_report.json
FIDELITY_RUNS_DIR = Path("data/fidelity_runs")


@dataclass(frozen=True)
class FidelityRunRef:
    underlying: str
    run_id: str
    report_path: Path


def _safe_underlying(underlying: str) -> str:
    u = (underlying or "").upper().strip()
    if u not in ("BTC", "ETH"):
        raise ValueError("underlying must be BTC or ETH")
    return u


def latest_report_path(underlying: str) -> Path:
    u = _safe_underlying(underlying)
    return FIDELITY_RUNS_DIR / u / "latest" / "fidelity_report.json"


def history_dir(underlying: str) -> Path:
    u = _safe_underlying(underlying)
    return FIDELITY_RUNS_DIR / u / "history"


def load_latest_report(underlying: str) -> Optional[Dict[str, Any]]:
    path = latest_report_path(underlying)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def list_recent_reports(underlying: str, limit: int = 30) -> List[Dict[str, Any]]:
    d = history_dir(underlying)
    if not d.exists():
        return []

    run_dirs = [p for p in d.iterdir() if p.is_dir()]
    run_dirs.sort(key=lambda p: p.name, reverse=True)

    out: List[Dict[str, Any]] = []
    for run_dir in run_dirs[: max(0, int(limit))]:
        report_path = run_dir / "fidelity_report.json"
        if not report_path.exists():
            continue
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                out.append(json.load(f))
        except Exception:
            continue

    return out
