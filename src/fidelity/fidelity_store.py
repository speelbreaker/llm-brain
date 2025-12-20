from __future__ import annotations

import json
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


# File-based store for fidelity reports.
#
# Layout:
#   data/fidelity_runs/{UNDERLYING}/latest/fidelity_report.json
#   data/fidelity_runs/{UNDERLYING}/history/{RUN_ID}/fidelity_report.json
FIDELITY_RUNS_DIR = Path("data/fidelity_runs")


def base_runs_dir() -> Path:
    return Path(os.getenv("FIDELITY_RUNS_DIR", str(FIDELITY_RUNS_DIR)))


def latest_index_path() -> Path:
    return base_runs_dir() / "latest.json"


def latest_index_path_for_underlying(underlying: str) -> Path:
    u = _safe_underlying(underlying)
    return base_runs_dir() / u / "latest.json"


def run_dir(run_id: str) -> Path:
    return base_runs_dir() / str(run_id)


def run_report_path(run_id: str) -> Path:
    return run_dir(run_id) / "fidelity_report.json"


def load_latest_index(underlying: Optional[str] = None) -> Optional[Dict[str, Any]]:
    path = latest_index_path_for_underlying(underlying) if underlying else latest_index_path()
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def load_latest_run_id(underlying: Optional[str] = None) -> Optional[str]:
    latest = load_latest_index(underlying)
    if latest and latest.get("run_id"):
        return str(latest.get("run_id"))
    return None


def load_report_by_id(run_id: str) -> Optional[Dict[str, Any]]:
    path = run_report_path(run_id)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def list_history_runs(limit: int = 30, *, underlying: Optional[str] = None) -> List[Dict[str, Any]]:
    base = base_runs_dir()
    if not base.exists():
        return []

    want_underlying = (underlying or "").upper().strip() if underlying else None

    # Only include run directories that contain a report file.
    dirs = [p for p in base.iterdir() if p.is_dir() and (p / "fidelity_report.json").exists()]
    dirs.sort(key=lambda p: p.name, reverse=True)

    out: List[Dict[str, Any]] = []
    cap = max(0, int(limit))
    for d in dirs:
        if len(out) >= cap:
            break
        try:
            with open(d / "fidelity_report.json", "r", encoding="utf-8") as f:
                report = json.load(f)

            if want_underlying and (str(report.get("underlying") or "").upper().strip() != want_underlying):
                continue

            out.append(
                {
                    "run_id": report.get("run_id") or d.name,
                    "timestamp": report.get("timestamp"),
                    "underlying": report.get("underlying"),
                    "overall_score": report.get("overall_score"),
                    "gate_label": report.get("gate_label") or report.get("gate"),
                }
            )
        except Exception:
            continue
    return out


def load_latest_report_mvp(*, underlying: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Load latest report from the run-id-scoped MVP store.

    If underlying is provided, uses the underlying-scoped latest index when available.
    Falls back to scanning recent runs for that underlying.
    """
    run_id = load_latest_run_id(underlying)
    if run_id:
        report = load_report_by_id(run_id)
        if report is not None:
            if underlying:
                want = (underlying or "").upper().strip()
                got = (str(report.get("underlying") or "").upper().strip())
                if got == want:
                    return report
            else:
                return report

    # Fallback: scan history for the most recent matching underlying.
    if underlying:
        runs = list_history_runs(limit=50, underlying=underlying)
        if runs:
            rid = str(runs[0].get("run_id") or "")
            if rid:
                return load_report_by_id(rid)
    return None


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
