"""Fidelity run persistence (canonical Lab store).

This is the single source of truth for ops-grade Fidelity artifacts.

Design goals:
- deterministic: latest resolution is stable (per-underlying supported)
- self-describing: reports contain thresholds + coverage + component scores
- ops-friendly: simple facts can be loaded without DB

Layout (base dir defaults to data/fidelity_runs; env override: FIDELITY_RUNS_DIR):
- <base>/<run_id>.json            (full fidelity report)
- <base>/latest.json              (full report for latest run; copy/symlink semantics)
- <base>/<UNDERLYING>/latest.json (full report for latest run for that underlying)

For UI endpoints and history browsing we additionally maintain a compact summary index:
- <base>/index.jsonl
- <base>/latest_summary.json
- <base>/<UNDERLYING>/latest_summary.json

Backward compatibility:
- Readers tolerate the older layout: <base>/<run_id>/report.json
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

_DEFAULT_FIDELITY_DIR = Path("data/fidelity_runs")
_lock = threading.Lock()


def _get_fidelity_dir(base_dir: str | Path | None = None) -> Path:
    if base_dir is not None:
        return Path(base_dir)
    override = os.environ.get("FIDELITY_RUNS_DIR")
    if override:
        return Path(override)
    return _DEFAULT_FIDELITY_DIR


def _get_index_file(base_dir: str | Path | None = None) -> Path:
    return _get_fidelity_dir(base_dir) / "index.jsonl"


def _get_latest_summary_file(base_dir: str | Path | None = None) -> Path:
    return _get_fidelity_dir(base_dir) / "latest_summary.json"


def _get_latest_report_file(base_dir: str | Path | None = None) -> Path:
    return _get_fidelity_dir(base_dir) / "latest.json"


def _safe_underlying(underlying: str) -> str:
    u = (underlying or "").upper().strip()
    # Keep this strict and deterministic.
    if u not in ("BTC", "ETH"):
        raise ValueError("underlying must be BTC or ETH")
    return u


def _get_latest_file_for_underlying(underlying: str, base_dir: str | Path | None = None) -> Path:
    u = _safe_underlying(underlying)
    # Full report (canonical)
    return _get_fidelity_dir(base_dir) / u / "latest.json"


def _get_latest_summary_file_for_underlying(underlying: str, base_dir: str | Path | None = None) -> Path:
    u = _safe_underlying(underlying)
    return _get_fidelity_dir(base_dir) / u / "latest_summary.json"


def _get_run_report_path(run_id: str, base_dir: str | Path | None = None) -> Path:
    rid = str(run_id or "").strip()
    if not rid:
        raise ValueError("run_id required")
    return _get_fidelity_dir(base_dir) / f"{rid}.json"


def _get_legacy_run_report_path(run_id: str, base_dir: str | Path | None = None) -> Path:
    rid = str(run_id or "").strip()
    if not rid:
        raise ValueError("run_id required")
    return _get_fidelity_dir(base_dir) / rid / "report.json"


def ensure_fidelity_dir(base_dir: str | Path | None = None) -> None:
    _get_fidelity_dir(base_dir).mkdir(parents=True, exist_ok=True)


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.replace(path)


def _append_index_entry(entry: Dict[str, Any], *, base_dir: str | Path | None = None) -> None:
    index_file = _get_index_file(base_dir)
    index_file.parent.mkdir(parents=True, exist_ok=True)
    with index_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, sort_keys=True) + "\n")


def write_fidelity_report(report: Dict[str, Any], *, base_dir: str | Path | None = None) -> Dict[str, Any]:
    """Write a full fidelity report and update index/latest.

    Canonical persistence:
    - <base>/<run_id>.json (full report)
    - <base>/latest.json (full report copy)
    - <base>/<UNDERLYING>/latest.json (full report copy)

    Also maintains summary pointers + index for UI/history.

    Returns the summary entry that was appended/written.
    """
    ensure_fidelity_dir(base_dir)

    run_id = str(report.get("run_id") or "").strip()
    if not run_id:
        raise ValueError("report missing run_id")

    created_at = report.get("created_at")
    if not created_at:
        created_at = datetime.now(timezone.utc).isoformat()
        report["created_at"] = created_at

    summary: Dict[str, Any] = {
        "run_id": run_id,
        "created_at": created_at,
        "underlying": report.get("underlying"),
        "overall_score": report.get("overall_score"),
        "gate_label": report.get("gate_label"),
        "component_scores": report.get("component_scores") or {},
        "coverage": report.get("coverage") or {},
    }

    underlying = report.get("underlying")
    if not underlying:
        raise ValueError("report missing underlying")
    u = _safe_underlying(str(underlying))

    report_path = _get_run_report_path(run_id, base_dir)
    legacy_report_path = _get_legacy_run_report_path(run_id, base_dir)

    latest_report_path = _get_latest_report_file(base_dir)
    latest_report_path_for_u = _get_latest_file_for_underlying(u, base_dir)

    latest_summary_path = _get_latest_summary_file(base_dir)
    latest_summary_path_for_u = _get_latest_summary_file_for_underlying(u, base_dir)

    with _lock:
        # Canonical flat-file report.
        _atomic_write_json(report_path, report)
        # Backward-compatible legacy layout (helps older tooling that expects /<run_id>/report.json).
        # This does not change the canonical read path.
        _atomic_write_json(legacy_report_path, report)

        # Latest full reports.
        _atomic_write_json(latest_report_path, report)
        _atomic_write_json(latest_report_path_for_u, report)

        # History + latest summary pointers for UI endpoints.
        _append_index_entry(summary, base_dir=base_dir)
        _atomic_write_json(latest_summary_path, summary)
        _atomic_write_json(latest_summary_path_for_u, summary)

    return summary


def load_latest(
    *,
    underlying: str | None = None,
    base_dir: str | Path | None = None,
) -> Optional[Dict[str, Any]]:
    """Load the latest fidelity *summary*.

    This is used by /api/fidelity/latest.

    If `underlying` is provided, prefers the per-underlying latest summary pointer.
    Falls back to scanning index.jsonl newest->oldest for that underlying.
    """
    if underlying:
        u = _safe_underlying(underlying)
        latest_file = _get_latest_summary_file_for_underlying(u, base_dir)
        if latest_file.exists():
            try:
                entry = json.loads(latest_file.read_text())
                # Validate that this pointer refers to a canonical run report.
                run_id = str((entry or {}).get("run_id") or "").strip()
                if run_id and _get_run_report_path(run_id, base_dir).exists():
                    return entry
            except Exception:
                pass

        # Fallback: scan index for the most recent matching underlying.
        index_file = _get_index_file(base_dir)
        if not index_file.exists():
            return None
        try:
            lines = index_file.read_text().splitlines()
        except Exception:
            return None
        for line in reversed(lines):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except Exception:
                continue
            if str(entry.get("underlying") or "").upper().strip() == u:
                return entry
        return None

    latest_file = _get_latest_summary_file(base_dir)
    if not latest_file.exists():
        return None
    try:
        return json.loads(latest_file.read_text())
    except Exception:
        return None


def load_latest_report(
    *,
    underlying: str | None = None,
    base_dir: str | Path | None = None,
) -> Optional[Dict[str, Any]]:
    """Load the latest fidelity *full report*.

    Canonical latest lives in:
    - <base>/latest.json (most recently written report, any underlying)
    - <base>/<UNDERLYING>/latest.json (per-underlying deterministic latest)
    """
    if underlying:
        u = _safe_underlying(underlying)
        p = _get_latest_file_for_underlying(u, base_dir)
    else:
        p = _get_latest_report_file(base_dir)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def load_report_by_run_id(
    run_id: str,
    *,
    base_dir: str | Path | None = None,
) -> Optional[Dict[str, Any]]:
    """Load a full report by run_id from the canonical store.

    Supports both canonical and legacy-on-disk layouts.
    """
    rid = str(run_id or "").strip()
    if not rid:
        return None
    p = _get_run_report_path(rid, base_dir)
    if not p.exists():
        p2 = _get_legacy_run_report_path(rid, base_dir)
        if not p2.exists():
            return None
        p = p2
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def load_latest_facts(
    *,
    underlying: str,
    base_dir: str | Path | None = None,
) -> Dict[str, Any]:
    """Load ops-friendly facts from the canonical lab store.

    Returns a stable dict shape:
      {available, run_id, created_at, gate_label, overall_score,
       component_scores, coverage, path, source="lab_store"}
    """
    u = _safe_underlying(underlying)
    report = load_latest_report(underlying=u, base_dir=base_dir)
    if not report:
        return {
            "available": False,
            "run_id": None,
            "created_at": None,
            "gate_label": None,
            "overall_score": None,
            "component_scores": {},
            "coverage": {},
            "path": None,
            "source": "lab_store",
            "underlying": u,
        }

    created_at = report.get("created_at") or report.get("timestamp")
    run_id = report.get("run_id")
    gate = report.get("gate_label") or report.get("gate")
    overall = report.get("overall_score")
    component_scores = report.get("component_scores") or {}
    coverage = report.get("coverage") or {}

    return {
        "available": True,
        "run_id": run_id,
        "created_at": created_at,
        "gate_label": gate,
        "overall_score": overall,
        "component_scores": component_scores,
        "coverage": coverage,
        "path": str(_get_latest_file_for_underlying(u, base_dir)),
        "source": "lab_store",
        "underlying": u,
    }


def load_history(limit: int = 30) -> List[Dict[str, Any]]:
    if limit <= 0:
        return []

    index_file = _get_index_file()
    if not index_file.exists():
        return []

    entries: List[Dict[str, Any]] = []
    # Newest are at the end (append-only). Read all, then reverse.
    try:
        lines = index_file.read_text().splitlines()
    except Exception:
        return []

    for line in reversed(lines):
        if not line.strip():
            continue
        try:
            entries.append(json.loads(line))
        except Exception:
            continue
        if len(entries) >= limit:
            break

    return entries
