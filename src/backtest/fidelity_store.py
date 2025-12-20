"""Fidelity run persistence (Lab-based).

Mirrors the conventions in src/backtest/run_store.py:
- Default base dir: data/fidelity_runs
- Env override: FIDELITY_RUNS_DIR
- Layout:
  - <base>/<run_id>/report.json
  - <base>/index.jsonl (append summaries)
  - <base>/latest.json (pointer to latest summary)

This store is intentionally file-based so:
- endpoints can serve data without DB access
- tests can sandbox storage via tmp_path + env override
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


def _get_latest_file(base_dir: str | Path | None = None) -> Path:
    return _get_fidelity_dir(base_dir) / "latest.json"


def _safe_underlying(underlying: str) -> str:
    u = (underlying or "").upper().strip()
    # Keep this strict and deterministic.
    if u not in ("BTC", "ETH"):
        raise ValueError("underlying must be BTC or ETH")
    return u


def _get_latest_file_for_underlying(underlying: str, base_dir: str | Path | None = None) -> Path:
    u = _safe_underlying(underlying)
    return _get_fidelity_dir(base_dir) / u / "latest.json"


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

    run_dir = _get_fidelity_dir(base_dir) / run_id
    report_path = run_dir / "report.json"

    with _lock:
        run_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(report_path, report)
        _append_index_entry(summary, base_dir=base_dir)
        # Global pointer (backward compatible)
        _atomic_write_json(_get_latest_file(base_dir), summary)
        # Per-underlying pointer (deterministic for BTC/ETH)
        _atomic_write_json(_get_latest_file_for_underlying(u, base_dir), summary)

    return summary


def load_latest(
    *,
    underlying: str | None = None,
    base_dir: str | Path | None = None,
) -> Optional[Dict[str, Any]]:
    """Load the latest fidelity summary.

    If `underlying` is provided, prefers the per-underlying latest pointer.
    Falls back to scanning index.jsonl newest->oldest for that underlying.
    """
    if underlying:
        u = _safe_underlying(underlying)
        latest_file = _get_latest_file_for_underlying(u, base_dir)
        if latest_file.exists():
            try:
                entry = json.loads(latest_file.read_text())
                # Validate that this pointer refers to a Lab run directory.
                run_id = str((entry or {}).get("run_id") or "").strip()
                if run_id:
                    lab_report = _get_fidelity_dir(base_dir) / run_id / "report.json"
                    if lab_report.exists():
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

    latest_file = _get_latest_file(base_dir)
    if not latest_file.exists():
        return None
    try:
        return json.loads(latest_file.read_text())
    except Exception:
        return None


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
