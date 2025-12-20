from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict


def _safe_underlying(underlying: str) -> str:
    u = (underlying or "").upper().strip()
    if u not in ("BTC", "ETH"):
        raise ValueError("underlying must be BTC or ETH")
    return u


def _lab_base_dir(base_dir: str | Path | None = None) -> Path:
    if base_dir is not None:
        return Path(base_dir)
    # Preferred canonical env (used by ops facts resolver)
    override = os.environ.get("FIDELITY_DIR")
    if override:
        return Path(override)
    # Back-compat
    return Path(os.environ.get("FIDELITY_RUNS_DIR") or "data/fidelity_runs")


def _lab_latest_path(*, underlying: str, base_dir: str | Path | None = None) -> Path:
    u = _safe_underlying(underlying)
    return _lab_base_dir(base_dir) / u / "latest.json"


def get_fidelity_facts(*, underlying: str, base_dir: str | Path | None = None) -> Dict[str, Any]:
    """Return latest fidelity gate facts using a single deterministic resolver.

    Policy:
    - Prefer Lab store (src.backtest.fidelity_store) if present
    - Fallback to legacy store (src.fidelity.fidelity_store) only if Lab store missing

    Returns a fact dict with stable keys used by gates/health:
    - available, gate_label, overall_score, run_id, created_at
    - source (lab_store|legacy|missing), path
    """
    u = _safe_underlying(underlying)

    lab_err: str | None = None
    try:
        from src.backtest import fidelity_store

        latest = fidelity_store.load_latest(underlying=u, base_dir=base_dir)
        if latest:
            return {
                "underlying": u,
                "available": True,
                "gate_label": (latest.get("gate_label") or latest.get("gate")),
                "overall_score": latest.get("overall_score"),
                "run_id": latest.get("run_id"),
                "created_at": latest.get("created_at") or latest.get("timestamp"),
                "source": "lab_store",
                "path": str(_lab_latest_path(underlying=u, base_dir=base_dir)),
                "base_dir": str(_lab_base_dir(base_dir)),
            }
    except Exception as e:
        lab_err = str(e)

    # Legacy fallback (only if Lab missing)
    try:
        import src.fidelity.fidelity_store as legacy_store

        report = legacy_store.load_latest_report_mvp(underlying=u)
        if report:
            return {
                "underlying": u,
                "available": True,
                "gate_label": report.get("gate_label") or report.get("gate"),
                "overall_score": report.get("overall_score"),
                "run_id": report.get("run_id"),
                "created_at": report.get("created_at") or report.get("timestamp"),
                "source": "legacy",
                "path": str(legacy_store.latest_index_path_for_underlying(u)),
                "base_dir": str(_lab_base_dir(base_dir)),
            }

        # Older legacy path, if present.
        report2 = None
        try:
            report2 = legacy_store.load_latest_report(u)  # type: ignore[attr-defined]
        except Exception:
            report2 = None
        if report2:
            return {
                "underlying": u,
                "available": True,
                "gate_label": report2.get("gate_label") or report2.get("gate"),
                "overall_score": report2.get("overall_score"),
                "run_id": report2.get("run_id"),
                "created_at": report2.get("created_at") or report2.get("timestamp"),
                "source": "legacy",
                "path": str(legacy_store.latest_index_path_for_underlying(u)),
                "base_dir": str(_lab_base_dir(base_dir)),
            }
    except Exception:
        pass

    out: Dict[str, Any] = {
        "underlying": u,
        "available": False,
        "gate_label": None,
        "overall_score": None,
        "run_id": None,
        "created_at": None,
        "source": "missing",
        "path": None,
        "base_dir": str(_lab_base_dir(base_dir)),
    }
    if lab_err:
        out["lab_error"] = lab_err
    return out
