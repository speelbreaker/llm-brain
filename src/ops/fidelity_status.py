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
    - Canonical source of truth is the Lab store in data/fidelity_runs
    - No legacy fallback here (health/gates should not guess)

    Returns a fact dict with stable keys used by gates/health:
    - available, gate_label, overall_score, run_id, created_at
    - source (lab_store), path
    """
    u = _safe_underlying(underlying)

    try:
        from src.backtest import fidelity_store

        facts = fidelity_store.load_latest_facts(underlying=u, base_dir=base_dir)
        # Add base_dir for traceability in ops payloads.
        facts["base_dir"] = str(_lab_base_dir(base_dir))
        return facts
    except Exception as e:
        return {
            "underlying": u,
            "available": False,
            "gate_label": None,
            "overall_score": None,
            "run_id": None,
            "created_at": None,
            "source": "lab_store",
            "path": None,
            "base_dir": str(_lab_base_dir(base_dir)),
            "lab_error": str(e),
        }
