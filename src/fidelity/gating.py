from __future__ import annotations

from typing import Any, Dict, Optional


def get_fidelity_gate_status(*, underlying: str = "BTC") -> Dict[str, Any]:
    """Return latest fidelity gate status for an underlying.

    This is intentionally defensive: if no report exists, it returns available=False.
    """
    u = (underlying or "BTC").upper().strip()
    report = None
    try:
        from src.fidelity.fidelity_store import load_latest_report_mvp

        report = load_latest_report_mvp(underlying=u)
    except Exception:
        report = None

    if not report:
        # Backward-compat fallback: legacy underlying-scoped store.
        try:
            from src.fidelity.fidelity_store import load_latest_report

            report = load_latest_report(u)
        except Exception:
            report = None

    if not report:
        return {
            "available": False,
            "underlying": u,
            "run_id": None,
            "overall_score": None,
            "gate_label": None,
        }

    return {
        "available": True,
        "underlying": u,
        "run_id": report.get("run_id"),
        "overall_score": report.get("overall_score"),
        "gate_label": report.get("gate_label") or report.get("gate"),
    }
