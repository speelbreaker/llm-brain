"""Lab-based Fidelity Suite Orchestrator.

TOP CONSTRAINT (TOC): trustworthiness.
This orchestrator intentionally runs Fidelity through the *existing Backtest Lab*
compare + diff utilities so the score reflects the same machinery users rely on.

STEP 0 — DISCOVERY NOTES (MANDATORY)
- Backtest lab compare runner:
  - src/backtest/compare.py
  - function: run_synthetic_vs_live_pair(...)
- Backtest lab diff utilities:
  - src/backtest/diff.py
  - function: compute_diff_for_runs(run_id_a, run_id_b, exit_style=...)
- Backtest lab run store conventions (pattern reference):
  - src/backtest/run_store.py
  - env override pattern, index.jsonl append, latest pointer
- FastAPI route registration:
  - src/web/routes_fidelity.py defines /api/fidelity/latest and /api/fidelity/history
  - src/web_app.py registers routers via app.include_router(...)
- Backtest Lab UI rendering:
  - src/web/dashboard.py (render_dashboard_html)

This module provides a deterministic, robust MVP scoring layer on top of Lab diff.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from src.backtest import compare, diff


METRIC_TOLERANCES: Dict[str, float] = {
    "net_profit_pct": 5.0,  # percentage points
    "max_drawdown_pct": 5.0,  # percentage points
    "win_rate": 10.0,  # percentage points
    "profit_factor": 0.30,  # absolute
    "avg_trade_usd": 25.0,  # USD
}


def _metric_score(diff_value: float, tolerance: float) -> float:
    if tolerance <= 0:
        return 0.0
    err_ratio = abs(float(diff_value)) / float(tolerance)
    return float(100.0 * math.exp(-1.2 * err_ratio))


def score_case_from_diff(diff_payload: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    """Compute a case score from a diff payload.

    Returns: (case_score, diagnostics)
    """
    metrics = (diff_payload or {}).get("metrics") or {}

    used: Dict[str, float] = {}
    skipped: List[str] = []
    scores: List[float] = []

    for field, tol in METRIC_TOLERANCES.items():
        m = metrics.get(field)
        if not isinstance(m, dict) or "diff" not in m:
            skipped.append(field)
            continue
        used[field] = tol
        scores.append(_metric_score(m.get("diff", 0.0), tol))

    if not scores:
        return 0.0, {"used_metrics": used, "skipped_metrics": skipped}

    return float(sum(scores) / len(scores)), {"used_metrics": used, "skipped_metrics": skipped}


def _safe_get_num_trades(diff_payload: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    try:
        metrics = (diff_payload or {}).get("metrics") or {}
        mt = metrics.get("num_trades") or {}
        # In diff.py: a = run_a (typically synthetic), b = run_b (typically live)
        a = mt.get("a")
        b = mt.get("b")
        if a is None or b is None:
            return None, None
        return int(a), int(b)
    except Exception:
        return None, None


@dataclass(frozen=True)
class FidelityCase:
    exit_style: str
    synth_run_id: Optional[str]
    live_run_id: Optional[str]
    diff_payload: Optional[Dict[str, Any]]
    num_trades_synth: Optional[int]
    num_trades_live: Optional[int]
    case_score: float
    valid: bool
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "exit_style": self.exit_style,
            "synth_run_id": self.synth_run_id,
            "live_run_id": self.live_run_id,
            "diff": self.diff_payload,
            "num_trades": {"synth": self.num_trades_synth, "live": self.num_trades_live},
            "case_score": self.case_score,
            "valid": self.valid,
        }
        if self.error:
            payload["error"] = self.error
        return payload


def run_fidelity_from_lab(
    underlying: str,
    start_ts: datetime,
    end_ts: datetime,
    decision_interval_minutes: int = 60,
    exit_styles: List[str] | None = None,
    min_trades_per_case: int = 5,
) -> Dict[str, Any]:
    """Run Lab-based fidelity across a set of exit styles.

    Runs synthetic vs live via src/backtest/compare.py and compares via src/backtest/diff.py.

    Returns a report dict suitable for persistence via src/backtest/fidelity_store.py.
    """

    if exit_styles is None:
        exit_styles = ["hold_to_expiry", "tp_and_roll"]

    created_at = datetime.now(timezone.utc).isoformat()
    run_id = f"fidelity_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{underlying}_{uuid.uuid4().hex[:8]}"

    cases: List[FidelityCase] = []

    for exit_style in exit_styles:
        synth_run_id: Optional[str] = None
        live_run_id: Optional[str] = None
        diff_payload: Optional[Dict[str, Any]] = None
        err: Optional[str] = None
        case_score = 0.0
        num_trades_synth: Optional[int] = None
        num_trades_live: Optional[int] = None
        valid = False

        try:
            synth_run_id, live_run_id = compare.run_synthetic_vs_live_pair(
                underlying=underlying,
                start_ts=start_ts,
                end_ts=end_ts,
                decision_interval_minutes=decision_interval_minutes,
                exit_style=exit_style,
                verbose=False,
            )

            diff_payload = diff.compute_diff_for_runs(
                run_id_a=synth_run_id,
                run_id_b=live_run_id,
                exit_style=exit_style,
            )

            num_trades_synth, num_trades_live = _safe_get_num_trades(diff_payload)

            case_score, _diag = score_case_from_diff(diff_payload)

            if num_trades_synth is not None and num_trades_live is not None:
                valid = (num_trades_synth >= min_trades_per_case) and (num_trades_live >= min_trades_per_case)
            else:
                valid = False

        except Exception as e:
            err = str(e)
            valid = False

        cases.append(
            FidelityCase(
                exit_style=exit_style,
                synth_run_id=synth_run_id,
                live_run_id=live_run_id,
                diff_payload=diff_payload,
                num_trades_synth=num_trades_synth,
                num_trades_live=num_trades_live,
                case_score=float(case_score),
                valid=bool(valid),
                error=err,
            )
        )

    total_cases = len(cases)
    valid_cases = sum(1 for c in cases if c.valid)
    coverage_ratio_cases = valid_cases / max(total_cases, 1)

    total_trades_synth = sum(int(c.num_trades_synth or 0) for c in cases)
    total_trades_live = sum(int(c.num_trades_live or 0) for c in cases)

    valid_case_scores = [c.case_score for c in cases if c.valid]
    strategy_pnl_parity = float(sum(valid_case_scores) / len(valid_case_scores)) if valid_case_scores else 0.0

    if valid_cases == 0 or coverage_ratio_cases < 0.5:
        gate_label = "UNTRUSTED"
    elif strategy_pnl_parity < 80.0:
        gate_label = "WARNING"
    elif strategy_pnl_parity >= 80.0 and coverage_ratio_cases >= 0.8:
        gate_label = "TRUSTED"
    else:
        gate_label = "WARNING"

    report: Dict[str, Any] = {
        "run_id": run_id,
        "created_at": created_at,
        "underlying": underlying,
        "start_ts": start_ts.isoformat(),
        "end_ts": end_ts.isoformat(),
        "decision_interval_minutes": decision_interval_minutes,
        "cases": [c.to_dict() for c in cases],
        "component_scores": {"strategy_pnl_parity": strategy_pnl_parity},
        "coverage": {
            "valid_cases": valid_cases,
            "total_cases": total_cases,
            "coverage_ratio_cases": coverage_ratio_cases,
            "total_trades_synth": total_trades_synth,
            "total_trades_live": total_trades_live,
            "min_trades_per_case": min_trades_per_case,
        },
        "overall_score": strategy_pnl_parity,
        "gate_label": gate_label,
        "notes": [
            "Lab-based: compare.run_synthetic_vs_live_pair + diff.compute_diff_for_runs",
            "MVP: run-level parity scoring on key metrics",
        ],
    }

    return report
