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

import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.backtest import compare, diff

from src.fidelity.metrics import exp_score


@dataclass(frozen=True)
class ParityMetricSpec:
    tolerance: float
    weight: float


# Explicit tolerances + weights (P0). These are evaluated on absolute diffs.
# Units match src/backtest/diff.py outputs:
# - *_pct diffs are in percentage points
# - *_usd diffs are USD
# - win_rate diffs are in percentage points
# - profit_factor diffs are absolute
DEFAULT_METRIC_SPECS: Dict[str, ParityMetricSpec] = {
    "net_profit_pct": ParityMetricSpec(tolerance=5.0, weight=0.25),
    "net_profit_usd": ParityMetricSpec(tolerance=250.0, weight=0.10),
    "max_drawdown_pct": ParityMetricSpec(tolerance=5.0, weight=0.20),
    "max_drawdown_usd": ParityMetricSpec(tolerance=250.0, weight=0.10),
    "win_rate": ParityMetricSpec(tolerance=10.0, weight=0.15),
    "profit_factor": ParityMetricSpec(tolerance=0.30, weight=0.15),
    "avg_trade_usd": ParityMetricSpec(tolerance=25.0, weight=0.05),
}


@dataclass(frozen=True)
class ParityCaseSpec:
    underlying: str
    start_ts: datetime
    end_ts: datetime
    decision_interval_minutes: int
    exit_style: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "underlying": self.underlying,
            "start_ts": self.start_ts.isoformat(),
            "end_ts": self.end_ts.isoformat(),
            "decision_interval_minutes": int(self.decision_interval_minutes),
            "exit_style": self.exit_style,
        }


def _safe_underlying(u: str) -> str:
    s = (u or "").upper().strip()
    if s not in ("BTC", "ETH"):
        raise ValueError("underlying must be BTC or ETH")
    return s


def score_case_from_diff(
    diff_payload: Dict[str, Any],
    *,
    metric_specs: Dict[str, ParityMetricSpec] | None = None,
) -> Tuple[float, Dict[str, Any]]:
    """Compute a per-case score from a Backtest Lab diff payload.

    The score is a weighted mean of metric scores, each mapped to 0-100.
    """
    specs = metric_specs or DEFAULT_METRIC_SPECS
    metrics = (diff_payload or {}).get("metrics") or {}

    used: Dict[str, Dict[str, Any]] = {}
    skipped: List[str] = []
    num = 0.0
    den = 0.0

    for field, spec in specs.items():
        m = metrics.get(field)
        if not isinstance(m, dict) or "diff" not in m:
            skipped.append(field)
            continue
        err = abs(float(m.get("diff") or 0.0))
        s = exp_score(err, spec.tolerance, k=1.0)
        w = float(spec.weight)
        num += w * float(s)
        den += w
        used[field] = {
            "tolerance": float(spec.tolerance),
            "weight": float(spec.weight),
            "error": float(err),
            "score": float(s),
        }

    score = float(num / den) if den > 0 else 0.0
    return score, {"used_metrics": used, "skipped_metrics": skipped}


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
    spec: ParityCaseSpec
    synth_run_id: Optional[str]
    live_run_id: Optional[str]
    diff_payload: Optional[Dict[str, Any]]
    num_trades_synth: Optional[int]
    num_trades_live: Optional[int]
    case_score: float
    valid: bool
    error: Optional[str] = None
    invalid_trades_missing_quote: int = 0
    invalid_trades_missing_close: int = 0
    scoring: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "case": self.spec.to_dict(),
            "synth_run_id": self.synth_run_id,
            "live_run_id": self.live_run_id,
            "diff": self.diff_payload,
            "num_trades": {"synth": self.num_trades_synth, "live": self.num_trades_live},
            "case_score": self.case_score,
            "valid": self.valid,
            "invalid_trades": {
                "missing_quote": int(self.invalid_trades_missing_quote),
                "missing_close": int(self.invalid_trades_missing_close),
            },
        }
        if self.scoring is not None:
            payload["scoring"] = self.scoring
        if self.error:
            payload["error"] = self.error
        return payload


PairRunner = Callable[..., Tuple[str, str]]
DiffComputer = Callable[..., Dict[str, Any]]


def _infer_invalid_trade_counters_from_error(err: str | None) -> Tuple[int, int]:
    """Best-effort invalid trade counters.

    The backtest lab does not currently expose per-trade data-quality counters,
    so we infer coarse counters from error text.
    """
    if not err:
        return 0, 0
    s = str(err).lower()
    missing_quote = 1 if re.search(r"missing[_\s-]?quote|no quote|stale quote|expired.*no quote", s) else 0
    missing_close = 1 if re.search(r"missing[_\s-]?close", s) else 0
    return missing_quote, missing_close


def run_strategy_pnl_parity_suite(
    *,
    cases: List[ParityCaseSpec],
    min_trades_per_case: int = 5,
    metric_specs: Dict[str, ParityMetricSpec] | None = None,
    pair_runner: Optional[PairRunner] = None,
    diff_computer: Optional[DiffComputer] = None,
    fixture_diffs: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Run real Strategy PnL Parity on Backtest Lab outputs.

    - For each case, runs a synthetic vs live pair (compare.run_synthetic_vs_live_pair)
    - Computes a diff (diff.compute_diff_for_runs)
    - Scores the case using explicit tolerances + weights
    - Enforces minimum trade count on *both* runs

    For tests, `fixture_diffs` can be provided to avoid running backtests.
    Keys must match a stable case key: "{UNDERLYING}|{start_iso}|{end_iso}|{interval}|{exit_style}".
    """
    if min_trades_per_case < 0:
        raise ValueError("min_trades_per_case must be >= 0")

    runner = pair_runner or compare.run_synthetic_vs_live_pair
    dcomp = diff_computer or diff.compute_diff_for_runs
    specs = metric_specs or DEFAULT_METRIC_SPECS

    results: List[FidelityCase] = []

    for cs in cases:
        u = _safe_underlying(cs.underlying)
        spec = ParityCaseSpec(
            underlying=u,
            start_ts=cs.start_ts,
            end_ts=cs.end_ts,
            decision_interval_minutes=int(cs.decision_interval_minutes),
            exit_style=str(cs.exit_style),
        )

        key = f"{u}|{spec.start_ts.isoformat()}|{spec.end_ts.isoformat()}|{spec.decision_interval_minutes}|{spec.exit_style}"

        synth_run_id: Optional[str] = None
        live_run_id: Optional[str] = None
        diff_payload: Optional[Dict[str, Any]] = None
        err: Optional[str] = None
        case_score = 0.0
        num_trades_synth: Optional[int] = None
        num_trades_live: Optional[int] = None
        valid = False
        inv_mq = 0
        inv_mc = 0
        scoring_diag: Optional[Dict[str, Any]] = None

        try:
            if fixture_diffs is not None and key in fixture_diffs:
                diff_payload = fixture_diffs[key]
                # In fixture mode, allow tests to supply trade counts explicitly.
                num_trades_synth, num_trades_live = _safe_get_num_trades(diff_payload)
            else:
                synth_run_id, live_run_id = runner(
                    underlying=u,
                    start_ts=spec.start_ts,
                    end_ts=spec.end_ts,
                    decision_interval_minutes=spec.decision_interval_minutes,
                    exit_style=spec.exit_style,
                    verbose=False,
                )

                diff_payload = dcomp(
                    run_id_a=synth_run_id,
                    run_id_b=live_run_id,
                    exit_style=spec.exit_style,
                )
                num_trades_synth, num_trades_live = _safe_get_num_trades(diff_payload)

            case_score, scoring_diag = score_case_from_diff(diff_payload or {}, metric_specs=specs)

            if num_trades_synth is not None and num_trades_live is not None:
                valid = (num_trades_synth >= min_trades_per_case) and (num_trades_live >= min_trades_per_case)
            else:
                valid = False
        except Exception as e:
            err = str(e)
            inv_mq, inv_mc = _infer_invalid_trade_counters_from_error(err)
            valid = False

        results.append(
            FidelityCase(
                spec=spec,
                synth_run_id=synth_run_id,
                live_run_id=live_run_id,
                diff_payload=diff_payload,
                num_trades_synth=num_trades_synth,
                num_trades_live=num_trades_live,
                case_score=float(case_score),
                valid=bool(valid),
                error=err,
                invalid_trades_missing_quote=int(inv_mq),
                invalid_trades_missing_close=int(inv_mc),
                scoring=scoring_diag,
            )
        )

    total_cases = len(results)
    valid_cases = sum(1 for c in results if c.valid)
    coverage_ratio_cases = valid_cases / max(total_cases, 1)

    total_trades_synth = sum(int(c.num_trades_synth or 0) for c in results)
    total_trades_live = sum(int(c.num_trades_live or 0) for c in results)
    invalid_missing_quote = sum(int(c.invalid_trades_missing_quote or 0) for c in results)
    invalid_missing_close = sum(int(c.invalid_trades_missing_close or 0) for c in results)

    valid_case_scores = [c.case_score for c in results if c.valid]
    component_score = float(sum(valid_case_scores) / len(valid_case_scores)) if valid_case_scores else 0.0

    return {
        "component_score": float(component_score),
        "cases": [c.to_dict() for c in results],
        "coverage": {
            "valid_cases": int(valid_cases),
            "total_cases": int(total_cases),
            "coverage_ratio_cases": float(coverage_ratio_cases),
            "total_trades_synth": int(total_trades_synth),
            "total_trades_live": int(total_trades_live),
            "min_trades_per_case": int(min_trades_per_case),
            "invalid_trades_missing_quote": int(invalid_missing_quote),
            "invalid_trades_missing_close": int(invalid_missing_close),
        },
        "metric_specs": {
            k: {"tolerance": float(v.tolerance), "weight": float(v.weight)} for k, v in (specs or {}).items()
        },
    }


def run_fidelity_from_lab(
    underlying: str,
    start_ts: datetime,
    end_ts: datetime,
    decision_interval_minutes: int = 60,
    exit_styles: List[str] | None = None,
    min_trades_per_case: int = 5,

) -> Dict[str, Any]:
    """Run Lab-based Strategy PnL Parity using real backtest outputs.

    This produces a report dict suitable for canonical persistence via
    src/backtest/fidelity_store.write_fidelity_report.
    """

    if exit_styles is None:
        exit_styles = ["hold_to_expiry", "tp_and_roll"]

    u = _safe_underlying(underlying)

    created_at = datetime.now(timezone.utc).isoformat()
    run_id = f"fidelity_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{u}_{uuid.uuid4().hex[:8]}"

    case_specs = [
        ParityCaseSpec(
            underlying=u,
            start_ts=start_ts,
            end_ts=end_ts,
            decision_interval_minutes=int(decision_interval_minutes),
            exit_style=str(es),
        )
        for es in exit_styles
    ]

    suite = run_strategy_pnl_parity_suite(
        cases=case_specs,
        min_trades_per_case=int(min_trades_per_case),
        metric_specs=DEFAULT_METRIC_SPECS,
    )

    strategy_pnl_parity = float(suite.get("component_score") or 0.0)
    coverage = suite.get("coverage") or {}

    valid_cases = int(coverage.get("valid_cases") or 0)
    total_cases = int(coverage.get("total_cases") or 0)
    coverage_ratio_cases = float(coverage.get("coverage_ratio_cases") or 0.0)

    # Gate label for this parity-only report is intentionally conservative.
    if valid_cases == 0 or coverage_ratio_cases < 0.5:
        gate_label = "UNTRUSTED"
    elif strategy_pnl_parity >= 80.0 and coverage_ratio_cases >= 0.8:
        gate_label = "TRUSTED"
    elif strategy_pnl_parity >= 65.0:
        gate_label = "WARNING"
    else:
        gate_label = "UNTRUSTED"

    report: Dict[str, Any] = {
        "run_id": run_id,
        "created_at": created_at,
        "underlying": u,
        "start_ts": start_ts.isoformat(),
        "end_ts": end_ts.isoformat(),
        "decision_interval_minutes": decision_interval_minutes,
        "cases": suite.get("cases") or [],
        "component_scores": {"strategy_pnl_parity": float(strategy_pnl_parity)},
        "coverage": dict(coverage),
        "parity": {
            "metric_specs": suite.get("metric_specs") or {},
        },
        "overall_score": strategy_pnl_parity,
        "gate_label": gate_label,
        "notes": [
            "Lab-based: compare.run_synthetic_vs_live_pair + diff.compute_diff_for_runs",
            "Ops-grade: explicit tolerances + weights + min-trades enforcement",
        ],
    }

    return report
