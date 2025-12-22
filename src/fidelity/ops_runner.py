from __future__ import annotations

import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.backtest.fidelity_suite import ParityCaseSpec, run_strategy_pnl_parity_suite
from src.backtest.fidelity_store import write_fidelity_report
from src.fidelity.scoring import FidelityGates, gate_label, gates_to_dict


DEFAULT_COMPONENT_WEIGHTS: Dict[str, float] = {
    "underlying_returns": 0.20,
    "iv_surface_level": 0.30,
    "spot_iv_coupling": 0.20,
    "strategy_pnl_parity": 0.30,
}


def _clamp_score_0_100(value: Any) -> float:
    try:
        v = float(value or 0.0)
    except Exception:
        v = 0.0
    return float(max(0.0, min(100.0, v)))


def _clamp_ratio_0_1(value: Any) -> float:
    try:
        v = float(value or 0.0)
    except Exception:
        v = 0.0
    return float(max(0.0, min(1.0, v)))


def _apply_strategy_parity_coverage_penalty(
    component_scores: Dict[str, float],
    *,
    parity_coverage: Dict[str, Any],
    penalty_threshold: float = 0.95,
) -> Dict[str, Any]:
    """Apply a conservative penalty to strategy_pnl_parity when coverage is imperfect.

    Policy:
    - If coverage_ratio_cases < penalty_threshold OR any invalid trade counters are > 0,
      multiply strategy_pnl_parity by coverage_ratio_cases (clamped 0..1).
    """
    cov_ratio = _clamp_ratio_0_1(parity_coverage.get("coverage_ratio_cases"))
    invalid_mq = int(parity_coverage.get("invalid_trades_missing_quote") or 0)
    invalid_mc = int(parity_coverage.get("invalid_trades_missing_close") or 0)

    reasons: List[str] = []
    if cov_ratio < float(penalty_threshold):
        reasons.append(f"coverage_ratio_cases<{penalty_threshold}")
    if invalid_mq > 0:
        reasons.append("invalid_trades_missing_quote>0")
    if invalid_mc > 0:
        reasons.append("invalid_trades_missing_close>0")

    before = float(component_scores.get("strategy_pnl_parity") or 0.0)
    applied = bool(reasons)
    after = float(before)
    if applied:
        after = float(before) * float(cov_ratio)
        component_scores["strategy_pnl_parity"] = float(after)

    return {
        "applied": applied,
        "penalty_ratio": float(cov_ratio),
        "penalty_threshold": float(penalty_threshold),
        "invalid_trades_missing_quote": int(invalid_mq),
        "invalid_trades_missing_close": int(invalid_mc),
        "before_score": float(before),
        "after_score": float(after),
        "reasons": reasons,
    }


def _weighted_overall_score(component_scores: Dict[str, float], *, weights: Dict[str, float]) -> float:
    num = 0.0
    den = 0.0
    for name, w in weights.items():
        if name not in component_scores:
            continue
        num += float(w) * float(component_scores.get(name) or 0.0)
        den += float(w)
    return float(num / den) if den > 0 else 0.0


def _validate_parity_coverage_schema(parity_cov: Dict[str, Any]) -> Dict[str, Any]:
    required = [
        "coverage_ratio_cases",
        "invalid_trades_missing_quote",
        "invalid_trades_missing_close",
        "total_trades_live",
        "total_trades_synth",
        "valid_cases",
        "total_cases",
    ]
    missing = [k for k in required if k not in (parity_cov or {})]
    return {
        "ok": len(missing) == 0,
        "missing_keys": missing,
        "required_keys": required,
    }


def run_ops_fidelity_suite(
    *,
    underlying: str,
    start_ts: int,
    end_ts: int,
    seed: int = 123,
    slippage_bps: float = 0.0,
    parity_cases: Optional[List[ParityCaseSpec]] = None,
    min_trades_per_case: int = 5,
    base_dir: str | Path | None = None,
    gates: Optional[FidelityGates] = None,
    fixture_parity_diffs: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Run ops-grade Fidelity and persist a canonical lab-store report.

    Components:
    - underlying_returns
    - iv_surface_level
    - spot_iv_coupling
    - strategy_pnl_parity (real backtest run parity via compare+diff)

    Storage:
    - Writes canonical report via src.backtest.fidelity_store.write_fidelity_report

    Notes:
    - We reuse the existing market-fidelity computations from src.fidelity.run_suite
      (which are deterministic + fixture-friendly) but run them in a temp dir so
      this runner does not mutate the legacy/MVP store.
    """
    u = (underlying or "").upper().strip() or "BTC"
    created_at = datetime.now(timezone.utc).isoformat()
    run_id = f"ops_fidelity_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{u}_{uuid.uuid4().hex[:8]}"

    # 1) Market-fidelity components (existing suite, temp scratch).
    from src.fidelity.run_suite import run_fidelity_suite

    with tempfile.TemporaryDirectory(prefix="fidelity_tmp_") as tmp:
        _ = run_fidelity_suite(
            start_ts=int(start_ts),
            end_ts=int(end_ts),
            underlying=u,
            seed=int(seed),
            out_dir=str(Path(tmp) / "mvp"),
            slippage_bps=float(slippage_bps),
        )
        market_component_scores = dict(_.component_scores or {})
        market_components = dict(_.components or {})
        market_component_status = dict(_.component_status or {})

    # 2) Real run parity component.
    if parity_cases is None:
        # Default: a small but meaningful matrix across common exit styles.
        # The caller can provide a richer matrix if desired.
        from datetime import datetime as dt

        start_dt = dt.fromtimestamp(int(start_ts), tz=timezone.utc)
        end_dt = dt.fromtimestamp(int(end_ts), tz=timezone.utc)
        parity_cases = [
            ParityCaseSpec(
                underlying=u,
                start_ts=start_dt,
                end_ts=end_dt,
                decision_interval_minutes=60,
                exit_style="tp_and_roll",
            ),
            ParityCaseSpec(
                underlying=u,
                start_ts=start_dt,
                end_ts=end_dt,
                decision_interval_minutes=60,
                exit_style="hold_to_expiry",
            ),
        ]

    parity = run_strategy_pnl_parity_suite(
        cases=list(parity_cases),
        min_trades_per_case=int(min_trades_per_case),
        fixture_diffs=fixture_parity_diffs,
    )
    parity_score = float(parity.get("component_score") or 0.0)
    parity_cov = parity.get("coverage") or {}
    parity_cov_schema = _validate_parity_coverage_schema(dict(parity_cov))

    raw_component_scores = {
        "underlying_returns": float(market_component_scores.get("underlying_returns") or 0.0),
        "iv_surface_level": float(market_component_scores.get("iv_surface_level") or 0.0),
        "spot_iv_coupling": float(market_component_scores.get("spot_iv_coupling") or 0.0),
        "strategy_pnl_parity": float(parity_score),
    }

    warnings: List[str] = []
    component_scores = {k: _clamp_score_0_100(v) for k, v in raw_component_scores.items()}
    for k, raw in raw_component_scores.items():
        clamped = component_scores.get(k)
        if clamped != float(raw):
            warnings.append(f"component_score_clamped:{k}:{raw}->{clamped}")

    weights = dict(DEFAULT_COMPONENT_WEIGHTS)
    errors: List[Dict[str, Any]] = []

    # Runtime schema enforcement: malformed coverage is an ops-critical bug.
    # Hard-fail to UNTRUSTED by forcing coverage_ratio=0 and parity score to 0.
    if not bool(parity_cov_schema.get("ok")):
        errors.append(
            {
                "code": "FIDELITY_PARITY_COVERAGE_SCHEMA_MISSING",
                "message": "Parity coverage schema missing required keys",
                "missing_keys": parity_cov_schema.get("missing_keys"),
            }
        )
        component_scores["strategy_pnl_parity"] = 0.0
        parity_cov = dict(parity_cov)
        parity_cov.setdefault("coverage_ratio_cases", 0.0)
        parity_cov.setdefault("invalid_trades_missing_quote", 0)
        parity_cov.setdefault("invalid_trades_missing_close", 0)

    # Conservative penalty: don't allow mediocre coverage to keep a high score.
    coverage_penalty = _apply_strategy_parity_coverage_penalty(component_scores, parity_coverage=dict(parity_cov))

    overall = _clamp_score_0_100(_weighted_overall_score(component_scores, weights=weights))

    cov_ratio_cases = _clamp_ratio_0_1(parity_cov.get("coverage_ratio_cases"))
    invalid_mq = int(parity_cov.get("invalid_trades_missing_quote") or 0)
    invalid_mc = int(parity_cov.get("invalid_trades_missing_close") or 0)

    g = gates or FidelityGates()
    gate = gate_label(
        overall_score=float(overall),
        coverage_ratio=float(cov_ratio_cases),
        invalid_trades_missing_quote=int(invalid_mq),
        invalid_trades_missing_close=int(invalid_mc),
        gates=g,
    )

    report: Dict[str, Any] = {
        "schema_version": 1,
        "score_scale": "0-100",
        "run_id": run_id,
        "created_at": created_at,
        "underlying": u,
        "window": {"start_ts": int(start_ts), "end_ts": int(end_ts)},
        "overall_score": float(overall),
        "gate_label": gate,
        "raw_component_scores": raw_component_scores,
        "component_scores": component_scores,
        "component_weights": {k: float(v) for k, v in weights.items()},
        "coverage_penalty": coverage_penalty,
        "warnings": warnings,
        "errors": errors,
        "coverage": {
            "strategy_pnl_parity": dict(parity_cov),
        },
        "thresholds": gates_to_dict(g),
        "components": {
            # Preserve market component meta (existing shapes) for debugging.
            "market": {
                "component_status": market_component_status,
                "components": market_components,
            },
            # Real run parity suite details.
            "strategy_pnl_parity": {
                "score": float(parity_score),
                "cases": parity.get("cases") or [],
                "metric_specs": parity.get("metric_specs") or {},
            },
        },
        "notes": [
            "Ops-grade fidelity report: weighted overall score + explicit gate thresholds.",
            "strategy_pnl_parity is computed from real Backtest Lab synthetic vs live runs.",
        ],
    }

    # Persist into the canonical lab store.
    write_fidelity_report(report, base_dir=base_dir)
    return report
