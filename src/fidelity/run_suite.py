from __future__ import annotations

# FOUND LOCATIONS (repo scan)
# - Canonical strategy runner: src/fidelity/canonical_strategies.py (run_strategy)
# - Suite runner + report writing: src/fidelity/run_suite.py (this file)
# - Scoring + gating: src/fidelity/scoring.py
# - Report schema + latest.json writer: src/fidelity/reporting.py
# - File store readers/writers: src/fidelity/fidelity_store.py
# - FastAPI routes: src/web/routes_positions.py
# - UI: src/web/dashboard.py
# Notes:
# - Removed duplicates/stubs to keep a single canonical implementation path.

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .canonical_strategies import canonical_strategies, run_strategy_with_diagnostics
from .market_replay import (
    detect_live_dataset,
    load_fixture_snapshots_jsonl,
    make_live_replay,
    make_synthetic_replay,
)
from .metrics import (
    ks_statistic,
    quantile_diffs,
    strategy_metrics_from_returns,
    _sorted,
)
from .reporting import FidelityReport, write_report_json, write_report_md, write_latest_index, write_latest_index_for_underlying
from .scoring import apply_coverage_penalty, gate_label, score_fidelity_components


def _to_ts(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def _parse_iso_or_ts(s: str) -> int:
    s = (s or "").strip()
    if s.isdigit():
        return int(s)
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    return _to_ts(dt)


def _corr(x: List[float], y: List[float]) -> Optional[float]:
    if len(x) < 3 or len(y) < 3 or len(x) != len(y):
        return None
    mx = sum(x) / len(x)
    my = sum(y) / len(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    denx = sum((a - mx) ** 2 for a in x)
    deny = sum((b - my) ** 2 for b in y)
    if denx <= 0 or deny <= 0:
        return None
    return float(num / ((denx ** 0.5) * (deny ** 0.5)))


def _bucket_iv_errors(
    *,
    live_snaps: List[Any],
    synth_snaps: List[Any],
    tenors: List[int],
    deltas: List[float],
) -> Dict[str, Any]:
    # Bucket by nearest tenor + nearest abs(delta).
    def nearest(items: List[float], v: float) -> float:
        return min(items, key=lambda x: abs(x - v))

    buckets: Dict[str, List[float]] = {}
    covered = 0
    total = 0

    for ls, ss in zip(live_snaps, synth_snaps):
        live_by_name = {q.instrument_name: q for q in ls.options}
        for q in ss.options:
            total += 1
            ql = live_by_name.get(q.instrument_name)
            if not ql:
                continue
            if q.mark_iv is None or ql.mark_iv is None:
                continue
            if q.delta is None or ql.delta is None:
                continue
            dte = max(0.0, (q.expiry_ts - ss.ts) / 86400.0)
            tenor = int(nearest([float(t) for t in tenors], float(dte)))
            ad = float(abs(q.delta))
            db = float(nearest(deltas, ad))
            key = f"tenor_{tenor}d_delta_{db:.2f}"
            buckets.setdefault(key, []).append(abs(float(q.mark_iv) - float(ql.mark_iv)))
            covered += 1

    mae_by_bucket = {k: (sum(v) / len(v) if v else 0.0) for k, v in buckets.items()}
    all_errs: List[float] = []
    for v in buckets.values():
        all_errs.extend(v)
    mae = sum(all_errs) / len(all_errs) if all_errs else None
    coverage = (covered / total) if total > 0 else 0.0

    return {"mae": mae, "mae_by_bucket": mae_by_bucket, "coverage": coverage}


def run_fidelity_suite(
    *,
    start_ts: int,
    end_ts: int,
    underlying: str = "BTC",
    seed: int = 123,
    out_dir: Optional[str] = None,
    fixture_dir: Optional[str] = None,
    slippage_bps: float = 0.0,
) -> FidelityReport:
    """Run the full MVP fidelity suite.

    - LiveReplay uses harvested data if found; otherwise falls back to fixtures.
    - SyntheticReplay uses existing synthetic state builder when harvested data is available.
      If running on fixture data, synthetic snapshots are generated deterministically from the seed.
    """
    u = (underlying or "BTC").upper().strip()

    fixture_dir = fixture_dir or str(Path("tests") / "fixtures" / "fidelity")
    fixture_path = Path(fixture_dir) / "live_snapshots.jsonl"

    # Detect harvested dataset.
    detected = detect_live_dataset(underlying=u)
    live_data_status = "ok" if detected.get("found") else "missing"

    if live_data_status == "missing":
        live_snaps = load_fixture_snapshots_jsonl(str(fixture_path), underlying=u, start_ts=start_ts, end_ts=end_ts)
        synth_market = make_synthetic_replay(live_snaps, underlying=u, seed=seed)
        synth_snaps = list(synth_market.iter_snapshots(start_ts=start_ts, end_ts=end_ts))
        live_meta = {"type": "fixture", **detected}
        synth_meta = {"type": "fixture_synth", "seed": seed}
    else:
        live_market = make_live_replay(underlying=u, detected=detected)
        live_snaps = list(live_market.iter_snapshots(start_ts=start_ts, end_ts=end_ts))
        # If harvested data exists for the underlying but none is available in the requested
        # time window, fall back to fixtures to keep the suite runnable and informative.
        if not live_snaps:
            live_data_status = "missing_window"
            live_snaps = load_fixture_snapshots_jsonl(
                str(fixture_path),
                underlying=u,
                start_ts=start_ts,
                end_ts=end_ts,
            )
            synth_market = make_synthetic_replay(live_snaps, underlying=u, seed=seed)
            synth_snaps = list(synth_market.iter_snapshots(start_ts=start_ts, end_ts=end_ts))
            live_meta = {"type": "fixture", **detected, "note": "no harvested snapshots in requested window"}
            synth_meta = {"type": "fixture_synth", "seed": seed}
        else:
            # Critical: generate synthetic snapshots from the *same* instrument universe as live.
            # This makes parity measurable even when synthetic_grid candidate sets differ.
            synth_market = make_synthetic_replay(live_snaps, underlying=u, seed=seed)
            synth_snaps = list(synth_market.iter_snapshots(start_ts=start_ts, end_ts=end_ts))
            live_meta = getattr(live_market, "meta", lambda: {"type": "live"})()
            synth_meta = getattr(synth_market, "meta", lambda: {"type": "synthetic"})()

    # Ensure alignment.
    n = min(len(live_snaps), len(synth_snaps))
    live_snaps = live_snaps[:n]
    synth_snaps = synth_snaps[:n]

    def _replay_diag(snaps: List[Any]) -> Dict[str, Any]:
        if not snaps:
            return {
                "snapshots_count": 0,
                "options_count_min": 0,
                "options_count_avg": 0.0,
                "options_count_max": 0,
                "first_snapshot": None,
            }
        counts = [len(getattr(s, "options", []) or []) for s in snaps]
        first = snaps[0]
        opts = list(getattr(first, "options", []) or [])
        sample = []
        for q in opts[:3]:
            sample.append(
                {
                    "instrument_name": getattr(q, "instrument_name", None),
                    "expiry_ts": getattr(q, "expiry_ts", None),
                    "strike": getattr(q, "strike", None),
                    "option_type": getattr(q, "option_type", None),
                    "mark_price": getattr(q, "mark_price", None),
                    "mark_iv": getattr(q, "mark_iv", None),
                    "delta": getattr(q, "delta", None),
                }
            )
        first_fields = {
            "instrument_name": any(getattr(q, "instrument_name", None) for q in opts),
            "expiry_ts": any(getattr(q, "expiry_ts", None) for q in opts),
            "strike": any((getattr(q, "strike", None) or 0) for q in opts),
            "option_type": any(getattr(q, "option_type", None) for q in opts),
            "mark_price": any((getattr(q, "mark_price", None) or 0) for q in opts),
            "mark_iv": any(getattr(q, "mark_iv", None) is not None for q in opts),
            "delta": any(getattr(q, "delta", None) is not None for q in opts),
        }
        return {
            "snapshots_count": int(len(snaps)),
            "options_count_min": int(min(counts) if counts else 0),
            "options_count_avg": float(sum(counts) / len(counts) if counts else 0.0),
            "options_count_max": int(max(counts) if counts else 0),
            "first_snapshot": {
                "ts": getattr(first, "ts", None),
                "spot": getattr(first, "spot", None),
                "options_count": int(len(opts)),
                "fields_present": first_fields,
                "sample_options": sample,
            },
        }

    replay_diagnostics = {
        "live": _replay_diag(live_snaps),
        "synthetic": _replay_diag(synth_snaps),
    }

    # ===== Underlying path fidelity (spot returns) =====
    live_spots = [float(s.spot or 0.0) for s in live_snaps if (s.spot or 0.0) > 0]
    synth_spots = [float(s.spot or 0.0) for s in synth_snaps if (s.spot or 0.0) > 0]

    def log_returns(spots: List[float]) -> List[float]:
        out: List[float] = []
        for i in range(1, len(spots)):
            if spots[i - 1] <= 0 or spots[i] <= 0:
                continue
            out.append(float((spots[i] / spots[i - 1]) - 1.0))
        return out

    live_rets = log_returns(live_spots)
    synth_rets = log_returns(synth_spots)

    underlying_quant_diffs = quantile_diffs(synth_rets, live_rets)
    underlying_tail_err = sum(abs(v) for v in underlying_quant_diffs.values()) / max(1, len(underlying_quant_diffs))
    # Simple realized vol level diff (annualized stdev).
    def rv_level(rs: List[float]) -> Optional[float]:
        if len(rs) < 3:
            return None
        m = sum(rs) / len(rs)
        var = sum((x - m) ** 2 for x in rs) / (len(rs) - 1)
        return float((var ** 0.5) * (365.0 ** 0.5))

    rv_live = rv_level(live_rets)
    rv_synth = rv_level(synth_rets)
    rv_err = abs((rv_synth or 0.0) - (rv_live or 0.0)) if (rv_live is not None and rv_synth is not None) else None

    underlying_comp = {
        "weight": 0.20,
        "status": "ok" if len(live_rets) >= 2 and len(synth_rets) >= 2 else "not_available",
        "metrics": {
            "tail_quantile_diff": {"error": float(underlying_tail_err), "tolerance": 0.02, "k": 1.0, "weight": 0.6},
            "rv_level_diff": {"error": float(rv_err or 0.0), "tolerance": 0.10, "k": 1.0, "weight": 0.4},
        },
        "meta": {
            "quantile_diffs": underlying_quant_diffs,
            "rv_live": rv_live,
            "rv_synth": rv_synth,
        },
    }

    # ===== IV surface fidelity (bucket IV errors) =====
    iv_errs = _bucket_iv_errors(
        live_snaps=live_snaps,
        synth_snaps=synth_snaps,
        tenors=[7, 14, 30],
        deltas=[0.10, 0.25, 0.50],
    )
    iv_status = "ok" if (iv_errs.get("mae") is not None and float(iv_errs.get("coverage") or 0.0) > 0.05) else "not_available"
    iv_comp = {
        "weight": 0.30,
        "status": iv_status,
        "metrics": {
            "iv_bucket_mae": {"error": float(iv_errs.get("mae") or 0.0), "tolerance": 0.05, "k": 1.0, "weight": 1.0},
        },
        "meta": iv_errs,
    }

    # ===== Spot–IV coupling =====
    # Approximate IV as mean abs-delta~0.5 option IV at tenor~30.
    def approx_atm_iv(snap: Any) -> Optional[float]:
        best = None
        for q in snap.options:
            if q.mark_iv is None or q.delta is None:
                continue
            dte = abs(_dte_days(snap.ts, q.expiry_ts) - 30.0)
            ad = abs(float(q.delta))
            score = abs(ad - 0.5) + 0.01 * dte
            if best is None or score < best[0]:
                best = (score, float(q.mark_iv))
        return best[1] if best else None

    def _dte_days(ts: int, expiry_ts: int) -> float:
        return max(0.0, (float(expiry_ts) - float(ts)) / 86400.0)

    live_atm = [approx_atm_iv(s) for s in live_snaps]
    synth_atm = [approx_atm_iv(s) for s in synth_snaps]

    live_div: List[float] = []
    synth_div: List[float] = []
    spot_r: List[float] = []
    for i in range(1, len(live_snaps)):
        s0 = float(live_snaps[i - 1].spot or 0.0)
        s1 = float(live_snaps[i].spot or 0.0)
        if s0 <= 0 or s1 <= 0:
            continue
        if live_atm[i - 1] is None or live_atm[i] is None:
            continue
        if synth_atm[i - 1] is None or synth_atm[i] is None:
            continue
        spot_r.append((s1 / s0) - 1.0)
        live_div.append(float(live_atm[i] - live_atm[i - 1]))
        synth_div.append(float(synth_atm[i] - synth_atm[i - 1]))

    corr_live = _corr(spot_r, live_div)
    corr_synth = _corr(spot_r, synth_div)
    corr_err = abs((corr_synth or 0.0) - (corr_live or 0.0)) if (corr_live is not None and corr_synth is not None) else None
    coupling_status = "ok" if (corr_live is not None and corr_synth is not None) else "not_available"
    coupling_comp = {
        "weight": 0.20,
        "status": coupling_status,
        "metrics": {
            "corr_spot_div_diff": {"error": float(corr_err or 0.0), "tolerance": 0.30, "k": 1.0, "weight": 1.0},
        },
        "meta": {"corr_live": corr_live, "corr_synth": corr_synth},
    }

    # ===== Strategy PnL parity =====
    per_strategy: Dict[str, Any] = {}
    parity_errors: List[float] = []
    ks_errors: List[float] = []
    es_errors: List[float] = []
    dd_errors: List[float] = []

    coverage_live_total = 0
    coverage_live_valid = 0
    coverage_live_invalid_missing = 0
    coverage_synth_total = 0
    coverage_synth_valid = 0
    coverage_synth_invalid_missing = 0

    def _accum_cov(trades: List[Any], *, side: str) -> None:
        nonlocal coverage_live_total, coverage_live_valid, coverage_live_invalid_missing
        nonlocal coverage_synth_total, coverage_synth_valid, coverage_synth_invalid_missing

        total = len(trades)
        valid = sum(1 for t in trades if getattr(t, "is_valid", True))
        invalid_missing = sum(
            1
            for t in trades
            if (not getattr(t, "is_valid", True))
            and str(getattr(t, "data_quality_status", ""))
            in ("missing_close_quote", "missing_open_quote", "stale_quote", "expired_no_quote")
        )

        if side == "live":
            coverage_live_total += total
            coverage_live_valid += valid
            coverage_live_invalid_missing += invalid_missing
        else:
            coverage_synth_total += total
            coverage_synth_valid += valid
            coverage_synth_invalid_missing += invalid_missing

    strategies = canonical_strategies()
    strategy_diagnostics: Dict[str, Any] = {}
    for spec in strategies:
        live_trades, live_diag = run_strategy_with_diagnostics(
            spec=spec,
            snapshots=live_snaps,
            slippage_bps=slippage_bps,
            use_mid=True,
        )
        synth_trades, synth_diag = run_strategy_with_diagnostics(
            spec=spec,
            snapshots=synth_snaps,
            slippage_bps=slippage_bps,
            use_mid=True,
        )

        strategy_diagnostics[spec.name] = {
            "live": live_diag,
            "synthetic": synth_diag,
        }

        _accum_cov(live_trades, side="live")
        _accum_cov(synth_trades, side="synth")

        live_returns = [t.pnl_pct for t in live_trades if getattr(t, "is_valid", True)]
        synth_returns = [t.pnl_pct for t in synth_trades if getattr(t, "is_valid", True)]

        live_m = strategy_metrics_from_returns(live_returns)
        synth_m = strategy_metrics_from_returns(synth_returns)
        qdiffs = quantile_diffs(synth_returns, live_returns)
        ks = ks_statistic(synth_returns, live_returns)

        per_strategy[spec.name] = {
            "live_metrics": live_m,
            "synthetic_metrics": synth_m,
            "parity_metrics": {
                "quantile_diffs": qdiffs,
                "ks": ks,
                "n_live": len(live_returns),
                "n_synth": len(synth_returns),
            },
        }

        if len(live_returns) >= 2 and len(synth_returns) >= 2:
            parity_errors.append(sum(abs(v) for v in qdiffs.values()) / max(1, len(qdiffs)))
            ks_errors.append(float(ks))
            es_errors.append(abs(float(synth_m.get("es_1pct", 0.0)) - float(live_m.get("es_1pct", 0.0))))
            dd_errors.append(abs(float(synth_m.get("max_drawdown", 0.0)) - float(live_m.get("max_drawdown", 0.0))))

    parity_status = "ok" if parity_errors else "not_available"
    parity_comp = {
        "weight": 0.30,
        "status": parity_status,
        "metrics": {
            "return_quantile_diff": {"error": float(sum(parity_errors) / len(parity_errors)) if parity_errors else 0.0, "tolerance": 0.02, "k": 1.0, "weight": 0.5},
            "ks": {"error": float(sum(ks_errors) / len(ks_errors)) if ks_errors else 0.0, "tolerance": 0.20, "k": 1.0, "weight": 0.2},
            "es_1pct_diff": {"error": float(sum(es_errors) / len(es_errors)) if es_errors else 0.0, "tolerance": 0.03, "k": 1.0, "weight": 0.2},
            "max_dd_diff": {"error": float(sum(dd_errors) / len(dd_errors)) if dd_errors else 0.0, "tolerance": 0.10, "k": 1.0, "weight": 0.1},
        },
        "meta": {"n_strategies": len(strategies)},
    }

    components = {
        "underlying_returns": underlying_comp,
        "iv_surface_level": iv_comp,
        "spot_iv_coupling": coupling_comp,
        "strategy_pnl_parity": parity_comp,
    }
    scored = score_fidelity_components(components=components)

    live_cov = (coverage_live_valid / max(coverage_live_total, 1))
    synth_cov = (coverage_synth_valid / max(coverage_synth_total, 1))
    invalid_missing_total = int(coverage_live_invalid_missing + coverage_synth_invalid_missing)
    penalty_cov = float(min(live_cov, synth_cov))

    scored = apply_coverage_penalty(
        scored,
        coverage_ratio=penalty_cov,
        invalid_trades_missing_quote=invalid_missing_total,
        invalid_trades_missing_close=0,
        component_name="strategy_pnl_parity",
    )

    overall = float(scored["overall_score"])
    gate = gate_label(
        overall_score=overall,
        coverage_ratio=penalty_cov,
        invalid_trades_missing_quote=invalid_missing_total,
    )

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    base_dir = Path(out_dir) if out_dir else Path(os.getenv("FIDELITY_RUNS_DIR", "data/fidelity_runs"))
    run_dir = base_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    report = FidelityReport(
        run_id=run_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        underlying=u,
        start_ts=int(start_ts),
        end_ts=int(end_ts),
        overall_score=overall,
        gate_label=gate,
        gate=gate,
        component_scores=scored["component_scores"],
        component_status=scored.get("component_status") or {},
        components=components,
        per_strategy=per_strategy,
        live_data_status=live_data_status,
        market_live_meta=live_meta,
        market_synth_meta=synth_meta,
        replay_diagnostics=replay_diagnostics,
        strategy_diagnostics=strategy_diagnostics,
        coverage={
            "total_trades_opened": int(coverage_live_total + coverage_synth_total),
            "valid_trades_closed": int(coverage_live_valid + coverage_synth_valid),
            "invalid_trades_missing_quote": int(invalid_missing_total),
            "coverage_ratio": float((coverage_live_valid + coverage_synth_valid) / max((coverage_live_total + coverage_synth_total), 1)),
            "penalty_ratio": float(penalty_cov),
            "live": {
                "total_trades_opened": int(coverage_live_total),
                "valid_trades_closed": int(coverage_live_valid),
                "invalid_trades_missing_quote": int(coverage_live_invalid_missing),
                "coverage_ratio": float(live_cov),
            },
            "synthetic": {
                "total_trades_opened": int(coverage_synth_total),
                "valid_trades_closed": int(coverage_synth_valid),
                "invalid_trades_missing_quote": int(coverage_synth_invalid_missing),
                "coverage_ratio": float(synth_cov),
            },
        },
        notes=["MVP: yardstick strategies + deterministic scoring"],
    )

    write_report_json(report, run_dir / "fidelity_report.json")
    write_report_md(report, run_dir / "fidelity_report.md")
    write_latest_index(report, base_dir / "latest.json")
    write_latest_index_for_underlying(report, base_dir / u / "latest.json")

    # Optionally write to legacy UI paths (underlying-scoped) for compatibility.
    write_legacy = (os.getenv("FIDELITY_WRITE_LEGACY") or "0").strip().lower() in ("1", "true", "yes")
    if write_legacy:
        legacy_latest = Path("data/fidelity_runs") / u / "latest"
        legacy_hist = Path("data/fidelity_runs") / u / "history" / run_id
        write_report_json(report, legacy_latest / "fidelity_report.json")
        write_report_md(report, legacy_latest / "fidelity_report.md")
        write_report_json(report, legacy_hist / "fidelity_report.json")
        write_report_md(report, legacy_hist / "fidelity_report.md")

    return report


def run_fidelity_suite_from_cli(
    *,
    start: str,
    end: str,
    underlying: str,
    seed: int,
    out_dir: Optional[str],
    slippage_bps: float,
) -> FidelityReport:
    return run_fidelity_suite(
        start_ts=_parse_iso_or_ts(start),
        end_ts=_parse_iso_or_ts(end),
        underlying=underlying,
        seed=int(seed),
        out_dir=out_dir,
        slippage_bps=float(slippage_bps),
    )
