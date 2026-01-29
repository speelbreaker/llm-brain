from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import math


@dataclass(frozen=True)
class TailRiskMetrics:
    var_1pct: float
    es_1pct: float


@dataclass(frozen=True)
class IVErrorBuckets:
    mae: Optional[float]
    mae_by_bucket: dict[str, float]
    coverage: float


@dataclass(frozen=True)
class StrategyParityMetrics:
    return_quantile_diff: float
    ks_stat: float
    es_1pct_diff: float
    max_dd_diff: float


def _sorted(xs: Iterable[float]) -> list[float]:
    ys = [float(x) for x in xs]
    ys.sort()
    return ys


def quantile(xs: Iterable[float], q: float) -> float:
    ys = _sorted(xs)
    if not ys:
        return 0.0
    q = min(1.0, max(0.0, float(q)))
    idx = int(round(q * (len(ys) - 1)))
    return float(ys[idx])


def var_es(xs: Iterable[float], alpha: float = 0.01) -> TailRiskMetrics:
    ys = _sorted(xs)
    if not ys:
        return TailRiskMetrics(var_1pct=0.0, es_1pct=0.0)

    a = min(0.5, max(1e-6, float(alpha)))
    cutoff = quantile(ys, a)
    tail = [x for x in ys if x <= cutoff]
    es = sum(tail) / len(tail) if tail else cutoff
    if abs(a - 0.01) < 1e-9:
        return TailRiskMetrics(var_1pct=float(cutoff), es_1pct=float(es))
    return TailRiskMetrics(var_1pct=float(cutoff), es_1pct=float(es))


def max_drawdown(equity: Iterable[float]) -> float:
    peak = None
    mdd = 0.0
    for v in equity:
        v = float(v)
        if peak is None or v > peak:
            peak = v
        if peak and peak > 0:
            dd = (peak - v) / peak
            if dd > mdd:
                mdd = dd
    return float(mdd)


def exp_score(error: float, tolerance: float, k: float = 1.0) -> float:
    """Map an error to a 0-100 score using an exponential decay.

    score = 100 * exp(-k * (error/tolerance))
    """
    tol = max(1e-12, float(tolerance))
    ratio = max(0.0, float(error) / tol)
    score = 100.0 * math.exp(-float(k) * ratio)
    if score < 0:
        return 0.0
    if score > 100:
        return 100.0
    return float(score)


def win_rate(returns: Iterable[float]) -> float:
    xs = [float(x) for x in returns]
    if not xs:
        return 0.0
    wins = sum(1 for x in xs if x > 0)
    return float(wins / len(xs))


def avg(xs: Iterable[float]) -> float:
    ys = [float(x) for x in xs]
    if not ys:
        return 0.0
    return float(sum(ys) / len(ys))


def median(xs: Iterable[float]) -> float:
    return quantile(xs, 0.50)


def profit_factor(returns: Iterable[float]) -> float:
    xs = [float(x) for x in returns]
    wins = sum(x for x in xs if x > 0)
    losses = -sum(x for x in xs if x < 0)
    if losses <= 0:
        return float("inf") if wins > 0 else 0.0
    return float(wins / losses)


def worst_trade(returns: Iterable[float]) -> float:
    xs = [float(x) for x in returns]
    return float(min(xs)) if xs else 0.0


def equity_curve_from_returns(returns: Iterable[float]) -> list[float]:
    eq = 1.0
    curve = [eq]
    for r in returns:
        eq = eq * (1.0 + float(r))
        curve.append(eq)
    return curve


def ks_statistic(a: Iterable[float], b: Iterable[float]) -> float:
    """Two-sample KS statistic in pure Python."""
    x = _sorted(a)
    y = _sorted(b)
    if not x or not y:
        return 0.0

    i = 0
    j = 0
    n = len(x)
    m = len(y)
    d = 0.0

    while i < n and j < m:
        if x[i] <= y[j]:
            i += 1
        else:
            j += 1
        fx = i / n
        fy = j / m
        d = max(d, abs(fx - fy))

    # Tail
    while i < n:
        i += 1
        d = max(d, abs(i / n - j / m))
    while j < m:
        j += 1
        d = max(d, abs(i / n - j / m))

    return float(d)


def quantile_diffs(a: Iterable[float], b: Iterable[float], qs: Optional[list[float]] = None) -> dict[str, float]:
    qs = qs or [0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    out: dict[str, float] = {}
    for q in qs:
        out[f"q{int(q*100):02d}"] = float(quantile(a, q) - quantile(b, q))
    return out


@dataclass(frozen=True)
class StrategyParityMetrics:
    return_quantile_diff: float
    ks_stat: float
    es_1pct_diff: float
    max_dd_diff: float


def compute_strategy_pnl_parity(
    *,
    live_returns: list[float],
    synth_returns: list[float],
) -> StrategyParityMetrics:
    """Compute parity metrics between live and synthetic strategy returns."""
    live_m = strategy_metrics_from_returns(live_returns)
    synth_m = strategy_metrics_from_returns(synth_returns)
    
    qdiffs = quantile_diffs(synth_returns, live_returns)
    ks = ks_statistic(synth_returns, live_returns)
    
    return_q_diff = sum(abs(v) for v in qdiffs.values()) / max(1, len(qdiffs))
    es_diff = abs(float(synth_m.get("es_1pct", 0.0)) - float(live_m.get("es_1pct", 0.0)))
    dd_diff = abs(float(synth_m.get("max_drawdown", 0.0)) - float(live_m.get("max_drawdown", 0.0)))
    
    return StrategyParityMetrics(
        return_quantile_diff=float(return_q_diff),
        ks_stat=float(ks),
        es_1pct_diff=float(es_diff),
        max_dd_diff=float(dd_diff),
    )


def strategy_metrics_from_returns(returns: Iterable[float]) -> dict[str, float]:
    xs = [float(x) for x in returns]
    tail = var_es(xs, alpha=0.01)
    curve = equity_curve_from_returns(xs)
    return {
        "win_rate": win_rate(xs),
        "avg_trade_return": avg(xs),
        "median_trade_return": median(xs),
        "profit_factor": profit_factor(xs),
        "max_drawdown": max_drawdown(curve),
        "worst_trade_return": worst_trade(xs),
        "var_1pct": float(tail.var_1pct),
        "es_1pct": float(tail.es_1pct),
    }


def compute_iv_surface_fidelity(
    *,
    live_options_by_snap: list[list[Any]],
    synth_options_by_snap: list[list[Any]],
    timestamps: list[int],
    tenors: list[int],
    deltas: list[float],
) -> IVErrorBuckets:
    """Bucket IV errors by nearest tenor + nearest abs(delta)."""
    def nearest(items: list[float], v: float) -> float:
        return min(items, key=lambda x: abs(x - v))

    buckets: dict[str, list[float]] = {}
    covered = 0
    total = 0

    for live_opts, synth_opts, ts in zip(live_options_by_snap, synth_options_by_snap, timestamps):
        live_by_name = {getattr(o, "instrument_name", ""): o for o in live_opts}
        for q in synth_opts:
            total += 1
            ql = live_by_name.get(getattr(q, "instrument_name", ""))
            if not ql:
                continue
            
            q_iv = getattr(q, "mark_iv", None)
            ql_iv = getattr(ql, "mark_iv", None)
            q_delta = getattr(q, "delta", None)
            ql_delta = getattr(ql, "delta", None)
            q_expiry = getattr(q, "expiry_ts", None)

            if q_iv is None or ql_iv is None: continue
            if q_delta is None or ql_delta is None: continue
            if q_expiry is None: continue

            dte = max(0.0, (float(q_expiry) - float(ts)) / 86400.0)
            tenor = int(nearest([float(t) for t in tenors], float(dte)))
            ad = float(abs(q_delta))
            db = float(nearest(deltas, ad))
            
            key = f"tenor_{tenor}d_delta_{db:.2f}"
            buckets.setdefault(key, []).append(abs(float(q_iv) - float(ql_iv)))
            covered += 1

    mae_by_bucket = {k: (sum(v) / len(v) if v else 0.0) for k, v in buckets.items()}
    all_errs: list[float] = []
    for v in buckets.values():
        all_errs.extend(v)
    
    mae = sum(all_errs) / len(all_errs) if all_errs else None
    coverage = (covered / total) if total > 0 else 0.0

    return IVErrorBuckets(mae=mae, mae_by_bucket=mae_by_bucket, coverage=coverage)

