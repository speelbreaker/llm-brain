from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import math


@dataclass(frozen=True)
class TailRiskMetrics:
    var_1pct: float
    es_1pct: float


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

