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
