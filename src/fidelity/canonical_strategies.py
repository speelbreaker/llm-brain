from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .types import MarketSnapshot, OptionQuote, TradeResult


def _is_call(q: OptionQuote) -> bool:
    return (q.option_type or "").lower().startswith("c") or (q.option_type or "").lower() == "call"


def _is_put(q: OptionQuote) -> bool:
    return (q.option_type or "").lower().startswith("p") or (q.option_type or "").lower() == "put"


def _abs_delta(q: OptionQuote) -> Optional[float]:
    if q.delta is None:
        return None
    return abs(float(q.delta))


def _dte_days(snapshot_ts: int, expiry_ts: int) -> float:
    return max(0.0, (float(expiry_ts) - float(snapshot_ts)) / 86400.0)


def _closest_by_delta_and_dte(
    *,
    options: Sequence[OptionQuote],
    snapshot_ts: int,
    want_call: bool,
    target_abs_delta: float,
    target_dte_days: float,
    delta_tol: float = 0.10,
    dte_tol_days: float = 10.0,
) -> Optional[OptionQuote]:
    best: Optional[Tuple[float, OptionQuote]] = None
    for q in options:
        if want_call and not _is_call(q):
            continue
        if (not want_call) and not _is_put(q):
            continue
        if q.expiry_ts <= snapshot_ts:
            continue
        ad = _abs_delta(q)
        if ad is None:
            continue
        dte = _dte_days(snapshot_ts, q.expiry_ts)
        if abs(ad - target_abs_delta) > float(delta_tol):
            continue
        if abs(dte - target_dte_days) > float(dte_tol_days):
            continue
        score = abs(ad - target_abs_delta) + 0.01 * abs(dte - target_dte_days)
        if best is None or score < best[0]:
            best = (score, q)
    return best[1] if best else None


def _price_for_fill(q: OptionQuote, *, side: str, slippage_bps: float, use_mid: bool) -> float:
    mark = float(q.mark_price or 0.0)
    if use_mid and q.bid is not None and q.ask is not None and q.bid > 0 and q.ask > 0:
        mark = 0.5 * (float(q.bid) + float(q.ask))
    if mark <= 0:
        return 0.0
    adj = float(slippage_bps) / 10_000.0
    if side == "buy":
        return mark * (1.0 + adj)
    return mark * (1.0 - adj)


def _find_same_instrument(options: Sequence[OptionQuote], name: str) -> Optional[OptionQuote]:
    for q in options:
        if q.instrument_name == name:
            return q
    return None


def _intrinsic_value(*, q: OptionQuote, spot: float) -> float:
    if _is_call(q):
        return max(0.0, float(spot) - float(q.strike))
    return max(0.0, float(q.strike) - float(spot))


@dataclass(frozen=True)
class StrategySpec:
    name: str

    # Targets are intentionally coarse; this is a yardstick, not an optimizer.
    entry_target_dte_days: float
    entry_target_abs_delta: float
    hold_days: int

    kind: str  # used to build legs


def canonical_strategies() -> List[StrategySpec]:
    return [
        StrategySpec(
            name="CoveredCall",
            entry_target_dte_days=30.0,
            entry_target_abs_delta=0.25,
            hold_days=3,
            kind="short_call",
        ),
        StrategySpec(
            name="CashSecuredPut",
            entry_target_dte_days=30.0,
            entry_target_abs_delta=0.25,
            hold_days=3,
            kind="short_put",
        ),
        StrategySpec(
            name="ShortStrangle",
            entry_target_dte_days=30.0,
            entry_target_abs_delta=0.16,
            hold_days=3,
            kind="short_strangle",
        ),
        StrategySpec(
            name="PutCreditSpread",
            entry_target_dte_days=30.0,
            entry_target_abs_delta=0.25,
            hold_days=3,
            kind="put_credit_spread",
        ),
        StrategySpec(
            name="CallDebitSpread",
            entry_target_dte_days=30.0,
            entry_target_abs_delta=0.25,
            hold_days=3,
            kind="call_debit_spread",
        ),
        StrategySpec(
            name="Calendar",
            entry_target_dte_days=14.0,
            entry_target_abs_delta=0.25,
            hold_days=3,
            kind="calendar_call",
        ),
    ]


def run_strategy(
    *,
    spec: StrategySpec,
    snapshots: Sequence[MarketSnapshot],
    slippage_bps: float = 0.0,
    use_mid: bool = True,
) -> List[TradeResult]:
    """Run a deterministic yardstick strategy.

    Entry schedule (MVP): first snapshot of each UTC day from the provided snapshots.
    Exit (MVP): close after N days or at expiry (whichever comes first).
    """
    if not snapshots:
        return []

    # We assume snapshots are already one-per-day in chronological order.
    trades: List[TradeResult] = []
    n = len(snapshots)

    for i, open_snap in enumerate(snapshots):
        spot0 = float(open_snap.spot or 0.0)
        if spot0 <= 0:
            continue

        legs: List[Tuple[str, OptionQuote]] = []  # (side, quote)

        if spec.kind == "short_call":
            q = _closest_by_delta_and_dte(
                options=open_snap.options,
                snapshot_ts=open_snap.ts,
                want_call=True,
                target_abs_delta=spec.entry_target_abs_delta,
                target_dte_days=spec.entry_target_dte_days,
            )
            if not q:
                continue
            legs = [("sell", q)]

        elif spec.kind == "short_put":
            q = _closest_by_delta_and_dte(
                options=open_snap.options,
                snapshot_ts=open_snap.ts,
                want_call=False,
                target_abs_delta=spec.entry_target_abs_delta,
                target_dte_days=spec.entry_target_dte_days,
            )
            if not q:
                continue
            legs = [("sell", q)]

        elif spec.kind == "short_strangle":
            qc = _closest_by_delta_and_dte(
                options=open_snap.options,
                snapshot_ts=open_snap.ts,
                want_call=True,
                target_abs_delta=spec.entry_target_abs_delta,
                target_dte_days=spec.entry_target_dte_days,
            )
            qp = _closest_by_delta_and_dte(
                options=open_snap.options,
                snapshot_ts=open_snap.ts,
                want_call=False,
                target_abs_delta=spec.entry_target_abs_delta,
                target_dte_days=spec.entry_target_dte_days,
            )
            if not qc or not qp:
                continue
            # Align expiries roughly by using the nearer expiry.
            legs = [("sell", qc), ("sell", qp)]

        elif spec.kind == "put_credit_spread":
            short_put = _closest_by_delta_and_dte(
                options=open_snap.options,
                snapshot_ts=open_snap.ts,
                want_call=False,
                target_abs_delta=spec.entry_target_abs_delta,
                target_dte_days=spec.entry_target_dte_days,
            )
            long_put = _closest_by_delta_and_dte(
                options=open_snap.options,
                snapshot_ts=open_snap.ts,
                want_call=False,
                target_abs_delta=0.10,
                target_dte_days=spec.entry_target_dte_days,
                delta_tol=0.15,
                dte_tol_days=20.0,
            )
            if not short_put or not long_put:
                continue
            legs = [("sell", short_put), ("buy", long_put)]

        elif spec.kind == "call_debit_spread":
            long_call = _closest_by_delta_and_dte(
                options=open_snap.options,
                snapshot_ts=open_snap.ts,
                want_call=True,
                target_abs_delta=spec.entry_target_abs_delta,
                target_dte_days=spec.entry_target_dte_days,
            )
            short_call = _closest_by_delta_and_dte(
                options=open_snap.options,
                snapshot_ts=open_snap.ts,
                want_call=True,
                target_abs_delta=0.10,
                target_dte_days=spec.entry_target_dte_days,
                delta_tol=0.15,
                dte_tol_days=20.0,
            )
            if not long_call or not short_call:
                continue
            legs = [("buy", long_call), ("sell", short_call)]

        elif spec.kind == "calendar_call":
            short_call = _closest_by_delta_and_dte(
                options=open_snap.options,
                snapshot_ts=open_snap.ts,
                want_call=True,
                target_abs_delta=spec.entry_target_abs_delta,
                target_dte_days=14.0,
                dte_tol_days=14.0,
            )
            long_call = _closest_by_delta_and_dte(
                options=open_snap.options,
                snapshot_ts=open_snap.ts,
                want_call=True,
                target_abs_delta=spec.entry_target_abs_delta,
                target_dte_days=30.0,
                dte_tol_days=20.0,
            )
            if not short_call or not long_call:
                continue
            legs = [("sell", short_call), ("buy", long_call)]

        else:
            continue

        # Close after N days OR at expiry (whichever comes first).
        # For multi-leg strategies, use the earliest expiry.
        min_expiry_ts = min(int(q.expiry_ts) for _, q in legs)
        close_idx_hold = min(n - 1, i + max(1, int(spec.hold_days)))
        close_idx_expiry = n - 1
        for j in range(i, n):
            if int(snapshots[j].ts) >= min_expiry_ts:
                close_idx_expiry = j
                break
        close_idx = min(close_idx_hold, close_idx_expiry)
        close_snap = snapshots[close_idx]

        entry_pnl = 0.0
        exit_pnl = 0.0
        meta: Dict[str, Any] = {"strategy": spec.name, "legs": []}

        is_valid = True
        dq_status = "ok"

        def mark_invalid(status: str) -> None:
            nonlocal is_valid, dq_status
            is_valid = False
            priority = {
                "missing_close_quote": 4,
                "missing_open_quote": 3,
                "stale_quote": 2,
                "expired_no_quote": 1,
                "ok": 0,
            }
            if priority.get(status, 0) >= priority.get(dq_status, 0):
                dq_status = status

        def close_price_for_leg(q: OptionQuote, *, side: str) -> Optional[float]:
            # If we're at/after expiry and have spot, price as intrinsic.
            if int(close_snap.ts) >= int(q.expiry_ts):
                spot_close = float(close_snap.spot or 0.0)
                if spot_close > 0:
                    return float(_intrinsic_value(q=q, spot=spot_close))
                mark_invalid("expired_no_quote")
                return None

            q2 = _find_same_instrument(close_snap.options, q.instrument_name)
            if not q2:
                mark_invalid("missing_close_quote")
                return None
            px = float(_price_for_fill(q2, side=side, slippage_bps=slippage_bps, use_mid=use_mid))
            if px <= 0:
                mark_invalid("stale_quote")
                return None
            return float(px)

        for side, q in legs:
            entry = _price_for_fill(q, side=side, slippage_bps=slippage_bps, use_mid=use_mid)
            if entry <= 0:
                mark_invalid("missing_open_quote")
            exit_side = "buy" if side == "sell" else "sell"
            exit_price = close_price_for_leg(q, side=exit_side)
            if is_valid and exit_price is not None:
                if side == "sell":
                    # Short option: receive premium, pay to buy back.
                    entry_pnl += entry
                    exit_pnl += float(exit_price)
                else:
                    # Long option: pay premium, receive on sell.
                    entry_pnl -= entry
                    exit_pnl -= float(exit_price)
            meta["legs"].append(
                {
                    "instrument": q.instrument_name,
                    "side": side,
                    "expiry_ts": q.expiry_ts,
                    "strike": q.strike,
                    "option_type": q.option_type,
                    "entry": entry,
                    "exit": exit_price,
                }
            )

        pnl = (entry_pnl - exit_pnl) if is_valid else 0.0
        pnl_pct = (pnl / max(1e-9, spot0)) if is_valid else 0.0

        trades.append(
            TradeResult(
                open_ts=int(open_snap.ts),
                close_ts=int(close_snap.ts),
                pnl=float(pnl),
                pnl_pct=float(pnl_pct),
                is_valid=bool(is_valid),
                data_quality_status=str(dq_status),
                metadata=meta,
            )
        )

    return trades
