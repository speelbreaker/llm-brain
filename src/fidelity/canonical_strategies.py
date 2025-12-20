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
        close_idx = min(n - 1, i + max(1, int(spec.hold_days)))
        close_snap = snapshots[close_idx]
        spot0 = float(open_snap.spot or 0.0)
        if spot0 <= 0:
            continue

        def close_price_for(name: str, expiry_ts: int, side: str) -> float:
            if close_snap.ts >= int(expiry_ts):
                return 0.0
            q2 = _find_same_instrument(close_snap.options, name)
            if not q2:
                return 0.0
            return _price_for_fill(q2, side=side, slippage_bps=slippage_bps, use_mid=use_mid)

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

        entry_pnl = 0.0
        exit_pnl = 0.0
        meta: Dict[str, Any] = {"strategy": spec.name, "legs": []}

        for side, q in legs:
            entry = _price_for_fill(q, side=side, slippage_bps=slippage_bps, use_mid=use_mid)
            exit_side = "buy" if side == "sell" else "sell"
            exit_price = close_price_for(q.instrument_name, q.expiry_ts, exit_side)
            if side == "sell":
                # Short option: receive premium, pay to buy back.
                entry_pnl += entry
                exit_pnl += exit_price
            else:
                # Long option: pay premium, receive on sell.
                entry_pnl -= entry
                exit_pnl -= exit_price
            meta["legs"].append(
                {
                    "instrument": q.instrument_name,
                    "side": side,
                    "expiry_ts": q.expiry_ts,
                    "entry": entry,
                    "exit": exit_price,
                }
            )

        pnl = entry_pnl - exit_pnl
        pnl_pct = pnl / max(1e-9, spot0)

        trades.append(
            TradeResult(
                open_ts=int(open_snap.ts),
                close_ts=int(close_snap.ts),
                pnl=float(pnl),
                pnl_pct=float(pnl_pct),
                metadata=meta,
            )
        )

    return trades
