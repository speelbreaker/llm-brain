"""Parallel paper portfolios for strategy comparison.

Keeps separate PositionTracker instances (rule/llm/debate) that are updated using
simulated execution results (no exchange orders).

Fill model (Phase 1): mark price +/- fixed bps (slippage).
"""

from __future__ import annotations

import logging

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal

from src.config import settings
from src.models import ActionType, AgentState, CandidateOption
from src.position_tracker import PositionTracker

Lane = Literal["rule", "llm", "debate"]

logger = logging.getLogger(__name__)

def _lane_path(lane: Lane) -> Path:
    return Path(f"data/paper_positions_{lane}.json")


_TRACKERS: dict[Lane, PositionTracker] = {}


def get_tracker(lane: Lane) -> PositionTracker:
    t = _TRACKERS.get(lane)
    if t is not None:
        return t
    t = PositionTracker(persistence_path=_lane_path(lane))
    _TRACKERS[lane] = t
    return t


def _fill_price(mark: float, *, side: Literal["buy", "sell"], slippage_bps: float) -> float:
    if mark <= 0:
        return 0.0
    adj = float(slippage_bps) / 10_000.0
    if side == "buy":
        return mark * (1.0 + adj)
    return mark * (1.0 - adj)


def _candidate_mark(c: CandidateOption) -> float:
    # CandidateOption has mid_price, bid/ask.
    if c.mid_price is not None and float(c.mid_price) > 0:
        return float(c.mid_price)
    bid = float(c.bid or 0.0)
    ask = float(c.ask or 0.0)
    if bid > 0 and ask > 0:
        return (bid + ask) / 2.0
    return max(bid, ask, 0.0)


def _has_open_symbol(tracker: PositionTracker, symbol: str) -> bool:
    payload = tracker.get_open_positions_payload(include_sandbox=True)
    for p in payload.get("positions") or []:
        if p.get("symbol") == symbol:
            return True
    return False


def _has_any_open_call(tracker: PositionTracker, underlying: str) -> bool:
    payload = tracker.get_open_positions_payload(include_sandbox=True)
    for p in payload.get("positions") or []:
        if p.get("underlying") == underlying and p.get("strategy") in (None, "COVERED_CALL"):
            return True
        if p.get("underlying") == underlying and (p.get("strategy") or p.get("strategy_type")) == "COVERED_CALL":
            return True
    return False


def refresh_marks_for_all(client: Any) -> None:
    for lane in ("rule", "llm", "debate"):
        get_tracker(lane).refresh_marks(client)


def apply_decision_to_lane(
    *,
    lane: Lane,
    state: AgentState,
    candidates: list[CandidateOption],
    decision: Dict[str, Any] | None,
    client: Any,
    refresh_marks: bool = True,
) -> None:
    """Apply a decision into a paper portfolio lane.

    This never sends orders. It only updates paper PositionTracker state.
    """
    if not getattr(settings, "paper_compare_enabled", False):
        return

    tracker = get_tracker(lane)
    if refresh_marks:
        tracker.refresh_marks(client)

    if not decision or not isinstance(decision, dict):
        return

    action = str(decision.get("action") or ActionType.DO_NOTHING.value)
    params = decision.get("params") or {}

    if action == ActionType.DO_NOTHING.value:
        return

    slippage_bps = float(getattr(settings, "paper_slippage_bps", 10.0))

    if action == ActionType.OPEN_COVERED_CALL.value:
        symbol = str(params.get("symbol") or "")
        if not symbol:
            return
        # don't spam opens in paper lane
        if _has_open_symbol(tracker, symbol) or _has_any_open_call(tracker, params.get("underlying") or ((getattr(state, "underlyings", None) or [None])[0] if (getattr(state, "underlyings", None) or []) else None)):
            return

        cand = next((c for c in candidates if c.symbol == symbol), None)
        mark = _candidate_mark(cand) if cand else 0.0
        if mark <= 0:
            logger.info("[Paper:%s] skip OPEN %s: invalid mark=%s", lane, symbol, mark)
            return
        price = _fill_price(mark, side="sell", slippage_bps=slippage_bps)
        size = float(params.get("size") or params.get("quantity") or settings.default_order_size)

        tracker.process_execution_result(
            {
                "status": "simulated",
                "action": action,
                "params": {**params, "symbol": symbol, "underlying": params.get("underlying") or (cand.underlying if cand else "BTC"), "size": size},
                "orders": [{"symbol": symbol, "size": size, "price": price}],
                "dry_run": True,
            }
        )
        return

    if action == ActionType.CLOSE_COVERED_CALL.value:
        symbol = str(params.get("symbol") or "")
        if not symbol:
            return
        try:
            ticker = client.get_ticker(symbol)
            mark = float(ticker.get("mark_price") or 0.0) if ticker else 0.0
        except Exception:
            mark = 0.0
        if mark <= 0:
            logger.info("[Paper:%s] skip OPEN %s: invalid mark=%s", lane, symbol, mark)
            return
        price = _fill_price(mark, side="buy", slippage_bps=slippage_bps)
        size = float(params.get("size") or params.get("quantity") or settings.default_order_size)
        tracker.process_execution_result(
            {
                "status": "simulated",
                "action": action,
                "params": {**params, "symbol": symbol, "size": size},
                "orders": [{"symbol": symbol, "size": size, "price": price}],
                "dry_run": True,
            }
        )
        return

    if action == ActionType.ROLL_COVERED_CALL.value:
        from_symbol = str(params.get("from_symbol") or "")
        to_symbol = str(params.get("to_symbol") or "")
        if not from_symbol or not to_symbol:
            return

        # buy back old
        try:
            ticker = client.get_ticker(from_symbol)
            from_mark = float(ticker.get("mark_price") or 0.0) if ticker else 0.0
        except Exception:
            from_mark = 0.0
        if from_mark <= 0:
            logger.info("[Paper:%s] skip ROLL close %s: invalid mark=%s", lane, from_symbol, from_mark)
            return
        buy_price = _fill_price(from_mark, side="buy", slippage_bps=slippage_bps)

        cand = next((c for c in candidates if c.symbol == to_symbol), None)
        to_mark = _candidate_mark(cand) if cand else 0.0
        if to_mark <= 0:
            logger.info("[Paper:%s] skip ROLL open %s: invalid mark=%s", lane, to_symbol, to_mark)
            return
        sell_price = _fill_price(to_mark, side="sell", slippage_bps=slippage_bps)

        size = float(params.get("size") or params.get("quantity") or settings.default_order_size)

        tracker.process_execution_result(
            {
                "status": "simulated",
                "action": action,
                "params": {**params, "from_symbol": from_symbol, "to_symbol": to_symbol, "size": size},
                "orders": [
                    {"symbol": from_symbol, "size": size, "price": buy_price},
                    {"symbol": to_symbol, "size": size, "price": sell_price},
                ],
                "dry_run": True,
            }
        )
        return
