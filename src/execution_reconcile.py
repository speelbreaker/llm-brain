from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.deribit_client import DeribitClient
from src.execution_ledger import ExecutionLedger


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_currency(currency: Optional[str], instrument_name: Optional[str]) -> Optional[str]:
    if currency:
        return str(currency).upper()
    if instrument_name and "-" in instrument_name:
        return instrument_name.split("-")[0].upper()
    return None


def reconcile_execution_ledger(client: DeribitClient, ledger: ExecutionLedger) -> None:
    """Reconcile ACTIVE intents by Deribit truth.

    Runs on startup and each tick before dispatching new orders.

    This is the recovery algorithm as specified (label-first when order_id is missing).
    """
    eps = 1e-9

    active = ledger.get_active_intents()

    # Corruption sweep: multiple ACTIVE intents for same (position_id,intent_type)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for it in active:
        key = (str(it.get("position_id")), str(it.get("intent_type")))
        groups.setdefault(key, []).append(it)

    for (pid, itype), items in groups.items():
        if pid and itype and len(items) > 1:
            # Fail-closed: cancel all known attempt labels and abort all intents.
            for it in items:
                iid = it.get("intent_id")
                ccy = _resolve_currency(it.get("currency"), None)
                legs = it.get("legs") or {}
                for leg_key in ("close", "open"):
                    for a in (legs.get(leg_key) or {}).get("attempts", []) or []:
                        lbl = a.get("label")
                        if lbl:
                            try:
                                if ccy:
                                    client.cancel_by_label(str(lbl), ccy)
                                else:
                                    client.cancel_by_label(str(lbl))
                            except Exception:
                                pass
                if iid:
                    ledger.abort_intent(intent_id=str(iid), reason="CORRUPT_MULTIPLE_ACTIVE")

    # Refresh list after aborts
    for intent in ledger.get_active_intents():
        intent_id = intent.get("intent_id")
        if not intent_id:
            continue

        intent_type = intent.get("intent_type")
        currency = intent.get("currency")
        legs = intent.get("legs") or {}
        close_leg = legs.get("close") or {}
        open_leg = legs.get("open") or {}

        close_outcome = close_leg.get("terminal_outcome")
        open_outcome = open_leg.get("terminal_outcome")

        # Determine active leg
        if close_outcome != "FILLED":
            active_leg_key = "close"
            active_leg_name = "CLOSE"
        elif open_outcome != "FILLED":
            active_leg_key = "open"
            active_leg_name = "OPEN"
        else:
            # only DONE when both FILLED for ROLL_CC
            if intent_type == "ROLL_CC":
                ledger.mark_done_if_terminal(intent_id=intent_id)
            continue

        # ROLL ordering invariant
        if intent_type == "ROLL_CC" and active_leg_name == "CLOSE":
            # If close not FILLED, open must not be dispatched
            if (open_leg.get("leg_state") in {"DISPATCHED", "OPEN", "TERMINAL"}) and close_outcome != "FILLED":
                ledger.abort_intent(intent_id=intent_id, reason="I5_OPEN_BEFORE_CLOSE")
                try:
                    # best-effort cleanup: cancel any in-flight open attempts
                    for a in (open_leg.get("attempts") or []):
                        lbl = a.get("label")
                        if lbl:
                            client.cancel_by_label(str(lbl))
                except Exception:
                    pass
                continue

        leg = legs.get(active_leg_key) or {}
        attempts: List[Dict[str, Any]] = list(leg.get("attempts") or [])
        if not attempts:
            continue

        # current attempt is the last appended
        a = attempts[-1]
        attempt_n = int(a.get("attempt") or 0)
        label = a.get("label")
        order_id = a.get("order_id")

        plan = leg.get("plan") or {}
        planned_instr = plan.get("instrument_name")
        ccy = _resolve_currency(currency, planned_instr or a.get("instrument_name"))
        if not ccy:
            ledger.abort_intent(intent_id=intent_id, reason="CURRENCY_RESOLVE_FAILED")
            continue

        truth: Dict[str, Any] = {
            "order_id": order_id,
            "instrument_name": None,
            "order_state": None,
            "amount": None,
            "filled_amount": None,
            "average_price": None,
            "last_update_timestamp_ms": None,
        }

        # 2.1 Acquire truth
        if order_id:
            try:
                st = client.get_order_state(str(order_id))
                o = st.get("order") if isinstance(st, dict) and "order" in st else st
                o = o or {}
                truth.update(
                    {
                        "order_id": o.get("order_id") or order_id,
                        "instrument_name": o.get("instrument_name"),
                        "order_state": o.get("order_state"),
                        "amount": float(o.get("amount") or 0.0),
                        "filled_amount": float(o.get("filled_amount") or 0.0),
                        "average_price": (float(o.get("average_price")) if o.get("average_price") is not None else None),
                        "last_update_timestamp_ms": o.get("last_update_timestamp"),
                    }
                )
            except Exception:
                pass
        else:
            # label-based recovery
            if not label:
                continue

            try:
                open_orders = client.get_open_orders_by_label(ccy, str(label))
            except Exception:
                open_orders = []

            if len(open_orders) > 1:
                # I2 violation
                try:
                    client.cancel_by_label(str(label), ccy)
                except Exception:
                    pass
                ledger.abort_intent(intent_id=intent_id, reason="I2_MULTIPLE_OPEN_ORDERS_PER_LABEL")
                continue

            if len(open_orders) == 1:
                o = open_orders[0] or {}
                oid = o.get("order_id")
                truth.update(
                    {
                        "order_id": oid,
                        "instrument_name": o.get("instrument_name"),
                        "order_state": o.get("order_state") or "open",
                        "amount": float(o.get("amount") or 0.0),
                        "filled_amount": float(o.get("filled_amount") or 0.0),
                        "average_price": (float(o.get("average_price")) if o.get("average_price") is not None else None),
                        "last_update_timestamp_ms": o.get("last_update_timestamp"),
                    }
                )
                # prefer polling by order_id
                if oid:
                    try:
                        st = client.get_order_state(str(oid))
                        o2 = st.get("order") if isinstance(st, dict) and "order" in st else st
                        o2 = o2 or {}
                        truth.update(
                            {
                                "order_id": o2.get("order_id") or oid,
                                "instrument_name": o2.get("instrument_name"),
                                "order_state": o2.get("order_state"),
                                "amount": float(o2.get("amount") or 0.0),
                                "filled_amount": float(o2.get("filled_amount") or 0.0),
                                "average_price": (float(o2.get("average_price")) if o2.get("average_price") is not None else None),
                                "last_update_timestamp_ms": o2.get("last_update_timestamp"),
                            }
                        )
                    except Exception:
                        pass
            else:
                # open_orders == 0 => get_order_state_by_label
                try:
                    states = client.get_order_state_by_label(ccy, str(label))
                except Exception:
                    states = []

                # Select most recent by last_update_timestamp
                best = None
                best_ts = -1
                for o in states or []:
                    ts = int(o.get("last_update_timestamp") or 0)
                    if ts >= best_ts:
                        best_ts = ts
                        best = o

                if best is None:
                    # remain SUBMIT_UNKNOWN
                    continue

                truth.update(
                    {
                        "order_id": best.get("order_id"),
                        "instrument_name": best.get("instrument_name"),
                        "order_state": best.get("order_state"),
                        "amount": float(best.get("amount") or 0.0),
                        "filled_amount": float(best.get("filled_amount") or 0.0),
                        "average_price": (float(best.get("average_price")) if best.get("average_price") is not None else None),
                        "last_update_timestamp_ms": best.get("last_update_timestamp"),
                    }
                )

        # 2.2 + 2.3 Update ledger + terminal checks (includes I6)
        ledger.update_attempt_from_truth(intent_id=intent_id, leg=active_leg_name, attempt=attempt_n, truth=truth)

        # Abort rules
        # close partial => abort for ROLL_CC
        if intent_type == "ROLL_CC" and active_leg_name == "CLOSE":
            filled = float(truth.get("filled_amount") or 0.0)
            amount = float(truth.get("amount") or 0.0)
            order_state = str(truth.get("order_state") or "").lower()
            is_terminal = order_state in {"filled", "cancelled", "rejected"} or (amount > 0 and filled >= amount - eps)
            if is_terminal and filled > eps and amount > 0 and filled < amount - eps:
                ledger.abort_intent(intent_id=intent_id, reason="I5_CLOSE_PARTIAL_ABORT")
                try:
                    if label:
                        client.cancel_by_label(str(label), ccy)
                except Exception:
                    pass
