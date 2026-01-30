from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Literal, Optional


IntentState = Literal["ACTIVE", "DONE", "ABORTED"]
LegName = Literal["CLOSE", "OPEN"]
LegChar = Literal["c", "o"]
LegState = Literal["NOT_PLANNED", "PLANNED", "DISPATCHED", "OPEN", "TERMINAL"]
DispatchState = Literal["PREWRITTEN", "ACKED", "SUBMIT_UNKNOWN"]
TerminalOutcome = Literal["FILLED", "CANCELLED", "REJECTED", "PARTIAL"]


class LedgerCorruptionError(RuntimeError):
    pass


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def make_label(intent_id: str, leg: LegName, attempt: int) -> str:
    """Label construction spec (verbatim).

    Format: cc|{intent20}|{leg_char}|a{attempt}
    Where intent20=intent_id[:20], leg_char='c' if CLOSE else 'o'.
    """
    leg_char: LegChar = "c" if leg == "CLOSE" else "o"
    intent20 = (intent_id or "")[:20]
    label = f"cc|{intent20}|{leg_char}|a{int(attempt)}"

    # Invariants I1
    if len(label) > 64:
        raise ValueError(f"label too long ({len(label)}): {label}")

    # ASCII-safe + restricted char set [a-z0-9|:_-] + digits
    allowed = set("abcdefghijklmnopqrstuvwxyz0123456789|:_-")
    for ch in label:
        if ch not in allowed:
            raise ValueError(f"label contains disallowed char {ch!r}: {label}")

    return label


@dataclass
class OrderPlan:
    instrument_name: str
    side: Literal["buy", "sell"]
    amount: float
    order_type: str
    price: Optional[float]
    post_only: bool
    reduce_only: bool


class ExecutionLedger:
    """Execution ledger / WAL for restart-safe order intents.

    Atomic writer pattern reused from PositionTracker._save_to_disk (tempfile+fsync+replace).
    """

    def __init__(self, path: Optional[Path] = None) -> None:
        self._lock = Lock()
        self._path = path or Path("data/execution_ledger.json")
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def _load_unlocked(self) -> Dict[str, Any]:
        if not self._path.exists():
            return {"version": 1, "intents": {}}
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"version": 1, "intents": {}}

    def _save_unlocked(self, data: Dict[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix=self._path.name + ".",
            suffix=".tmp",
            dir=str(self._path.parent),
        )
        tmp_path = Path(tmp_name)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                pass
        os.replace(tmp_path, self._path)

    def prewrite_attempt(
        self,
        *,
        intent_id: str,
        position_id: str,
        intent_type: str,
        currency: str,
        leg: LegName,
        attempt: int,
        plan: OrderPlan,
        now: Optional[datetime] = None,
    ) -> str:
        """WAL write-before-dispatch: commit PREWRITTEN attempt before any network call."""
        ts = (now or _utc_now()).isoformat()
        label = make_label(intent_id, leg, attempt)

        with self._lock:
            data = self._load_unlocked()
            intents = data.setdefault("intents", {})
            intent = intents.get(intent_id)
            if intent is None:
                intent = {
                    "position_id": position_id,
                    "intent_type": intent_type,
                    "state": "ACTIVE",
                    "currency": currency,
                    "legs": {
                        "close": {"leg_state": "NOT_PLANNED", "terminal_outcome": None, "plan": None, "attempts": []},
                        "open": {"leg_state": "NOT_PLANNED", "terminal_outcome": None, "plan": None, "attempts": []},
                    },
                    "attempts": 0,
                    "last_error": None,
                    "abort_reason": None,
                }
                intents[intent_id] = intent

            if intent.get("state") != "ACTIVE":
                raise ValueError(f"intent {intent_id} not ACTIVE (state={intent.get('state')})")

            leg_key = "close" if leg == "CLOSE" else "open"
            leg_rec = intent["legs"][leg_key]

            # Overwrite plan for this attempt
            leg_rec["plan"] = {
                "instrument_name": plan.instrument_name,
                "side": plan.side,
                "amount": float(plan.amount),
                "type": plan.order_type,
                "price": float(plan.price) if plan.price is not None else None,
                "post_only": bool(plan.post_only),
                "reduce_only": bool(plan.reduce_only),
            }

            # Idempotent prewrite: if attempt N already exists, return its label and do NOT append.
            existing_attempts = leg_rec.get("attempts", []) or []
            for a in existing_attempts:
                if int(a.get("attempt", -1)) == int(attempt):
                    return str(a.get("label") or label)

            # Append attempt record
            attempt_rec = {
                "attempt": int(attempt),
                "label": label,
                "order_id": None,
                "dispatch_state": "PREWRITTEN",
                "submitted_at": ts,
                "last_checked_at": None,
                "instrument_name": plan.instrument_name,
                "amount": float(plan.amount),
                "filled_amount": 0.0,
                "average_price": None,
                "order_state": "unknown",
                "last_update_timestamp_ms": None,
                "error": None,
                "terminal_outcome": None,
            }

            # I1 attempt-unique label: ensure no other attempt uses it
            for k in ("close", "open"):
                for a in intent["legs"][k].get("attempts", []) or []:
                    if a.get("label") == label:
                        raise ValueError(f"label reuse detected for intent {intent_id}: {label}")

            leg_rec.setdefault("attempts", []).append(attempt_rec)
            leg_rec["leg_state"] = "PLANNED"
            intent["attempts"] = int(intent.get("attempts", 0) or 0)

            self._save_unlocked(data)

        return label

    def commit_dispatch_result(
        self,
        *,
        intent_id: str,
        leg: LegName,
        attempt: int,
        ok: bool,
        order_id: Optional[str],
        error: Optional[str],
        now: Optional[datetime] = None,
    ) -> None:
        ts = (now or _utc_now()).isoformat()
        with self._lock:
            data = self._load_unlocked()
            intent = (data.get("intents") or {}).get(intent_id)
            if not intent:
                return
            leg_key = "close" if leg == "CLOSE" else "open"
            leg_rec = intent["legs"][leg_key]

            attempts = leg_rec.get("attempts", []) or []
            tgt = None
            for a in attempts:
                if int(a.get("attempt", -1)) == int(attempt):
                    tgt = a
                    break
            if tgt is None:
                return

            if ok:
                tgt["order_id"] = order_id
                tgt["dispatch_state"] = "ACKED"
            else:
                tgt["dispatch_state"] = "SUBMIT_UNKNOWN"
                tgt["error"] = error
                intent["last_error"] = error

            leg_rec["leg_state"] = "DISPATCHED"
            # last_checked_at remains None; reconcile loop sets it.
            self._save_unlocked(data)

    def abort_intent(self, *, intent_id: str, reason: str) -> None:
        with self._lock:
            data = self._load_unlocked()
            intent = (data.get("intents") or {}).get(intent_id)
            if not intent:
                return
            intent["state"] = "ABORTED"
            intent["abort_reason"] = reason
            self._save_unlocked(data)

    def mark_done_if_terminal(self, *, intent_id: str) -> None:
        with self._lock:
            data = self._load_unlocked()
            intent = (data.get("intents") or {}).get(intent_id)
            if not intent:
                return
            legs = intent.get("legs") or {}
            c = (legs.get("close") or {}).get("terminal_outcome")
            o = (legs.get("open") or {}).get("terminal_outcome")
            if c == "FILLED" and o == "FILLED":
                intent["state"] = "DONE"
                self._save_unlocked(data)

    def get_active_intents(self) -> List[Dict[str, Any]]:
        with self._lock:
            data = self._load_unlocked()
            intents = data.get("intents") or {}
            out = []
            for intent_id, rec in intents.items():
                if rec.get("state") == "ACTIVE":
                    out.append({"intent_id": intent_id, **rec})
            return out

    def get_active_intent_id(self, *, position_id: str, intent_type: str) -> Optional[str]:
        """Return the active intent_id for a position+type.

        - Returns None if none exists.
        - Raises LedgerCorruptionError if multiple ACTIVE intents exist for the same (position_id,intent_type).
        """
        with self._lock:
            data = self._load_unlocked()
            intents = data.get("intents") or {}
            found = None
            for iid, rec in intents.items():
                if rec.get("state") != "ACTIVE":
                    continue
                if str(rec.get("position_id")) != str(position_id):
                    continue
                if str(rec.get("intent_type")) != str(intent_type):
                    continue
                if found and found != iid:
                    raise LedgerCorruptionError(
                        f"multiple ACTIVE intents for position_id={position_id}, intent_type={intent_type}: {found}, {iid}"
                    )
                found = iid
            return found

    def get_latest_attempt(self, *, intent_id: str, leg: LegName) -> Optional[Dict[str, Any]]:
        with self._lock:
            data = self._load_unlocked()
            intent = (data.get("intents") or {}).get(intent_id)
            if not intent:
                return None
            leg_key = "close" if leg == "CLOSE" else "open"
            leg_rec = (intent.get("legs") or {}).get(leg_key) or {}
            attempts = list(leg_rec.get("attempts") or [])
            if not attempts:
                return None
            return attempts[-1]

    def update_attempt_from_truth(
        self,
        *,
        intent_id: str,
        leg: LegName,
        attempt: int,
        truth: Dict[str, Any],
        now: Optional[datetime] = None,
    ) -> None:
        """Idempotent attempt update from Deribit truth."""
        eps = 1e-9
        ts = (now or _utc_now()).isoformat()

        with self._lock:
            data = self._load_unlocked()
            intent = (data.get("intents") or {}).get(intent_id)
            if not intent:
                return

            leg_key = "close" if leg == "CLOSE" else "open"
            leg_rec = intent["legs"][leg_key]
            attempts = leg_rec.get("attempts", []) or []
            tgt = None
            for a in attempts:
                if int(a.get("attempt", -1)) == int(attempt):
                    tgt = a
                    break
            if tgt is None:
                return

            tgt["last_checked_at"] = ts
            for k in (
                "order_state",
                "filled_amount",
                "amount",
                "average_price",
                "order_id",
                "last_update_timestamp_ms",
                "instrument_name",
            ):
                if k in truth and truth[k] is not None:
                    tgt[k] = truth[k]

            # I6 instrument match
            planned_instr = (leg_rec.get("plan") or {}).get("instrument_name")
            obs_instr = truth.get("instrument_name")
            if planned_instr and obs_instr and str(planned_instr) != str(obs_instr):
                intent["state"] = "ABORTED"
                intent["abort_reason"] = "I6_INSTRUMENT_MISMATCH"
                self._save_unlocked(data)
                return

            order_state = str(tgt.get("order_state") or "unknown").lower()
            filled = float(tgt.get("filled_amount") or 0.0)
            amount = float(tgt.get("amount") or 0.0)

            terminal = False
            if order_state in {"filled", "cancelled", "rejected"}:
                terminal = True
            if amount > 0 and filled >= amount - eps:
                terminal = True

            # Promote PREWRITTEN/SUBMIT_UNKNOWN to ACKED when truth shows an order exists (crash-window safety).
            if (tgt.get("dispatch_state") in {"PREWRITTEN", "SUBMIT_UNKNOWN"}) and (
                tgt.get("order_id") is not None or tgt.get("order_state") not in (None, "unknown")
            ):
                tgt["dispatch_state"] = "ACKED"

            if terminal:
                leg_rec["leg_state"] = "TERMINAL"
                if filled >= amount - eps and amount > 0:
                    tgt["terminal_outcome"] = "FILLED"
                    leg_rec["terminal_outcome"] = "FILLED"
                elif order_state == "cancelled" and filled <= eps:
                    tgt["terminal_outcome"] = "CANCELLED"
                    leg_rec["terminal_outcome"] = "CANCELLED"
                elif order_state == "rejected" and filled <= eps:
                    tgt["terminal_outcome"] = "REJECTED"
                    leg_rec["terminal_outcome"] = "REJECTED"
                elif filled > eps and amount > 0 and filled < amount - eps:
                    tgt["terminal_outcome"] = "PARTIAL"
                    leg_rec["terminal_outcome"] = "PARTIAL"
                else:
                    # Default conservative classification
                    if filled > eps:
                        tgt["terminal_outcome"] = "PARTIAL"
                        leg_rec["terminal_outcome"] = "PARTIAL"
                    else:
                        tgt["terminal_outcome"] = "CANCELLED"
                        leg_rec["terminal_outcome"] = "CANCELLED"
            elif order_state == "open":
                leg_rec["leg_state"] = "OPEN"
            else:
                # unknown-ish
                leg_rec["leg_state"] = "DISPATCHED"

            self._save_unlocked(data)
