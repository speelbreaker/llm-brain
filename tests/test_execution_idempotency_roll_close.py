from __future__ import annotations

import json
from pathlib import Path

import pytest


class FakeDeribitClient:
    def __init__(self):
        self.place_calls = []
        self.cancel_calls = []
        self._order_state = {
            "order_id": "oid-1",
            "instrument_name": "BTC-26DEC25-100000-C",
            "order_state": "open",
            "amount": 1.0,
            "filled_amount": 0.0,
            "average_price": None,
            "last_update_timestamp": 1,
        }
        self._ticker = {
            "best_bid_price": 0.01,
            "best_ask_price": 0.011,
            "mark_price": 0.0105,
        }

    def get_ticker(self, instrument_name: str):
        return dict(self._ticker)

    def place_order(self, **kwargs):
        self.place_calls.append(kwargs)
        return {"order": {"order_id": self._order_state["order_id"]}}

    def cancel_order(self, order_id: str):
        self.cancel_calls.append(order_id)
        return {"result": True}

    def get_order_state(self, order_id: str):
        return {"order": dict(self._order_state)}

    # label endpoints used by reconcile (not required by these unit tests)
    def get_open_orders_by_label(self, currency: str, label: str):
        return []

    def get_order_state_by_label(self, currency: str, label: str):
        return []

    def cancel_by_label(self, label: str, currency: str | None = None):
        return {"result": True}


def _read_ledger(path: Path) -> dict:
    return json.loads(path.read_text())


def test_submit_unknown_gate_prevents_second_dispatch(tmp_path: Path, monkeypatch):
    """If first dispatch throws and ledger is SUBMIT_UNKNOWN, a second execute must not call place_order."""
    import src.execution as ex
    from src.execution_ledger import ExecutionLedger

    ledger_path = tmp_path / "execution_ledger.json"
    ex._ledger = ExecutionLedger(path=ledger_path)

    # Make position_id stable
    monkeypatch.setattr(ex.position_tracker, "get_open_position_id_for_symbol", lambda sym: "pid-1")

    client = FakeDeribitClient()

    # First call: dispatch throws => SUBMIT_UNKNOWN
    def _raise(**kwargs):
        client.place_calls.append(kwargs)
        raise RuntimeError("network down")

    client.place_order = _raise

    action = {
        "action": "ROLL_COVERED_CALL",
        "params": {
            "from_symbol": "BTC-26DEC25-100000-C",
            "to_symbol": "BTC-27DEC25-110000-C",
            "size": 1.0,
            "underlying": "BTC",
        },
        "reasoning": "test",
    }

    r1 = ex.execute_action(client, action, config=ex.settings)
    assert r1["status"] == "error"
    assert "SUBMIT_UNKNOWN" in " ".join(r1.get("errors", []))
    assert len(client.place_calls) == 1

    # Second call: must return in_flight, no dispatch
    r2 = ex.execute_action(client, action, config=ex.settings)
    assert r2["status"] == "in_flight"
    assert len(client.place_calls) == 1


def test_acked_gate_prevents_second_dispatch(tmp_path: Path, monkeypatch):
    """If ledger has ACKED attempt, a second execute must not call place_order again."""
    import src.execution as ex
    from src.execution_ledger import ExecutionLedger

    ledger_path = tmp_path / "execution_ledger.json"
    ex._ledger = ExecutionLedger(path=ledger_path)

    monkeypatch.setattr(ex.position_tracker, "get_open_position_id_for_symbol", lambda sym: "pid-1")

    client = FakeDeribitClient()

    # Make poller return immediate timeout so we don't wait 30s
    monkeypatch.setattr(
        ex,
        "_poll_order_until_terminal_or_timeout",
        lambda *a, **k: (ex.OrderPollStatus.OPEN_TIMEOUT_UNFILLED, {"order_state": "open", "amount": 1.0, "filled_amount": 0.0, "average_price": None}),
    )

    action = {
        "action": "ROLL_COVERED_CALL",
        "params": {
            "from_symbol": "BTC-26DEC25-100000-C",
            "to_symbol": "BTC-27DEC25-110000-C",
            "size": 1.0,
            "underlying": "BTC",
        },
        "reasoning": "test",
    }

    r1 = ex.execute_action(client, action, config=ex.settings)
    assert len(client.place_calls) == 1

    r2 = ex.execute_action(client, action, config=ex.settings)
    assert r2["status"] == "in_flight"
    assert len(client.place_calls) == 1


def test_active_intent_reuse_by_position_id(tmp_path: Path, monkeypatch):
    """If there is already an ACTIVE ROLL_CC intent for position_id, execution must reuse it (no new uuid4)."""
    import src.execution as ex
    from src.execution_ledger import ExecutionLedger

    ledger_path = tmp_path / "execution_ledger.json"
    ex._ledger = ExecutionLedger(path=ledger_path)

    monkeypatch.setattr(ex.position_tracker, "get_open_position_id_for_symbol", lambda sym: "pid-1")

    # Create an active intent by running a SUBMIT_UNKNOWN dispatch once.
    client = FakeDeribitClient()

    def _raise(**kwargs):
        raise RuntimeError("network down")

    client.place_order = _raise

    action = {
        "action": "ROLL_COVERED_CALL",
        "params": {
            "from_symbol": "BTC-26DEC25-100000-C",
            "to_symbol": "BTC-27DEC25-110000-C",
            "size": 1.0,
            "underlying": "BTC",
        },
        "reasoning": "test",
    }

    r1 = ex.execute_action(client, action, config=ex.settings)
    iid1 = r1.get("intent_id")
    assert iid1

    # Next call should reuse iid1 (and return in_flight due to SUBMIT_UNKNOWN)
    r2 = ex.execute_action(client, action, config=ex.settings)
    assert r2.get("intent_id") == iid1
    assert r2["status"] == "in_flight"


def test_i6_instrument_mismatch_aborts_intent(tmp_path: Path):
    from src.execution_ledger import ExecutionLedger, OrderPlan

    ledger = ExecutionLedger(path=tmp_path / "execution_ledger.json")

    iid = "a" * 32
    label, created_now = ledger.prewrite_attempt(
        intent_id=iid,
        position_id="pid-1",
        intent_type="ROLL_CC",
        currency="BTC",
        leg="CLOSE",
        attempt=0,
        plan=OrderPlan(
            instrument_name="BTC-26DEC25-100000-C",
            side="buy",
            amount=1.0,
            order_type="limit",
            price=0.01,
            post_only=False,
            reduce_only=True,
        ),
    )
    assert created_now is True
    assert label.startswith("cc|")

    # Now update with mismatched instrument_name
    ledger.update_attempt_from_truth(
        intent_id=iid,
        leg="CLOSE",
        attempt=0,
        truth={
            "instrument_name": "BTC-27DEC25-110000-C",
            "order_state": "open",
            "amount": 1.0,
            "filled_amount": 0.0,
            "average_price": None,
        },
    )

    data = _read_ledger(tmp_path / "execution_ledger.json")
    assert data["intents"][iid]["state"] == "ABORTED"
    assert data["intents"][iid].get("abort_reason") == "I6_INSTRUMENT_MISMATCH"
