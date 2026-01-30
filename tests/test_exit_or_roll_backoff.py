from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path


def test_exit_or_roll_failure_backoff_exact_minutes(tmp_path: Path, monkeypatch):
    """Cooldown backoff uses exit_or_roll_failures.

    cooldown = base * min(4, 2**failures)
      failures=0 => 1x
      failures=1 => 2x
      failures=2 => 4x
      failures>=2 => capped at 4x
    """

    import src.position_tracker as pt
    from src.position_tracker import PositionTracker

    now = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    # Monkeypatch the module-level _utc_now used by PositionTracker
    monkeypatch.setattr(pt, "_utc_now", lambda: now)

    tracker = PositionTracker(persistence_path=tmp_path / "positions.json")

    tracker.process_execution_result(
        {
            "status": "simulated",
            "dry_run": True,
            "action": "OPEN_COVERED_CALL",
            "params": {
                "underlying": "BTC",
                "strategy_type": "COVERED_CALL",
                "symbol": "BTC-26DEC25-100000-C",
                "size": 1.0,
            },
            "orders": [{"symbol": "BTC-26DEC25-100000-C", "size": 1.0, "price": 100.0}],
        }
    )

    pid = tracker.get_open_position_id_for_symbol("BTC-26DEC25-100000-C")
    assert pid is not None

    base = 15

    # failures=0 => 15
    ok = tracker.set_exit_or_roll_cooldown_for_position(position_id=pid, cooldown_minutes=base, increment_failures=False)
    assert ok
    chain = tracker._chains[pid]  # type: ignore[attr-defined]
    assert chain.exit_or_roll_failures == 0
    assert chain.exit_or_roll_cooldown_until == now + timedelta(minutes=15)

    # failures=1 => 30
    ok = tracker.set_exit_or_roll_cooldown_for_position(position_id=pid, cooldown_minutes=base, increment_failures=True)
    assert ok
    chain = tracker._chains[pid]  # type: ignore[attr-defined]
    assert chain.exit_or_roll_failures == 1
    assert chain.exit_or_roll_cooldown_until == now + timedelta(minutes=30)

    # failures=2 => 60
    ok = tracker.set_exit_or_roll_cooldown_for_position(position_id=pid, cooldown_minutes=base, increment_failures=True)
    assert ok
    chain = tracker._chains[pid]  # type: ignore[attr-defined]
    assert chain.exit_or_roll_failures == 2
    assert chain.exit_or_roll_cooldown_until == now + timedelta(minutes=60)

    # failures=3 => still capped at 60
    ok = tracker.set_exit_or_roll_cooldown_for_position(position_id=pid, cooldown_minutes=base, increment_failures=True)
    assert ok
    chain = tracker._chains[pid]  # type: ignore[attr-defined]
    assert chain.exit_or_roll_failures == 3
    assert chain.exit_or_roll_cooldown_until == now + timedelta(minutes=60)
