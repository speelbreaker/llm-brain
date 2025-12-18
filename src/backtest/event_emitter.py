"""
Event emitter infrastructure for backtest strategy analytics.

Provides a consistent interface for strategies (especially GregBot) to emit
events during backtests. Events are stored in the database and can be
aggregated for strategy breakdown summaries.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional
from enum import Enum


class EventType(str, Enum):
    """Standard event types for strategy analytics."""

    DECISION = "DECISION"
    OPEN = "OPEN"
    CLOSE = "CLOSE"
    TAKE_PROFIT = "TAKE_PROFIT"
    ROLL = "ROLL"
    SKIP = "SKIP"
    STOP_LOSS = "STOP_LOSS"
    EXPIRY = "EXPIRY"


@dataclass
class BacktestEventRecord:
    """In-memory representation of a backtest event."""

    event_time: datetime
    selector_name: str
    strategy_key: str
    event_type: str
    trade_id: Optional[str] = None
    position_id: Optional[str] = None
    pnl: Optional[float] = None
    reason_json: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_time": self.event_time.isoformat() if self.event_time else None,
            "selector_name": self.selector_name,
            "strategy_key": self.strategy_key,
            "event_type": self.event_type,
            "trade_id": self.trade_id,
            "position_id": self.position_id,
            "pnl": self.pnl,
            "reason_json": self.reason_json,
        }


@dataclass
class StrategySummary:
    """Summary statistics for a single strategy."""

    strategy_key: str
    opens: int = 0
    closes: int = 0
    total_pnl: float = 0.0
    avg_pnl: float = 0.0
    win_rate: float = 0.0
    avg_hold_time_hours: float = 0.0
    decisions: int = 0
    skips: int = 0
    rolls: int = 0
    take_profits: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BacktestEventEmitter:
    """
    Collects events during a backtest run.

    NOTE: This emitter is NOT thread-safe. Use only from a single thread/task
    or add external synchronization if needed in concurrent contexts.
    Events are accumulated in memory and can be persisted at the end of the run.
    """

    def __init__(self, run_id: int, selector_name: str = "generic_covered_call"):
        self.run_id = run_id
        self.selector_name = selector_name
        self._events: List[BacktestEventRecord] = []
        self._position_open_times: Dict[str, datetime] = {}

    def emit(
        self,
        event_time: datetime,
        strategy_key: str,
        event_type: str | EventType,
        trade_id: Optional[str] = None,
        position_id: Optional[str] = None,
        pnl: Optional[float] = None,
        reason: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Emit a strategy event."""
        if isinstance(event_type, EventType):
            event_type = event_type.value

        event = BacktestEventRecord(
            event_time=event_time,
            selector_name=self.selector_name,
            strategy_key=strategy_key,
            event_type=event_type,
            trade_id=trade_id,
            position_id=position_id,
            pnl=pnl,
            reason_json=reason,
        )
        self._events.append(event)

        if event_type == EventType.OPEN.value and position_id:
            self._position_open_times[position_id] = event_time

    def emit_decision(
        self,
        event_time: datetime,
        strategy_key: str,
        reason: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Emit a strategy decision event."""
        self.emit(event_time, strategy_key, EventType.DECISION, reason=reason)

    def emit_open(
        self,
        event_time: datetime,
        strategy_key: str,
        trade_id: str,
        position_id: Optional[str] = None,
        reason: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Emit a position open event."""
        self.emit(
            event_time,
            strategy_key,
            EventType.OPEN,
            trade_id=trade_id,
            position_id=position_id,
            reason=reason,
        )

    def emit_close(
        self,
        event_time: datetime,
        strategy_key: str,
        trade_id: str,
        pnl: float,
        close_type: str = "CLOSE",
        position_id: Optional[str] = None,
        reason: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Emit a position close event.

        close_type can be CLOSE, TAKE_PROFIT, STOP_LOSS, ROLL, or EXPIRY.
        """
        event_type = close_type.upper()
        if event_type not in [e.value for e in EventType]:
            event_type = EventType.CLOSE.value

        self.emit(
            event_time,
            strategy_key,
            event_type,
            trade_id=trade_id,
            position_id=position_id,
            pnl=pnl,
            reason=reason,
        )

    def emit_skip(
        self,
        event_time: datetime,
        strategy_key: str,
        reason: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Emit a skip event (no trade taken)."""
        self.emit(event_time, strategy_key, EventType.SKIP, reason=reason)

    def get_events(self) -> List[BacktestEventRecord]:
        """Get all recorded events."""
        return self._events.copy()

    def compute_strategy_summary(self) -> List[StrategySummary]:
        """
        Compute summary statistics grouped by strategy_key.

        Returns a list of StrategySummary objects with aggregated metrics.
        """
        from collections import defaultdict

        strategy_stats: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "opens": 0,
                "closes": 0,
                "total_pnl": 0.0,
                "wins": 0,
                "decisions": 0,
                "skips": 0,
                "rolls": 0,
                "take_profits": 0,
                "hold_times": [],
            }
        )

        for event in self._events:
            key = event.strategy_key
            stats = strategy_stats[key]

            if event.event_type == EventType.DECISION.value:
                stats["decisions"] += 1
            elif event.event_type == EventType.OPEN.value:
                stats["opens"] += 1
            elif event.event_type == EventType.SKIP.value:
                stats["skips"] += 1
            elif event.event_type == EventType.ROLL.value:
                stats["rolls"] += 1
            elif event.event_type == EventType.TAKE_PROFIT.value:
                stats["take_profits"] += 1
                stats["closes"] += 1
                if event.pnl is not None:
                    stats["total_pnl"] += event.pnl
                    if event.pnl > 0:
                        stats["wins"] += 1
                if event.position_id and event.position_id in self._position_open_times:
                    open_time = self._position_open_times[event.position_id]
                    hold_hours = (event.event_time - open_time).total_seconds() / 3600
                    stats["hold_times"].append(hold_hours)
            elif event.event_type in (
                EventType.CLOSE.value,
                EventType.STOP_LOSS.value,
                EventType.EXPIRY.value,
            ):
                stats["closes"] += 1
                if event.pnl is not None:
                    stats["total_pnl"] += event.pnl
                    if event.pnl > 0:
                        stats["wins"] += 1

                if event.position_id and event.position_id in self._position_open_times:
                    open_time = self._position_open_times[event.position_id]
                    hold_hours = (event.event_time - open_time).total_seconds() / 3600
                    stats["hold_times"].append(hold_hours)

        summaries = []
        for key, stats in strategy_stats.items():
            closes = stats["closes"]
            summaries.append(
                StrategySummary(
                    strategy_key=key,
                    opens=stats["opens"],
                    closes=closes,
                    total_pnl=stats["total_pnl"],
                    avg_pnl=stats["total_pnl"] / closes if closes > 0 else 0.0,
                    win_rate=stats["wins"] / closes if closes > 0 else 0.0,
                    avg_hold_time_hours=(
                        sum(stats["hold_times"]) / len(stats["hold_times"])
                        if stats["hold_times"]
                        else 0.0
                    ),
                    decisions=stats["decisions"],
                    skips=stats["skips"],
                    rolls=stats["rolls"],
                    take_profits=stats["take_profits"],
                )
            )

        return sorted(summaries, key=lambda s: s.total_pnl, reverse=True)

    def persist_events(self, session) -> int:
        """
        Persist all events to the database.

        Args:
            session: SQLAlchemy session

        Returns:
            Number of events persisted
        """
        from src.db.models_backtest import BacktestEvent

        count = 0
        for event in self._events:
            db_event = BacktestEvent(
                run_id=self.run_id,
                event_time=event.event_time,
                selector_name=event.selector_name,
                strategy_key=event.strategy_key,
                event_type=event.event_type,
                trade_id=event.trade_id,
                position_id=event.position_id,
                pnl=event.pnl,
                reason_json=event.reason_json,
            )
            session.add(db_event)
            count += 1

        return count


GREG_EVENT_KEYS = {
    "vrp_harvest_straddle": "greg.vrp_harvest.straddle",
    "vrp_harvest_strangle": "greg.vrp_harvest.strangle",
    "vrp_harvest_short_call": "greg.vrp_harvest.short_call",
    "vrp_harvest_short_put": "greg.vrp_harvest.short_put",
    "calendar_spread": "greg.calendar_spread",
    "iron_fly": "greg.iron_fly",
    "take_profit": "greg.take_profit",
    "roll": "greg.roll",
    "skip_no_edge": "greg.skip.no_edge",
    "skip_safety": "greg.skip.safety",
    "skip_no_signal": "greg.skip.no_signal",
    "expiry": "greg.expiry",
}


def get_greg_event_key(strategy_name: str) -> str:
    """
    Get a stable event key for GregBot events.

    Args:
        strategy_name: The GregBot strategy name (e.g., "STRATEGY_A_STRADDLE")

    Returns:
        Stable event key like "greg.vrp_harvest.straddle"
    """
    name_lower = strategy_name.lower()

    if "straddle" in name_lower:
        base = "greg.vrp_harvest.straddle"
    elif "strangle" in name_lower:
        base = "greg.vrp_harvest.strangle"
    elif "short_call" in name_lower:
        base = "greg.vrp_harvest.short_call"
    elif "short_put" in name_lower:
        base = "greg.vrp_harvest.short_put"
    elif "calendar" in name_lower:
        base = "greg.calendar_spread"
    elif "iron_fly" in name_lower or "iron_condor" in name_lower:
        base = "greg.iron_fly"
    elif "no_trade" in name_lower or "skip" in name_lower:
        base = "greg.skip.no_signal"
    else:
        base = f"greg.{name_lower.replace(' ', '_')}"

    return base
