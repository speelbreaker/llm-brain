"""Tests for backtest event emitter infrastructure."""

from datetime import datetime
from src.backtest.event_emitter import (
    BacktestEventEmitter,
    BacktestEventRecord,
    EventType,
    StrategySummary,
    get_greg_event_key,
)


class TestBacktestEventEmitter:
    """Tests for BacktestEventEmitter class."""

    def test_emitter_initialization(self):
        """Test emitter initializes with correct defaults."""
        emitter = BacktestEventEmitter(run_id=123, selector_name="gregbot")

        assert emitter.run_id == 123
        assert emitter.selector_name == "gregbot"
        assert len(emitter.get_events()) == 0

    def test_emit_basic_event(self):
        """Test emitting a basic event."""
        emitter = BacktestEventEmitter(run_id=1)
        now = datetime.now()

        emitter.emit(
            event_time=now,
            strategy_key="greg.vrp_harvest.straddle",
            event_type=EventType.OPEN,
            trade_id="trade_001",
        )

        events = emitter.get_events()
        assert len(events) == 1
        assert events[0].event_type == "OPEN"
        assert events[0].strategy_key == "greg.vrp_harvest.straddle"
        assert events[0].trade_id == "trade_001"

    def test_emit_decision_event(self):
        """Test emitting a decision event."""
        emitter = BacktestEventEmitter(run_id=1)
        now = datetime.now()

        emitter.emit_decision(
            event_time=now,
            strategy_key="greg.vrp_harvest.straddle",
            reason={"signal_strength": 0.85},
        )

        events = emitter.get_events()
        assert len(events) == 1
        assert events[0].event_type == "DECISION"
        assert events[0].reason_json == {"signal_strength": 0.85}

    def test_emit_open_event(self):
        """Test emitting an open event."""
        emitter = BacktestEventEmitter(run_id=1)
        now = datetime.now()

        emitter.emit_open(
            event_time=now,
            strategy_key="greg.vrp_harvest.straddle",
            trade_id="trade_001",
            position_id="pos_001",
        )

        events = emitter.get_events()
        assert len(events) == 1
        assert events[0].event_type == "OPEN"
        assert events[0].trade_id == "trade_001"
        assert events[0].position_id == "pos_001"

    def test_emit_close_event_with_pnl(self):
        """Test emitting a close event with PnL."""
        emitter = BacktestEventEmitter(run_id=1)
        now = datetime.now()

        emitter.emit_close(
            event_time=now,
            strategy_key="greg.vrp_harvest.straddle",
            trade_id="trade_001",
            pnl=150.50,
            close_type="TAKE_PROFIT",
            position_id="pos_001",
        )

        events = emitter.get_events()
        assert len(events) == 1
        assert events[0].event_type == "TAKE_PROFIT"
        assert events[0].pnl == 150.50

    def test_emit_skip_event(self):
        """Test emitting a skip event."""
        emitter = BacktestEventEmitter(run_id=1)
        now = datetime.now()

        emitter.emit_skip(
            event_time=now,
            strategy_key="greg.skip.no_edge",
            reason={"vrp": 0.02, "threshold": 0.05},
        )

        events = emitter.get_events()
        assert len(events) == 1
        assert events[0].event_type == "SKIP"


class TestStrategySummaryComputation:
    """Tests for strategy summary computation."""

    def test_compute_empty_summary(self):
        """Test summary computation with no events."""
        emitter = BacktestEventEmitter(run_id=1)
        summaries = emitter.compute_strategy_summary()

        assert len(summaries) == 0

    def test_compute_summary_with_trades(self):
        """Test summary computation with multiple trades."""
        emitter = BacktestEventEmitter(run_id=1)
        now = datetime.now()

        emitter.emit_open(now, "greg.straddle", "t1", "p1")
        emitter.emit_close(now, "greg.straddle", "t1", pnl=100.0, position_id="p1")
        emitter.emit_open(now, "greg.straddle", "t2", "p2")
        emitter.emit_close(now, "greg.straddle", "t2", pnl=-50.0, position_id="p2")
        emitter.emit_skip(now, "greg.straddle")

        summaries = emitter.compute_strategy_summary()

        assert len(summaries) == 1
        summary = summaries[0]
        assert summary.strategy_key == "greg.straddle"
        assert summary.opens == 2
        assert summary.closes == 2
        assert summary.total_pnl == 50.0
        assert summary.avg_pnl == 25.0
        assert summary.win_rate == 0.5
        assert summary.skips == 1

    def test_compute_summary_multiple_strategies(self):
        """Test summary groups by strategy_key."""
        emitter = BacktestEventEmitter(run_id=1)
        now = datetime.now()

        emitter.emit_open(now, "greg.straddle", "t1")
        emitter.emit_close(now, "greg.straddle", "t1", pnl=100.0)
        emitter.emit_open(now, "greg.strangle", "t2")
        emitter.emit_close(now, "greg.strangle", "t2", pnl=200.0)

        summaries = emitter.compute_strategy_summary()

        assert len(summaries) == 2
        straddle = next(s for s in summaries if s.strategy_key == "greg.straddle")
        strangle = next(s for s in summaries if s.strategy_key == "greg.strangle")

        assert straddle.total_pnl == 100.0
        assert strangle.total_pnl == 200.0


class TestGregEventKeys:
    """Tests for get_greg_event_key function."""

    def test_straddle_key(self):
        """Test straddle strategy returns correct key."""
        key = get_greg_event_key("STRATEGY_A_STRADDLE")
        assert key == "greg.vrp_harvest.straddle"

    def test_strangle_key(self):
        """Test strangle strategy returns correct key."""
        key = get_greg_event_key("VRP_STRANGLE")
        assert key == "greg.vrp_harvest.strangle"

    def test_short_call_key(self):
        """Test short call strategy returns correct key."""
        key = get_greg_event_key("covered_short_call")
        assert key == "greg.vrp_harvest.short_call"

    def test_calendar_key(self):
        """Test calendar spread returns correct key."""
        key = get_greg_event_key("calendar_spread_front")
        assert key == "greg.calendar_spread"

    def test_no_trade_key(self):
        """Test no trade signal returns skip key."""
        key = get_greg_event_key("NO_TRADE")
        assert key == "greg.skip.no_signal"

    def test_unknown_strategy_key(self):
        """Test unknown strategy returns formatted key."""
        key = get_greg_event_key("custom_strategy")
        assert key.startswith("greg.")


class TestBacktestEventRecord:
    """Tests for BacktestEventRecord dataclass."""

    def test_to_dict(self):
        """Test event record serialization."""
        now = datetime.now()
        record = BacktestEventRecord(
            event_time=now,
            selector_name="gregbot",
            strategy_key="greg.straddle",
            event_type="OPEN",
            trade_id="t1",
            position_id="p1",
            pnl=None,
            reason_json={"signal": True},
        )

        d = record.to_dict()

        assert d["selector_name"] == "gregbot"
        assert d["strategy_key"] == "greg.straddle"
        assert d["event_type"] == "OPEN"
        assert d["trade_id"] == "t1"
        assert d["reason_json"] == {"signal": True}


class TestStrategySummary:
    """Tests for StrategySummary dataclass."""

    def test_to_dict(self):
        """Test summary serialization."""
        summary = StrategySummary(
            strategy_key="greg.straddle",
            opens=5,
            closes=4,
            total_pnl=250.0,
            avg_pnl=62.5,
            win_rate=0.75,
            avg_hold_time_hours=24.5,
            decisions=10,
            skips=2,
            rolls=1,
            take_profits=3,
        )

        d = summary.to_dict()

        assert d["strategy_key"] == "greg.straddle"
        assert d["opens"] == 5
        assert d["closes"] == 4
        assert d["total_pnl"] == 250.0
        assert d["win_rate"] == 0.75
        assert d["take_profits"] == 3
