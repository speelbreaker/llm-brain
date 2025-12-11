"""
Unit tests for the Delta Hedging Engine.
"""
import pytest
from datetime import datetime, timezone

from src.hedging.hedge_engine import (
    HedgeEngine,
    HedgeRules,
    HedgeOrder,
    HedgeResult,
    GregPosition,
    load_greg_hedge_rules,
)


class TestHedgeRules:
    """Tests for HedgeRules dataclass."""
    
    def test_from_dict_dynamic_delta(self):
        """Test parsing DYNAMIC_DELTA mode."""
        data = {
            "mode": "DYNAMIC_DELTA",
            "delta_abs_threshold": 0.15,
            "target_delta": 0.0,
            "check_frequency": "60s",
        }
        rules = HedgeRules.from_dict(data)
        assert rules.mode == "DYNAMIC_DELTA"
        assert rules.delta_abs_threshold == 0.15
        assert rules.target_delta == 0.0
    
    def test_from_dict_none_mode(self):
        """Test parsing NONE mode."""
        data = {"mode": "NONE"}
        rules = HedgeRules.from_dict(data)
        assert rules.mode == "NONE"
    
    def test_from_dict_none_input(self):
        """Test handling None input."""
        rules = HedgeRules.from_dict(None)
        assert rules.mode == "NONE"
        assert rules.delta_abs_threshold == 999.0
    
    def test_from_dict_legacy_mode(self):
        """Test handling legacy mode names."""
        data = {"mode": "delta_hedge_perp"}
        rules = HedgeRules.from_dict(data)
        assert rules.mode == "DYNAMIC_DELTA"


class TestGregPosition:
    """Tests for GregPosition dataclass."""
    
    def test_is_hedgeable_straddle(self):
        """Test straddle is hedgeable."""
        pos = GregPosition(
            position_id="test-1",
            strategy_type="STRATEGY_A_STRADDLE",
            underlying="BTC",
            option_legs=[],
        )
        assert pos.is_hedgeable is True
    
    def test_is_hedgeable_strangle(self):
        """Test strangle is hedgeable."""
        pos = GregPosition(
            position_id="test-2",
            strategy_type="STRATEGY_A_STRANGLE",
            underlying="ETH",
            option_legs=[],
        )
        assert pos.is_hedgeable is True
    
    def test_is_hedgeable_calendar(self):
        """Test calendar is hedgeable."""
        pos = GregPosition(
            position_id="test-3",
            strategy_type="STRATEGY_B_CALENDAR",
            underlying="BTC",
            option_legs=[],
        )
        assert pos.is_hedgeable is True
    
    def test_is_hedgeable_iron_fly(self):
        """Test iron fly is hedgeable."""
        pos = GregPosition(
            position_id="test-4",
            strategy_type="STRATEGY_D_IRON_BUTTERFLY",
            underlying="ETH",
            option_legs=[],
        )
        assert pos.is_hedgeable is True
    
    def test_not_hedgeable_short_put(self):
        """Test short put is NOT hedgeable."""
        pos = GregPosition(
            position_id="test-5",
            strategy_type="STRATEGY_C_SHORT_PUT",
            underlying="BTC",
            option_legs=[],
        )
        assert pos.is_hedgeable is False
    
    def test_not_hedgeable_credit_spread(self):
        """Test credit spreads are NOT hedgeable."""
        pos = GregPosition(
            position_id="test-6",
            strategy_type="STRATEGY_F_BULL_PUT_SPREAD",
            underlying="BTC",
            option_legs=[],
        )
        assert pos.is_hedgeable is False


class TestHedgeEngine:
    """Tests for HedgeEngine class."""
    
    @pytest.fixture
    def engine(self):
        """Create a test engine."""
        return HedgeEngine(dry_run=True)
    
    def test_compute_net_delta_options_only(self, engine):
        """Test net delta with options only."""
        delta = engine.compute_net_delta([0.5, -0.3, 0.1], perp_delta=0.0)
        assert delta == pytest.approx(0.3, abs=0.001)
    
    def test_compute_net_delta_with_perp(self, engine):
        """Test net delta with perp hedge."""
        delta = engine.compute_net_delta([0.5, -0.3], perp_delta=-0.2)
        assert delta == pytest.approx(0.0, abs=0.001)
    
    def test_compute_net_delta_for_position(self, engine):
        """Test net delta computation from GregPosition."""
        pos = GregPosition(
            position_id="test-1",
            strategy_type="STRATEGY_A_STRADDLE",
            underlying="BTC",
            option_legs=[
                {"delta": -0.50, "size": 1.0},
                {"delta": 0.50, "size": 1.0},
            ],
            hedge_perp_size=0.0,
        )
        delta = engine.compute_net_delta_for_position(pos)
        assert delta == pytest.approx(0.0, abs=0.001)
    
    def test_compute_net_delta_for_position_with_perp(self, engine):
        """Test net delta with existing perp hedge."""
        pos = GregPosition(
            position_id="test-2",
            strategy_type="STRATEGY_A_STRADDLE",
            underlying="BTC",
            option_legs=[
                {"delta": -0.50, "size": 1.0},
                {"delta": 0.65, "size": 1.0},
            ],
            hedge_perp_size=-0.15,
        )
        delta = engine.compute_net_delta_for_position(pos)
        assert delta == pytest.approx(0.0, abs=0.001)
    
    def test_compute_net_delta_for_position_weighted_by_size(self, engine):
        """Test net delta is weighted by contract size."""
        pos = GregPosition(
            position_id="test-weighted",
            strategy_type="STRATEGY_A_STRADDLE",
            underlying="BTC",
            option_legs=[
                {"delta": -0.50, "size": 2.0},
                {"delta": 0.50, "size": 1.0},
            ],
            hedge_perp_size=0.0,
        )
        delta = engine.compute_net_delta_for_position(pos)
        assert delta == pytest.approx(-0.50, abs=0.001)
    
    def test_compute_net_delta_for_position_default_size(self, engine):
        """Test net delta defaults to size=1 if not specified."""
        pos = GregPosition(
            position_id="test-default-size",
            strategy_type="STRATEGY_A_STRADDLE",
            underlying="BTC",
            option_legs=[
                {"delta": -0.50},
                {"delta": 0.50},
            ],
            hedge_perp_size=0.0,
        )
        delta = engine.compute_net_delta_for_position(pos)
        assert delta == pytest.approx(0.0, abs=0.001)
    
    def test_build_hedge_order_no_hedge_needed(self, engine):
        """Test no hedge when delta within threshold."""
        pos = GregPosition(
            position_id="test-1",
            strategy_type="STRATEGY_A_STRADDLE",
            underlying="BTC",
            option_legs=[
                {"delta": -0.50},
                {"delta": 0.55},
            ],
            hedge_perp_size=0.0,
        )
        rules = HedgeRules(
            mode="DYNAMIC_DELTA",
            delta_abs_threshold=0.15,
            target_delta=0.0,
        )
        order = engine.build_hedge_order(pos, rules)
        assert order is None
    
    def test_build_hedge_order_hedge_needed(self, engine):
        """Test hedge order when delta exceeds threshold."""
        pos = GregPosition(
            position_id="test-2",
            strategy_type="STRATEGY_A_STRADDLE",
            underlying="BTC",
            option_legs=[
                {"delta": -0.35},
                {"delta": 0.65},
            ],
            hedge_perp_size=0.0,
        )
        rules = HedgeRules(
            mode="DYNAMIC_DELTA",
            delta_abs_threshold=0.15,
            target_delta=0.0,
        )
        order = engine.build_hedge_order(pos, rules)
        assert order is not None
        assert order.underlying == "BTC"
        assert order.instrument == "BTC-PERPETUAL"
        assert order.side == "sell"
        assert order.size == pytest.approx(0.30, abs=0.01)
        assert order.net_delta_before == pytest.approx(0.30, abs=0.01)
        assert order.net_delta_after == pytest.approx(0.0, abs=0.01)
    
    def test_build_hedge_order_buy_side(self, engine):
        """Test hedge order creates buy order for negative delta."""
        pos = GregPosition(
            position_id="test-3",
            strategy_type="STRATEGY_A_STRANGLE",
            underlying="ETH",
            option_legs=[
                {"delta": -0.25},
                {"delta": 0.05},
            ],
            hedge_perp_size=0.0,
        )
        rules = HedgeRules(
            mode="DYNAMIC_DELTA",
            delta_abs_threshold=0.15,
            target_delta=0.0,
        )
        order = engine.build_hedge_order(pos, rules)
        assert order is not None
        assert order.underlying == "ETH"
        assert order.instrument == "ETH-PERPETUAL"
        assert order.side == "buy"
        assert order.size == pytest.approx(0.20, abs=0.01)
    
    def test_build_hedge_order_none_mode(self, engine):
        """Test no hedge when mode is NONE."""
        pos = GregPosition(
            position_id="test-4",
            strategy_type="STRATEGY_C_SHORT_PUT",
            underlying="BTC",
            option_legs=[{"delta": -0.80}],
            hedge_perp_size=0.0,
        )
        rules = HedgeRules(
            mode="NONE",
            delta_abs_threshold=0.15,
        )
        order = engine.build_hedge_order(pos, rules)
        assert order is None
    
    def test_apply_hedge_dry_run(self, engine):
        """Test hedge application in dry run mode."""
        order = HedgeOrder(
            underlying="BTC",
            instrument="BTC-PERPETUAL",
            side="sell",
            size=0.30,
            net_delta_before=0.30,
            net_delta_after=0.0,
            strategy_position_id="test-1",
            strategy_type="STRATEGY_A_STRADDLE",
        )
        result = engine.apply_hedge(order)
        assert result.success is True
        assert result.dry_run is True
        assert result.executed is False
        assert result.order is order
    
    def test_apply_hedge_no_client(self):
        """Test hedge fails without client in live mode."""
        engine = HedgeEngine(dry_run=False, deribit_client=None)
        order = HedgeOrder(
            underlying="BTC",
            instrument="BTC-PERPETUAL",
            side="sell",
            size=0.30,
            net_delta_before=0.30,
            net_delta_after=0.0,
            strategy_position_id="test-1",
            strategy_type="STRATEGY_A_STRADDLE",
        )
        result = engine.apply_hedge(order)
        assert result.success is False
        assert result.error is not None
    
    def test_step_non_hedgeable_position(self, engine):
        """Test step returns None for non-hedgeable position."""
        pos = GregPosition(
            position_id="test-5",
            strategy_type="STRATEGY_C_SHORT_PUT",
            underlying="BTC",
            option_legs=[{"delta": -0.80}],
        )
        result = engine.step(pos)
        assert result is None
    
    def test_step_hedgeable_position(self, engine):
        """Test step processes hedgeable position."""
        pos = GregPosition(
            position_id="test-6",
            strategy_type="STRATEGY_A_STRADDLE",
            underlying="BTC",
            option_legs=[
                {"delta": -0.35},
                {"delta": 0.65},
            ],
            hedge_perp_size=0.0,
        )
        result = engine.step(pos)
        assert result is not None
        assert result.success is True
        assert result.dry_run is True
    
    def test_hedge_all_positions(self, engine):
        """Test hedging multiple positions."""
        positions = [
            GregPosition(
                position_id="test-1",
                strategy_type="STRATEGY_A_STRADDLE",
                underlying="BTC",
                option_legs=[{"delta": -0.50}, {"delta": 0.50}],
            ),
            GregPosition(
                position_id="test-2",
                strategy_type="STRATEGY_A_STRADDLE",
                underlying="BTC",
                option_legs=[{"delta": -0.35}, {"delta": 0.65}],
            ),
            GregPosition(
                position_id="test-3",
                strategy_type="STRATEGY_C_SHORT_PUT",
                underlying="BTC",
                option_legs=[{"delta": -0.80}],
            ),
        ]
        results = engine.hedge_all_positions(positions)
        assert len(results) == 1
        assert results[0].order.strategy_position_id == "test-2"
    
    def test_get_hedge_history(self, engine):
        """Test hedge history retrieval."""
        pos = GregPosition(
            position_id="test-7",
            strategy_type="STRATEGY_A_STRADDLE",
            underlying="BTC",
            option_legs=[{"delta": -0.35}, {"delta": 0.65}],
        )
        engine.step(pos)
        history = engine.get_hedge_history(limit=10)
        assert len(history) == 1
        assert history[0]["order"]["strategy_position_id"] == "test-7"
    
    def test_set_dry_run(self, engine):
        """Test dry run toggle."""
        assert engine.dry_run is True
        engine.set_dry_run(False)
        assert engine.dry_run is False
        engine.set_dry_run(True)
        assert engine.dry_run is True


class TestLoadGregHedgeRules:
    """Tests for loading Greg hedge rules from JSON."""
    
    def test_load_rules_file(self):
        """Test loading rules from file."""
        rules = load_greg_hedge_rules()
        assert "strategies" in rules
        assert "global_definitions" in rules
    
    def test_straddle_rules(self):
        """Test straddle hedge rules are correct."""
        rules = load_greg_hedge_rules()
        straddle = rules.get("strategies", {}).get("STRATEGY_A_STRADDLE", {})
        hedge = straddle.get("hedge", {})
        assert hedge.get("mode") == "DYNAMIC_DELTA"
        assert hedge.get("delta_abs_threshold") == 0.15
    
    def test_short_put_rules(self):
        """Test short put has no hedging."""
        rules = load_greg_hedge_rules()
        short_put = rules.get("strategies", {}).get("STRATEGY_C_SHORT_PUT", {})
        hedge = short_put.get("hedge", {})
        assert hedge.get("mode") == "NONE"
