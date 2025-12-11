"""
Greg Selector Scenario Tests - ENTRY_ENGINE v8.0 Positive Cases.

These tests validate that the selector makes the right choice in canonical
market environments that Greg Mandolini's strategy targets.
"""
import pytest
from typing import Dict, Any

from src.strategies.greg_selector import (
    GregSelectorSensors,
    evaluate_greg_selector,
    load_greg_spec,
    get_calibration_spec,
)


def base_metrics(**overrides: float) -> GregSelectorSensors:
    """
    Create a baseline sensor dict with neutral defaults.
    Override specific keys for each scenario.
    """
    defaults = {
        "vrp_30d": 12.0,
        "vrp_7d": 8.0,
        "front_rv_iv_ratio": 0.7,
        "chop_factor_7d": 0.5,
        "iv_rank_6m": 0.40,
        "term_structure_spread": 2.0,
        "skew_25d": 0.0,
        "adx_14d": 15.0,
        "rsi_14d": 50.0,
        "price_vs_ma200": 500.0,
        "predicted_funding_rate": 0.0,
    }
    defaults.update(overrides)
    return GregSelectorSensors(**defaults)


class TestHighVRPCalmNeutralSkew:
    """
    High VRP + calm market + neutral skew = sell neutral short-vol.
    This is Greg's bread-and-butter environment.
    """

    def test_premium_calm_neutral_selects_straddle(self):
        """Very high VRP, very calm, neutral skew - should select Straddle."""
        sensors = base_metrics(
            vrp_30d=28.0,
            adx_14d=10.0,
            skew_25d=0.0,
            iv_rank_6m=0.35,
            chop_factor_7d=0.5,
        )
        decision = evaluate_greg_selector(sensors)
        assert decision.selected_strategy == "STRATEGY_A_STRADDLE", (
            f"High VRP, calm, neutral skew should select Straddle. "
            f"Got: {decision.selected_strategy}"
        )

    def test_straddle_conditions_met_exactly(self):
        """Conditions at exact thresholds for straddle."""
        cal = get_calibration_spec()
        
        sensors = base_metrics(
            vrp_30d=cal["straddle_vrp_min"] + 0.1,
            adx_14d=cal["straddle_adx_max"] - 0.1,
            chop_factor_7d=cal["straddle_chop_max"] - 0.01,
            skew_25d=0.0,
        )
        decision = evaluate_greg_selector(sensors)
        assert decision.selected_strategy == "STRATEGY_A_STRADDLE", (
            f"Conditions at straddle thresholds should select Straddle. "
            f"Got: {decision.selected_strategy}"
        )


class TestModerateVRPDriftingMarket:
    """
    Moderate VRP + mild trend + neutral skew = Strangle preferred.
    Strangle gives more room for market drift than straddle.
    """

    def test_moderate_vrp_mild_trend_selects_strangle(self):
        """Moderate VRP, higher ADX (but under threshold), neutral skew - Strangle."""
        sensors = base_metrics(
            vrp_30d=12.0,
            adx_14d=22.0,
            skew_25d=0.0,
            iv_rank_6m=0.35,
            chop_factor_7d=0.7,
        )
        decision = evaluate_greg_selector(sensors)
        assert decision.selected_strategy == "STRATEGY_A_STRANGLE", (
            f"Moderate VRP, mild drift should select Strangle. "
            f"Got: {decision.selected_strategy}"
        )

    def test_strangle_selected_when_vrp_below_straddle_threshold(self):
        """VRP is good but below straddle min - should fall through to strangle."""
        cal = get_calibration_spec()
        
        sensors = base_metrics(
            vrp_30d=cal["straddle_vrp_min"] - 1.0,
            adx_14d=15.0,
            chop_factor_7d=0.5,
            skew_25d=0.0,
        )
        decision = evaluate_greg_selector(sensors)
        assert decision.selected_strategy == "STRATEGY_A_STRANGLE", (
            f"VRP below straddle threshold should select Strangle. "
            f"Got: {decision.selected_strategy}"
        )


class TestCalendarEnvironment:
    """
    Steep term structure + rich front IV vs RV = Calendar spread.
    """

    def test_calendar_selected_with_term_structure(self):
        """Term spread steep, front IV > RV, VRP positive - Calendar."""
        sensors = base_metrics(
            vrp_30d=15.0,
            vrp_7d=8.0,
            term_structure_spread=6.0,
            front_rv_iv_ratio=0.6,
            adx_14d=15.0,
            skew_25d=0.0,
        )
        decision = evaluate_greg_selector(sensors)
        assert decision.selected_strategy == "STRATEGY_B_CALENDAR", (
            f"Steep term structure, favorable RV/IV should select Calendar. "
            f"Got: {decision.selected_strategy}"
        )


class TestBullishAccumulation:
    """
    Above MA200 + put skew (fear) + positive VRP = Short Put or Bull Put Spread.
    This is the accumulation play where Greg gets paid to buy.
    """

    def test_short_put_bullish_fear_skew(self):
        """Above MA200, put skew, positive VRP - Short Put."""
        sensors = base_metrics(
            vrp_30d=10.0,
            price_vs_ma200=500.0,
            skew_25d=+6.0,
            rsi_14d=45.0,
            adx_14d=18.0,
            chop_factor_7d=0.5,
        )
        decision = evaluate_greg_selector(sensors)
        assert decision.selected_strategy == "STRATEGY_C_SHORT_PUT", (
            f"Bullish trend with fear skew should select Short Put. "
            f"Got: {decision.selected_strategy}"
        )

    def test_bull_put_spread_oversold_fear(self):
        """Oversold (low RSI) + fear skew + positive VRP - Bull Put Spread.
        
        Note: Short Put (step 6) requires price > MA200. By setting price_vs_ma200
        to a negative value, we force the selector to skip Short Put and match
        Bull Put Spread at step 7 instead.
        """
        sensors = base_metrics(
            vrp_30d=10.0,
            skew_25d=+6.0,
            rsi_14d=25.0,
            adx_14d=20.0,
            chop_factor_7d=0.6,
            price_vs_ma200=-500.0,
        )
        decision = evaluate_greg_selector(sensors)
        assert decision.selected_strategy == "STRATEGY_F_BULL_PUT_SPREAD", (
            f"Oversold + fear skew (price below MA200) should select Bull Put Spread. "
            f"Got: {decision.selected_strategy}"
        )


class TestBearishOverbought:
    """
    High RSI (overbought) + negative skew (FOMO) + positive VRP = Bear Call Spread.
    """

    def test_bear_call_spread_overbought_fomo(self):
        """High RSI, negative skew (call skew), positive VRP - Bear Call Spread."""
        sensors = base_metrics(
            vrp_30d=10.0,
            price_vs_ma200=500.0,
            skew_25d=-6.0,
            rsi_14d=75.0,
            adx_14d=20.0,
            chop_factor_7d=0.6,
        )
        decision = evaluate_greg_selector(sensors)
        assert decision.selected_strategy == "STRATEGY_F_BEAR_CALL_SPREAD", (
            f"Overbought + FOMO skew should select Bear Call Spread. "
            f"Got: {decision.selected_strategy}"
        )


class TestIronButterflyExtremeIV:
    """
    Extreme IV rank + positive VRP = Iron Butterfly (defined risk).
    When IV is extremely high, use wings for protection.
    """

    def test_iron_fly_extreme_iv(self):
        """Very high IV rank, positive VRP - Iron Butterfly."""
        sensors = base_metrics(
            vrp_30d=15.0,
            iv_rank_6m=0.85,
            adx_14d=20.0,
            chop_factor_7d=0.5,
            skew_25d=0.0,
        )
        decision = evaluate_greg_selector(sensors)
        assert decision.selected_strategy == "STRATEGY_D_IRON_BUTTERFLY", (
            f"Extreme IV rank should select Iron Butterfly. "
            f"Got: {decision.selected_strategy}"
        )


class TestSafetyFilters:
    """
    Safety filters should block trades when market is too trending or choppy.
    """
    
    def test_adx_safety_filter(self):
        """ADX > 35 should trigger NO_TRADE."""
        sensors = base_metrics(
            adx_14d=40.0,
            vrp_30d=20.0,
            chop_factor_7d=0.5,
        )
        decision = evaluate_greg_selector(sensors)
        assert decision.selected_strategy == "NO_TRADE"
        assert decision.step_name == "SAFETY_FILTER"
    
    def test_chop_safety_filter(self):
        """Chop > 0.85 should trigger NO_TRADE."""
        sensors = base_metrics(
            chop_factor_7d=0.9,
            vrp_30d=20.0,
            adx_14d=15.0,
        )
        decision = evaluate_greg_selector(sensors)
        assert decision.selected_strategy == "NO_TRADE"
        assert decision.step_name == "SAFETY_FILTER"


class TestDefaultFallback:
    """
    When no conditions match, selector should return NO_TRADE.
    """

    def test_no_trade_when_nothing_matches(self):
        """All metrics at mediocre levels - should be NO_TRADE."""
        sensors = base_metrics(
            vrp_30d=3.0,
            vrp_7d=2.0,
            adx_14d=25.0,
            chop_factor_7d=0.6,
            skew_25d=2.0,
            iv_rank_6m=0.30,
            term_structure_spread=1.0,
            front_rv_iv_ratio=0.9,
            rsi_14d=50.0,
        )
        decision = evaluate_greg_selector(sensors)
        assert decision.selected_strategy == "NO_TRADE", (
            f"Mediocre metrics should result in NO_TRADE. "
            f"Got: {decision.selected_strategy}"
        )


class TestDecisionTreeOrder:
    """
    Verify the decision waterfall order is correct.
    Earlier rules should take precedence.
    """

    def test_safety_first_in_waterfall(self):
        """Safety filter must be checked before all other strategies."""
        sensors = base_metrics(
            adx_14d=40.0,
            term_structure_spread=8.0,
            front_rv_iv_ratio=0.5,
            vrp_7d=20.0,
        )
        decision = evaluate_greg_selector(sensors)
        assert decision.selected_strategy == "NO_TRADE", (
            "Safety filter should trigger before Calendar. "
            f"Got: {decision.selected_strategy}"
        )
        assert decision.rule_index == 0, (
            f"Safety filter should be rule 0. Got rule_index: {decision.rule_index}"
        )

    def test_calendar_before_straddle(self):
        """Calendar should be checked before Straddle in waterfall."""
        sensors = base_metrics(
            vrp_30d=20.0,
            adx_14d=15.0,
            chop_factor_7d=0.5,
            skew_25d=0.0,
            term_structure_spread=6.0,
            front_rv_iv_ratio=0.6,
            vrp_7d=6.0,
        )
        decision = evaluate_greg_selector(sensors)
        assert decision.selected_strategy == "STRATEGY_B_CALENDAR", (
            "Calendar should win over Straddle when term structure is favorable. "
            f"Got: {decision.selected_strategy}"
        )

    def test_iron_fly_before_straddle(self):
        """Iron Butterfly (extreme IV) should be checked before Straddle."""
        sensors = base_metrics(
            vrp_30d=20.0,
            iv_rank_6m=0.85,
            adx_14d=15.0,
            chop_factor_7d=0.5,
            skew_25d=0.0,
        )
        decision = evaluate_greg_selector(sensors)
        assert decision.selected_strategy == "STRATEGY_D_IRON_BUTTERFLY", (
            "Iron Butterfly should win when IV rank is extreme. "
            f"Got: {decision.selected_strategy}"
        )


class TestV8DirectionalVRPFloor:
    """
    v8.0 introduced min_vrp_directional (2.0) for directional spreads.
    """

    def test_bull_put_requires_vrp_above_directional_floor(self):
        """Bull Put Spread should require VRP > min_vrp_directional."""
        sensors = base_metrics(
            vrp_30d=1.5,
            skew_25d=6.0,
            rsi_14d=25.0,
            adx_14d=20.0,
            chop_factor_7d=0.5,
        )
        decision = evaluate_greg_selector(sensors)
        assert decision.selected_strategy != "STRATEGY_F_BULL_PUT_SPREAD", (
            "Bull Put Spread should not trigger with VRP below directional floor. "
            f"Got: {decision.selected_strategy}"
        )

    def test_bear_call_requires_vrp_above_directional_floor(self):
        """Bear Call Spread should require VRP > min_vrp_directional."""
        sensors = base_metrics(
            vrp_30d=1.5,
            skew_25d=-6.0,
            rsi_14d=75.0,
            adx_14d=20.0,
            chop_factor_7d=0.5,
        )
        decision = evaluate_greg_selector(sensors)
        assert decision.selected_strategy != "STRATEGY_F_BEAR_CALL_SPREAD", (
            "Bear Call Spread should not trigger with VRP below directional floor. "
            f"Got: {decision.selected_strategy}"
        )

    def test_short_put_requires_vrp_above_directional_floor(self):
        """Short Put should require VRP > min_vrp_directional."""
        sensors = base_metrics(
            vrp_30d=1.5,
            skew_25d=6.0,
            price_vs_ma200=500.0,
            adx_14d=20.0,
            chop_factor_7d=0.5,
        )
        decision = evaluate_greg_selector(sensors)
        assert decision.selected_strategy != "STRATEGY_C_SHORT_PUT", (
            "Short Put should not trigger with VRP below directional floor. "
            f"Got: {decision.selected_strategy}"
        )
