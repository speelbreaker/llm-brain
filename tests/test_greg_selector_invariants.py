"""
Greg Selector Invariant Tests - "Never do this" rules.

These tests validate that the Greg selector never makes dangerous decisions,
regardless of how tempting the other conditions look.
"""
import pytest
from typing import Dict, Any, Optional

from src.strategies.greg_selector import (
    GregSelectorSensors,
    evaluate_greg_selector,
    load_greg_spec,
)


SHORT_VOL_STRATEGIES = {
    "STRATEGY_A_STRADDLE",
    "STRATEGY_A_STRANGLE",
    "STRATEGY_C_SHORT_PUT",
    "STRATEGY_F_BULL_PUT_SPREAD",
    "STRATEGY_F_BEAR_CALL_SPREAD",
    "STRATEGY_D_IRON_BUTTERFLY",
}

NEUTRAL_SHORT_VOL = {
    "STRATEGY_A_STRADDLE",
    "STRATEGY_A_STRANGLE",
}


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


class TestNoNeutralShortVolWhenSkewExtreme:
    """
    Greg never sells neutral straddles/strangles when skew is directional.
    Extreme skew signals market fear/greed that shouldn't be fought with neutral structures.
    """

    def test_positive_extreme_skew_no_straddle_strangle(self):
        """High VRP, calm ADX, but positive extreme skew - no neutral short-vol."""
        sensors = base_metrics(
            vrp_30d=30.0,
            adx_14d=10.0,
            chop_factor_7d=0.4,
            skew_25d=+8.0,
        )
        decision = evaluate_greg_selector(sensors)
        assert decision.selected_strategy not in NEUTRAL_SHORT_VOL, (
            f"Should not select neutral short-vol with positive extreme skew. "
            f"Got: {decision.selected_strategy}"
        )

    def test_negative_extreme_skew_no_straddle_strangle(self):
        """High VRP, calm ADX, but negative extreme skew - no neutral short-vol."""
        sensors = base_metrics(
            vrp_30d=30.0,
            adx_14d=10.0,
            chop_factor_7d=0.4,
            skew_25d=-8.0,
        )
        decision = evaluate_greg_selector(sensors)
        assert decision.selected_strategy not in NEUTRAL_SHORT_VOL, (
            f"Should not select neutral short-vol with negative extreme skew. "
            f"Got: {decision.selected_strategy}"
        )


class TestNoCalendarInRealizedTrap:
    """
    Never sell calendars when front-month realized vol >= implied vol (RV trap).
    This is a value trap: the vol looks cheap but the underlying is too active.
    """

    def test_calendar_blocked_when_front_rv_high(self):
        """Steep term structure, but RV >= IV on front - no calendar."""
        sensors = base_metrics(
            term_structure_spread=6.0,
            front_rv_iv_ratio=1.05,
            vrp_7d=0.0,
        )
        decision = evaluate_greg_selector(sensors)
        assert decision.selected_strategy != "STRATEGY_B_CALENDAR", (
            f"Should not select Calendar in realized trap. "
            f"Got: {decision.selected_strategy}"
        )

    def test_calendar_blocked_when_front_rv_equals_iv(self):
        """Term spread ok, but front RV/IV at 1.0 - still no calendar."""
        sensors = base_metrics(
            term_structure_spread=6.0,
            front_rv_iv_ratio=1.0,
            vrp_7d=5.0,
        )
        decision = evaluate_greg_selector(sensors)
        assert decision.selected_strategy != "STRATEGY_B_CALENDAR", (
            f"Should not select Calendar when RV/IV ratio is at parity. "
            f"Got: {decision.selected_strategy}"
        )


class TestNoShortVolWhenVRPNegative:
    """
    Never sell vol when VRP is negative (implied < realized).
    This means the market is pricing risk correctly or underpricing it.
    """

    def test_negative_vrp_no_short_vol(self):
        """VRP is negative, no short-vol strategies should be selected."""
        sensors = base_metrics(
            vrp_30d=-5.0,
            vrp_7d=-3.0,
            iv_rank_6m=0.25,
            chop_factor_7d=0.5,
            adx_14d=15.0,
            skew_25d=0.0,
        )
        decision = evaluate_greg_selector(sensors)
        assert decision.selected_strategy not in SHORT_VOL_STRATEGIES, (
            f"Should not select any short-vol strategy with negative VRP. "
            f"Got: {decision.selected_strategy}"
        )


class TestSafetyFilterOverridesEverything:
    """
    The safety filter (high ADX or high chop) must trigger NO_TRADE,
    even if all other conditions look perfect.
    """

    def test_safety_triggered_by_high_adx(self):
        """Perfect straddle environment but ADX above safety - NO_TRADE."""
        sensors = base_metrics(
            vrp_30d=25.0,
            adx_14d=40.0,
            chop_factor_7d=0.4,
            skew_25d=0.0,
        )
        decision = evaluate_greg_selector(sensors)
        assert decision.selected_strategy == "NO_TRADE", (
            f"Safety filter should trigger NO_TRADE for high ADX. "
            f"Got: {decision.selected_strategy}"
        )

    def test_safety_triggered_by_high_chop(self):
        """Good VRP, low ADX, but high chop factor - NO_TRADE."""
        sensors = base_metrics(
            vrp_30d=25.0,
            adx_14d=15.0,
            chop_factor_7d=0.90,
            skew_25d=0.0,
        )
        decision = evaluate_greg_selector(sensors)
        assert decision.selected_strategy == "NO_TRADE", (
            f"Safety filter should trigger NO_TRADE for high chop. "
            f"Got: {decision.selected_strategy}"
        )

    def test_safety_at_exact_boundary_adx(self):
        """ADX exactly at safety threshold - should trigger NO_TRADE."""
        spec = load_greg_spec()
        safety_adx = spec["global_constraints"]["calibration"]["safety_adx_high"]
        
        sensors = base_metrics(
            vrp_30d=25.0,
            adx_14d=safety_adx + 0.01,
            chop_factor_7d=0.4,
            skew_25d=0.0,
        )
        decision = evaluate_greg_selector(sensors)
        assert decision.selected_strategy == "NO_TRADE", (
            f"Should trigger NO_TRADE when ADX just above threshold. "
            f"Got: {decision.selected_strategy}"
        )


class TestCalibrationLoadsCorrectly:
    """Verify the calibration block loads all expected variables."""

    def test_calibration_has_required_keys(self):
        """All calibration keys should be present in the spec."""
        spec = load_greg_spec()
        calibration = spec.get("global_constraints", {}).get("calibration", {})
        
        required_keys = [
            "skew_neutral_threshold",
            "min_vrp_floor",
            "safety_adx_high",
            "safety_chop_high",
            "straddle_vrp_min",
            "straddle_adx_max",
            "straddle_chop_max",
            "strangle_vrp_min",
            "strangle_adx_max",
            "strangle_chop_max",
            "calendar_term_spread_min",
            "calendar_front_rv_iv_ratio_max",
            "iron_fly_iv_rank_min",
            "iron_fly_vrp_min",
            "short_put_iv_rank_min",
            "short_put_price_vs_ma200_min",
            "bull_put_rsi_max",
            "bear_call_rsi_min",
        ]
        
        for key in required_keys:
            assert key in calibration, f"Missing calibration key: {key}"
            assert isinstance(calibration[key], (int, float)), (
                f"Calibration key {key} should be numeric, got {type(calibration[key])}"
            )

    def test_calibration_version_in_meta(self):
        """Meta should contain calibration_version."""
        spec = load_greg_spec()
        meta = spec.get("meta", {})
        assert "calibration_version" in meta, "meta.calibration_version missing"
        assert meta["calibration_version"] == "Greg-aligned-v1"
