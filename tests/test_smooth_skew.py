"""
Tests for smooth parametric skew curve (quadratic in log-space).

Verifies that:
1. Output equals anchor ratios at anchor deltas (within tolerance)
2. Function is smooth/monotone-ish between anchors (no large jumps)
3. Falls back to linear interpolation for <3 anchors
"""
import pytest
import numpy as np

from src.synthetic.vol_surface import (
    VolSurfaceConfig,
    SkewTemplate,
    reset_vol_surface_config,
)


@pytest.fixture(autouse=True)
def reset_vol_surface():
    """Reset vol surface config before and after each test."""
    reset_vol_surface_config()
    yield
    reset_vol_surface_config()


class TestSmoothSkewAnchorAccuracy:
    """Tests that curve passes through anchor points."""

    def test_anchor_ratios_at_exact_deltas(self):
        """Curve should pass through anchor points (within small tolerance)."""
        skew = SkewTemplate(
            enabled=True,
            anchor_ratios={"0.15": 1.10, "0.25": 1.05, "0.35": 1.02},
            mode="put_heavy",
            scale=1.0,
        )
        config = VolSurfaceConfig(skew=skew)
        
        ratio_15 = config.get_skew_anchor_ratio(0.15)
        ratio_25 = config.get_skew_anchor_ratio(0.25)
        ratio_35 = config.get_skew_anchor_ratio(0.35)
        
        assert abs(ratio_15 - 1.10) < 0.05, f"At delta=0.15, expected ~1.10, got {ratio_15:.4f}"
        assert abs(ratio_25 - 1.05) < 0.05, f"At delta=0.25, expected ~1.05, got {ratio_25:.4f}"
        assert abs(ratio_35 - 1.02) < 0.05, f"At delta=0.35, expected ~1.02, got {ratio_35:.4f}"

    def test_more_anchors_still_accurate(self):
        """With more anchors, should still be reasonably accurate."""
        skew = SkewTemplate(
            enabled=True,
            anchor_ratios={"0.10": 1.15, "0.20": 1.08, "0.30": 1.03, "0.40": 1.00},
            mode="put_heavy",
            scale=1.0,
        )
        config = VolSurfaceConfig(skew=skew)
        
        for delta_str, expected in [("0.10", 1.15), ("0.20", 1.08), ("0.30", 1.03), ("0.40", 1.00)]:
            delta = float(delta_str)
            ratio = config.get_skew_anchor_ratio(delta)
            assert abs(ratio - expected) < 0.15, f"At delta={delta}, expected ~{expected}, got {ratio:.4f}"


class TestSmoothSkewMonotonicity:
    """Tests that curve is smooth and monotone-ish."""

    def test_no_large_jumps_between_anchors(self):
        """Values between anchors should not have large discontinuities."""
        skew = SkewTemplate(
            enabled=True,
            anchor_ratios={"0.15": 1.10, "0.25": 1.05, "0.35": 1.02},
            mode="put_heavy",
            scale=1.0,
        )
        config = VolSurfaceConfig(skew=skew)
        
        deltas = np.linspace(0.10, 0.40, 31)
        ratios = [config.get_skew_anchor_ratio(d) for d in deltas]
        
        for i in range(1, len(ratios)):
            diff = abs(ratios[i] - ratios[i-1])
            assert diff < 0.05, f"Large jump between delta={deltas[i-1]:.3f} and {deltas[i]:.3f}: {diff:.4f}"

    def test_generally_monotone_for_put_heavy_skew(self):
        """Put-heavy skew should generally decrease as delta increases."""
        skew = SkewTemplate(
            enabled=True,
            anchor_ratios={"0.15": 1.10, "0.25": 1.05, "0.35": 1.02},
            mode="put_heavy",
            scale=1.0,
        )
        config = VolSurfaceConfig(skew=skew)
        
        ratio_low = config.get_skew_anchor_ratio(0.15)
        ratio_high = config.get_skew_anchor_ratio(0.35)
        
        assert ratio_low > ratio_high, (
            f"Put-heavy skew should decrease with delta: "
            f"ratio(0.15)={ratio_low:.4f} should be > ratio(0.35)={ratio_high:.4f}"
        )

    def test_curve_is_smooth_second_derivative(self):
        """Curve should be smooth (bounded second derivative)."""
        skew = SkewTemplate(
            enabled=True,
            anchor_ratios={"0.15": 1.10, "0.25": 1.05, "0.35": 1.02},
            mode="put_heavy",
            scale=1.0,
        )
        config = VolSurfaceConfig(skew=skew)
        
        deltas = np.linspace(0.10, 0.40, 101)
        ratios = np.array([config.get_skew_anchor_ratio(d) for d in deltas])
        
        first_deriv = np.diff(ratios)
        second_deriv = np.diff(first_deriv)
        
        max_second = np.max(np.abs(second_deriv))
        assert max_second < 0.01, f"Second derivative too large: {max_second:.6f}"


class TestLinearFallback:
    """Tests that <3 anchors falls back to linear interpolation."""

    def test_two_anchors_uses_linear(self):
        """With only 2 anchors, should use linear interpolation."""
        skew = SkewTemplate(
            enabled=True,
            anchor_ratios={"0.15": 1.10, "0.35": 1.02},
            mode="put_heavy",
            scale=1.0,
        )
        config = VolSurfaceConfig(skew=skew)
        
        ratio_mid = config.get_skew_anchor_ratio(0.25)
        expected_mid = (1.10 + 1.02) / 2
        
        assert abs(ratio_mid - expected_mid) < 0.01, (
            f"With 2 anchors, midpoint should be ~{expected_mid:.4f}, got {ratio_mid:.4f}"
        )

    def test_one_anchor_returns_that_value(self):
        """With 1 anchor, should return that value everywhere."""
        skew = SkewTemplate(
            enabled=True,
            anchor_ratios={"0.25": 1.05},
            mode="put_heavy",
            scale=1.0,
        )
        config = VolSurfaceConfig(skew=skew)
        
        for delta in [0.10, 0.20, 0.30, 0.40]:
            ratio = config.get_skew_anchor_ratio(delta)
            assert abs(ratio - 1.05) < 0.01, f"Single anchor should apply everywhere, got {ratio:.4f} at {delta}"


class TestModeAndScalePreservation:
    """Tests that mode and scale behavior is preserved."""

    def test_neutral_mode_dampens_skew(self):
        """Neutral mode should dampen skew toward 1.0."""
        skew_put = SkewTemplate(
            enabled=True,
            anchor_ratios={"0.15": 1.10, "0.25": 1.05, "0.35": 1.02},
            mode="put_heavy",
            scale=1.0,
        )
        skew_neutral = SkewTemplate(
            enabled=True,
            anchor_ratios={"0.15": 1.10, "0.25": 1.05, "0.35": 1.02},
            mode="neutral",
            scale=1.0,
        )
        
        config_put = VolSurfaceConfig(skew=skew_put)
        config_neutral = VolSurfaceConfig(skew=skew_neutral)
        
        ratio_put = config_put.get_skew_anchor_ratio(0.15)
        ratio_neutral = config_neutral.get_skew_anchor_ratio(0.15)
        
        assert abs(ratio_put - 1.0) > abs(ratio_neutral - 1.0), (
            f"Neutral mode should dampen skew: put={ratio_put:.4f}, neutral={ratio_neutral:.4f}"
        )

    def test_scale_multiplies_ratio(self):
        """Scale should multiply the ratio."""
        skew = SkewTemplate(
            enabled=True,
            anchor_ratios={"0.15": 1.10, "0.25": 1.05, "0.35": 1.02},
            mode="put_heavy",
            scale=0.5,
        )
        config = VolSurfaceConfig(skew=skew)
        
        ratio = config.get_skew_anchor_ratio(0.25)
        
        assert ratio < 1.0, f"Scale 0.5 should reduce ratio below base, got {ratio:.4f}"

    def test_ratio_clamped_to_safe_range(self):
        """Ratio should be clamped to [0.5, 2.0] for safety."""
        skew = SkewTemplate(
            enabled=True,
            anchor_ratios={"0.15": 5.0, "0.25": 1.0, "0.35": 0.1},
            mode="put_heavy",
            scale=1.0,
        )
        config = VolSurfaceConfig(skew=skew)
        
        ratio_high = config.get_skew_anchor_ratio(0.10)
        ratio_low = config.get_skew_anchor_ratio(0.40)
        
        assert ratio_high <= 2.0, f"Ratio should be clamped to 2.0 max, got {ratio_high:.4f}"
        assert ratio_low >= 0.5, f"Ratio should be clamped to 0.5 min, got {ratio_low:.4f}"
