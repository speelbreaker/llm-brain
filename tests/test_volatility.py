"""Tests for src/metrics/volatility.py - IVRV ratio calculation."""
import pytest
from src.metrics.volatility import compute_ivrv_ratio


class TestComputeIvrvRatio:
    """Tests for the compute_ivrv_ratio function."""

    def test_normal_case(self):
        """Normal case: iv=0.8, rv=0.4 should give ratio=2.0."""
        result = compute_ivrv_ratio(0.8, 0.4)
        assert result == pytest.approx(2.0)

    def test_iv_greater_than_rv(self):
        """IV higher than RV indicates expensive options."""
        result = compute_ivrv_ratio(0.70, 0.50)
        assert result == pytest.approx(1.4)

    def test_iv_less_than_rv(self):
        """IV lower than RV indicates cheap options."""
        result = compute_ivrv_ratio(0.30, 0.50)
        assert result == pytest.approx(0.6)

    def test_realized_vol_zero_returns_default(self):
        """When realized vol is 0, return default (1.0)."""
        result = compute_ivrv_ratio(0.8, 0.0)
        assert result == 1.0

    def test_realized_vol_negative_returns_default(self):
        """When realized vol is negative, return default (1.0)."""
        result = compute_ivrv_ratio(0.8, -0.1)
        assert result == 1.0

    def test_iv_none_returns_default(self):
        """When iv is None, return default (1.0)."""
        result = compute_ivrv_ratio(None, 0.4)
        assert result == 1.0

    def test_rv_none_returns_default(self):
        """When realized vol is None, return default (1.0)."""
        result = compute_ivrv_ratio(0.8, None)
        assert result == 1.0

    def test_both_none_returns_default(self):
        """When both inputs are None, return default (1.0)."""
        result = compute_ivrv_ratio(None, None)
        assert result == 1.0

    def test_iv_zero_returns_default(self):
        """When iv is 0, return default (1.0)."""
        result = compute_ivrv_ratio(0.0, 0.5)
        assert result == 1.0

    def test_custom_default(self):
        """Custom default value should be returned for invalid inputs."""
        result = compute_ivrv_ratio(None, 0.5, default=2.5)
        assert result == 2.5
