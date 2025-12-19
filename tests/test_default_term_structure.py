"""
Tests for default term-structure realism (weekly vs monthly IV multipliers).

Verifies that:
1. Default DTE bands are applied when no calibration override is set
2. Weekly options (3-10 DTE) use 1.0x multiplier
3. Monthly options (20-40 DTE) use 1.10x multiplier
4. Calibration overrides take precedence over defaults
"""
from datetime import datetime, timedelta
from typing import List, Tuple

import pytest

from src.synthetic.vol_surface import (
    get_vol_surface_config,
    set_vol_surface_config,
    reset_vol_surface_config,
    VolSurfaceConfig,
    DteBand,
)
from src.backtest.pricing import get_sigma_for_option
from src.backtest.types import CallSimulationConfig


@pytest.fixture(autouse=True)
def reset_vol_surface():
    """Reset vol surface config before and after each test."""
    reset_vol_surface_config()
    yield
    reset_vol_surface_config()


def make_spot_history(days: int = 30) -> List[Tuple[datetime, float]]:
    """Create fake spot history for RV calculation."""
    base = datetime(2024, 1, 1)
    return [(base + timedelta(days=i), 40000.0 + i * 10) for i in range(days)]


class TestDefaultDteBands:
    """Tests for default DTE band configuration."""

    def test_default_config_has_dte_bands(self):
        """Default config should include DTE bands."""
        config = get_vol_surface_config()
        assert config.dte_bands is not None
        assert len(config.dte_bands) == 2

    def test_weekly_band_settings(self):
        """Weekly band should be 3-10 DTE with 1.0x multiplier."""
        config = get_vol_surface_config()
        weekly = next(b for b in config.dte_bands if b.name == "weekly")
        
        assert weekly.min_dte == 3.0
        assert weekly.max_dte == 10.0
        assert weekly.iv_multiplier == 1.0

    def test_monthly_band_settings(self):
        """Monthly band should be 20-40 DTE with 1.10x multiplier."""
        config = get_vol_surface_config()
        monthly = next(b for b in config.dte_bands if b.name == "monthly")
        
        assert monthly.min_dte == 20.0
        assert monthly.max_dte == 40.0
        assert monthly.iv_multiplier == 1.10


class TestTermStructureRealism:
    """Tests for term structure affecting sigma selection."""

    def test_weekly_vs_monthly_multiplier_difference(self):
        """Weekly and monthly DTEs should produce different sigmas due to term structure."""
        spot_history = make_spot_history(30)
        as_of = datetime(2024, 1, 15)
        
        cfg = CallSimulationConfig(
            underlying="BTC",
            start=datetime(2024, 1, 1),
            end=datetime(2024, 2, 1),
            target_dte=7,
            timeframe="1d",
            decision_interval_bars=1,
            initial_spot_position=1.0,
            contract_size=1.0,
            fee_rate=0.0,
        )
        
        sigma_weekly = get_sigma_for_option(
            config=cfg,
            spot_history=spot_history,
            as_of=as_of,
            dte_days=7.0,
            abs_delta=0.25,
        )
        
        sigma_monthly = get_sigma_for_option(
            config=cfg,
            spot_history=spot_history,
            as_of=as_of,
            dte_days=30.0,
            abs_delta=0.25,
        )
        
        assert sigma_monthly > sigma_weekly, (
            f"Monthly sigma ({sigma_monthly:.4f}) should be higher than "
            f"weekly sigma ({sigma_weekly:.4f}) due to term structure"
        )
        
        expected_ratio = 1.10
        actual_ratio = sigma_monthly / sigma_weekly
        assert abs(actual_ratio - expected_ratio) < 0.01, (
            f"Monthly/weekly ratio should be ~{expected_ratio}, got {actual_ratio:.4f}"
        )

    def test_multiplier_applied_via_get_iv_multiplier_for_dte(self):
        """get_iv_multiplier_for_dte should return correct multipliers."""
        config = get_vol_surface_config()
        
        assert config.get_iv_multiplier_for_dte(7.0) == 1.0
        assert config.get_iv_multiplier_for_dte(30.0) == 1.10
        
        assert config.get_iv_multiplier_for_dte(15.0) == config.iv_multiplier


class TestCalibrationOverride:
    """Tests for calibration overriding default DTE bands."""

    def test_calibration_override_replaces_defaults(self):
        """Calibration config should completely override default bands."""
        custom_bands = [
            DteBand(name="short", min_dte=1.0, max_dte=5.0, iv_multiplier=0.95),
            DteBand(name="medium", min_dte=10.0, max_dte=20.0, iv_multiplier=1.15),
        ]
        custom_config = VolSurfaceConfig(
            iv_multiplier=1.2,
            dte_bands=custom_bands,
        )
        set_vol_surface_config(custom_config)
        
        config = get_vol_surface_config()
        
        assert len(config.dte_bands) == 2
        assert config.dte_bands[0].name == "short"
        assert config.dte_bands[0].iv_multiplier == 0.95
        assert config.get_iv_multiplier_for_dte(3.0) == 0.95
        assert config.get_iv_multiplier_for_dte(15.0) == 1.15
        assert config.get_iv_multiplier_for_dte(25.0) == 1.2

    def test_calibration_with_no_bands_uses_global(self):
        """Calibration with no DTE bands should use global multiplier."""
        custom_config = VolSurfaceConfig(
            iv_multiplier=1.25,
            dte_bands=None,
        )
        set_vol_surface_config(custom_config)
        
        config = get_vol_surface_config()
        
        assert config.dte_bands is None
        assert config.get_iv_multiplier_for_dte(7.0) == 1.25
        assert config.get_iv_multiplier_for_dte(30.0) == 1.25

    def test_reset_restores_defaults(self):
        """After reset, defaults should be restored."""
        custom_config = VolSurfaceConfig(iv_multiplier=2.0, dte_bands=[])
        set_vol_surface_config(custom_config)
        
        reset_vol_surface_config()
        
        config = get_vol_surface_config()
        assert config.dte_bands is not None
        assert len(config.dte_bands) == 2
        assert config.get_iv_multiplier_for_dte(7.0) == 1.0
        assert config.get_iv_multiplier_for_dte(30.0) == 1.10
