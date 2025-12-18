"""
Tests for calibration → pricing wiring.

Verifies that VolSurfaceConfig settings are used by get_sigma_for_option()
for IV multiplier selection and skew adjustments.
"""
import pytest
from datetime import datetime, timezone, timedelta

from src.backtest.pricing import get_sigma_for_option
from src.backtest.types import CallSimulationConfig
from src.synthetic.vol_surface import (
    VolSurfaceConfig,
    DteBand,
    SkewTemplate,
    get_vol_surface_config,
    set_vol_surface_config,
)


@pytest.fixture
def reset_vol_surface():
    """Reset vol surface config before and after each test."""
    original = get_vol_surface_config()
    yield
    set_vol_surface_config(original)


@pytest.fixture
def base_config():
    """Create a minimal CallSimulationConfig for testing."""
    return CallSimulationConfig(
        underlying="BTC",
        start=datetime(2024, 6, 1, tzinfo=timezone.utc),
        end=datetime(2024, 6, 30, tzinfo=timezone.utc),
        timeframe="1d",
        decision_interval_bars=1,
        initial_spot_position=1.0,
        contract_size=1.0,
        fee_rate=0.0005,
        target_dte=7,
        dte_tolerance=2,
    )


@pytest.fixture
def spot_history():
    """Create spot history with known RV pattern."""
    as_of = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
    return [(as_of - timedelta(days=i), 60000.0) for i in range(30)]


class TestGlobalIVMultiplier:
    """Test that global IV multiplier from VolSurfaceConfig is used."""
    
    def test_iv_multiplier_1_2_increases_sigma_by_20pct(
        self, reset_vol_surface, base_config, spot_history
    ):
        """When VolSurfaceConfig.iv_multiplier=1.2, sigma increases by 20% vs 1.0."""
        as_of = spot_history[0][0]
        
        set_vol_surface_config(VolSurfaceConfig(iv_multiplier=1.0))
        sigma_base = get_sigma_for_option(
            config=base_config,
            spot_history=spot_history,
            as_of=as_of,
            skew_source="none",
            dte_days=7.0,
        )
        
        set_vol_surface_config(VolSurfaceConfig(iv_multiplier=1.2))
        sigma_scaled = get_sigma_for_option(
            config=base_config,
            spot_history=spot_history,
            as_of=as_of,
            skew_source="none",
            dte_days=7.0,
        )
        
        ratio = sigma_scaled / sigma_base
        assert 1.19 < ratio < 1.21, f"Expected ~1.2x increase, got {ratio}"
    
    def test_iv_multiplier_0_8_decreases_sigma(
        self, reset_vol_surface, base_config, spot_history
    ):
        """When VolSurfaceConfig.iv_multiplier=0.8, sigma decreases by 20%."""
        as_of = spot_history[0][0]
        
        set_vol_surface_config(VolSurfaceConfig(iv_multiplier=1.0))
        sigma_base = get_sigma_for_option(
            config=base_config,
            spot_history=spot_history,
            as_of=as_of,
            skew_source="none",
            dte_days=7.0,
        )
        
        set_vol_surface_config(VolSurfaceConfig(iv_multiplier=0.8))
        sigma_scaled = get_sigma_for_option(
            config=base_config,
            spot_history=spot_history,
            as_of=as_of,
            skew_source="none",
            dte_days=7.0,
        )
        
        ratio = sigma_scaled / sigma_base
        assert 0.79 < ratio < 0.81, f"Expected ~0.8x decrease, got {ratio}"


class TestDteBandMultiplier:
    """Test that DTE-band specific multipliers are used when applicable."""
    
    def test_dte_band_multiplier_used_when_in_range(
        self, reset_vol_surface, base_config, spot_history
    ):
        """When DTE is within a band, that band's multiplier is used."""
        as_of = spot_history[0][0]
        
        vs = VolSurfaceConfig(
            iv_multiplier=1.0,
            dte_bands=[
                DteBand(name="weekly", min_dte=5.0, max_dte=10.0, iv_multiplier=1.5),
                DteBand(name="monthly", min_dte=20.0, max_dte=35.0, iv_multiplier=1.2),
            ]
        )
        set_vol_surface_config(vs)
        
        set_vol_surface_config(VolSurfaceConfig(iv_multiplier=1.0))
        sigma_global = get_sigma_for_option(
            config=base_config,
            spot_history=spot_history,
            as_of=as_of,
            skew_source="none",
            dte_days=7.0,
        )
        
        set_vol_surface_config(vs)
        sigma_band = get_sigma_for_option(
            config=base_config,
            spot_history=spot_history,
            as_of=as_of,
            skew_source="none",
            dte_days=7.0,  # Falls within 'weekly' band (5-10)
        )
        
        ratio = sigma_band / sigma_global
        assert 1.49 < ratio < 1.51, f"Expected ~1.5x from weekly band, got {ratio}"
    
    def test_dte_outside_band_uses_global_multiplier(
        self, reset_vol_surface, base_config, spot_history
    ):
        """When DTE is outside all bands, global multiplier is used."""
        as_of = spot_history[0][0]
        
        vs = VolSurfaceConfig(
            iv_multiplier=1.1,
            dte_bands=[
                DteBand(name="weekly", min_dte=5.0, max_dte=10.0, iv_multiplier=1.5),
            ]
        )
        set_vol_surface_config(vs)
        
        sigma = get_sigma_for_option(
            config=base_config,
            spot_history=spot_history,
            as_of=as_of,
            skew_source="none",
            dte_days=15.0,  # Outside the weekly band
        )
        
        set_vol_surface_config(VolSurfaceConfig(iv_multiplier=1.1))
        sigma_global = get_sigma_for_option(
            config=base_config,
            spot_history=spot_history,
            as_of=as_of,
            skew_source="none",
            dte_days=15.0,
        )
        
        assert abs(sigma - sigma_global) < 0.001, \
            f"Expected global multiplier when outside band, got sigma={sigma}, expected={sigma_global}"


class TestVolSurfaceSkew:
    """Test that VolSurfaceConfig.skew is used for skew adjustments."""
    
    def test_skew_applied_when_enabled_and_delta_in_range(
        self, reset_vol_surface, base_config, spot_history
    ):
        """When skew is enabled and abs_delta provided, get_skew_anchor_ratio is used."""
        as_of = spot_history[0][0]
        
        vs_no_skew = VolSurfaceConfig(iv_multiplier=1.0)
        set_vol_surface_config(vs_no_skew)
        
        sigma_no_skew = get_sigma_for_option(
            config=base_config,
            spot_history=spot_history,
            as_of=as_of,
            abs_delta=0.25,
            skew_source="none",
            dte_days=7.0,
        )
        
        vs_with_skew = VolSurfaceConfig(
            iv_multiplier=1.0,
            skew=SkewTemplate(
                enabled=True,
                min_dte=3.0,
                max_dte=14.0,
                anchor_ratios={"0.15": 1.1, "0.25": 1.05, "0.35": 1.02},
            )
        )
        set_vol_surface_config(vs_with_skew)
        
        sigma_with_skew = get_sigma_for_option(
            config=base_config,
            spot_history=spot_history,
            as_of=as_of,
            abs_delta=0.25,
            skew_source="none",
            dte_days=7.0,  # Within skew DTE range (3-14)
        )
        
        ratio = sigma_with_skew / sigma_no_skew
        assert 1.04 < ratio < 1.06, f"Expected ~1.05x from skew anchor, got {ratio}"
    
    def test_skew_not_applied_when_dte_outside_range(
        self, reset_vol_surface, base_config, spot_history
    ):
        """When DTE is outside skew range, skew is not applied."""
        as_of = spot_history[0][0]
        
        vs_with_skew = VolSurfaceConfig(
            iv_multiplier=1.0,
            skew=SkewTemplate(
                enabled=True,
                min_dte=3.0,
                max_dte=14.0,
                anchor_ratios={"0.15": 1.1, "0.25": 1.05, "0.35": 1.02},
            )
        )
        set_vol_surface_config(vs_with_skew)
        
        sigma_outside_range = get_sigma_for_option(
            config=base_config,
            spot_history=spot_history,
            as_of=as_of,
            abs_delta=0.25,
            skew_source="none",
            dte_days=20.0,  # Outside skew DTE range (3-14)
        )
        
        vs_no_skew = VolSurfaceConfig(iv_multiplier=1.0)
        set_vol_surface_config(vs_no_skew)
        
        sigma_no_skew = get_sigma_for_option(
            config=base_config,
            spot_history=spot_history,
            as_of=as_of,
            abs_delta=0.25,
            skew_source="none",
            dte_days=20.0,
        )
        
        assert abs(sigma_outside_range - sigma_no_skew) < 0.001, \
            f"Skew should not apply outside DTE range"
    
    def test_skew_not_applied_when_disabled(
        self, reset_vol_surface, base_config, spot_history
    ):
        """When skew.enabled=False, skew is not applied."""
        as_of = spot_history[0][0]
        
        vs_skew_disabled = VolSurfaceConfig(
            iv_multiplier=1.0,
            skew=SkewTemplate(
                enabled=False,
                min_dte=3.0,
                max_dte=14.0,
                anchor_ratios={"0.25": 1.5},
            )
        )
        set_vol_surface_config(vs_skew_disabled)
        
        sigma_disabled = get_sigma_for_option(
            config=base_config,
            spot_history=spot_history,
            as_of=as_of,
            abs_delta=0.25,
            skew_source="none",
            dte_days=7.0,
        )
        
        vs_no_skew = VolSurfaceConfig(iv_multiplier=1.0)
        set_vol_surface_config(vs_no_skew)
        
        sigma_no_skew = get_sigma_for_option(
            config=base_config,
            spot_history=spot_history,
            as_of=as_of,
            abs_delta=0.25,
            skew_source="none",
            dte_days=7.0,
        )
        
        assert abs(sigma_disabled - sigma_no_skew) < 0.001, \
            f"Skew should not apply when disabled"


class TestNoNetworkCallsInBacktest:
    """Verify no network calls are made during backtest sigma calculation."""
    
    def test_no_deribit_calls_with_vol_surface_config(
        self, reset_vol_surface, base_config, spot_history
    ):
        """VolSurfaceConfig wiring should not trigger any network calls."""
        from unittest.mock import patch
        
        as_of = spot_history[0][0]
        
        vs = VolSurfaceConfig(
            iv_multiplier=1.2,
            dte_bands=[
                DteBand(name="weekly", min_dte=5.0, max_dte=10.0, iv_multiplier=1.3),
            ],
            skew=SkewTemplate(
                enabled=True,
                min_dte=3.0,
                max_dte=14.0,
                anchor_ratios={"0.25": 1.05},
            )
        )
        set_vol_surface_config(vs)
        
        with patch('src.synthetic_skew.compute_live_skew_anchors') as mock_live:
            mock_live.side_effect = RuntimeError("Network call detected in backtest!")
            
            sigma = get_sigma_for_option(
                config=base_config,
                spot_history=spot_history,
                as_of=as_of,
                abs_delta=0.25,
                skew_source="none",
                dte_days=7.0,
            )
            
            assert sigma > 0
            mock_live.assert_not_called()


class TestCalibrationImmediacy:
    """Test that calibration changes are reflected immediately."""
    
    def test_changing_multiplier_reflects_immediately(
        self, reset_vol_surface, base_config, spot_history
    ):
        """Changing VolSurfaceConfig.iv_multiplier is immediately reflected in sigma."""
        as_of = spot_history[0][0]
        
        set_vol_surface_config(VolSurfaceConfig(iv_multiplier=1.0))
        sigma_1 = get_sigma_for_option(
            config=base_config,
            spot_history=spot_history,
            as_of=as_of,
            skew_source="none",
            dte_days=7.0,
        )
        
        set_vol_surface_config(VolSurfaceConfig(iv_multiplier=1.5))
        sigma_2 = get_sigma_for_option(
            config=base_config,
            spot_history=spot_history,
            as_of=as_of,
            skew_source="none",
            dte_days=7.0,
        )
        
        set_vol_surface_config(VolSurfaceConfig(iv_multiplier=0.7))
        sigma_3 = get_sigma_for_option(
            config=base_config,
            spot_history=spot_history,
            as_of=as_of,
            skew_source="none",
            dte_days=7.0,
        )
        
        assert sigma_2 > sigma_1, "1.5x multiplier should yield higher sigma than 1.0x"
        assert sigma_3 < sigma_1, "0.7x multiplier should yield lower sigma than 1.0x"
        
        ratio_1_to_2 = sigma_2 / sigma_1
        ratio_1_to_3 = sigma_3 / sigma_1
        
        assert 1.49 < ratio_1_to_2 < 1.51, f"Expected 1.5x ratio, got {ratio_1_to_2}"
        assert 0.69 < ratio_1_to_3 < 0.71, f"Expected 0.7x ratio, got {ratio_1_to_3}"
