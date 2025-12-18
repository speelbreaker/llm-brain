"""
Tests for the unified SyntheticVolEngine.

Verifies that the engine:
1. Reproduces current behavior for default settings
2. Applies calibration multipliers and skew correctly
3. Blocks look-ahead bias (live skew for historical dates)
"""
import pytest
from datetime import datetime, timezone, timedelta

from src.synthetic.engine import (
    SyntheticVolEngine,
    SyntheticVolConfig,
    get_iv,
    get_synthetic_vol_engine,
)
from src.synthetic.vol_surface import (
    VolSurfaceConfig,
    DteBand,
    SkewTemplate,
    set_vol_surface_config,
    get_vol_surface_config,
)


@pytest.fixture
def reset_vol_surface():
    """Reset vol surface config before and after each test."""
    original = get_vol_surface_config()
    yield
    set_vol_surface_config(original)


@pytest.fixture
def spot_history():
    """Create spot history with known RV pattern."""
    as_of = datetime.now(timezone.utc)
    return [(as_of - timedelta(days=i), 60000.0) for i in range(30)]


class TestEngineBasics:
    """Test basic engine functionality."""
    
    def test_engine_returns_valid_iv(self, spot_history):
        """Engine should return a valid IV value."""
        engine = SyntheticVolEngine()
        as_of = datetime.now(timezone.utc)
        
        iv = engine.get_iv(
            underlying="BTC",
            as_of=as_of,
            dte_days=7.0,
            abs_delta=0.25,
            option_type="call",
            spot_history=spot_history,
            config=SyntheticVolConfig(skew_source="none"),
        )
        
        assert 0.01 <= iv <= 5.0, f"IV {iv} out of bounds"
    
    def test_get_atm_iv_from_rv(self, spot_history):
        """Engine should compute ATM IV from realized volatility."""
        engine = SyntheticVolEngine()
        as_of = datetime.now(timezone.utc)
        
        atm_iv = engine.get_atm_iv(
            underlying="BTC",
            as_of=as_of,
            dte_days=7.0,
            spot_history=spot_history,
            config=SyntheticVolConfig(sigma_mode="rv_x_multiplier"),
        )
        
        assert atm_iv > 0, "ATM IV should be positive"
    
    def test_get_skew_anchor_ratio_flat_by_default(self, reset_vol_surface):
        """With no config, skew should be flat (1.0)."""
        engine = SyntheticVolEngine()
        as_of = datetime.now(timezone.utc)
        
        set_vol_surface_config(VolSurfaceConfig())
        
        ratio = engine.get_skew_anchor_ratio(
            underlying="BTC",
            as_of=as_of,
            dte_days=7.0,
            abs_delta=0.25,
            option_type="call",
            config=SyntheticVolConfig(skew_source="none"),
        )
        
        assert ratio == 1.0, "Flat skew should return 1.0"


class TestCalibrationWiring:
    """Test that calibration config affects engine output."""
    
    def test_multiplier_affects_iv(self, reset_vol_surface, spot_history):
        """Changing calibration multiplier should affect IV."""
        engine = SyntheticVolEngine()
        as_of = datetime.now(timezone.utc)
        
        set_vol_surface_config(VolSurfaceConfig(iv_multiplier=1.0))
        iv_base = engine.get_iv(
            underlying="BTC",
            as_of=as_of,
            dte_days=7.0,
            abs_delta=0.25,
            spot_history=spot_history,
            config=SyntheticVolConfig(skew_source="none"),
        )
        
        set_vol_surface_config(VolSurfaceConfig(iv_multiplier=1.5))
        iv_scaled = engine.get_iv(
            underlying="BTC",
            as_of=as_of,
            dte_days=7.0,
            abs_delta=0.25,
            spot_history=spot_history,
            config=SyntheticVolConfig(skew_source="none"),
        )
        
        ratio = iv_scaled / iv_base
        assert 1.45 < ratio < 1.55, f"Expected ~1.5x multiplier effect, got {ratio}"
    
    def test_dte_band_multiplier_used(self, reset_vol_surface, spot_history):
        """DTE-band specific multiplier should be used when DTE is in range."""
        engine = SyntheticVolEngine()
        as_of = datetime.now(timezone.utc)
        
        vs = VolSurfaceConfig(
            iv_multiplier=1.0,
            dte_bands=[
                DteBand(name="weekly", min_dte=5.0, max_dte=10.0, iv_multiplier=2.0),
            ]
        )
        set_vol_surface_config(vs)
        
        iv_in_band = engine.get_iv(
            underlying="BTC",
            as_of=as_of,
            dte_days=7.0,  # In weekly band
            abs_delta=0.25,
            spot_history=spot_history,
            config=SyntheticVolConfig(skew_source="none"),
        )
        
        iv_out_band = engine.get_iv(
            underlying="BTC",
            as_of=as_of,
            dte_days=15.0,  # Outside band
            abs_delta=0.25,
            spot_history=spot_history,
            config=SyntheticVolConfig(skew_source="none"),
        )
        
        ratio = iv_in_band / iv_out_band
        assert 1.9 < ratio < 2.1, f"Expected ~2x for DTE band, got {ratio}"
    
    def test_calibrated_skew_applied(self, reset_vol_surface, spot_history):
        """Calibrated skew anchors should affect IV."""
        engine = SyntheticVolEngine()
        as_of = datetime.now(timezone.utc)
        
        vs_no_skew = VolSurfaceConfig(iv_multiplier=1.0)
        set_vol_surface_config(vs_no_skew)
        
        iv_no_skew = engine.get_iv(
            underlying="BTC",
            as_of=as_of,
            dte_days=7.0,
            abs_delta=0.25,
            spot_history=spot_history,
            config=SyntheticVolConfig(skew_source="none"),
        )
        
        vs_with_skew = VolSurfaceConfig(
            iv_multiplier=1.0,
            skew=SkewTemplate(
                enabled=True,
                min_dte=3.0,
                max_dte=14.0,
                anchor_ratios={"0.25": 1.2},
            )
        )
        set_vol_surface_config(vs_with_skew)
        
        iv_with_skew = engine.get_iv(
            underlying="BTC",
            as_of=as_of,
            dte_days=7.0,
            abs_delta=0.25,
            spot_history=spot_history,
            config=SyntheticVolConfig(skew_source="none"),
        )
        
        ratio = iv_with_skew / iv_no_skew
        assert 1.15 < ratio < 1.25, f"Expected ~1.2x for skew, got {ratio}"


class TestNoLookaheadBias:
    """Test that the engine prevents look-ahead bias."""
    
    def test_live_skew_blocked_for_historical(self, spot_history):
        """Live skew should be blocked for historical dates."""
        engine = SyntheticVolEngine()
        
        historical_as_of = datetime.now(timezone.utc) - timedelta(days=30)
        
        with pytest.raises(ValueError, match="Look-ahead bias"):
            engine.get_iv(
                underlying="BTC",
                as_of=historical_as_of,
                dte_days=7.0,
                abs_delta=0.25,
                spot_history=spot_history,
                config=SyntheticVolConfig(skew_source="live"),
            )
    
    def test_live_skew_allowed_for_current(self, reset_vol_surface, spot_history):
        """Live skew should be allowed for current/recent dates."""
        from unittest.mock import patch
        
        engine = SyntheticVolEngine()
        current_as_of = datetime.now(timezone.utc)
        
        set_vol_surface_config(VolSurfaceConfig())
        
        with patch('src.synthetic_skew.get_skew_factor', return_value=1.0):
            iv = engine.get_iv(
                underlying="BTC",
                as_of=current_as_of,
                dte_days=7.0,
                abs_delta=0.25,
                spot_history=spot_history,
                config=SyntheticVolConfig(skew_source="live"),
            )
            
            assert iv > 0
    
    def test_harvested_skew_allowed_for_historical(self, reset_vol_surface, spot_history):
        """Harvested skew should be allowed for historical dates."""
        from unittest.mock import patch
        
        engine = SyntheticVolEngine()
        historical_as_of = datetime.now(timezone.utc) - timedelta(days=30)
        
        set_vol_surface_config(VolSurfaceConfig())
        
        with patch('src.synthetic_skew.get_skew_factor', return_value=1.0):
            iv = engine.get_iv(
                underlying="BTC",
                as_of=historical_as_of,
                dte_days=7.0,
                abs_delta=0.25,
                spot_history=spot_history,
                config=SyntheticVolConfig(skew_source="harvested"),
            )
            
            assert iv > 0


class TestConvenienceFunctions:
    """Test module-level convenience functions."""
    
    def test_get_iv_convenience_function(self, reset_vol_surface, spot_history):
        """get_iv() convenience function should work."""
        set_vol_surface_config(VolSurfaceConfig(iv_multiplier=1.0))
        
        iv = get_iv(
            underlying="BTC",
            as_of=datetime.now(timezone.utc),
            dte_days=7.0,
            abs_delta=0.25,
            spot_history=spot_history,
            skew_source="none",
        )
        
        assert 0.01 <= iv <= 5.0
    
    def test_singleton_engine(self):
        """get_synthetic_vol_engine() should return singleton."""
        engine1 = get_synthetic_vol_engine()
        engine2 = get_synthetic_vol_engine()
        
        assert engine1 is engine2


class TestIVWithDetails:
    """Test the get_iv_with_details audit trail."""
    
    def test_returns_audit_trail(self, reset_vol_surface, spot_history):
        """get_iv_with_details should return full audit trail."""
        engine = SyntheticVolEngine()
        as_of = datetime.now(timezone.utc)
        
        set_vol_surface_config(VolSurfaceConfig(
            iv_multiplier=1.2,
            skew=SkewTemplate(
                enabled=True,
                min_dte=3.0,
                max_dte=14.0,
                anchor_ratios={"0.25": 1.05},
            )
        ))
        
        result = engine.get_iv_with_details(
            underlying="BTC",
            as_of=as_of,
            dte_days=7.0,
            abs_delta=0.25,
            spot_history=spot_history,
            config=SyntheticVolConfig(skew_source="none"),
        )
        
        assert result.iv > 0
        assert result.atm_iv > 0
        assert result.multiplier == 1.2
        assert 1.04 < result.skew_ratio < 1.06
        assert result.sigma_mode == "rv_x_multiplier"
        assert result.skew_source == "none"
        assert result.surface_source == "calibration"
        assert result.dte_days == 7.0
