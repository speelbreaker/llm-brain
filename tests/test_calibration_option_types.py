"""
Tests for option type filtering and default bands in extended calibration.
"""
import pytest
from datetime import datetime, timezone

from src.calibration_config import (
    CalibrationConfig,
    BandConfig,
    DEFAULT_TERM_BANDS,
    OptionTypeMetrics,
    DteBandResult,
)


class TestCalibrationConfig:
    """Tests for CalibrationConfig option_types field."""
    
    def test_default_option_types_is_calls_only(self):
        """Default option_types should be ['C'] for backward compatibility."""
        config = CalibrationConfig(underlying="BTC")
        assert config.option_types == ["C"]
    
    def test_custom_option_types_calls_only(self):
        """Should accept calls only."""
        config = CalibrationConfig(underlying="BTC", option_types=["C"])
        assert config.option_types == ["C"]
    
    def test_custom_option_types_puts_only(self):
        """Should accept puts only."""
        config = CalibrationConfig(underlying="BTC", option_types=["P"])
        assert config.option_types == ["P"]
    
    def test_custom_option_types_both(self):
        """Should accept both calls and puts."""
        config = CalibrationConfig(underlying="BTC", option_types=["C", "P"])
        assert "C" in config.option_types
        assert "P" in config.option_types


class TestDefaultTermBands:
    """Tests for DEFAULT_TERM_BANDS constant."""
    
    def test_default_bands_exist(self):
        """Default bands should be defined."""
        assert DEFAULT_TERM_BANDS is not None
        assert len(DEFAULT_TERM_BANDS) == 3
    
    def test_default_bands_names(self):
        """Default bands should have weekly, monthly, quarterly names."""
        names = [b.name for b in DEFAULT_TERM_BANDS]
        assert "weekly" in names
        assert "monthly" in names
        assert "quarterly" in names
    
    def test_weekly_band_range(self):
        """Weekly band should be 3-10 DTE."""
        weekly = next(b for b in DEFAULT_TERM_BANDS if b.name == "weekly")
        assert weekly.min_dte == 3
        assert weekly.max_dte == 10
    
    def test_monthly_band_range(self):
        """Monthly band should be 20-40 DTE."""
        monthly = next(b for b in DEFAULT_TERM_BANDS if b.name == "monthly")
        assert monthly.min_dte == 20
        assert monthly.max_dte == 40
    
    def test_quarterly_band_range(self):
        """Quarterly band should be 60-100 DTE."""
        quarterly = next(b for b in DEFAULT_TERM_BANDS if b.name == "quarterly")
        assert quarterly.min_dte == 60
        assert quarterly.max_dte == 100


class TestOptionTypeMetrics:
    """Tests for OptionTypeMetrics model."""
    
    def test_create_call_metrics(self):
        """Should create metrics for calls."""
        metrics = OptionTypeMetrics(
            option_type="C",
            count=100,
            mae_pct=2.5,
            bias_pct=0.3,
        )
        assert metrics.option_type == "C"
        assert metrics.count == 100
        assert metrics.mae_pct == 2.5
    
    def test_create_put_metrics(self):
        """Should create metrics for puts."""
        metrics = OptionTypeMetrics(
            option_type="P",
            count=50,
            mae_pct=3.1,
            bias_pct=1.2,
        )
        assert metrics.option_type == "P"
        assert metrics.count == 50
    
    def test_metrics_with_bands(self):
        """Should include band-level metrics."""
        band = DteBandResult(
            name="weekly",
            min_dte=3,
            max_dte=10,
            count=30,
            mae_pct=2.0,
            bias_pct=0.5,
            option_type="C",
        )
        metrics = OptionTypeMetrics(
            option_type="C",
            count=30,
            mae_pct=2.0,
            bias_pct=0.5,
            bands=[band],
        )
        assert metrics.bands is not None
        assert len(metrics.bands) == 1
        assert metrics.bands[0].name == "weekly"


class TestDteBandResultWithOptionType:
    """Tests for DteBandResult with option_type field."""
    
    def test_band_with_call_type(self):
        """Band can have call option type."""
        band = DteBandResult(
            name="weekly",
            min_dte=3,
            max_dte=10,
            count=50,
            mae_pct=2.5,
            bias_pct=0.3,
            option_type="C",
        )
        assert band.option_type == "C"
    
    def test_band_with_put_type(self):
        """Band can have put option type."""
        band = DteBandResult(
            name="weekly",
            min_dte=3,
            max_dte=10,
            count=30,
            mae_pct=3.0,
            bias_pct=0.8,
            option_type="P",
        )
        assert band.option_type == "P"
    
    def test_band_without_option_type(self):
        """Band can exist without option type (backward compatible)."""
        band = DteBandResult(
            name="weekly",
            min_dte=3,
            max_dte=10,
            count=80,
            mae_pct=2.7,
            bias_pct=0.5,
        )
        assert band.option_type is None


class TestCalibrationConfigBackwardCompatibility:
    """Tests to ensure backward compatibility."""
    
    def test_old_style_config_works(self):
        """Config without option_types should work."""
        config = CalibrationConfig(
            underlying="BTC",
            min_dte=3,
            max_dte=10,
            iv_multiplier=1.1,
        )
        assert config.underlying == "BTC"
        assert config.option_types == ["C"]
    
    def test_old_style_config_with_bands_works(self):
        """Config with bands but without option_types should work."""
        bands = [BandConfig(name="short", min_dte=3, max_dte=7)]
        config = CalibrationConfig(
            underlying="ETH",
            bands=bands,
        )
        assert config.bands is not None
        assert len(config.bands) == 1
        assert config.option_types == ["C"]
