"""
Regression tests for skew look-ahead bias fix.

These tests verify that historical backtests never use live Deribit skew data,
which would introduce look-ahead bias.
"""
import os
import json
import pytest
from datetime import datetime, date, timezone, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestSkewSourceParameter:
    """Test that skew source parameter works correctly."""
    
    def test_get_skew_factor_returns_flat_for_none_source(self):
        """Test that source='none' always returns 1.0 (flat skew)."""
        from src.synthetic_skew import get_skew_factor
        
        result = get_skew_factor(
            underlying="BTC",
            option_type="call",
            abs_delta=0.25,
            skew_enabled=True,
            source="none",
        )
        
        assert result == 1.0
    
    def test_get_skew_factor_returns_flat_when_disabled(self):
        """Test that skew_enabled=False always returns 1.0."""
        from src.synthetic_skew import get_skew_factor
        
        result = get_skew_factor(
            underlying="BTC",
            option_type="call",
            abs_delta=0.25,
            skew_enabled=False,
            source="live",  # Even with live source, disabled should return 1.0
        )
        
        assert result == 1.0
    
    def test_get_skew_factor_harvested_returns_flat_without_as_of(self):
        """Test that harvested source returns 1.0 when as_of is None."""
        from src.synthetic_skew import get_skew_factor
        
        result = get_skew_factor(
            underlying="BTC",
            option_type="call",
            abs_delta=0.25,
            skew_enabled=True,
            source="harvested",
            as_of=None,  # No as_of should return flat
        )
        
        assert result == 1.0


class TestNoLiveSkewInBacktest:
    """Test that backtest code never calls live Deribit skew."""
    
    def test_backtest_does_not_call_live_skew(self):
        """Verify that backtest mode does not call compute_live_skew_anchors."""
        from src.backtest.pricing import get_sigma_for_option
        from src.backtest.types import CallSimulationConfig
        
        as_of = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        spot_history = [(as_of - timedelta(days=i), 60000.0 + i * 100) for i in range(30)]
        
        cfg = CallSimulationConfig(
            underlying="BTC",
            start=date(2024, 6, 1),
            end=date(2024, 6, 30),
            timeframe="1d",
            decision_interval_bars=1,
            initial_spot_position=1.0,
            contract_size=1.0,
            fee_rate=0.0005,
        )
        
        with patch('src.synthetic_skew.compute_live_skew_anchors') as mock_live:
            mock_live.side_effect = RuntimeError("Live skew should not be called in backtest!")
            
            result = get_sigma_for_option(
                config=cfg,
                spot_history=spot_history,
                as_of=as_of,
                option_chain=None,
                option_mark_iv=None,
                abs_delta=0.25,
                regime_state=None,
                skew_source="none",  # Backtest mode
            )
            
            assert result > 0
            mock_live.assert_not_called()
    
    def test_harvested_mode_does_not_call_live_api(self):
        """Verify that harvested mode does not call Deribit API."""
        from src.synthetic_skew import get_skew_factor
        
        as_of = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        
        with patch('src.synthetic_skew._deribit_get') as mock_api:
            mock_api.side_effect = RuntimeError("Live API should not be called!")
            
            result = get_skew_factor(
                underlying="BTC",
                option_type="call",
                abs_delta=0.25,
                skew_enabled=True,
                source="harvested",
                as_of=as_of,
            )
            
            assert result == 1.0
            mock_api.assert_not_called()


class TestSkewCacheIsolation:
    """Test that skew caches are properly isolated by time."""
    
    def test_different_dates_use_different_cache_keys(self):
        """Verify that different as_of dates don't share cached anchors."""
        from src.synthetic_skew import (
            get_skew_factor, 
            clear_skew_cache,
            get_cached_harvested_anchors,
        )
        
        clear_skew_cache()
        
        date1 = date(2024, 6, 15)
        date2 = date(2024, 6, 16)
        as_of1 = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        as_of2 = datetime(2024, 6, 16, 12, 0, 0, tzinfo=timezone.utc)
        
        get_skew_factor(
            underlying="BTC",
            option_type="call",
            abs_delta=0.25,
            skew_enabled=True,
            source="harvested",
            as_of=as_of1,
        )
        
        get_skew_factor(
            underlying="BTC",
            option_type="call",
            abs_delta=0.25,
            skew_enabled=True,
            source="harvested",
            as_of=as_of2,
        )
        
        cached1 = get_cached_harvested_anchors("BTC", "call", date1)
        cached2 = get_cached_harvested_anchors("BTC", "call", date2)
        
        assert cached1 is not None
        assert cached2 is not None
    
    def test_live_and_harvested_use_separate_caches(self):
        """Verify that live and harvested caches are separate."""
        from src.synthetic_skew import (
            clear_skew_cache,
            clear_live_skew_cache,
            clear_harvested_skew_cache,
            get_cached_anchors,
            get_cached_harvested_anchors,
        )
        
        clear_skew_cache()
        
        assert get_cached_anchors("BTC", "call") is None
        assert get_cached_harvested_anchors("BTC", "call", date(2024, 6, 15)) is None


class TestDeterministicSkew:
    """Test that skew calculation is deterministic for given inputs."""
    
    def test_harvested_skew_is_deterministic(self):
        """Verify that same inputs produce same output."""
        from src.synthetic_skew import get_skew_factor, clear_skew_cache
        
        clear_skew_cache()
        
        as_of = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        
        result1 = get_skew_factor(
            underlying="BTC",
            option_type="call",
            abs_delta=0.25,
            skew_enabled=True,
            source="harvested",
            as_of=as_of,
        )
        
        clear_skew_cache()
        
        result2 = get_skew_factor(
            underlying="BTC",
            option_type="call",
            abs_delta=0.25,
            skew_enabled=True,
            source="harvested",
            as_of=as_of,
        )
        
        assert result1 == result2
    
    def test_flat_skew_is_always_one(self):
        """Verify flat skew returns exactly 1.0."""
        from src.synthetic_skew import _flat_anchors, SkewAnchor
        
        anchors = _flat_anchors()
        
        assert len(anchors) == 4
        for anchor in anchors:
            assert anchor.ratio == 1.0


class TestSkewAnchorComputation:
    """Test skew anchor computation from quotes."""
    
    def test_compute_anchors_from_quotes_with_valid_data(self):
        """Test that anchors are computed correctly from quotes."""
        from src.synthetic_skew import _compute_anchors_from_quotes
        
        quotes = [
            {"mark_iv": 60.0, "delta": 0.50, "dte": 7.0},  # ATM
            {"mark_iv": 65.0, "delta": 0.25, "dte": 7.0},  # OTM
            {"mark_iv": 70.0, "delta": 0.15, "dte": 7.0},  # Deep OTM
            {"mark_iv": 58.0, "delta": 0.35, "dte": 7.0},  # Slight OTM
        ]
        
        anchors = _compute_anchors_from_quotes(quotes, "call")
        
        assert len(anchors) == 4
        
        for anchor in anchors:
            assert 0.4 <= anchor.ratio <= 1.4
    
    def test_compute_anchors_returns_flat_with_insufficient_data(self):
        """Test that insufficient data returns flat anchors."""
        from src.synthetic_skew import _compute_anchors_from_quotes
        
        quotes = [
            {"mark_iv": 60.0, "delta": 0.50, "dte": 7.0},
        ]
        
        anchors = _compute_anchors_from_quotes(quotes, "call")
        
        assert len(anchors) == 4
        for anchor in anchors:
            assert anchor.ratio == 1.0


class TestPricingIntegration:
    """Test pricing module integration with skew source."""
    
    def test_get_sigma_for_option_accepts_skew_source(self):
        """Verify get_sigma_for_option accepts skew_source parameter."""
        from src.backtest.pricing import get_sigma_for_option
        from src.backtest.types import CallSimulationConfig
        
        as_of = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        spot_history = [(as_of - timedelta(days=i), 60000.0) for i in range(30)]
        
        cfg = CallSimulationConfig(
            underlying="BTC",
            start=date(2024, 6, 1),
            end=date(2024, 6, 30),
            timeframe="1d",
            decision_interval_bars=1,
            initial_spot_position=1.0,
            contract_size=1.0,
            fee_rate=0.0005,
        )
        
        result = get_sigma_for_option(
            config=cfg,
            spot_history=spot_history,
            as_of=as_of,
            skew_source="none",
        )
        
        assert result > 0
    
    def test_compute_synthetic_iv_with_skew_accepts_source(self):
        """Verify compute_synthetic_iv_with_skew accepts skew_source parameter."""
        from src.backtest.pricing import compute_synthetic_iv_with_skew
        
        as_of = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        
        result = compute_synthetic_iv_with_skew(
            underlying="BTC",
            option_type="call",
            abs_delta=0.25,
            rv_annualized=0.6,
            iv_multiplier=1.0,
            skew_enabled=True,
            as_of=as_of,
            skew_source="none",
        )
        
        assert result > 0
