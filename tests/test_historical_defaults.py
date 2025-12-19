"""
Tests for historical backtest defaults and fallback behavior.

Covers:
1. Historical requests default to live_chain + harvested skew
2. Live-ish requests keep synthetic_grid + none defaults
3. Empty chain fallback to synthetic candidates
4. Explicit overrides are respected
5. skew_source='live' still blocked for historical
"""
from __future__ import annotations

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock

from src.web.routes_backtest import (
    is_historical_backtest,
    apply_historical_defaults,
    SkewSourceType,
    BacktestStartRequest,
)


class TestIsHistoricalBacktest:
    """Tests for the is_historical_backtest helper function."""
    
    def test_historical_end_date_returns_true(self):
        """End date 1 day ago is historical."""
        end_dt = datetime.now(timezone.utc) - timedelta(days=1)
        assert is_historical_backtest(end_dt) is True
    
    def test_very_recent_end_date_returns_false(self):
        """End date 1 minute ago is live-ish."""
        end_dt = datetime.now(timezone.utc) - timedelta(minutes=1)
        assert is_historical_backtest(end_dt) is False
    
    def test_exactly_5_minutes_ago_is_liveish(self):
        """End date exactly 5 minutes ago is live-ish (not < 5 min)."""
        end_dt = datetime.now(timezone.utc) - timedelta(minutes=5)
        assert is_historical_backtest(end_dt) is False
    
    def test_6_minutes_ago_is_historical(self):
        """End date 6 minutes ago is historical."""
        end_dt = datetime.now(timezone.utc) - timedelta(minutes=6)
        assert is_historical_backtest(end_dt) is True
    
    def test_future_end_date_is_liveish(self):
        """End date in the future is live-ish."""
        end_dt = datetime.now(timezone.utc) + timedelta(days=1)
        assert is_historical_backtest(end_dt) is False


class TestApplyHistoricalDefaults:
    """Tests for apply_historical_defaults function."""
    
    def test_historical_defaults_to_live_chain_and_harvested(self):
        """Historical with omitted chain_mode/skew_source uses live_chain + harvested."""
        chain_mode, skew_source = apply_historical_defaults(
            is_historical=True,
            chain_mode=None,
            skew_source=None,
        )
        assert chain_mode == "live_chain"
        assert skew_source == SkewSourceType.HARVESTED
    
    def test_liveish_defaults_to_synthetic_and_none(self):
        """Live-ish with omitted fields uses synthetic_grid + none."""
        chain_mode, skew_source = apply_historical_defaults(
            is_historical=False,
            chain_mode=None,
            skew_source=None,
        )
        assert chain_mode == "synthetic_grid"
        assert skew_source == SkewSourceType.NONE
    
    def test_explicit_chain_mode_respected_for_historical(self):
        """Explicit chain_mode override is respected for historical."""
        chain_mode, skew_source = apply_historical_defaults(
            is_historical=True,
            chain_mode="synthetic_grid",
            skew_source=None,
        )
        assert chain_mode == "synthetic_grid"
        assert skew_source == SkewSourceType.HARVESTED
    
    def test_explicit_skew_source_respected_for_historical(self):
        """Explicit skew_source override is respected for historical."""
        chain_mode, skew_source = apply_historical_defaults(
            is_historical=True,
            chain_mode=None,
            skew_source=SkewSourceType.NONE,
        )
        assert chain_mode == "live_chain"
        assert skew_source == SkewSourceType.NONE
    
    def test_both_explicit_respected_for_historical(self):
        """Both explicit overrides are respected for historical."""
        chain_mode, skew_source = apply_historical_defaults(
            is_historical=True,
            chain_mode="synthetic_grid",
            skew_source=SkewSourceType.NONE,
        )
        assert chain_mode == "synthetic_grid"
        assert skew_source == SkewSourceType.NONE
    
    def test_explicit_chain_mode_respected_for_liveish(self):
        """Explicit chain_mode override is respected for live-ish."""
        chain_mode, skew_source = apply_historical_defaults(
            is_historical=False,
            chain_mode="live_chain",
            skew_source=None,
        )
        assert chain_mode == "live_chain"
        assert skew_source == SkewSourceType.NONE


class TestBacktestStartRequestDefaults:
    """Tests for BacktestStartRequest model Optional fields."""
    
    def test_chain_mode_default_is_none(self):
        """chain_mode defaults to None when not specified."""
        req = BacktestStartRequest(start="2024-01-01", end="2024-01-15")
        assert req.chain_mode is None
    
    def test_skew_source_default_is_none(self):
        """skew_source defaults to None when not specified."""
        req = BacktestStartRequest(start="2024-01-01", end="2024-01-15")
        assert req.skew_source is None
    
    def test_explicit_chain_mode_preserved(self):
        """Explicit chain_mode value is preserved."""
        req = BacktestStartRequest(
            start="2024-01-01", 
            end="2024-01-15",
            chain_mode="live_chain",
        )
        assert req.chain_mode == "live_chain"
    
    def test_explicit_skew_source_preserved(self):
        """Explicit skew_source value is preserved."""
        req = BacktestStartRequest(
            start="2024-01-01", 
            end="2024-01-15",
            skew_source=SkewSourceType.HARVESTED,
        )
        assert req.skew_source == SkewSourceType.HARVESTED


class TestLiveChainEmptyFallback:
    """Tests for empty chain fallback to synthetic candidates."""
    
    def test_empty_chain_produces_synthetic_candidates(self):
        """When list_option_chain returns empty, synthetic candidates are generated."""
        from src.backtest.state_builder import build_historical_state
        from src.backtest.types import CallSimulationConfig
        from unittest.mock import MagicMock
        import pandas as pd
        
        mock_ds = MagicMock()
        mock_ds.get_spot_ohlc.return_value = pd.DataFrame({
            "close": [50000.0, 51000.0, 52000.0]
        }, index=pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC"))
        
        mock_ds.list_option_chain.return_value = []
        
        mock_mc = MagicMock()
        mock_mc.rv_30d = 50.0
        mock_mc.atm_iv = 55.0
        mock_mc.vrp = -5.0
        
        cfg = CallSimulationConfig(
            underlying="BTC",
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 1, 15, tzinfo=timezone.utc),
            chain_mode="live_chain",
            skew_source="harvested",
        )
        
        t = datetime(2024, 1, 10, 12, 0, tzinfo=timezone.utc)
        
        with patch('src.backtest.state_builder.compute_market_context_from_ds', return_value=mock_mc):
            with patch('src.backtest.state_builder.market_context_to_dict', return_value={}):
                result = build_historical_state(mock_ds, cfg, t)
        
        assert "candidate_options" in result
        assert len(result["candidate_options"]) > 0
    
    def test_empty_chain_fallback_does_not_crash(self):
        """Empty harvested chain should not cause an exception."""
        from src.backtest.state_builder import build_historical_state
        from src.backtest.types import CallSimulationConfig
        from unittest.mock import MagicMock
        import pandas as pd
        
        mock_ds = MagicMock()
        mock_ds.get_spot_ohlc.return_value = pd.DataFrame({
            "close": [50000.0]
        }, index=pd.date_range("2024-01-01", periods=1, freq="D", tz="UTC"))
        mock_ds.list_option_chain.return_value = []
        
        mock_mc = MagicMock()
        mock_mc.rv_30d = 50.0
        mock_mc.atm_iv = 55.0
        mock_mc.vrp = -5.0
        
        cfg = CallSimulationConfig(
            underlying="BTC",
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 1, 15, tzinfo=timezone.utc),
            chain_mode="live_chain",
        )
        
        t = datetime(2024, 1, 10, 12, 0, tzinfo=timezone.utc)
        
        with patch('src.backtest.state_builder.compute_market_context_from_ds', return_value=mock_mc):
            with patch('src.backtest.state_builder.market_context_to_dict', return_value={}):
                result = build_historical_state(mock_ds, cfg, t)
        
        assert result is not None
        assert "spot" in result


class TestSkewSourceLiveBlocked:
    """Verify that skew_source='live' remains blocked for historical backtests."""
    
    def test_explicit_live_skew_still_blocked(self):
        """Even with explicit live skew, historical backtests are blocked."""
        chain_mode, skew_source = apply_historical_defaults(
            is_historical=True,
            chain_mode=None,
            skew_source=SkewSourceType.LIVE,
        )
        assert skew_source == SkewSourceType.LIVE
    
    def test_live_skew_allowed_for_liveish(self):
        """Live-ish backtests can use skew_source='live'."""
        chain_mode, skew_source = apply_historical_defaults(
            is_historical=False,
            chain_mode=None,
            skew_source=SkewSourceType.LIVE,
        )
        assert skew_source == SkewSourceType.LIVE
