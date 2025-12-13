"""Unit tests for hybrid synthetic mode implementation."""

import pytest
from datetime import datetime, timezone, timedelta

from src.backtest.types import (
    CallSimulationConfig,
    OptionSnapshot,
    SigmaMode,
    ChainMode,
)
from src.backtest.pricing import get_atm_iv_from_chain


def make_config(**kwargs) -> CallSimulationConfig:
    """Create a CallSimulationConfig with required defaults."""
    defaults = {
        "underlying": "BTC",
        "start": datetime(2024, 1, 1, tzinfo=timezone.utc),
        "end": datetime(2024, 2, 1, tzinfo=timezone.utc),
        "timeframe": "1d",
        "decision_interval_bars": 24,
        "initial_spot_position": 1.0,
        "contract_size": 1.0,
        "fee_rate": 0.0005,
    }
    defaults.update(kwargs)
    return CallSimulationConfig(**defaults)


class TestSyntheticModeEnums:
    """Test SigmaMode and ChainMode type literals."""

    def test_sigma_mode_values(self):
        """Test that SigmaMode accepts valid values."""
        valid_modes: list[SigmaMode] = [
            "rv_x_multiplier",
            "atm_iv_x_multiplier", 
            "mark_iv_x_multiplier",
        ]
        for mode in valid_modes:
            config = make_config(sigma_mode=mode)
            assert config.sigma_mode == mode

    def test_chain_mode_values(self):
        """Test that ChainMode accepts valid values."""
        valid_modes: list[ChainMode] = [
            "synthetic_grid",
            "live_chain",
        ]
        for mode in valid_modes:
            config = make_config(chain_mode=mode)
            assert config.chain_mode == mode

    def test_default_modes(self):
        """Test default values for sigma_mode and chain_mode."""
        config = make_config()
        assert config.sigma_mode == "rv_x_multiplier"
        assert config.chain_mode == "synthetic_grid"


class TestSyntheticModePresets:
    """Test UI preset mappings for synthetic modes."""

    def test_pure_synthetic_preset(self):
        """Test Pure Synthetic preset configuration."""
        config = make_config(
            sigma_mode="rv_x_multiplier",
            chain_mode="synthetic_grid",
        )
        assert config.sigma_mode == "rv_x_multiplier"
        assert config.chain_mode == "synthetic_grid"

    def test_live_iv_synthetic_preset(self):
        """Test Live IV Synthetic Grid preset configuration."""
        config = make_config(
            sigma_mode="atm_iv_x_multiplier",
            chain_mode="synthetic_grid",
        )
        assert config.sigma_mode == "atm_iv_x_multiplier"
        assert config.chain_mode == "synthetic_grid"

    def test_live_chain_preset(self):
        """Test Live Chain + Live IV preset configuration."""
        config = make_config(
            sigma_mode="mark_iv_x_multiplier",
            chain_mode="live_chain",
        )
        assert config.sigma_mode == "mark_iv_x_multiplier"
        assert config.chain_mode == "live_chain"


class TestGetAtmIvFromChain:
    """Test get_atm_iv_from_chain timestamp handling."""

    def test_uses_as_of_timestamp_not_now(self):
        """Test that DTE calculation uses provided as_of, not datetime.now()."""
        # Create a historical timestamp (30 days ago)
        historical_time = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        
        # Create an option expiring 7 days after the historical time
        expiry = historical_time + timedelta(days=7)
        
        option = OptionSnapshot(
            instrument_name="BTC-8JUN24-70000-C",
            underlying="BTC",
            kind="call",
            strike=70000,
            expiry=expiry,
            delta=0.45,
            iv=0.55,
            mark_price=1000.0,
            settlement_ccy="BTC",
            margin_type="PM",
        )
        
        # When using the historical timestamp, option should be ~7 DTE
        # and match our target_dte=7 with tolerance=2
        result = get_atm_iv_from_chain(
            option_chain=[option],
            spot=68000,
            target_dte=7,
            dte_tolerance=2,
            as_of=historical_time,
        )
        
        # Should find the option since DTE matches target
        assert result is not None
        assert result == 0.55

    def test_fallback_to_now_when_as_of_is_none(self):
        """Test that as_of=None falls back to datetime.now()."""
        # Create an option with future expiry
        future_expiry = datetime.now(timezone.utc) + timedelta(days=7)
        
        option = OptionSnapshot(
            instrument_name="BTC-FUTURE-70000-C",
            underlying="BTC",
            kind="call",
            strike=70000,
            expiry=future_expiry,
            delta=0.48,
            iv=0.60,
            mark_price=1200.0,
            settlement_ccy="BTC",
            margin_type="PM",
        )
        
        # Without as_of parameter, should use current time
        result = get_atm_iv_from_chain(
            option_chain=[option],
            spot=68000,
            target_dte=7,
            dte_tolerance=2,
        )
        
        # Should find the option since DTE is ~7 from now
        assert result is not None
        assert result == 0.60


class TestLiveChainMarkPreservation:
    """Test that live chain mode preserves Deribit mark values."""

    def test_multiplier_1_preserves_original_marks(self):
        """When multiplier=1.0, original Deribit marks should be preserved."""
        from src.backtest.state_builder import _generate_live_chain_candidates, _filter_option_chain
        from src.backtest.pricing import bs_call_price
        from unittest.mock import MagicMock
        
        # Create config with multiplier=1.0 (Live Chain + Live IV mode)
        cfg = make_config(
            sigma_mode="mark_iv_x_multiplier",
            chain_mode="live_chain",
            synthetic_iv_multiplier=1.0,
            delta_min=0.10,
            delta_max=0.50,
            min_dte=1,
            max_dte=30,
        )
        
        t = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        expiry = t + timedelta(days=7)
        original_mark_price = 1500.0
        original_iv = 0.55
        original_delta = 0.35
        
        # Create a mock option with original Deribit values
        mock_option = OptionSnapshot(
            instrument_name="BTC-8JUN24-70000-C",
            underlying="BTC",
            kind="call",
            strike=70000,
            expiry=expiry,
            delta=original_delta,
            iv=original_iv,
            mark_price=original_mark_price,
            settlement_ccy="BTC",
            margin_type="PM",
        )
        
        # Mock the data source
        ds = MagicMock()
        ds.list_option_chain.return_value = [mock_option]
        
        # Generate candidates
        spot = 68000
        spot_history = [(t - timedelta(days=i), 67000 + i * 100) for i in range(30)]
        result = _generate_live_chain_candidates(ds, spot, t, cfg, spot_history)
        
        # With multiplier=1.0, original values should be preserved
        assert len(result) == 1
        assert result[0].mark_price == original_mark_price
        assert result[0].iv == original_iv
        assert result[0].delta == original_delta

    def test_multiplier_not_1_recalculates_marks(self):
        """When multiplier != 1.0, marks should be recalculated with scaled IV."""
        from src.backtest.state_builder import _generate_live_chain_candidates
        from src.backtest.pricing import bs_call_price, bs_call_delta
        from unittest.mock import MagicMock
        
        # Create config with multiplier=1.2 (stress scenario)
        multiplier = 1.2
        cfg = make_config(
            sigma_mode="mark_iv_x_multiplier",
            chain_mode="live_chain",
            synthetic_iv_multiplier=multiplier,
            delta_min=0.10,
            delta_max=0.50,
            min_dte=1,
            max_dte=30,
        )
        
        t = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        expiry = t + timedelta(days=7)
        original_iv = 0.55
        
        mock_option = OptionSnapshot(
            instrument_name="BTC-8JUN24-70000-C",
            underlying="BTC",
            kind="call",
            strike=70000,
            expiry=expiry,
            delta=0.35,
            iv=original_iv,
            mark_price=1500.0,
            settlement_ccy="BTC",
            margin_type="PM",
        )
        
        ds = MagicMock()
        ds.list_option_chain.return_value = [mock_option]
        
        spot = 68000
        spot_history = [(t - timedelta(days=i), 67000 + i * 100) for i in range(30)]
        result = _generate_live_chain_candidates(ds, spot, t, cfg, spot_history)
        
        # With multiplier != 1.0, IV should be scaled and marks recalculated
        assert len(result) == 1
        expected_scaled_iv = original_iv * multiplier
        assert abs(result[0].iv - expected_scaled_iv) < 1e-6
        
        # Mark price should be recalculated using BS with scaled IV
        dte = 7
        t_years = dte / 365.0
        expected_price = bs_call_price(spot, 70000, t_years, expected_scaled_iv, cfg.risk_free_rate)
        assert abs(result[0].mark_price - expected_price) < 0.01


class TestLiveChainDebugSamples:
    """Test that debug sample collection captures Deribit mark vs BS price comparison."""

    def test_debug_sample_collects_comparison_data(self):
        """Debug samples should capture Deribit mark price and BS-calculated price."""
        from src.backtest.state_builder import _generate_live_chain_candidates
        from src.backtest.types import LiveChainDebugSample
        from src.backtest.pricing import bs_call_price
        from unittest.mock import MagicMock
        
        cfg = make_config(
            sigma_mode="mark_iv_x_multiplier",
            chain_mode="live_chain",
            synthetic_iv_multiplier=1.0,
            delta_min=0.10,
            delta_max=0.50,
            min_dte=1,
            max_dte=30,
        )
        
        t = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        expiry = t + timedelta(days=7)
        original_mark_price = 1500.0
        original_iv = 0.55
        original_delta = 0.35
        spot = 68000
        
        mock_option = OptionSnapshot(
            instrument_name="BTC-8JUN24-70000-C",
            underlying="BTC",
            kind="call",
            strike=70000,
            expiry=expiry,
            delta=original_delta,
            iv=original_iv,
            mark_price=original_mark_price,
            settlement_ccy="BTC",
            margin_type="inverse",
        )
        
        ds = MagicMock()
        ds.list_option_chain.return_value = [mock_option]
        
        spot_history = [(t - timedelta(days=i), 67000 + i * 100) for i in range(30)]
        
        candidates, debug_samples = _generate_live_chain_candidates(
            ds, spot, t, cfg, spot_history, collect_debug_samples=True
        )
        
        assert len(candidates) == 1
        assert candidates[0].mark_price == original_mark_price
        
        assert len(debug_samples) >= 1
        sample = debug_samples[0]
        assert isinstance(sample, LiveChainDebugSample)
        assert sample.instrument_name == "BTC-8JUN24-70000-C"
        assert sample.deribit_mark_price == original_mark_price
        assert sample.dte_days == 7.0
        assert sample.strike == 70000
        
        expected_bs_price = bs_call_price(spot, 70000, 7.0 / 365.0, original_iv, cfg.risk_free_rate)
        assert abs(sample.engine_price - expected_bs_price) < 0.01
        
        expected_diff_pct = abs(expected_bs_price - original_mark_price) / original_mark_price * 100.0
        assert abs(sample.abs_diff_pct - expected_diff_pct) < 0.01

    def test_debug_sample_to_dict(self):
        """Test that LiveChainDebugSample.to_dict() returns correct format."""
        from src.backtest.types import LiveChainDebugSample
        
        sample = LiveChainDebugSample(
            instrument_name="BTC-TEST-70000-C",
            dte_days=7.123456,
            strike=70000,
            deribit_mark_price=1500.123456,
            engine_price=1450.654321,
            abs_diff_pct=3.295,
        )
        
        d = sample.to_dict()
        assert d["instrument_name"] == "BTC-TEST-70000-C"
        assert d["dte_days"] == 7.12
        assert d["strike"] == 70000
        assert d["deribit_mark_price"] == 1500.123456
        assert d["engine_price"] == 1450.654321
        assert d["abs_diff_pct"] == 3.295
