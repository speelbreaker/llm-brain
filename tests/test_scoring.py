"""Tests for src/scoring/candidates.py - candidate option scoring."""
import pytest
from src.scoring.candidates import score_option_candidate


class TestScoreOptionCandidate:
    """Tests for the score_option_candidate function."""

    def test_returns_float(self):
        """Function should return a float score."""
        features = {
            "delta": 0.25,
            "dte": 7,
            "ivrv": 1.2,
            "premium_usd": 500.0,
            "premium_pct": 0.5,
        }
        result = score_option_candidate(features, profile="backtest")
        assert isinstance(result, float)

    def test_backtest_profile_basic(self):
        """Backtest profile should score candidates in [0, 10] range."""
        features = {
            "delta": 0.25,
            "dte": 7,
            "ivrv": 1.3,
            "premium_pct": 0.8,
            "otm_pct": 5.0,
            "regime": 0,
            "return_7d_pct": 0.0,
            "return_30d_pct": 0.0,
            "realized_vol_7d": 0.5,
            "pct_from_200d_ma": 0.0,
        }
        score = score_option_candidate(features, profile="backtest")
        assert 0.0 <= score <= 10.0

    def test_live_profile_returns_score(self):
        """Live profile should return a numeric score."""
        features = {
            "delta": 0.25,
            "dte": 7,
            "ivrv": 1.2,
            "premium_usd": 500.0,
        }
        score = score_option_candidate(features, profile="live")
        assert isinstance(score, float)

    def test_conservative_profile(self):
        """Conservative profile should work with valid features."""
        features = {
            "delta": 0.20,
            "dte": 7,
            "ivrv": 1.1,
            "premium_usd": 300.0,
        }
        score = score_option_candidate(features, profile="conservative")
        assert isinstance(score, float)

    def test_aggressive_profile(self):
        """Aggressive profile should work with valid features."""
        features = {
            "delta": 0.40,
            "dte": 7,
            "ivrv": 1.1,
            "premium_usd": 800.0,
        }
        score = score_option_candidate(features, profile="aggressive")
        assert isinstance(score, float)

    def test_higher_ivrv_improves_backtest_score(self):
        """Higher IVRV should result in higher backtest score."""
        base_features = {
            "delta": 0.25,
            "dte": 7,
            "premium_pct": 0.5,
            "otm_pct": 5.0,
            "regime": 0,
            "return_7d_pct": 0.0,
            "return_30d_pct": 0.0,
            "realized_vol_7d": 0.5,
            "pct_from_200d_ma": 0.0,
        }
        
        low_ivrv = {**base_features, "ivrv": 1.0}
        high_ivrv = {**base_features, "ivrv": 1.4}
        
        score_low = score_option_candidate(low_ivrv, profile="backtest")
        score_high = score_option_candidate(high_ivrv, profile="backtest")
        
        assert score_high > score_low

    def test_missing_features_use_defaults(self):
        """Function should handle missing features gracefully."""
        minimal_features = {"delta": 0.25}
        score = score_option_candidate(minimal_features, profile="backtest")
        assert isinstance(score, float)

    def test_config_overrides(self):
        """Config overrides should be applied."""
        features = {
            "delta": 0.30,
            "dte": 7,
            "ivrv": 1.2,
            "premium_pct": 0.5,
        }
        score = score_option_candidate(
            features,
            profile="backtest",
            config_overrides={"target_delta": 0.30},
        )
        assert isinstance(score, float)
