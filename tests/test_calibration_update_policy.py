"""
Unit tests for the calibration update policy module.

Tests cover:
- CalibrationUpdatePolicy configuration
- should_apply_update decision logic
- get_smoothed_multipliers EWMA computation
- History storage read/write
"""
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List
from unittest.mock import patch

import pytest
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.calibration_update_policy import (
    CalibrationUpdatePolicy,
    CalibrationRunRecord,
    CurrentAppliedMultipliers,
    BandMultiplier,
    UpdateDecision,
    should_apply_update,
    get_smoothed_multipliers,
    record_calibration_result,
    load_recent_calibration_history,
    _ewma,
    CALIBRATION_RUNS_DIR,
)


class TestCalibrationUpdatePolicy:
    """Tests for CalibrationUpdatePolicy configuration."""
    
    def test_default_policy_values(self):
        """Test default policy has sensible values."""
        policy = CalibrationUpdatePolicy()
        
        assert policy.min_delta_global == 0.03
        assert policy.min_delta_band == 0.03
        assert policy.min_sample_size == 50
        assert policy.min_vega_sum == 100.0
        assert policy.smoothing_window_days == 14
        assert policy.ewma_alpha == 0.3
    
    def test_custom_policy_values(self):
        """Test custom policy configuration."""
        policy = CalibrationUpdatePolicy(
            min_delta_global=0.05,
            min_sample_size=100,
            smoothing_window_days=30,
        )
        
        assert policy.min_delta_global == 0.05
        assert policy.min_sample_size == 100
        assert policy.smoothing_window_days == 30
    
    def test_policy_to_dict(self):
        """Test policy serialization."""
        policy = CalibrationUpdatePolicy()
        d = policy.to_dict()
        
        assert "min_delta_global" in d
        assert "min_sample_size" in d
        assert "smoothing_window_days" in d


class TestShouldApplyUpdate:
    """Tests for should_apply_update decision logic."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.policy = CalibrationUpdatePolicy(
            min_delta_global=0.03,
            min_delta_band=0.03,
            min_sample_size=50,
            min_vega_sum=100.0,
        )
        
        self.current = CurrentAppliedMultipliers(
            global_multiplier=1.0,
            band_multipliers=[
                BandMultiplier(name="weekly", min_dte=3, max_dte=10, iv_multiplier=1.0),
            ],
        )
    
    def test_large_global_delta_applies(self):
        """Test that large global delta triggers apply."""
        decision = should_apply_update(
            current_applied=self.current,
            smoothed_global=1.05,  # Delta = 0.05 > 0.03
            smoothed_bands=None,
            policy=self.policy,
            sample_size=100,
            vega_sum=200.0,
        )
        
        assert decision.should_apply is True
        assert "global" in decision.reason.lower()
    
    def test_small_global_delta_rejects(self):
        """Test that small global delta is rejected."""
        decision = should_apply_update(
            current_applied=self.current,
            smoothed_global=1.02,  # Delta = 0.02 < 0.03
            smoothed_bands=None,
            policy=self.policy,
            sample_size=100,
            vega_sum=200.0,
        )
        
        assert decision.should_apply is False
        assert "too small" in decision.reason.lower()
    
    def test_insufficient_sample_size_rejects(self):
        """Test that insufficient sample size is rejected."""
        decision = should_apply_update(
            current_applied=self.current,
            smoothed_global=1.10,  # Large delta
            smoothed_bands=None,
            policy=self.policy,
            sample_size=30,  # Below min_sample_size=50
            vega_sum=200.0,
        )
        
        assert decision.should_apply is False
        assert "sample" in decision.reason.lower()
    
    def test_insufficient_vega_rejects(self):
        """Test that insufficient vega sum is rejected."""
        decision = should_apply_update(
            current_applied=self.current,
            smoothed_global=1.10,
            smoothed_bands=None,
            policy=self.policy,
            sample_size=100,
            vega_sum=50.0,  # Below min_vega_sum=100
        )
        
        assert decision.should_apply is False
        assert "vega" in decision.reason.lower()
    
    def test_band_delta_applies(self):
        """Test that large band delta triggers apply."""
        smoothed_bands = [
            BandMultiplier(name="weekly", min_dte=3, max_dte=10, iv_multiplier=1.05),
        ]
        
        decision = should_apply_update(
            current_applied=self.current,
            smoothed_global=1.01,  # Small global delta
            smoothed_bands=smoothed_bands,
            policy=self.policy,
            sample_size=100,
            vega_sum=200.0,
        )
        
        assert decision.should_apply is True
        assert "band" in decision.reason.lower()


class TestGetSmoothedMultipliers:
    """Tests for EWMA smoothing computation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.policy = CalibrationUpdatePolicy(
            smoothing_window_days=14,
            ewma_alpha=0.3,
        )
    
    def test_empty_history_returns_current(self):
        """Test that empty history returns current recommended value."""
        smoothed_global, smoothed_bands = get_smoothed_multipliers(
            history=[],
            current_recommended=1.05,
            current_recommended_bands=None,
            policy=self.policy,
        )
        
        assert smoothed_global == 1.05
        assert smoothed_bands is None
    
    def test_single_history_point_smooths(self):
        """Test smoothing with one history point."""
        now = datetime.now(timezone.utc)
        history = [
            CalibrationRunRecord(
                timestamp=now - timedelta(days=1),
                underlying="BTC",
                source="live",
                recommended_iv_multiplier=1.0,
                sample_size=100,
                vega_sum=200.0,
            ),
        ]
        
        smoothed_global, _ = get_smoothed_multipliers(
            history=history,
            current_recommended=1.10,
            current_recommended_bands=None,
            policy=self.policy,
        )
        
        assert 1.0 < smoothed_global < 1.10
    
    def test_old_history_excluded(self):
        """Test that history beyond window is excluded."""
        now = datetime.now(timezone.utc)
        history = [
            CalibrationRunRecord(
                timestamp=now - timedelta(days=30),  # Beyond 14-day window
                underlying="BTC",
                source="live",
                recommended_iv_multiplier=0.5,  # Very different value
                sample_size=100,
                vega_sum=200.0,
            ),
        ]
        
        smoothed_global, _ = get_smoothed_multipliers(
            history=history,
            current_recommended=1.10,
            current_recommended_bands=None,
            policy=self.policy,
        )
        
        assert smoothed_global == 1.10


class TestEwma:
    """Tests for EWMA helper function."""
    
    def test_single_value(self):
        """Test EWMA with single value."""
        result = _ewma([1.0], alpha=0.3)
        assert result == 1.0
    
    def test_two_values(self):
        """Test EWMA with two values."""
        result = _ewma([1.0, 0.5], alpha=0.3)
        assert 0.5 < result < 1.0
    
    def test_empty_list_returns_default(self):
        """Test EWMA with empty list."""
        result = _ewma([], alpha=0.3)
        assert result == 1.0


class TestHistoryStorage:
    """Tests for file-based history storage."""
    
    def setup_method(self):
        """Set up temporary directory for tests."""
        self.temp_dir = tempfile.mkdtemp()
        self.original_dir = CALIBRATION_RUNS_DIR
    
    def teardown_method(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_record_and_load(self):
        """Test recording and loading calibration result."""
        with patch('src.calibration_update_policy.CALIBRATION_RUNS_DIR', Path(self.temp_dir)):
            record = record_calibration_result(
                underlying="BTC",
                source="live",
                recommended_iv_multiplier=1.05,
                recommended_band_multipliers=None,
                sample_size=100,
                vega_sum=200.0,
                applied=True,
                applied_reason="Test apply",
            )
            
            assert record.underlying == "BTC"
            assert record.recommended_iv_multiplier == 1.05
            assert record.applied is True
            
            json_files = list(Path(self.temp_dir).glob("*.json"))
            assert len(json_files) == 1
            
            with open(json_files[0]) as f:
                data = json.load(f)
            assert data["underlying"] == "BTC"
            assert data["applied"] is True


class TestUpdateDecision:
    """Tests for UpdateDecision dataclass."""
    
    def test_creation(self):
        """Test UpdateDecision creation."""
        decision = UpdateDecision(
            should_apply=True,
            reason="Test reason",
            details={"key": "value"},
        )
        
        assert decision.should_apply is True
        assert decision.reason == "Test reason"
        assert decision.details["key"] == "value"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
