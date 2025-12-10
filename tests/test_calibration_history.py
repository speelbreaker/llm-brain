"""
Tests for calibration_history database helpers.

Verifies that calibration history entries can be inserted and retrieved correctly.
"""
from __future__ import annotations

import pytest
from datetime import datetime, timezone

from src.db import init_db, get_db_session
from src.db.models_calibration import (
    CalibrationHistory,
    CalibrationHistoryEntry,
    insert_calibration_history,
    get_latest_calibration,
    list_recent_calibrations,
)


@pytest.fixture(scope="module", autouse=True)
def setup_db():
    """Ensure database tables exist before running tests."""
    try:
        init_db()
    except Exception:
        pytest.skip("Database not available")


class TestCalibrationHistory:
    """Tests for calibration history database operations."""
    
    def test_insert_and_get_latest(self):
        """Test inserting a calibration entry and retrieving it."""
        entry = CalibrationHistoryEntry(
            underlying="BTC",
            dte_min=3,
            dte_max=10,
            lookback_days=14,
            multiplier=1.1234,
            mae_pct=7.85,
            num_samples=1243,
        )
        
        row_id = insert_calibration_history(entry)
        assert row_id is not None
        assert row_id > 0
        
        latest = get_latest_calibration("BTC", dte_min=3, dte_max=10)
        assert latest is not None
        assert latest.underlying == "BTC"
        assert latest.dte_min == 3
        assert latest.dte_max == 10
        assert latest.lookback_days == 14
        assert abs(latest.multiplier - 1.1234) < 0.0001
        assert abs(latest.mae_pct - 7.85) < 0.01
        assert latest.num_samples == 1243
        assert latest.created_at is not None
    
    def test_get_latest_returns_most_recent(self):
        """Test that get_latest_calibration returns the most recent entry."""
        entry1 = CalibrationHistoryEntry(
            underlying="ETH",
            dte_min=5,
            dte_max=15,
            lookback_days=7,
            multiplier=0.95,
            mae_pct=5.5,
            num_samples=500,
        )
        insert_calibration_history(entry1)
        
        entry2 = CalibrationHistoryEntry(
            underlying="ETH",
            dte_min=5,
            dte_max=15,
            lookback_days=14,
            multiplier=1.05,
            mae_pct=4.2,
            num_samples=1000,
        )
        insert_calibration_history(entry2)
        
        latest = get_latest_calibration("ETH", dte_min=5, dte_max=15)
        assert latest is not None
        assert abs(latest.multiplier - 1.05) < 0.0001
        assert latest.lookback_days == 14
    
    def test_get_latest_returns_none_for_missing(self):
        """Test that get_latest_calibration returns None for non-existent entries."""
        result = get_latest_calibration("XYZ", dte_min=99, dte_max=999)
        assert result is None
    
    def test_list_recent_calibrations(self):
        """Test listing recent calibration entries."""
        for i in range(3):
            entry = CalibrationHistoryEntry(
                underlying="BTC",
                dte_min=1,
                dte_max=5,
                lookback_days=7 + i,
                multiplier=1.0 + i * 0.1,
                mae_pct=5.0 + i,
                num_samples=100 + i * 100,
            )
            insert_calibration_history(entry)
        
        entries = list_recent_calibrations("BTC", limit=5)
        assert len(entries) >= 3
        
        for entry in entries:
            assert entry.underlying == "BTC"
            assert entry.id is not None
            assert entry.created_at is not None
    
    def test_case_insensitive_underlying(self):
        """Test that underlying is case-insensitive."""
        entry = CalibrationHistoryEntry(
            underlying="btc",
            dte_min=20,
            dte_max=30,
            lookback_days=21,
            multiplier=1.15,
            mae_pct=6.0,
            num_samples=800,
        )
        insert_calibration_history(entry)
        
        latest = get_latest_calibration("BTC", dte_min=20, dte_max=30)
        assert latest is not None
        assert latest.underlying == "BTC"
