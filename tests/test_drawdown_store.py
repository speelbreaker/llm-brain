"""
Tests for the drawdown store module.
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pytest

from src.ops.drawdown_store import (
    DailyDrawdownState,
    load_daily_drawdown_state,
    save_daily_drawdown_state,
    reset_daily_drawdown_state,
    get_daily_drawdown_status,
    DEFAULT_STATE_FILE,
)


@pytest.fixture
def temp_state_dir(tmp_path):
    """Use a temporary directory for state files."""
    temp_file = tmp_path / "drawdown_state.json"
    with patch.object(
        __import__('src.ops.drawdown_store', fromlist=['_get_state_file_path']),
        'DEFAULT_STATE_FILE',
        temp_file,
    ):
        # Also need to patch the function that uses it
        import src.ops.drawdown_store as module
        original_file = module.DEFAULT_STATE_FILE
        module.DEFAULT_STATE_FILE = temp_file
        yield temp_file
        module.DEFAULT_STATE_FILE = original_file


class TestDailyDrawdownState:
    """Tests for DailyDrawdownState dataclass."""
    
    def test_to_dict(self):
        """Should convert to dict correctly."""
        state = DailyDrawdownState(
            date=date(2025, 12, 22),
            max_equity_usd=100000.0,
        )
        d = state.to_dict()
        
        assert d["date"] == "2025-12-22"
        assert d["max_equity_usd"] == 100000.0
    
    def test_from_dict(self):
        """Should create from dict correctly."""
        d = {
            "date": "2025-12-22",
            "max_equity_usd": 100000.0,
        }
        state = DailyDrawdownState.from_dict(d)
        
        assert state.date == date(2025, 12, 22)
        assert state.max_equity_usd == 100000.0


class TestDrawdownStore:
    """Tests for drawdown store persistence."""
    
    def test_load_nonexistent_file(self, temp_state_dir):
        """Loading from nonexistent file should return None."""
        result = load_daily_drawdown_state()
        assert result is None
    
    def test_save_and_load(self, temp_state_dir):
        """Should save and load state correctly."""
        state = DailyDrawdownState(
            date=date(2025, 12, 22),
            max_equity_usd=50000.0,
        )
        
        success = save_daily_drawdown_state(state)
        assert success is True
        
        loaded = load_daily_drawdown_state()
        assert loaded is not None
        assert loaded.date == state.date
        assert loaded.max_equity_usd == state.max_equity_usd
    
    def test_atomic_write(self, temp_state_dir):
        """Save should use atomic write."""
        state = DailyDrawdownState(
            date=date(2025, 12, 22),
            max_equity_usd=75000.0,
        )
        
        save_daily_drawdown_state(state)
        
        # Verify file exists and has correct content
        assert temp_state_dir.exists()
        with open(temp_state_dir) as f:
            data = json.load(f)
        assert data["max_equity_usd"] == 75000.0
    
    def test_reset_state(self, temp_state_dir):
        """Reset should delete state file."""
        state = DailyDrawdownState(
            date=date(2025, 12, 22),
            max_equity_usd=60000.0,
        )
        save_daily_drawdown_state(state)
        assert temp_state_dir.exists()
        
        success = reset_daily_drawdown_state()
        assert success is True
        assert not temp_state_dir.exists()
    
    def test_reset_nonexistent(self, temp_state_dir):
        """Reset should succeed even if file doesn't exist."""
        assert not temp_state_dir.exists()
        success = reset_daily_drawdown_state()
        assert success is True
    
    def test_get_status_no_state(self, temp_state_dir):
        """Status should indicate no state when file doesn't exist."""
        status = get_daily_drawdown_status()
        
        assert status["has_state"] is False
        assert status["date"] is None
    
    def test_get_status_with_state(self, temp_state_dir):
        """Status should include state info when file exists."""
        from datetime import datetime, timezone
        
        today = datetime.now(timezone.utc).date()
        state = DailyDrawdownState(
            date=today,
            max_equity_usd=80000.0,
        )
        save_daily_drawdown_state(state)
        
        status = get_daily_drawdown_status()
        
        assert status["has_state"] is True
        assert status["date"] == today.isoformat()
        assert status["max_equity_usd"] == 80000.0
        assert status["is_current_day"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

