"""
Tests for position reconciliation enforcement.
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.ops.reconciliation_state import (
    ReconciliationResult,
    ReconciliationStatus,
    PositionMismatch,
    get_reconciliation_status,
    set_reconciliation_status,
    set_trading_blocked,
    is_trading_blocked_by_reconciliation,
    clear_reconciliation_block,
    build_mismatches_from_diff,
    reset_reconciliation_state,
)


@pytest.fixture(autouse=True)
def reset_state():
    """Reset reconciliation state before each test."""
    reset_reconciliation_state()
    yield
    reset_reconciliation_state()


class TestReconciliationState:
    """Tests for reconciliation state tracking."""
    
    def test_initial_state_is_pending(self):
        """Initial state should be PENDING and not blocking."""
        status = get_reconciliation_status()
        
        assert status.result == ReconciliationResult.PENDING
        assert status.is_clean is True
        assert status.trading_blocked is False
    
    def test_set_clean_status(self):
        """Setting clean status should update state."""
        status = set_reconciliation_status(
            result=ReconciliationResult.CLEAN,
            is_clean=True,
            mismatches=[],
            exchange_count=5,
            local_count=5,
            action_taken="none",
            trading_blocked=False,
        )
        
        assert status.result == ReconciliationResult.CLEAN
        assert status.is_clean is True
        assert status.exchange_count == 5
        assert status.local_count == 5
        assert status.trading_blocked is False
        assert status.last_run_time is not None
    
    def test_set_divergent_status(self):
        """Setting divergent status should update state."""
        mismatches = [
            PositionMismatch(
                symbol="BTC-25DEC25-100000-C",
                mismatch_type="size_mismatch",
                local_size=1.0,
                exchange_size=2.0,
            ),
        ]
        
        status = set_reconciliation_status(
            result=ReconciliationResult.DIVERGENT,
            is_clean=False,
            mismatches=mismatches,
            exchange_count=3,
            local_count=2,
            action_taken="halt",
            trading_blocked=True,
        )
        
        assert status.result == ReconciliationResult.DIVERGENT
        assert status.is_clean is False
        assert len(status.mismatches) == 1
        assert status.trading_blocked is True
    
    def test_trading_blocked_flag(self):
        """Trading blocked flag should work correctly."""
        assert is_trading_blocked_by_reconciliation() is False
        
        set_trading_blocked(True)
        assert is_trading_blocked_by_reconciliation() is True
        
        clear_reconciliation_block()
        assert is_trading_blocked_by_reconciliation() is False
    
    def test_to_dict_serialization(self):
        """Status should serialize to dict correctly."""
        set_reconciliation_status(
            result=ReconciliationResult.CLEAN,
            is_clean=True,
            exchange_count=3,
            local_count=3,
        )
        
        status = get_reconciliation_status()
        d = status.to_dict()
        
        assert isinstance(d, dict)
        assert d["result"] == "clean"
        assert d["is_clean"] is True
        assert "last_run_time" in d


class TestBuildMismatchesFromDiff:
    """Tests for building mismatches from diff dictionary."""
    
    def test_empty_diff(self):
        """Empty diff should return empty list."""
        mismatches = build_mismatches_from_diff({})
        assert mismatches == []
    
    def test_missing_in_local(self):
        """Should create mismatch for untracked positions."""
        diff = {
            "missing_in_local": ["BTC-25DEC25-100000-C", "BTC-25DEC25-110000-C"],
        }
        
        mismatches = build_mismatches_from_diff(diff)
        
        assert len(mismatches) == 2
        assert mismatches[0].symbol == "BTC-25DEC25-100000-C"
        assert mismatches[0].mismatch_type == "untracked_on_exchange"
    
    def test_missing_in_exchange(self):
        """Should create mismatch for missing on exchange."""
        diff = {
            "missing_in_exchange": ["BTC-25DEC25-90000-C"],
        }
        
        mismatches = build_mismatches_from_diff(diff)
        
        assert len(mismatches) == 1
        assert mismatches[0].symbol == "BTC-25DEC25-90000-C"
        assert mismatches[0].mismatch_type == "missing_on_exchange"
    
    def test_size_mismatches(self):
        """Should create mismatch for size differences."""
        diff = {
            "size_mismatches": [
                ("BTC-25DEC25-100000-C", 1.0, 2.0),
                ("BTC-25DEC25-110000-C", 0.5, 0.75),
            ],
        }
        
        mismatches = build_mismatches_from_diff(diff)
        
        assert len(mismatches) == 2
        assert mismatches[0].symbol == "BTC-25DEC25-100000-C"
        assert mismatches[0].mismatch_type == "size_mismatch"
        assert mismatches[0].local_size == 1.0
        assert mismatches[0].exchange_size == 2.0


class TestPositionMismatch:
    """Tests for PositionMismatch dataclass."""
    
    def test_to_dict(self):
        """Should serialize to dict correctly."""
        mismatch = PositionMismatch(
            symbol="BTC-25DEC25-100000-C",
            mismatch_type="size_mismatch",
            local_size=1.0,
            exchange_size=2.0,
                side="SHORT",
            diff_usd=1500.50,
        )
        
        d = mismatch.to_dict()
        
        assert d["symbol"] == "BTC-25DEC25-100000-C"
        assert d["mismatch_type"] == "size_mismatch"
        assert d["local_size"] == 1.0
        assert d["exchange_size"] == 2.0
        assert d["diff_usd"] == 1500.50


class TestReconciliationEnforcement:
    """Tests for reconciliation enforcement in trading flow."""
    
    def test_halt_blocks_trading(self):
        """When divergent with halt action, trading should be blocked."""
        set_reconciliation_status(
            result=ReconciliationResult.DIVERGENT,
            is_clean=False,
            mismatches=[
                PositionMismatch(symbol="TEST", mismatch_type="size_mismatch"),
            ],
            action_taken="halt",
            trading_blocked=True,
        )
        
        assert is_trading_blocked_by_reconciliation() is True
    
    def test_auto_heal_allows_trading(self):
        """After auto-heal, trading should be allowed."""
        # First set divergent
        set_reconciliation_status(
            result=ReconciliationResult.DIVERGENT,
            is_clean=False,
            mismatches=[
                PositionMismatch(symbol="TEST", mismatch_type="size_mismatch"),
            ],
            action_taken="halt",
            trading_blocked=True,
        )
        
        assert is_trading_blocked_by_reconciliation() is True
        
        # Then simulate auto-heal
        set_reconciliation_status(
            result=ReconciliationResult.CLEAN,
            is_clean=True,
            mismatches=[],
            action_taken="auto_heal",
            trading_blocked=False,
        )
        
        assert is_trading_blocked_by_reconciliation() is False
    
    def test_manual_clear_allows_trading(self):
        """Manual clear should allow trading to resume."""
        set_reconciliation_status(
            result=ReconciliationResult.DIVERGENT,
            is_clean=False,
            action_taken="halt",
            trading_blocked=True,
        )
        
        assert is_trading_blocked_by_reconciliation() is True
        
        clear_reconciliation_block()
        
        assert is_trading_blocked_by_reconciliation() is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
