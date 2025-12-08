"""
Tests for position reconciliation logic.

Tests the diff_positions function and related reconciliation helpers.
All tests are pure/no network.
"""
from __future__ import annotations

import pytest
from typing import Dict, Any, List

from src.reconciliation import (
    diff_positions,
    reconcile_positions,
    normalize_symbol,
    PositionReconciliationDiff,
    PositionSizeMismatch,
    get_reconciliation_status,
    format_reconciliation_summary,
)


def _make_exchange_position(
    symbol: str,
    size: float,
    direction: str = "sell",
    underlying: str = "BTC",
) -> Dict[str, Any]:
    """Create a mock exchange position."""
    return {
        "symbol": symbol,
        "instrument_name": symbol,
        "size": size,
        "direction": direction,
        "underlying": underlying,
        "average_price": 0.01,
        "mark_price": 0.012,
        "delta": -0.25,
    }


def _make_tracked_position(
    symbol: str,
    quantity: float,
    side: str = "SHORT",
    underlying: str = "BTC",
) -> Dict[str, Any]:
    """Create a mock tracked (local) position."""
    return {
        "symbol": symbol,
        "quantity": quantity,
        "side": side,
        "underlying": underlying,
        "entry_price": 0.01,
        "mark_price": 0.012,
    }


class TestDiffPositions:
    """Tests for the diff_positions function."""
    
    def test_identical_positions_is_clean(self):
        """Identical tracked vs exchange positions should result in is_clean=True."""
        exchange = [
            _make_exchange_position("BTC-27DEC24-100000-C", 0.1, "sell"),
            _make_exchange_position("ETH-27DEC24-4000-C", 0.5, "sell", "ETH"),
        ]
        tracked = [
            _make_tracked_position("BTC-27DEC24-100000-C", 0.1, "SHORT"),
            _make_tracked_position("ETH-27DEC24-4000-C", 0.5, "SHORT", "ETH"),
        ]
        
        diff = diff_positions(tracked, exchange, tolerance_usd=10.0)
        
        assert diff.is_clean is True
        assert len(diff.untracked_on_exchange) == 0
        assert len(diff.missing_on_exchange) == 0
        assert len(diff.size_mismatches) == 0
    
    def test_empty_positions_is_clean(self):
        """Both empty should be is_clean=True."""
        diff = diff_positions([], [], tolerance_usd=10.0)
        
        assert diff.is_clean is True
        assert diff.exchange_count == 0
        assert diff.local_count == 0
    
    def test_extra_exchange_position_is_untracked(self):
        """Position only on exchange should appear in untracked_on_exchange."""
        exchange = [
            _make_exchange_position("BTC-27DEC24-100000-C", 0.1, "sell"),
        ]
        tracked = []
        
        diff = diff_positions(tracked, exchange, tolerance_usd=10.0)
        
        assert diff.is_clean is False
        assert len(diff.untracked_on_exchange) == 1
        assert diff.untracked_on_exchange[0]["symbol"] == "BTC-27DEC24-100000-C"
        assert len(diff.missing_on_exchange) == 0
    
    def test_extra_tracked_position_is_missing(self):
        """Position only in tracker should appear in missing_on_exchange."""
        exchange = []
        tracked = [
            _make_tracked_position("BTC-27DEC24-100000-C", 0.1, "SHORT"),
        ]
        
        diff = diff_positions(tracked, exchange, tolerance_usd=10.0)
        
        assert diff.is_clean is False
        assert len(diff.untracked_on_exchange) == 0
        assert len(diff.missing_on_exchange) == 1
        assert diff.missing_on_exchange[0]["symbol"] == "BTC-27DEC24-100000-C"
    
    def test_size_mismatch_beyond_tolerance(self):
        """Size difference beyond tolerance should appear in size_mismatches."""
        exchange = [
            _make_exchange_position("BTC-27DEC24-100000-C", 0.2, "sell"),
        ]
        tracked = [
            _make_tracked_position("BTC-27DEC24-100000-C", 0.1, "SHORT"),
        ]
        spot_prices = {"BTC": 100000.0}
        
        diff = diff_positions(tracked, exchange, tolerance_usd=10.0, spot_prices=spot_prices)
        
        assert diff.is_clean is False
        assert len(diff.size_mismatches) == 1
        mismatch = diff.size_mismatches[0]
        assert mismatch.instrument_name == "BTC-27DEC24-100000-C"
        assert mismatch.size_tracker == 0.1
        assert mismatch.size_exchange == 0.2
        assert mismatch.diff_usd > 10.0
    
    def test_size_mismatch_within_tolerance_is_clean(self):
        """Size difference within tolerance should NOT appear in size_mismatches."""
        exchange = [
            _make_exchange_position("BTC-27DEC24-100000-C", 0.10001, "sell"),
        ]
        tracked = [
            _make_tracked_position("BTC-27DEC24-100000-C", 0.1, "SHORT"),
        ]
        spot_prices = {"BTC": 100000.0}
        
        diff = diff_positions(tracked, exchange, tolerance_usd=100.0, spot_prices=spot_prices)
        
        assert diff.is_clean is True
        assert len(diff.size_mismatches) == 0
    
    def test_multiple_issues_detected(self):
        """Multiple issues should all be detected."""
        exchange = [
            _make_exchange_position("BTC-27DEC24-100000-C", 0.1, "sell"),
            _make_exchange_position("BTC-27DEC24-110000-C", 0.3, "sell"),
        ]
        tracked = [
            _make_tracked_position("BTC-27DEC24-100000-C", 0.1, "SHORT"),
            _make_tracked_position("ETH-27DEC24-4000-C", 0.5, "SHORT", "ETH"),
        ]
        
        diff = diff_positions(tracked, exchange, tolerance_usd=10.0)
        
        assert diff.is_clean is False
        assert len(diff.untracked_on_exchange) == 1
        assert len(diff.missing_on_exchange) == 1
        assert diff.exchange_count == 2
        assert diff.local_count == 2
    
    def test_symbol_normalization(self):
        """Symbols should be normalized (uppercase, stripped)."""
        exchange = [
            _make_exchange_position("btc-27dec24-100000-c", 0.1, "sell"),
        ]
        tracked = [
            _make_tracked_position("BTC-27DEC24-100000-C", 0.1, "SHORT"),
        ]
        
        diff = diff_positions(tracked, exchange, tolerance_usd=10.0)
        
        assert diff.is_clean is True


class TestReconcilePositions:
    """Tests for the reconcile_positions function."""
    
    def test_halt_action_returns_original_local(self):
        """With action='halt', should return original local positions unchanged."""
        exchange = [_make_exchange_position("BTC-27DEC24-100000-C", 0.1)]
        local = [_make_tracked_position("ETH-27DEC24-4000-C", 0.5, "SHORT", "ETH")]
        
        new_local, stats = reconcile_positions(exchange, local, action="halt")
        
        assert len(new_local) == 1
        assert new_local[0]["symbol"] == "ETH-27DEC24-4000-C"
        assert stats["divergent"] is True
    
    def test_auto_heal_rebuilds_from_exchange(self):
        """With action='auto_heal', should rebuild from exchange when divergent."""
        exchange = [_make_exchange_position("BTC-27DEC24-100000-C", 0.1)]
        local = [_make_tracked_position("ETH-27DEC24-4000-C", 0.5, "SHORT", "ETH")]
        
        new_local, stats = reconcile_positions(exchange, local, action="auto_heal")
        
        assert len(new_local) == 1
        assert new_local[0]["symbol"] == "BTC-27DEC24-100000-C"
        assert stats["divergent"] is True
    
    def test_in_sync_returns_original(self):
        """When in sync, should return original local positions."""
        exchange = [_make_exchange_position("BTC-27DEC24-100000-C", 0.1)]
        local = [_make_tracked_position("BTC-27DEC24-100000-C", 0.1)]
        
        new_local, stats = reconcile_positions(exchange, local, action="auto_heal")
        
        assert stats["divergent"] is False
        assert stats["is_clean"] is True


class TestPositionReconciliationDiff:
    """Tests for the PositionReconciliationDiff dataclass."""
    
    def test_is_clean_property(self):
        """is_clean should be True only when all lists are empty."""
        diff = PositionReconciliationDiff()
        assert diff.is_clean is True
        
        diff.untracked_on_exchange.append({"symbol": "test"})
        assert diff.is_clean is False
    
    def test_to_dict_method(self):
        """to_dict should produce a serializable dictionary."""
        diff = PositionReconciliationDiff(
            exchange_count=2,
            local_count=1,
            tolerance_usd=10.0,
        )
        diff.untracked_on_exchange.append({"symbol": "BTC-TEST-C"})
        diff.size_mismatches.append(
            PositionSizeMismatch(
                instrument_name="ETH-TEST-C",
                side="SHORT",
                size_tracker=0.1,
                size_exchange=0.2,
                diff_usd=350.0,
            )
        )
        
        result = diff.to_dict()
        
        assert result["is_clean"] is False
        assert result["divergent"] is True
        assert result["exchange_count"] == 2
        assert result["local_count"] == 1
        assert "BTC-TEST-C" in result["untracked_on_exchange"]
        assert len(result["size_mismatches"]) == 1


class TestHelperFunctions:
    """Tests for helper functions."""
    
    def test_normalize_symbol(self):
        """normalize_symbol should uppercase and strip."""
        assert normalize_symbol("btc-test-c") == "BTC-TEST-C"
        assert normalize_symbol("  BTC-TEST-C  ") == "BTC-TEST-C"
        assert normalize_symbol("Btc-TeSt-C") == "BTC-TEST-C"
    
    def test_get_reconciliation_status(self):
        """get_reconciliation_status should return 'clean' or 'out_of_sync'."""
        clean_diff = PositionReconciliationDiff()
        assert get_reconciliation_status(clean_diff) == "clean"
        
        dirty_diff = PositionReconciliationDiff()
        dirty_diff.untracked_on_exchange.append({"symbol": "test"})
        assert get_reconciliation_status(dirty_diff) == "out_of_sync"
    
    def test_format_reconciliation_summary(self):
        """format_reconciliation_summary should produce readable output."""
        stats = {
            "divergent": True,
            "is_clean": False,
            "exchange_count": 2,
            "local_count": 1,
            "missing_in_local": ["BTC-27DEC24-100000-C"],
            "missing_in_exchange": [],
            "size_mismatches": [("ETH-27DEC24-4000-C", 0.1, 0.2)],
        }
        
        summary = format_reconciliation_summary(stats)
        
        assert "DIVERGENT" in summary
        assert "Exchange positions: 2" in summary
        assert "Missing in local" in summary
        assert "BTC-27DEC24-100000-C" in summary
        assert "Size mismatches" in summary


class TestAutoHealEdgeCases:
    """Regression tests for auto_heal edge cases."""
    
    def test_auto_heal_does_not_overwrite_when_clean(self):
        """Auto-heal should NOT overwrite local when diff is clean."""
        local = [{"symbol": "BTC-20DEC24-100000-C", "quantity": 0.1, "side": "SHORT"}]
        exchange = [{"symbol": "BTC-20DEC24-100000-C", "size": 0.1, "direction": "sell"}]
        
        new_local, stats = reconcile_positions(
            exchange_positions=exchange,
            local_positions=local,
            action="auto_heal",
            tolerance_usd=10.0,
        )
        
        assert stats["is_clean"] is True
        assert len(new_local) == 1
        assert new_local[0]["symbol"] == "BTC-20DEC24-100000-C"
        assert "healed_from_exchange" not in new_local[0]
    
    def test_auto_heal_preserves_local_on_clean_diff(self):
        """When diff is clean, auto_heal returns original local positions, not rebuilt ones."""
        local = [
            {"symbol": "BTC-20DEC24-100000-C", "quantity": 0.1, "side": "SHORT", "custom_field": "preserved"}
        ]
        exchange = [{"symbol": "BTC-20DEC24-100000-C", "size": 0.1, "direction": "sell"}]
        
        new_local, stats = reconcile_positions(
            exchange_positions=exchange,
            local_positions=local,
            action="auto_heal",
            tolerance_usd=10.0,
        )
        
        assert stats["is_clean"] is True
        assert new_local[0].get("custom_field") == "preserved"


class TestPositionSizeMismatch:
    """Tests for the PositionSizeMismatch dataclass."""
    
    def test_size_diff_property(self):
        """size_diff should return absolute difference."""
        mismatch = PositionSizeMismatch(
            instrument_name="TEST",
            side="SHORT",
            size_tracker=0.1,
            size_exchange=0.3,
        )
        assert abs(mismatch.size_diff - 0.2) < 1e-9
        
        mismatch2 = PositionSizeMismatch(
            instrument_name="TEST",
            side="SHORT",
            size_tracker=0.5,
            size_exchange=0.1,
        )
        assert abs(mismatch2.size_diff - 0.4) < 1e-9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
