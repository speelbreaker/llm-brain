"""
Unit tests for Greg decision logging and execute endpoint safety gates.
"""
from __future__ import annotations

import pytest
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

from src.config import GregTradingMode


class TestGregDecisionLogging:
    """Tests for the GregDecisionLog model and helper functions."""
    
    def test_greg_action_type_enum(self):
        """Test that all action types are defined."""
        from src.db.models_greg_decision import GregActionType
        
        assert GregActionType.OPEN.value == "OPEN"
        assert GregActionType.HEDGE.value == "HEDGE"
        assert GregActionType.TAKE_PROFIT.value == "TAKE_PROFIT"
        assert GregActionType.ASSIGN.value == "ASSIGN"
        assert GregActionType.ROLL.value == "ROLL"
        assert GregActionType.CLOSE.value == "CLOSE"
        assert GregActionType.HOLD.value == "HOLD"
        assert GregActionType.IGNORE.value == "IGNORE"
    
    def test_greg_decision_log_model_fields(self):
        """Test that GregDecisionLog has all expected fields."""
        from src.db.models_greg_decision import GregDecisionLog
        
        columns = GregDecisionLog.__table__.columns
        column_names = [c.name for c in columns]
        
        expected_fields = [
            "id", "timestamp", "underlying", "strategy_type", "position_id",
            "action_type", "mode", "suggested", "executed", "reason",
            "vrp_30d", "chop_factor_7d", "adx_14d", "term_structure_spread",
            "skew_25d", "rsi_14d", "pnl_pct", "pnl_usd", "net_delta",
            "spot_price", "order_ids", "extra_info",
        ]
        
        for field in expected_fields:
            assert field in column_names, f"Missing field: {field}"
    
    @patch("src.db.models_greg_decision.get_db_session")
    def test_log_greg_decision_creates_entry(self, mock_session):
        """Test that log_greg_decision creates the expected entry."""
        from src.db.models_greg_decision import log_greg_decision
        
        mock_ctx = MagicMock()
        mock_session.return_value.__enter__ = MagicMock(return_value=mock_ctx)
        mock_session.return_value.__exit__ = MagicMock(return_value=False)
        
        result = log_greg_decision(
            underlying="BTC",
            strategy_type="STRATEGY_A_STRADDLE",
            position_id="test:123",
            action_type="HEDGE",
            mode="advice_only",
            suggested=True,
            executed=False,
            reason="Delta exceeded threshold",
            pnl_pct=0.15,
        )
        
        mock_ctx.add.assert_called_once()
        mock_ctx.commit.assert_called_once()


class TestExecuteSuggestionSafetyGates:
    """Tests for the execute suggestion API safety gates."""
    
    def test_greg_trading_mode_enum(self):
        """Test that GregTradingMode enum has expected values."""
        assert GregTradingMode.ADVICE_ONLY.value == "advice_only"
        assert GregTradingMode.PAPER.value == "paper"
        assert GregTradingMode.LIVE.value == "live"
    
    def test_mutable_store_default_mode(self):
        """Test that mutable store starts in ADVICE_ONLY mode."""
        from src.greg_trading_store import GregTradingStore
        
        store = GregTradingStore()
        assert store.get_mode() == GregTradingMode.ADVICE_ONLY
    
    def test_mutable_store_mode_change(self):
        """Test that mode changes take effect in mutable store."""
        from src.greg_trading_store import GregTradingStore
        
        store = GregTradingStore()
        store.set_mode(GregTradingMode.PAPER, "Test change")
        assert store.get_mode() == GregTradingMode.PAPER
        
        store.set_mode(GregTradingMode.ADVICE_ONLY, "Reset")
        assert store.get_mode() == GregTradingMode.ADVICE_ONLY
    
    def test_mutable_store_can_execute_advice_only(self):
        """Test that ADVICE_ONLY mode blocks execution."""
        from src.greg_trading_store import GregTradingStore
        
        store = GregTradingStore()
        can_exec, reason = store.can_execute("STRATEGY_A_STRADDLE")
        assert not can_exec
        assert "advice-only" in reason.lower()
    
    def test_mutable_store_can_execute_paper(self):
        """Test that PAPER mode allows execution."""
        from src.greg_trading_store import GregTradingStore
        
        store = GregTradingStore()
        store.set_mode(GregTradingMode.PAPER, "Test")
        can_exec, reason = store.can_execute("STRATEGY_A_STRADDLE")
        assert can_exec
        assert "paper" in reason.lower() or "dry_run" in reason.lower()
    
    def test_mutable_store_live_requires_master_switch(self):
        """Test that LIVE mode requires master switch."""
        from src.greg_trading_store import GregTradingStore
        
        store = GregTradingStore()
        store.set_mode(GregTradingMode.LIVE, "Test")
        
        can_exec, reason = store.can_execute("STRATEGY_A_STRADDLE")
        assert not can_exec
        assert "master switch" in reason.lower()
    
    def test_mutable_store_live_requires_strategy_flag(self):
        """Test that LIVE mode requires per-strategy flag."""
        from src.greg_trading_store import GregTradingStore
        
        store = GregTradingStore()
        store.set_mode(GregTradingMode.LIVE, "Test")
        store.set_enable_live(True)
        
        can_exec, reason = store.can_execute("STRATEGY_A_STRADDLE")
        assert not can_exec
        assert "not enabled" in reason.lower()
        
        store.set_strategy_enabled("STRATEGY_A_STRADDLE", True)
        can_exec, reason = store.can_execute("STRATEGY_A_STRADDLE")
        assert can_exec
    
    def test_atomic_execute_check_paper_requires_testnet(self):
        """Test that atomic check verifies deribit env for PAPER mode."""
        from src.greg_trading_store import GregTradingStore
        
        store = GregTradingStore()
        store.set_mode(GregTradingMode.PAPER, "Test")
        
        can_exec, reason, is_dry_run = store.atomic_execute_check("STRATEGY_A_STRADDLE", "testnet")
        assert can_exec
        assert is_dry_run
        
        can_exec, reason, is_dry_run = store.atomic_execute_check("STRATEGY_A_STRADDLE", "mainnet")
        assert not can_exec
        assert "testnet" in reason.lower()
    
    def test_atomic_execute_check_live_requires_mainnet(self):
        """Test that atomic check verifies deribit env for LIVE mode."""
        from src.greg_trading_store import GregTradingStore
        
        store = GregTradingStore()
        store.set_mode(GregTradingMode.LIVE, "Test")
        store.set_enable_live(True)
        store.set_strategy_enabled("STRATEGY_A_STRADDLE", True)
        
        can_exec, reason, is_dry_run = store.atomic_execute_check("STRATEGY_A_STRADDLE", "mainnet")
        assert can_exec
        assert not is_dry_run
        
        can_exec, reason, is_dry_run = store.atomic_execute_check("STRATEGY_A_STRADDLE", "testnet")
        assert not can_exec
        assert "mainnet" in reason.lower()
    
    def test_all_strategies_disabled_by_default(self):
        """Test that all strategies are disabled for live execution by default."""
        from src.greg_trading_store import GregTradingStore
        
        store = GregTradingStore()
        flags = store.get_all_strategy_flags()
        
        for strategy, enabled in flags.items():
            assert not enabled, f"{strategy} should be disabled by default"
    
    def test_default_mode_is_advice_only(self):
        """Test that default mode is ADVICE_ONLY in config."""
        from src.config import Settings
        
        fresh_settings = Settings()
        assert fresh_settings.greg_trading_mode == GregTradingMode.ADVICE_ONLY
        assert not fresh_settings.greg_enable_live_execution


class TestDecisionReportScript:
    """Tests for the decision report script."""
    
    def test_parse_date(self):
        """Test date parsing."""
        from scripts.greg_decision_report import parse_date
        
        result = parse_date("2025-01-15")
        assert result.year == 2025
        assert result.month == 1
        assert result.day == 15
    
    def test_format_date_with_datetime(self):
        """Test date formatting."""
        from scripts.greg_decision_report import format_date
        
        dt = datetime(2025, 6, 15, 10, 30, 0)
        result = format_date(dt)
        assert "2025-06-15" in result
        assert "10:30:00" in result
    
    def test_format_date_with_none(self):
        """Test date formatting with None."""
        from scripts.greg_decision_report import format_date
        
        result = format_date(None)
        assert result == "N/A"
    
    @patch("scripts.greg_decision_report.get_decision_history")
    @patch("scripts.greg_decision_report.get_decision_stats")
    def test_generate_report_text(self, mock_stats, mock_history):
        """Test text report generation."""
        from scripts.greg_decision_report import generate_report
        
        mock_history.return_value = [
            {
                "timestamp": "2025-01-15T10:00:00",
                "underlying": "BTC",
                "strategy_type": "STRATEGY_A_STRADDLE",
                "action_type": "HEDGE",
                "mode": "paper",
                "suggested": True,
                "executed": True,
                "pnl_pct": 0.05,
            }
        ]
        mock_stats.return_value = {
            "total_suggestions": 10,
            "total_executed": 5,
            "by_strategy": {},
            "by_action": {},
        }
        
        report = generate_report(output_format="text")
        
        assert "GREG DECISION REPORT" in report
        assert "Total Suggestions: 10" in report
        assert "Total Executed: 5" in report
        assert "Follow Rate: 50.0%" in report


class TestEnvMatrixTests:
    """Tests for the environment matrix test updates."""
    
    def test_all_env_tests_have_required_sensors(self):
        """Test that all environment tests have the required sensors."""
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        from scripts.smoke_greg_strategies import ENV_TESTS
        
        required_sensors = [
            "vrp_30d", "chop_factor_7d", "adx_14d",
            "term_structure_spread", "skew_25d", "rsi_14d",
            "iv_rank_6m", "price_vs_ma200", "vrp_7d",
        ]
        
        for test in ENV_TESTS:
            env = test["env"]
            for sensor in required_sensors:
                assert sensor in env, f"Missing {sensor} in test: {test['name']}"
    
    def test_env_tests_count(self):
        """Test that we have 6 environment tests."""
        from scripts.smoke_greg_strategies import ENV_TESTS
        
        assert len(ENV_TESTS) == 6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
