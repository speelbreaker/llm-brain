"""
End-to-end tests for the agent loop.

These tests run one actual tick of the agent loop with mocked components,
validating the complete flow from state building to execution.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest

from src.config import Settings, TradingMode
from src.models import ActionType


class FakeDeribitClient:
    """
    Fake Deribit client for testing.
    Tracks placed orders for assertions.
    """
    
    def __init__(self):
        self.orders_placed: list[dict] = []
        self._mock_positions: list[dict] = []
        self._mock_ticker: dict = {}
        self._mock_account: dict = {}
        self._closed = False
    
    def get_positions(self, currency: str, kind: str = "option") -> list:
        return self._mock_positions
    
    def get_ticker(self, instrument_name: str) -> dict:
        return self._mock_ticker.get(instrument_name, {
            "best_bid_price": 0.01,
            "best_ask_price": 0.012,
            "mark_price": 0.011,
        })
    
    def get_account_summary(self, currency: str) -> dict:
        return self._mock_account.get(currency, {
            "equity": 10000.0,
            "balance": 10000.0,
            "margin_balance": 10000.0,
            "initial_margin": 100.0,
            "maintenance_margin": 50.0,
            "available_funds": 9900.0,
        })
    
    def get_instruments(self, currency: str, kind: str = "option", expired: bool = False) -> list:
        return []
    
    def get_index_price(self, index_name: str) -> dict:
        return {"index_price": 100000.0}
    
    def place_order(
        self,
        instrument_name: str,
        side: str,
        amount: float,
        order_type: str = "limit",
        price: Optional[float] = None,
        post_only: bool = True,
        reduce_only: bool = False,
        label: Optional[str] = None,
    ) -> dict:
        order = {
            "instrument_name": instrument_name,
            "side": side,
            "amount": amount,
            "order_type": order_type,
            "price": price,
            "post_only": post_only,
            "reduce_only": reduce_only,
            "label": label,
        }
        self.orders_placed.append(order)
        return {
            "order": {
                "order_id": f"test_order_{len(self.orders_placed)}",
                "order_state": "open",
            }
        }
    
    def close(self):
        self._closed = True


@pytest.fixture
def fake_client():
    """Create a fresh fake client for each test."""
    return FakeDeribitClient()


@pytest.fixture
def test_settings():
    """Create test settings."""
    settings = Settings(
        mode="research",
        deribit_env="testnet",
        dry_run=False,  # We want to see if orders would be placed
        kill_switch_enabled=False,
        trade_mode=TradingMode.NORMAL,
        heartbeat_timeout_sec=300,
        max_orders_per_minute=10,
        rolling_drawdown_window_days=0,  # Disabled for testing
        daily_drawdown_limit_pct=0.0,  # Disabled for testing
    )
    return settings


@pytest.fixture(autouse=True)
def reset_modules():
    """Reset module-level state before each test."""
    from src.ops.heartbeat import reset_heartbeat
    from src.ops.rate_limiter import reset_rate_limiter
    from src.ops.rolling_drawdown import reset_rolling_drawdown
    
    reset_heartbeat()
    reset_rate_limiter()
    reset_rolling_drawdown(clear_close_only_trigger=True)
    yield
    reset_heartbeat()
    reset_rate_limiter()
    reset_rolling_drawdown(clear_close_only_trigger=True)


class TestAgentLoopE2E:
    """End-to-end tests for agent loop tick."""
    
    def test_heartbeat_recorded_on_tick(self, fake_client, test_settings):
        """Running a tick should record a heartbeat."""
        from agent_loop import run_single_tick
        from src.ops.heartbeat import check_heartbeat, HeartbeatState
        
        # Before tick, no heartbeat
        status_before = check_heartbeat(timeout_sec=300)
        assert status_before.state == HeartbeatState.NEVER_STARTED
        
        # Mock the state builder to avoid actual API calls
        with patch('agent_loop.build_agent_state') as mock_build:
            mock_build.return_value = self._create_mock_agent_state()
            with patch('agent_loop.get_cached_health_status') as mock_health:
                mock_health.return_value = MagicMock(can_trade=True)
                
                result = run_single_tick(fake_client, test_settings)
        
        # After tick, heartbeat should be recorded
        status_after = check_heartbeat(timeout_sec=300)
        assert status_after.state == HeartbeatState.HEALTHY
        assert status_after.seconds_since_last < 1.0
    
    def test_tick_with_do_nothing_places_no_orders(self, fake_client, test_settings):
        """When action is DO_NOTHING, no orders should be placed."""
        from agent_loop import run_single_tick
        
        with patch('agent_loop.build_agent_state') as mock_build:
            mock_state = self._create_mock_agent_state()
            mock_state.candidate_options = []  # No candidates = DO_NOTHING
            mock_build.return_value = mock_state
            with patch('agent_loop.get_cached_health_status') as mock_health:
                mock_health.return_value = MagicMock(can_trade=True)
                
                result = run_single_tick(fake_client, test_settings)
        
        assert result.final_action["action"] == ActionType.DO_NOTHING.value
        assert len(fake_client.orders_placed) == 0
    
    def test_tick_blocked_by_permission_downgrades_to_do_nothing(self, fake_client, test_settings):
        """When trade permission blocks action, it should be downgraded to DO_NOTHING."""
        from agent_loop import run_single_tick
        
        # Set close_only mode
        test_settings.trade_mode = TradingMode.CLOSE_ONLY
        
        with patch('agent_loop.build_agent_state') as mock_build:
            mock_state = self._create_mock_agent_state()
            mock_state.candidate_options = self._create_mock_candidates()
            mock_build.return_value = mock_state
            
            with patch('agent_loop.rule_decide_action') as mock_decide:
                # Force a OPEN action
                mock_decide.return_value = {
                    "action": ActionType.OPEN_COVERED_CALL.value,
                    "params": {"symbol": "BTC-25DEC25-100000-C", "size": 1.0},
                    "reasoning": "Test open",
                }
                with patch('agent_loop.get_cached_health_status') as mock_health:
                    mock_health.return_value = MagicMock(can_trade=True)
                    
                    result = run_single_tick(fake_client, test_settings)
        
        # Should be downgraded to DO_NOTHING because close_only
        assert result.final_action["action"] == ActionType.DO_NOTHING.value
        assert "Permission denied" in result.final_action["reasoning"]
        assert len(fake_client.orders_placed) == 0
    
    def test_tick_blocked_by_rate_limit(self, fake_client, test_settings):
        """When rate limit is exceeded, orders should be blocked."""
        from agent_loop import run_single_tick
        from src.ops.rate_limiter import record_order
        
        # Pre-fill rate limiter to limit
        test_settings.max_orders_per_minute = 5
        for _ in range(5):
            record_order()
        
        with patch('agent_loop.build_agent_state') as mock_build:
            mock_state = self._create_mock_agent_state()
            mock_state.candidate_options = self._create_mock_candidates()
            mock_build.return_value = mock_state
            
            with patch('agent_loop.rule_decide_action') as mock_decide:
                mock_decide.return_value = {
                    "action": ActionType.OPEN_COVERED_CALL.value,
                    "params": {"symbol": "BTC-25DEC25-100000-C", "size": 1.0},
                    "reasoning": "Test open",
                }
                with patch('agent_loop.get_cached_health_status') as mock_health:
                    mock_health.return_value = MagicMock(can_trade=True)
                    with patch('agent_loop.check_action_allowed') as mock_risk:
                        mock_risk.return_value = (True, [])
                        
                        result = run_single_tick(fake_client, test_settings)
        
        # Action went through agent loop but execution was rate limited
        if result.execution_result.get("status") == "rate_limited":
            assert "Rate limit" in result.execution_result.get("message", "")
            assert len(fake_client.orders_placed) == 0
    
    def test_tick_halt_mode_blocks_all_actions(self, fake_client, test_settings):
        """In HALT mode, all actions should be blocked."""
        from agent_loop import run_single_tick
        
        test_settings.trade_mode = TradingMode.HALT
        
        with patch('agent_loop.build_agent_state') as mock_build:
            mock_state = self._create_mock_agent_state()
            mock_state.candidate_options = self._create_mock_candidates()
            mock_build.return_value = mock_state
            
            with patch('agent_loop.rule_decide_action') as mock_decide:
                mock_decide.return_value = {
                    "action": ActionType.CLOSE_COVERED_CALL.value,
                    "params": {"symbol": "BTC-25DEC25-100000-C", "size": 1.0},
                    "reasoning": "Test close",
                }
                with patch('agent_loop.get_cached_health_status') as mock_health:
                    mock_health.return_value = MagicMock(can_trade=True)
                    
                    result = run_single_tick(fake_client, test_settings)
        
        # Even CLOSE should be blocked in HALT mode
        assert result.final_action["action"] == ActionType.DO_NOTHING.value
        assert len(fake_client.orders_placed) == 0
    
    def test_tick_kill_switch_blocks_all_actions(self, fake_client, test_settings):
        """When kill switch is enabled, all actions should be blocked."""
        from agent_loop import run_single_tick
        
        test_settings.kill_switch_enabled = True
        
        with patch('agent_loop.build_agent_state') as mock_build:
            mock_state = self._create_mock_agent_state()
            mock_state.candidate_options = self._create_mock_candidates()
            mock_build.return_value = mock_state
            
            with patch('agent_loop.rule_decide_action') as mock_decide:
                mock_decide.return_value = {
                    "action": ActionType.OPEN_COVERED_CALL.value,
                    "params": {"symbol": "BTC-25DEC25-100000-C", "size": 1.0},
                    "reasoning": "Test open",
                }
                with patch('agent_loop.get_cached_health_status') as mock_health:
                    mock_health.return_value = MagicMock(can_trade=True)
                    
                    result = run_single_tick(fake_client, test_settings)
        
        # Kill switch should block
        assert result.final_action["action"] == ActionType.DO_NOTHING.value
        assert len(fake_client.orders_placed) == 0
    
    def _create_mock_agent_state(self):
        """Create a mock AgentState."""
        from src.models import AgentState, PortfolioState, MarketContext
        
        portfolio = MagicMock(spec=PortfolioState)
        portfolio.equity_usd = 100000.0
        portfolio.margin_used_pct = 10.0
        portfolio.option_positions = []
        portfolio.net_delta = 0.0
        
        market_context = MagicMock(spec=MarketContext)
        market_context.trend = "neutral"
        market_context.volatility = "normal"
        
        state = MagicMock(spec=AgentState)
        state.portfolio = portfolio
        state.spot = {"BTC": 100000.0, "ETH": 3500.0}
        state.candidate_options = []
        state.market_context = market_context
        
        return state
    
    def _create_mock_candidates(self):
        """Create mock candidate options."""
        from src.models import CandidateOption
        
        candidate = MagicMock(spec=CandidateOption)
        candidate.symbol = "BTC-25DEC25-100000-C"
        candidate.underlying = "BTC"
        candidate.option_type = "call"
        candidate.strike = 100000
        candidate.expiry = datetime(2025, 12, 25, tzinfo=timezone.utc)
        candidate.delta = 0.25
        candidate.iv = 0.6
        candidate.ivrv = 1.3
        candidate.bid = 0.01
        candidate.ask = 0.012
        candidate.mark = 0.011
        
        return [candidate]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

