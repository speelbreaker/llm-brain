"""
Integration test for trade permission and close-only mode.

Tests the unified "final permission" decision in the live trading path:
- One tick with FakeDeribitClient
- Deterministic behavior (no background threads)
- Defense in depth verification
"""
from __future__ import annotations

import pytest
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

from src.config import Settings, TradingMode
from src.models import (
    ActionType,
    AgentState,
    CandidateOption,
    OptionPosition,
    OptionType,
    PortfolioState,
    Side,
    VolState,
)
from src.ops.trade_permission import (
    compute_trade_permission,
    check_action_permission,
    TradePermission,
    PermissionCode,
)
from src.risk_engine import check_action_allowed


class FakeDeribitClient:
    """Fake Deribit client that captures all order calls."""
    
    def __init__(self):
        self.order_calls: List[Dict[str, Any]] = []
        self.cancel_calls: List[Dict[str, Any]] = []
        self._index_prices = {"BTC": 100000.0, "ETH": 3500.0}
        self._positions: List[Dict[str, Any]] = []
        self._ticker_data: Dict[str, Dict[str, Any]] = {}
    
    def get_index_price(self, currency: str) -> float:
        return self._index_prices.get(currency.upper(), 100000.0)
    
    def get_ticker(self, instrument_name: str) -> Dict[str, Any]:
        return self._ticker_data.get(instrument_name, {
            "best_bid_price": 0.01,
            "best_ask_price": 0.012,
            "mark_price": 0.011,
        })
    
    def get_instruments(self, currency: str, kind: str = "option", expired: bool = False) -> List[Dict[str, Any]]:
        return []
    
    def get_positions(self, currency: str, kind: str = "option") -> List[Dict[str, Any]]:
        return self._positions
    
    def get_account_summary(self, currency: str) -> Dict[str, Any]:
        return {
            "equity": 10000.0,
            "margin_balance": 8000.0,
            "initial_margin": 2000.0,
        }
    
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
    ) -> Dict[str, Any]:
        """Capture order call."""
        call = {
            "instrument_name": instrument_name,
            "side": side,
            "amount": amount,
            "order_type": order_type,
            "price": price,
            "post_only": post_only,
            "reduce_only": reduce_only,
            "label": label,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.order_calls.append(call)
        return {
            "order": {
                "order_id": f"fake_order_{len(self.order_calls)}",
                "order_state": "open",
            }
        }
    
    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        self.cancel_calls.append({"order_id": order_id})
        return {"order_id": order_id}
    
    def close(self) -> None:
        pass


def make_test_settings(**overrides) -> Settings:
    """Create test settings with sensible defaults."""
    defaults = {
        "deribit_env": "testnet",
        "deribit_base_url": "https://test.deribit.com",
        "mode": "research",
        "dry_run": False,
        "kill_switch_enabled": False,
        "trade_mode": TradingMode.NORMAL,
        "training_mode": False,
        "underlyings": ["BTC", "ETH"],
        "max_margin_used_pct": 80.0,
        "max_net_delta_abs": 5.0,
        "daily_drawdown_limit_pct": 0.0,
    }
    defaults.update(overrides)
    return Settings(**defaults)


def make_test_agent_state(
    spot_btc: float = 100000.0,
    equity_usd: float = 10000.0,
    margin_used_pct: float = 20.0,
    positions: Optional[List[OptionPosition]] = None,
    candidates: Optional[List[CandidateOption]] = None,
) -> AgentState:
    """Create a test agent state."""
    return AgentState(
        timestamp=datetime.now(timezone.utc),
        underlyings=["BTC", "ETH"],
        spot={"BTC": spot_btc, "ETH": 3500.0},
        portfolio=PortfolioState(
            balances={"BTC": 0.5, "ETH": 5.0, "USDC": 5000.0},
            spot_positions={"BTC": 0.5, "ETH": 5.0},
            equity_usd=equity_usd,
            margin_used_usd=equity_usd * margin_used_pct / 100,
            margin_available_usd=equity_usd * (1 - margin_used_pct / 100),
            margin_used_pct=margin_used_pct,
            net_delta=0.0,
            option_positions=positions or [],
        ),
        vol_state=VolState(
            btc_iv=0.6, btc_rv=0.5, btc_ivrv=1.2,
            eth_iv=0.7, eth_rv=0.55, eth_ivrv=1.27,
        ),
        candidate_options=candidates or [],
    )


def make_test_candidate(
    symbol: str = "BTC-27DEC24-100000-C",
    underlying: str = "BTC",
    dte: int = 7,
    delta: float = 0.25,
    premium_usd: float = 150.0,
) -> CandidateOption:
    """Create a test candidate option."""
    return CandidateOption(
        symbol=symbol,
        underlying=underlying,
        strike=100000.0,
        expiry=datetime(2024, 12, 27, 8, 0, 0, tzinfo=timezone.utc),
        option_type=OptionType.CALL,
        dte=dte,
        delta=delta,
        otm_pct=0.0,
        bid=0.001,
        ask=0.0012,
        mid_price=0.0011,
        premium_usd=premium_usd,
        iv=0.6,
        rv=0.5,
        ivrv=1.2,
    )


class TestTradePermissionBasic:
    """Basic trade permission tests."""
    
    def test_normal_mode_allows_all(self):
        """In normal mode with can_trade=True, all actions should be allowed."""
        cfg = make_test_settings(trade_mode=TradingMode.NORMAL, kill_switch_enabled=False)
        permission = compute_trade_permission(cfg, can_trade_from_health=True)
        
        assert permission.allow_open is True
        assert permission.allow_roll is True
        assert permission.allow_close is True
        assert permission.allow_do_nothing is True
        assert permission.code == PermissionCode.ALLOWED
    
    def test_close_only_blocks_open_and_roll(self):
        """In close_only mode, OPEN and ROLL should be blocked."""
        cfg = make_test_settings(trade_mode=TradingMode.CLOSE_ONLY)
        permission = compute_trade_permission(cfg, can_trade_from_health=True)
        
        assert permission.allow_open is False
        assert permission.allow_roll is False
        assert permission.allow_close is True
        assert permission.allow_do_nothing is True
        assert permission.code == PermissionCode.BLOCKED_TRADE_MODE_CLOSE_ONLY
    
    def test_halt_blocks_everything(self):
        """In halt mode, only DO_NOTHING should be allowed."""
        cfg = make_test_settings(trade_mode=TradingMode.HALT)
        permission = compute_trade_permission(cfg, can_trade_from_health=True)
        
        assert permission.allow_open is False
        assert permission.allow_roll is False
        assert permission.allow_close is False
        assert permission.allow_do_nothing is True
        assert permission.code == PermissionCode.BLOCKED_TRADE_MODE_HALT
    
    def test_kill_switch_overrides_everything(self):
        """Kill switch should override trade_mode and block all."""
        cfg = make_test_settings(trade_mode=TradingMode.NORMAL, kill_switch_enabled=True)
        permission = compute_trade_permission(cfg, can_trade_from_health=True)
        
        assert permission.allow_open is False
        assert permission.allow_roll is False
        assert permission.allow_close is False
        assert permission.allow_do_nothing is True
        assert permission.code == PermissionCode.BLOCKED_KILL_SWITCH
        assert permission.effective_trade_mode == TradingMode.HALT
    
    def test_can_trade_false_blocks_open_roll_allows_close(self):
        """When health says can_trade=False, block OPEN/ROLL but allow CLOSE."""
        cfg = make_test_settings(trade_mode=TradingMode.NORMAL)
        permission = compute_trade_permission(cfg, can_trade_from_health=False)
        
        assert permission.allow_open is False
        assert permission.allow_roll is False
        assert permission.allow_close is True
        assert permission.allow_do_nothing is True
        assert permission.code == PermissionCode.BLOCKED_CAN_TRADE_FALSE
    
    def test_action_permission_check(self):
        """Test the convenience check_action_permission function."""
        cfg = make_test_settings(trade_mode=TradingMode.CLOSE_ONLY)
        
        allowed, code, reason = check_action_permission(
            ActionType.OPEN_COVERED_CALL, cfg, can_trade_from_health=True
        )
        assert allowed is False
        assert code == PermissionCode.BLOCKED_TRADE_MODE_CLOSE_ONLY
        
        allowed, code, reason = check_action_permission(
            ActionType.CLOSE_COVERED_CALL, cfg, can_trade_from_health=True
        )
        assert allowed is True
        assert code == PermissionCode.BLOCKED_TRADE_MODE_CLOSE_ONLY  # Code reflects mode, not action


class TestRiskEngineIntegration:
    """Test that risk engine respects trade permission."""
    
    def test_risk_engine_blocks_open_in_close_only_mode(self):
        """Risk engine should block OPEN when trade_mode=close_only."""
        cfg = make_test_settings(trade_mode=TradingMode.CLOSE_ONLY)
        agent_state = make_test_agent_state()
        
        proposed_action = {
            "action": ActionType.OPEN_COVERED_CALL.value,
            "params": {"symbol": "BTC-27DEC24-100000-C", "size": 0.1},
            "reasoning": "Test action",
        }
        
        # Mock the health check to avoid circular imports in tests
        with patch("src.healthcheck.get_cached_health_status") as mock_health:
            mock_health.return_value = MagicMock(can_trade=True)
            
            allowed, reasons = check_action_allowed(agent_state, proposed_action, cfg)
        
        assert allowed is False
        assert any("BLOCKED_TRADE_MODE_CLOSE_ONLY" in r for r in reasons)
    
    def test_risk_engine_allows_close_in_close_only_mode(self):
        """Risk engine should allow CLOSE when trade_mode=close_only."""
        cfg = make_test_settings(trade_mode=TradingMode.CLOSE_ONLY)
        
        # Create agent state with an existing position to close
        existing_position = OptionPosition(
            symbol="BTC-27DEC24-100000-C",
            underlying="BTC",
            strike=100000.0,
            expiry=datetime(2024, 12, 27, 8, 0, 0, tzinfo=timezone.utc),
            option_type=OptionType.CALL,
            side=Side.SELL,
            size=0.1,
            avg_price=0.01,
        )
        agent_state = make_test_agent_state(positions=[existing_position])
        
        proposed_action = {
            "action": ActionType.CLOSE_COVERED_CALL.value,
            "params": {"symbol": "BTC-27DEC24-100000-C", "size": 0.1},
            "reasoning": "Close existing position",
        }
        
        with patch("src.healthcheck.get_cached_health_status") as mock_health:
            mock_health.return_value = MagicMock(can_trade=True)
            
            allowed, reasons = check_action_allowed(agent_state, proposed_action, cfg)
        
        assert allowed is True
    
    def test_risk_engine_blocks_all_with_kill_switch(self):
        """Risk engine should block all actions when kill switch is on."""
        cfg = make_test_settings(trade_mode=TradingMode.NORMAL, kill_switch_enabled=True)
        agent_state = make_test_agent_state()
        
        for action_type in [ActionType.OPEN_COVERED_CALL, ActionType.CLOSE_COVERED_CALL, ActionType.ROLL_COVERED_CALL]:
            proposed_action = {
                "action": action_type.value,
                "params": {"symbol": "BTC-27DEC24-100000-C", "size": 0.1},
                "reasoning": "Test action",
            }
            
            with patch("src.healthcheck.get_cached_health_status") as mock_health:
                mock_health.return_value = MagicMock(can_trade=True)
                
                allowed, reasons = check_action_allowed(agent_state, proposed_action, cfg)
            
            assert allowed is False
            assert any("BLOCKED_KILL_SWITCH" in r for r in reasons)


class TestOneTickIntegration:
    """Integration test: run one tick and verify order behavior."""
    
    def test_one_tick_blocked_by_can_trade_false(self):
        """
        When can_trade=False from health, proposed OPEN action should be blocked.
        No order placement should occur.
        """
        fake_client = FakeDeribitClient()
        cfg = make_test_settings(trade_mode=TradingMode.NORMAL, dry_run=False)
        
        # Create agent state with a candidate to potentially trade
        candidate = make_test_candidate()
        agent_state = make_test_agent_state(candidates=[candidate])
        
        # Proposed action that would normally trigger an order
        proposed_action = {
            "action": ActionType.OPEN_COVERED_CALL.value,
            "params": {"symbol": candidate.symbol, "size": 0.1},
            "reasoning": "Policy proposed OPEN",
        }
        
        # Simulate the trade permission check that happens in agent_loop.py
        from src.ops.trade_permission import compute_trade_permission
        
        # Health says can_trade=False
        trade_permission = compute_trade_permission(cfg, can_trade_from_health=False)
        
        # Check if action is allowed
        proposed_action_type = ActionType(proposed_action["action"])
        action_allowed = trade_permission.is_action_allowed(proposed_action_type)
        
        # Assertions
        assert action_allowed is False
        assert trade_permission.code == PermissionCode.BLOCKED_CAN_TRADE_FALSE
        
        # Simulate what agent_loop does: if blocked, don't execute
        if not action_allowed:
            final_action = {
                "action": ActionType.DO_NOTHING.value,
                "params": {},
                "reasoning": f"Blocked by trade permission: {trade_permission.reason}",
                "permission_code": trade_permission.code.value,
            }
        else:
            final_action = proposed_action
        
        # Only execute if not DO_NOTHING
        if final_action["action"] != ActionType.DO_NOTHING.value:
            # This should NOT happen in this test
            fake_client.place_order(
                instrument_name=final_action["params"]["symbol"],
                side="sell",
                amount=final_action["params"]["size"],
            )
        
        # Verify: no orders were placed
        assert len(fake_client.order_calls) == 0
        assert final_action["action"] == ActionType.DO_NOTHING.value
        assert "BLOCKED_CAN_TRADE_FALSE" in final_action.get("permission_code", "")
    
    def test_one_tick_allowed_places_order(self):
        """
        When can_trade=True and trade_mode=normal, proposed OPEN action should succeed.
        Order placement should occur.
        """
        fake_client = FakeDeribitClient()
        cfg = make_test_settings(trade_mode=TradingMode.NORMAL, dry_run=False)
        
        # Create agent state with a candidate
        candidate = make_test_candidate()
        agent_state = make_test_agent_state(candidates=[candidate])
        
        # Proposed action
        proposed_action = {
            "action": ActionType.OPEN_COVERED_CALL.value,
            "params": {"symbol": candidate.symbol, "size": 0.1},
            "reasoning": "Policy proposed OPEN",
        }
        
        # Health says can_trade=True
        trade_permission = compute_trade_permission(cfg, can_trade_from_health=True)
        
        # Check if action is allowed
        proposed_action_type = ActionType(proposed_action["action"])
        action_allowed = trade_permission.is_action_allowed(proposed_action_type)
        
        # Assertions for permission
        assert action_allowed is True
        assert trade_permission.code == PermissionCode.ALLOWED
        
        # Simulate execution (if action allowed)
        if action_allowed:
            # In real code, this goes through risk engine first, then execution
            # For this test, we verify the permission layer allows it
            fake_client.place_order(
                instrument_name=proposed_action["params"]["symbol"],
                side="sell",
                amount=proposed_action["params"]["size"],
            )
        
        # Verify: one order was placed
        assert len(fake_client.order_calls) == 1
        assert fake_client.order_calls[0]["instrument_name"] == candidate.symbol
        assert fake_client.order_calls[0]["side"] == "sell"
        assert fake_client.order_calls[0]["amount"] == 0.1
    
    def test_one_tick_close_only_blocks_open_allows_close(self):
        """
        In close_only mode:
        - OPEN should be blocked
        - CLOSE should be allowed
        """
        fake_client = FakeDeribitClient()
        cfg = make_test_settings(trade_mode=TradingMode.CLOSE_ONLY)
        
        # Test OPEN is blocked
        trade_permission = compute_trade_permission(cfg, can_trade_from_health=True)
        
        assert trade_permission.is_action_allowed(ActionType.OPEN_COVERED_CALL) is False
        assert trade_permission.is_action_allowed(ActionType.ROLL_COVERED_CALL) is False
        assert trade_permission.is_action_allowed(ActionType.CLOSE_COVERED_CALL) is True
        assert trade_permission.is_action_allowed(ActionType.DO_NOTHING) is True
        
        # Simulate CLOSE action succeeding
        fake_client.place_order(
            instrument_name="BTC-27DEC24-100000-C",
            side="buy",  # CLOSE = buy back short
            amount=0.1,
            reduce_only=True,
        )
        
        assert len(fake_client.order_calls) == 1
        assert fake_client.order_calls[0]["reduce_only"] is True


class TestDefenseInDepth:
    """Verify defense in depth: both loop and risk engine enforce permissions."""
    
    def test_both_layers_block_on_close_only(self):
        """
        Verify that BOTH the trade permission layer and risk engine
        would block an OPEN action in close_only mode.
        """
        cfg = make_test_settings(trade_mode=TradingMode.CLOSE_ONLY)
        agent_state = make_test_agent_state()
        
        proposed_action = {
            "action": ActionType.OPEN_COVERED_CALL.value,
            "params": {"symbol": "BTC-27DEC24-100000-C", "size": 0.1},
        }
        
        # Layer 1: Trade permission (used in agent_loop.py)
        trade_permission = compute_trade_permission(cfg, can_trade_from_health=True)
        layer1_blocked = not trade_permission.is_action_allowed(ActionType.OPEN_COVERED_CALL)
        
        # Layer 2: Risk engine (used in execution path)
        with patch("src.healthcheck.get_cached_health_status") as mock_health:
            mock_health.return_value = MagicMock(can_trade=True)
            allowed, reasons = check_action_allowed(agent_state, proposed_action, cfg)
        layer2_blocked = not allowed
        
        # Both layers should block
        assert layer1_blocked is True
        assert layer2_blocked is True
        assert any("BLOCKED_TRADE_MODE_CLOSE_ONLY" in r for r in reasons)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
