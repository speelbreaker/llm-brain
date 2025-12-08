"""
Unit tests for validate_llm_decision in src/agent_brain_llm.py.

Tests the LLM decision validation logic that ensures:
1. Actions are valid ActionType values
2. Params is a dict
3. Symbol references match candidates or open positions
4. Size is within limits
"""
from __future__ import annotations

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock

from src.agent_brain_llm import validate_llm_decision
from src.models import (
    ActionType,
    AgentState,
    CandidateOption,
    OptionPosition,
    PortfolioState,
    VolState,
    Side,
    OptionType,
)
from src.config import Settings


def _make_settings(default_order_size: float = 0.1) -> Settings:
    """Create a Settings mock with default values."""
    settings = MagicMock(spec=Settings)
    settings.default_order_size = default_order_size
    settings.llm_validation_strict = True
    return settings


def _make_candidate(symbol: str = "BTC-27DEC24-100000-C") -> CandidateOption:
    """Create a minimal CandidateOption for testing."""
    return CandidateOption(
        symbol=symbol,
        underlying="BTC",
        strike=100000.0,
        expiry=datetime(2024, 12, 27, 8, 0, 0, tzinfo=timezone.utc),
        dte=10,
        delta=0.25,
        iv=0.65,
        ivrv=1.2,
        premium_usd=500.0,
        bid=0.01,
        ask=0.012,
        mid_price=0.011,
        option_type=OptionType.CALL,
        otm_pct=10.0,
    )


def _make_position(symbol: str = "BTC-20DEC24-95000-C") -> OptionPosition:
    """Create a minimal OptionPosition for testing."""
    return OptionPosition(
        symbol=symbol,
        underlying="BTC",
        strike=95000.0,
        expiry=datetime(2024, 12, 20, 8, 0, 0, tzinfo=timezone.utc),
        expiry_dte=5,
        option_type=OptionType.CALL,
        side=Side.SELL,
        size=0.1,
        avg_price=0.015,
        mark_price=0.012,
        unrealized_pnl=30.0,
        delta=-0.3,
        moneyness="OTM",
    )


def _make_agent_state(positions: list[OptionPosition] | None = None) -> AgentState:
    """Create a minimal AgentState for testing."""
    if positions is None:
        positions = []
    
    portfolio = PortfolioState(
        balances={"USDC": 10000.0},
        equity_usd=10000.0,
        margin_used_pct=5.0,
        margin_available_usd=9500.0,
        net_delta=0.0,
        option_positions=positions,
    )
    
    vol_state = VolState(
        btc_iv=0.65,
        btc_rv=0.55,
        btc_ivrv=1.18,
        eth_iv=0.70,
        eth_rv=0.60,
        eth_ivrv=1.16,
    )
    
    return AgentState(
        timestamp=datetime.now(timezone.utc),
        underlyings=["BTC", "ETH"],
        spot={"BTC": 100000.0, "ETH": 3500.0},
        portfolio=portfolio,
        vol_state=vol_state,
        candidate_options=[],
        market_context=None,
    )


class TestValidateLlmDecision:
    """Tests for validate_llm_decision function."""
    
    def test_valid_open_covered_call_passes(self):
        """Valid OPEN_COVERED_CALL with symbol in candidates passes."""
        candidate = _make_candidate("BTC-27DEC24-100000-C")
        state = _make_agent_state()
        settings = _make_settings()
        
        decision = {
            "action": "OPEN_COVERED_CALL",
            "params": {"symbol": "BTC-27DEC24-100000-C"},
            "reasoning": "Good IVRV",
        }
        
        result = validate_llm_decision(decision, state, [candidate], settings)
        
        assert result["action"] == "OPEN_COVERED_CALL"
        assert result["params"]["symbol"] == "BTC-27DEC24-100000-C"
        assert result["params"]["size"] == 0.1
    
    def test_do_nothing_always_passes(self):
        """DO_NOTHING action passes without symbol validation."""
        state = _make_agent_state()
        settings = _make_settings()
        
        decision = {
            "action": "DO_NOTHING",
            "params": {},
            "reasoning": "Market conditions unfavorable",
        }
        
        result = validate_llm_decision(decision, state, [], settings)
        
        assert result["action"] == "DO_NOTHING"
    
    def test_invalid_action_raises_value_error(self):
        """Invalid action string raises ValueError."""
        state = _make_agent_state()
        settings = _make_settings()
        
        decision = {
            "action": "BUY_THE_DIP",
            "params": {},
            "reasoning": "YOLO",
        }
        
        with pytest.raises(ValueError, match="invalid_action"):
            validate_llm_decision(decision, state, [], settings)
    
    def test_symbol_not_in_candidates_raises(self):
        """OPEN_COVERED_CALL with symbol not in candidates raises."""
        candidate = _make_candidate("BTC-27DEC24-100000-C")
        state = _make_agent_state()
        settings = _make_settings()
        
        decision = {
            "action": "OPEN_COVERED_CALL",
            "params": {"symbol": "BTC-27DEC24-999999-C"},
            "reasoning": "Hallucinated symbol",
        }
        
        with pytest.raises(ValueError, match="invalid_symbol_reference"):
            validate_llm_decision(decision, state, [candidate], settings)
    
    def test_missing_symbol_for_open_raises(self):
        """OPEN_COVERED_CALL without symbol raises."""
        candidate = _make_candidate("BTC-27DEC24-100000-C")
        state = _make_agent_state()
        settings = _make_settings()
        
        decision = {
            "action": "OPEN_COVERED_CALL",
            "params": {},
            "reasoning": "No symbol provided",
        }
        
        with pytest.raises(ValueError, match="invalid_symbol_reference"):
            validate_llm_decision(decision, state, [candidate], settings)
    
    def test_close_with_valid_position_passes(self):
        """CLOSE_COVERED_CALL with symbol in positions passes."""
        position = _make_position("BTC-20DEC24-95000-C")
        state = _make_agent_state(positions=[position])
        settings = _make_settings()
        
        decision = {
            "action": "CLOSE_COVERED_CALL",
            "params": {"symbol": "BTC-20DEC24-95000-C"},
            "reasoning": "Take profit",
        }
        
        result = validate_llm_decision(decision, state, [], settings)
        
        assert result["action"] == "CLOSE_COVERED_CALL"
        assert result["params"]["symbol"] == "BTC-20DEC24-95000-C"
    
    def test_close_with_invalid_position_raises(self):
        """CLOSE_COVERED_CALL with symbol not in positions raises."""
        state = _make_agent_state(positions=[])
        settings = _make_settings()
        
        decision = {
            "action": "CLOSE_COVERED_CALL",
            "params": {"symbol": "BTC-20DEC24-95000-C"},
            "reasoning": "Close non-existent",
        }
        
        with pytest.raises(ValueError, match="invalid_symbol_reference"):
            validate_llm_decision(decision, state, [], settings)
    
    def test_roll_with_valid_symbols_passes(self):
        """ROLL_COVERED_CALL with valid from/to symbols passes."""
        position = _make_position("BTC-20DEC24-95000-C")
        candidate = _make_candidate("BTC-27DEC24-100000-C")
        state = _make_agent_state(positions=[position])
        settings = _make_settings()
        
        decision = {
            "action": "ROLL_COVERED_CALL",
            "params": {
                "from_symbol": "BTC-20DEC24-95000-C",
                "to_symbol": "BTC-27DEC24-100000-C",
            },
            "reasoning": "Roll up and out",
        }
        
        result = validate_llm_decision(decision, state, [candidate], settings)
        
        assert result["action"] == "ROLL_COVERED_CALL"
    
    def test_roll_with_invalid_from_symbol_raises(self):
        """ROLL_COVERED_CALL with from_symbol not in positions raises."""
        candidate = _make_candidate("BTC-27DEC24-100000-C")
        state = _make_agent_state(positions=[])
        settings = _make_settings()
        
        decision = {
            "action": "ROLL_COVERED_CALL",
            "params": {
                "from_symbol": "BTC-20DEC24-95000-C",
                "to_symbol": "BTC-27DEC24-100000-C",
            },
            "reasoning": "Roll from nothing",
        }
        
        with pytest.raises(ValueError, match="invalid_symbol_reference"):
            validate_llm_decision(decision, state, [candidate], settings)
    
    def test_size_clamped_to_max(self):
        """Size larger than default_order_size is clamped."""
        candidate = _make_candidate("BTC-27DEC24-100000-C")
        state = _make_agent_state()
        settings = _make_settings(default_order_size=0.1)
        
        decision = {
            "action": "OPEN_COVERED_CALL",
            "params": {"symbol": "BTC-27DEC24-100000-C", "size": 1.0},
            "reasoning": "Too large",
        }
        
        result = validate_llm_decision(decision, state, [candidate], settings)
        
        assert result["params"]["size"] == 0.1
    
    def test_missing_params_creates_empty_dict(self):
        """Missing params field creates empty dict."""
        state = _make_agent_state()
        settings = _make_settings()
        
        decision = {
            "action": "DO_NOTHING",
            "reasoning": "No params",
        }
        
        result = validate_llm_decision(decision, state, [], settings)
        
        assert result["params"] == {}
    
    def test_invalid_params_type_raises(self):
        """Non-dict params raises ValueError."""
        state = _make_agent_state()
        settings = _make_settings()
        
        decision = {
            "action": "DO_NOTHING",
            "params": "not a dict",
            "reasoning": "Bad params",
        }
        
        with pytest.raises(ValueError, match="invalid_params"):
            validate_llm_decision(decision, state, [], settings)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
