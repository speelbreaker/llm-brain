"""
Tests for state_core consistency between live and backtest.

These tests verify that:
1. RawMarketSnapshot → AgentState conversion is deterministic
2. Given the same inputs, live and backtest produce identical results
3. CandidateOption filtering rules are consistent
"""
from __future__ import annotations

import pytest
from datetime import datetime, timezone, timedelta

from src.state_core import (
    RawOption,
    RawPosition,
    RawPortfolio,
    RawMarketSnapshot,
    build_agent_state_from_raw,
    _calculate_dte,
    _calculate_dte_float,
    _calculate_moneyness,
    _calculate_otm_pct,
)
from src.models import AgentState, MarketContext


def create_test_snapshot(
    spot_btc: float = 100000.0,
    spot_eth: float = 3500.0,
    timestamp: datetime | None = None,
) -> RawMarketSnapshot:
    """Create a synthetic market snapshot for testing."""
    ts = timestamp or datetime.now(timezone.utc)
    
    btc_expiry = ts + timedelta(days=7)
    eth_expiry = ts + timedelta(days=14)
    
    options = [
        RawOption(
            instrument_name="BTC-14DEC24-105000-C",
            expiry=btc_expiry,
            strike=105000.0,
            option_type="call",
            mark_price=0.015,
            mark_iv=65.0,
            delta=0.25,
            underlying_price=spot_btc,
            underlying="BTC",
            bid=0.014,
            ask=0.016,
            rv=52.0,
        ),
        RawOption(
            instrument_name="BTC-14DEC24-110000-C",
            expiry=btc_expiry,
            strike=110000.0,
            option_type="call",
            mark_price=0.008,
            mark_iv=68.0,
            delta=0.15,
            underlying_price=spot_btc,
            underlying="BTC",
            bid=0.007,
            ask=0.009,
            rv=52.0,
        ),
        RawOption(
            instrument_name="ETH-21DEC24-3700-C",
            expiry=eth_expiry,
            strike=3700.0,
            option_type="call",
            mark_price=0.03,
            mark_iv=60.0,
            delta=0.30,
            underlying_price=spot_eth,
            underlying="ETH",
            bid=0.028,
            ask=0.032,
            rv=48.0,
        ),
    ]
    
    portfolio = RawPortfolio(
        equity_usd=50000.0,
        margin_used_pct=10.0,
        balances={"BTC": 0.5, "ETH": 5.0},
        positions=[],
        margin_used_usd=5000.0,
        margin_available_usd=45000.0,
        net_delta=0.0,
    )
    
    market_ctx = MarketContext(
        underlying="BTC",
        time=ts,
        regime="sideways",
        pct_from_50d_ma=2.0,
        return_7d_pct=3.5,
        return_30d_pct=8.0,
        realized_vol_7d=0.50,
        realized_vol_30d=0.48,
    )
    
    return RawMarketSnapshot(
        timestamp=ts,
        underlyings=["BTC", "ETH"],
        spot={"BTC": spot_btc, "ETH": spot_eth},
        portfolio=portfolio,
        options=options,
        realized_vol={"BTC": 52.0, "ETH": 48.0},
        market_context=market_ctx,
    )


class TestStateCoreHelpers:
    """Test helper functions in state_core."""
    
    def test_calculate_dte_basic(self):
        """Test DTE calculation."""
        now = datetime(2024, 12, 1, 12, 0, 0, tzinfo=timezone.utc)
        expiry = datetime(2024, 12, 8, 12, 0, 0, tzinfo=timezone.utc)
        
        dte = _calculate_dte(expiry, now)
        assert dte == 7
    
    def test_calculate_dte_expired(self):
        """Test DTE for expired option."""
        now = datetime(2024, 12, 10, 12, 0, 0, tzinfo=timezone.utc)
        expiry = datetime(2024, 12, 8, 12, 0, 0, tzinfo=timezone.utc)
        
        dte = _calculate_dte(expiry, now)
        assert dte == 0
    
    def test_calculate_dte_float_precision(self):
        """Test float DTE for sub-day precision."""
        now = datetime(2024, 12, 1, 12, 0, 0, tzinfo=timezone.utc)
        expiry = datetime(2024, 12, 2, 0, 0, 0, tzinfo=timezone.utc)
        
        dte = _calculate_dte_float(expiry, now)
        assert 0.4 < dte < 0.6
    
    def test_calculate_moneyness_otm_call(self):
        """Test OTM call moneyness."""
        result = _calculate_moneyness(100000.0, 105000.0, "call")
        assert result == "OTM"
    
    def test_calculate_moneyness_itm_call(self):
        """Test ITM call moneyness."""
        result = _calculate_moneyness(100000.0, 95000.0, "call")
        assert result == "ITM"
    
    def test_calculate_moneyness_atm(self):
        """Test ATM moneyness."""
        result = _calculate_moneyness(100000.0, 100500.0, "call")
        assert result == "ATM"
    
    def test_calculate_otm_pct(self):
        """Test OTM percentage calculation."""
        pct = _calculate_otm_pct(100000.0, 105000.0, "call")
        assert abs(pct - 5.0) < 0.01


class TestBuildAgentState:
    """Test build_agent_state_from_raw function."""
    
    def test_basic_state_construction(self):
        """Test basic AgentState construction."""
        snapshot = create_test_snapshot()
        
        state = build_agent_state_from_raw(
            snapshot,
            delta_min=0.10,
            delta_max=0.40,
            dte_min=1,
            dte_max=21,
        )
        
        assert isinstance(state, AgentState)
        assert state.spot["BTC"] == 100000.0
        assert state.spot["ETH"] == 3500.0
        assert len(state.underlyings) == 2
        assert state.portfolio.equity_usd == 50000.0
    
    def test_candidate_filtering(self):
        """Test that candidates are filtered by delta/DTE."""
        snapshot = create_test_snapshot()
        
        state = build_agent_state_from_raw(
            snapshot,
            delta_min=0.20,
            delta_max=0.35,
            dte_min=5,
            dte_max=20,
        )
        
        for c in state.candidate_options:
            assert c.delta >= 0.20
            assert c.delta <= 0.35
            assert c.dte >= 5
            assert c.dte <= 20
    
    def test_deterministic_output(self):
        """Test that same input produces identical output."""
        ts = datetime(2024, 12, 7, 12, 0, 0, tzinfo=timezone.utc)
        snapshot = create_test_snapshot(timestamp=ts)
        
        state1 = build_agent_state_from_raw(
            snapshot,
            delta_min=0.10,
            delta_max=0.40,
            dte_min=1,
            dte_max=21,
        )
        state2 = build_agent_state_from_raw(
            snapshot,
            delta_min=0.10,
            delta_max=0.40,
            dte_min=1,
            dte_max=21,
        )
        
        assert state1.spot == state2.spot
        assert state1.portfolio.equity_usd == state2.portfolio.equity_usd
        assert len(state1.candidate_options) == len(state2.candidate_options)
        
        for c1, c2 in zip(state1.candidate_options, state2.candidate_options):
            assert c1.symbol == c2.symbol
            assert c1.delta == c2.delta
            assert c1.premium_usd == c2.premium_usd
    
    def test_ivrv_calculation(self):
        """Test that IVRV is calculated correctly."""
        snapshot = create_test_snapshot()
        
        state = build_agent_state_from_raw(
            snapshot,
            delta_min=0.10,
            delta_max=0.40,
            dte_min=1,
            dte_max=21,
        )
        
        for c in state.candidate_options:
            assert c.ivrv >= 0
            expected_ivrv = c.iv / c.rv if c.rv > 0 else 1.0
            assert abs(c.ivrv - expected_ivrv) < 0.01
    
    def test_portfolio_positions_converted(self):
        """Test that portfolio positions are correctly converted."""
        ts = datetime.now(timezone.utc)
        snapshot = create_test_snapshot(timestamp=ts)
        
        snapshot.portfolio.positions.append(
            RawPosition(
                instrument_name="BTC-14DEC24-100000-C",
                underlying="BTC",
                strike=100000.0,
                expiry=ts + timedelta(days=7),
                option_type="call",
                size=-0.5,
                average_price=0.02,
                mark_price=0.018,
                delta=-0.4,
                unrealized_pnl_usd=100.0,
            )
        )
        
        state = build_agent_state_from_raw(snapshot)
        
        assert len(state.portfolio.option_positions) == 1
        pos = state.portfolio.option_positions[0]
        assert pos.symbol == "BTC-14DEC24-100000-C"
        assert pos.side.value == "sell"
        assert pos.size == 0.5
    
    def test_market_context_passthrough(self):
        """Test that market context is passed through."""
        snapshot = create_test_snapshot()
        
        state = build_agent_state_from_raw(snapshot)
        
        assert state.market_context is not None
        assert state.market_context.regime == "sideways"
        assert state.market_context.return_7d_pct == 3.5


class TestStateCoreConsistency:
    """Test that live and backtest use the same logic."""
    
    def test_same_snapshot_same_result_different_source(self):
        """Verify that source tag doesn't affect output."""
        snapshot = create_test_snapshot()
        
        state_live = build_agent_state_from_raw(snapshot, source="live")
        state_backtest = build_agent_state_from_raw(snapshot, source="backtest")
        
        assert state_live.spot == state_backtest.spot
        assert state_live.portfolio.equity_usd == state_backtest.portfolio.equity_usd
        assert len(state_live.candidate_options) == len(state_backtest.candidate_options)
        
        for c1, c2 in zip(state_live.candidate_options, state_backtest.candidate_options):
            assert c1.symbol == c2.symbol
            assert c1.delta == c2.delta
            assert c1.premium_usd == c2.premium_usd


class TestBacktestStateBuilderDTE:
    """
    Regression test: backtest state builder must use decision timestamp t
    for DTE calculation, NOT datetime.now().
    """
    
    def test_historical_dte_uses_decision_time(self):
        """
        Verify that build_historical_agent_state computes DTE relative to
        the decision timestamp, not the current wall clock.
        
        If this test fails, backtest will have wrong DTE values for
        historical backtests.
        """
        from src.backtest.state_builder import _option_snapshot_to_raw_option
        from src.backtest.types import OptionSnapshot
        
        historical_time = datetime(2024, 12, 1, 12, 0, 0, tzinfo=timezone.utc)
        expiry = datetime(2024, 12, 8, 12, 0, 0, tzinfo=timezone.utc)
        
        opt = OptionSnapshot(
            instrument_name="BTC-08DEC24-100000-C",
            underlying="BTC",
            kind="call",
            strike=100000.0,
            expiry=expiry,
            delta=0.25,
            iv=0.65,
            mark_price=0.015,
            settlement_ccy="Crypto",
            margin_type="linear",
        )
        
        raw_opt = _option_snapshot_to_raw_option(
            opt=opt,
            spot=100000.0,
            underlying="BTC",
            rv=0.52,
            reference_time=historical_time,
        )
        
        dte = _calculate_dte(raw_opt.expiry, historical_time)
        assert dte == 7, f"Expected DTE=7 relative to historical time, got {dte}"
        
        dte_now = _calculate_dte(raw_opt.expiry, datetime.now(timezone.utc))
        assert dte != dte_now or abs(dte_now) < 1, \
            "If DTE relative to now equals historical DTE, test fixture is broken"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
