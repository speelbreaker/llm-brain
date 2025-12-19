"""Regression test for blocking Deribit API live_chain in historical backtests.

DeribitDataSource does not provide true historical option chain snapshots; using it
with chain_mode="live_chain" for timestamps in the past is look-ahead.

We enforce a guard in src.backtest.state_builder.build_historical_state() that
falls back to synthetic candidates and annotates provenance.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

import pandas as pd


def test_deribit_data_source_live_chain_is_blocked_historically(monkeypatch):
    from src.backtest.state_builder import build_historical_state
    from src.backtest.types import CallSimulationConfig
    from src.backtest.pricing import RegimeState

    # Prevent the test from needing 60d of candles / numpy logic.
    monkeypatch.setattr(
        "src.backtest.state_builder.compute_market_context_from_ds",
        lambda *args, **kwargs: None,
    )

    # Class name must be exactly "DeribitDataSource" to trigger the guard.
    class DeribitDataSource:
        def get_spot_ohlc(self, underlying: str, start: datetime, end: datetime, timeframe: str):
            idx = pd.DatetimeIndex([end])
            return pd.DataFrame({"close": [60000.0]}, index=idx)

        def list_option_chain(self, *args, **kwargs):
            raise AssertionError("DeribitDataSource.list_option_chain() must not be called in this path")

    ds = DeribitDataSource()

    t = datetime(2020, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    cfg = CallSimulationConfig(
        underlying="BTC",
        start=t,
        end=t,
        timeframe="1h",
        decision_interval_bars=1,
        initial_spot_position=1.0,
        contract_size=1.0,
        fee_rate=0.0005,
    )

    cfg = replace(cfg, chain_mode="live_chain")

    # Ensure sigma is not clamped to an ultra-low value (which can produce zero candidates).
    regime_state = RegimeState(regime=object(), iv_atm=60.0)

    state = build_historical_state(ds, cfg, t, regime_state=regime_state)

    assert state["provenance"]["chain_source"] == "live_chain_blocked_deribit_api"
    assert state["provenance"]["used_synthetic_fallback"] is True
    assert len(state["candidate_options"]) > 0
