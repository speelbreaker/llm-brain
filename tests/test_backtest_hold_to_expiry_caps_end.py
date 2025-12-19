from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd


def test_hold_to_expiry_caps_spot_ohlc_end_to_cfg_end():
    from src.backtest.covered_call_simulator import CoveredCallSimulator
    from src.backtest.types import CallSimulationConfig, OptionSnapshot

    class FakeDS:
        def __init__(self):
            self.calls = []

        def get_spot_ohlc(self, *, underlying, start, end, timeframe):
            self.calls.append({"underlying": underlying, "start": start, "end": end, "timeframe": timeframe})
            idx = pd.date_range(start=start, end=end, freq="D", tz=timezone.utc)
            if len(idx) == 0:
                return pd.DataFrame()
            return pd.DataFrame({"close": [100.0] * len(idx)}, index=idx)

        def get_option_ohlc(self, *args, **kwargs):
            return pd.DataFrame()

        def close(self):
            return None

    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    cfg_end = datetime(2024, 1, 5, tzinfo=timezone.utc)
    expiry = datetime(2024, 1, 20, tzinfo=timezone.utc)
    decision_time = datetime(2024, 1, 2, tzinfo=timezone.utc)

    cfg = CallSimulationConfig(
        underlying="BTC",
        start=start,
        end=cfg_end,
        timeframe="1d",
        decision_interval_bars=1,
        initial_spot_position=1.0,
        contract_size=1.0,
        fee_rate=0.0,
        pricing_mode="synthetic_bs",
    )

    ds = FakeDS()
    sim = CoveredCallSimulator(data_source=ds, config=cfg)
    # Avoid extra lookback call altering the last recorded call ordering.
    sim._spot_history_cache = [(start + timedelta(days=i), 100.0 + i) for i in range(10)]

    opt = OptionSnapshot(
        instrument_name="BTC-TEST",
        underlying="BTC",
        kind="call",
        strike=110.0,
        expiry=expiry,
        delta=0.25,
        iv=0.7,
        mark_price=0.01,
        settlement_ccy="USDC",
        margin_type="linear",
    )

    trade = sim._simulate_call_hold_to_expiry(decision_time, opt)
    assert trade is not None

    # Find the call for the spot_df request (timeframe == cfg.timeframe, start == decision_time)
    matching = [c for c in ds.calls if c["timeframe"] == cfg.timeframe and c["start"] == decision_time]
    assert matching, "expected a spot_ohlc call for the hold-to-expiry window"
    assert matching[0]["end"] == cfg_end
