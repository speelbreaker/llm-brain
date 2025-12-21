from __future__ import annotations

from datetime import date, datetime, timezone

import pandas as pd
import pytest

from src.backtest.live_deribit_data_source import LiveDeribitDataSource


def test_live_deribit_datasource_prefers_usd_columns(monkeypatch) -> None:
    snap_time = datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc)
    expiry_ts = datetime(2025, 1, 8, 0, 0, tzinfo=timezone.utc).timestamp()

    # mark_price is BTC-quoted (0.001 BTC) => $50 at $50k spot
    df = pd.DataFrame(
        [
            {
                "harvest_time": snap_time,
                "instrument_name": "BTC-08JAN25-60000-C",
                "underlying": "BTC",
                "expiry_timestamp": expiry_ts,
                "option_type": "C",
                "strike": 60000.0,
                "underlying_price": 50000.0,
                "mark_price": 0.001,
                "mark_price_usd": 50.0,
                "mark_iv": 50.0,
                "greek_delta": 0.25,
            }
        ]
    )

    def _fake_build_live_deribit_exam_dataset(**kwargs):
        return df, {"ok": True}

    import src.data.live_deribit_exam as exam_mod

    monkeypatch.setattr(exam_mod, "build_live_deribit_exam_dataset", _fake_build_live_deribit_exam_dataset)

    ds = LiveDeribitDataSource(
        underlying="BTC",
        start_date=date(2025, 1, 1),
        end_date=date(2025, 1, 2),
    )

    chain = ds.list_option_chain("BTC", snap_time)
    assert len(chain) == 1
    assert chain[0].mark_price == 50.0

    ohlc = ds.get_option_ohlc("BTC-08JAN25-60000-C", snap_time, snap_time, timeframe="1h")
    assert not ohlc.empty
    assert float(ohlc["close"].iloc[0]) == 50.0
