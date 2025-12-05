"""
MarketDataSource implementation using Deribit's public API.
Suitable for live chain snapshots and OHLC pulls.
For bulk historical backtest, cache this or use offline data.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Literal, Optional

import pandas as pd

from .data_source import MarketDataSource, Timeframe
from .types import OptionSnapshot
from .deribit_client import DeribitPublicClient


class DeribitDataSource(MarketDataSource):
    """
    MarketDataSource using Deribit's public API.

    NOTE: For heavy historical backtesting, you'll likely want to cache or replace
    this with a local DB / CSV-based implementation, but the interface stays the same.
    """

    def __init__(self, client: Optional[DeribitPublicClient] = None):
        self.client = client or DeribitPublicClient()

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self.client.close()

    @staticmethod
    def _timeframe_to_resolution(timeframe: Timeframe) -> str:
        """Convert timeframe string to Deribit resolution."""
        mapping = {
            "1m": "1",
            "5m": "5",
            "15m": "15",
            "1h": "60",
            "4h": "240",
            "1d": "1D",
        }
        return mapping[timeframe]

    def get_spot_ohlc(
        self,
        underlying: str,
        start: datetime,
        end: datetime,
        timeframe: Timeframe,
    ) -> pd.DataFrame:
        """
        Use Deribit index as spot proxy, e.g. BTC -> btc_usd index.
        Falls back to perpetual futures if index fails.
        """
        index_name = f"{underlying.lower()}_usd"
        
        try:
            res = self.client.get_tradingview_chart_data(
                instrument_name=index_name,
                start=start,
                end=end,
                resolution=self._timeframe_to_resolution(timeframe),
            )
        except Exception:
            perp_name = f"{underlying}-PERPETUAL"
            res = self.client.get_tradingview_chart_data(
                instrument_name=perp_name,
                start=start,
                end=end,
                resolution=self._timeframe_to_resolution(timeframe),
            )

        if not res.get("ticks"):
            return pd.DataFrame()

        timestamps = [
            datetime.fromtimestamp(ts / 1000, tz=timezone.utc) for ts in res["ticks"]
        ]
        df = pd.DataFrame(
            {
                "open": res["open"],
                "high": res["high"],
                "low": res["low"],
                "close": res["close"],
                "volume": res["volume"],
            },
            index=pd.DatetimeIndex(timestamps, name="timestamp"),
        )
        return df

    def list_option_chain(
        self,
        underlying: str,
        as_of: datetime,
        settlement_ccy: str = "USDC",
        margin_type: Literal["linear", "inverse"] = "linear",
    ) -> List[OptionSnapshot]:
        """
        Chain snapshot at ~as_of:
        - list non-expired options for underlying
        - attach delta/IV/mark via ticker
        - filter by settlement currency and margin type
        
        Args:
            underlying: "BTC" or "ETH"
            as_of: Timestamp for filtering expired options
            settlement_ccy: Settlement currency ("USDC" for linear, "BTC"/"ETH" for inverse)
            margin_type: "linear" or "inverse"
        """
        instruments = self.client.get_instruments(currency=underlying, kind="option")
        snapshots: List[OptionSnapshot] = []
        
        total = 0
        after_expiry_filter = 0
        after_margin_filter = 0
        after_settlement_filter = 0

        for inst in instruments:
            name = inst["instrument_name"]
            parts = name.split("-")
            if len(parts) < 4:
                continue

            total += 1
            
            cur = parts[0]
            expiry_ts = inst.get("expiration_timestamp")
            if expiry_ts is None:
                continue
            expiry = datetime.fromtimestamp(expiry_ts / 1000, tz=timezone.utc)

            if expiry <= as_of:
                continue
            
            after_expiry_filter += 1
            
            inst_settlement = inst.get("settlement_currency", "").upper()
            is_linear = inst_settlement == "USDC" or inst_settlement == "USD"
            
            if margin_type == "linear" and not is_linear:
                continue
            if margin_type == "inverse" and is_linear:
                continue
            
            after_margin_filter += 1
            
            if settlement_ccy.upper() != "ANY":
                if is_linear and inst_settlement not in ["USDC", "USD"]:
                    continue
                if not is_linear and inst_settlement.upper() != cur.upper():
                    continue
            
            after_settlement_filter += 1

            strike_str = parts[2]
            cp_flag = parts[3].upper()
            kind: Literal["call", "put"] = "call" if cp_flag == "C" else "put"

            try:
                strike = float(strike_str)
            except ValueError:
                continue

            try:
                ticker = self.client.get_ticker(name)
                greeks = ticker.get("greeks") or {}
                delta = greeks.get("delta")
                iv = ticker.get("mark_iv")
                mark = ticker.get("mark_price")

                snapshots.append(
                    OptionSnapshot(
                        instrument_name=name,
                        underlying=cur,
                        kind=kind,
                        strike=strike,
                        expiry=expiry,
                        delta=float(delta) if delta is not None else None,
                        iv=float(iv) if iv is not None else None,
                        mark_price=float(mark) if mark is not None else None,
                        settlement_ccy=inst_settlement if inst_settlement else "USDC",
                        margin_type="linear" if is_linear else "inverse",
                    )
                )
            except Exception:
                continue

        print(
            f"[BACKTEST] {underlying} @ {as_of} (margin={margin_type}, settle={settlement_ccy}) - "
            f"total={total}, after_expiry={after_expiry_filter}, "
            f"after_margin={after_margin_filter}, after_settlement={after_settlement_filter}, "
            f"snapshots={len(snapshots)}",
            flush=True
        )
        return snapshots

    def get_option_ohlc(
        self,
        instrument_name: str,
        start: datetime,
        end: datetime,
        timeframe: Timeframe,
    ) -> pd.DataFrame:
        """
        Get OHLC data for a specific option instrument.
        """
        try:
            res = self.client.get_tradingview_chart_data(
                instrument_name=instrument_name,
                start=start,
                end=end,
                resolution=self._timeframe_to_resolution(timeframe),
            )
        except Exception:
            return pd.DataFrame()

        if not res.get("ticks"):
            return pd.DataFrame()

        timestamps = [
            datetime.fromtimestamp(ts / 1000, tz=timezone.utc) for ts in res["ticks"]
        ]
        df = pd.DataFrame(
            {
                "open": res["open"],
                "high": res["high"],
                "low": res["low"],
                "close": res["close"],
                "volume": res["volume"],
            },
            index=pd.DatetimeIndex(timestamps, name="timestamp"),
        )
        return df
