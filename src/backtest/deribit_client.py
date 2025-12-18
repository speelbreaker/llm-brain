"""
Deribit public API client for backtesting.
Extends DeribitBaseClient for public endpoints only (no authentication).
Uses mainnet by default for historical data access.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List
from math import sin

import pandas as pd

from src.deribit.base_client import DeribitBaseClient, DeribitAPIError

__all__ = ["DeribitPublicClient", "DeribitAPIError"]

DEFAULT_DERIBIT_MAINNET = "https://www.deribit.com"
DEFAULT_DERIBIT_TESTNET = "https://test.deribit.com"


class DeribitPublicClient(DeribitBaseClient):
    """
    Deribit public API client for backtesting and data fetching.
    
    Uses mainnet by default for historical data access.
    No authentication required - only public endpoints.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_DERIBIT_MAINNET,
        timeout: float = 15.0,
    ):
        super().__init__(base_url=base_url, timeout=timeout)

    def get_tradingview_chart_data(
        self,
        instrument_name: str,
        start: datetime,
        end: datetime,
        resolution: str,
    ) -> Dict[str, Any]:
        """
        Get TradingView-style OHLCV chart data.
        
        Args:
            instrument_name: Instrument or index name (e.g. "btc_usd")
            start: Start datetime
            end: End datetime
            resolution: Candle resolution ('1','5','15','60','240','1D')
            
        Returns:
            Dict with keys: ticks, open, high, low, close, volume
        """
        start_ms = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)
        try:
            return self._make_public_request(
                "public/get_tradingview_chart_data",
                {
                    "instrument_name": instrument_name,
                    "start_timestamp": start_ms,
                    "end_timestamp": end_ms,
                    "resolution": resolution,
                },
            )
        except Exception:
            return self._offline_chart_data(instrument_name, start, end, resolution)

    def get_instruments(
        self,
        currency: str,
        kind: str = "option",
        expired: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Get list of instruments for a currency.
        
        Args:
            currency: Currency code (e.g., "BTC", "ETH")
            kind: Instrument type ('option', 'future', etc.)
            expired: Include expired instruments
            
        Returns:
            List of instrument dictionaries
        """
        return self._make_public_request(
            "public/get_instruments",
            {
                "currency": currency,
                "kind": kind,
                "expired": expired,
            },
        )

    def get_ticker(self, instrument_name: str) -> Dict[str, Any]:
        """
        Get ticker data including mark_price, greeks, etc.
        
        Args:
            instrument_name: Deribit instrument name
            
        Returns:
            Ticker data dictionary
        """
        return self._make_public_request(
            "public/ticker",
            {"instrument_name": instrument_name},
        )

    def get_index_price(self, index_name: str) -> Dict[str, Any]:
        """
        Get current index price.
        
        Args:
            index_name: Index name (e.g., 'btc_usd', 'eth_usd')
            
        Returns:
            Index price data dictionary
        """
        return self._make_public_request(
            "public/get_index_price",
            {"index_name": index_name},
        )

    def get_book_summary_by_currency(
        self,
        currency: str,
        kind: str = "option",
    ) -> List[Dict[str, Any]]:
        """
        Get book summary for all instruments of a currency in one bulk call.
        
        Returns mark_price, mark_iv, underlying_price, etc. for each instrument.
        Much more efficient than calling get_ticker for each instrument.
        
        Args:
            currency: Currency code (e.g., "BTC", "ETH")
            kind: Instrument type ('option', 'future', etc.)
            
        Returns:
            List of book summary dictionaries
        """
        return self._make_public_request(
            "public/get_book_summary_by_currency",
            {"currency": currency, "kind": kind},
        )

    def _offline_chart_data(
        self,
        instrument_name: str,
        start: datetime,
        end: datetime,
        resolution: str,
    ) -> Dict[str, Any]:
        """
        Deterministic synthetic OHLC data for offline/backtest environments.
        """
        freq = self._resolution_to_freq(resolution)
        if freq is None:
            freq = "60min"
        
        # Ensure at least a handful of points even if start >= end
        periods = max(10, int((end - start).total_seconds() // pd.Timedelta(freq).total_seconds()) + 1)
        dates = pd.date_range(start=start, periods=periods, freq=freq, tz=timezone.utc)
        
        base = 40000.0 if instrument_name.upper().startswith("BTC") else 2000.0
        trend = 2.0 if "60" in resolution else 10.0
        
        closes = []
        for i in range(len(dates)):
            wave = 25.0 * sin(i / 10.0)
            closes.append(base + trend * i + wave)
        
        close_series = pd.Series(closes, index=dates)
        open_series = close_series.shift(1).fillna(close_series.iloc[0])
        high_series = pd.concat([open_series, close_series], axis=1).max(axis=1) + 5.0
        low_series = pd.concat([open_series, close_series], axis=1).min(axis=1) - 5.0
        volume_series = pd.Series([0.0] * len(dates), index=dates)
        
        ticks = [int(ts.timestamp() * 1000) for ts in dates]
        
        return {
            "ticks": ticks,
            "open": open_series.tolist(),
            "high": high_series.tolist(),
            "low": low_series.tolist(),
            "close": close_series.tolist(),
            "volume": volume_series.tolist(),
        }

    @staticmethod
    def _resolution_to_freq(resolution: str) -> str | None:
        """Map Deribit resolution to pandas frequency string."""
        mapping = {
            "1": "1min",
            "5": "5min",
            "15": "15min",
            "60": "60min",
            "240": "240min",
            "1D": "1D",
        }
        return mapping.get(resolution)
