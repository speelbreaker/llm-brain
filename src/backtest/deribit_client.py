"""
Deribit public API client for backtesting.
Extends DeribitBaseClient for public endpoints only (no authentication).
Uses mainnet by default for historical data access.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

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
        return self._make_public_request(
            "public/get_tradingview_chart_data",
            {
                "instrument_name": instrument_name,
                "start_timestamp": start_ms,
                "end_timestamp": end_ms,
                "resolution": resolution,
            },
        )

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
