"""
Minimal wrapper around Deribit's public endpoints used for backtesting.
Uses mainnet by default for historical data access.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

import httpx


DEFAULT_DERIBIT_MAINNET = "https://www.deribit.com/api/v2"
DEFAULT_DERIBIT_TESTNET = "https://test.deribit.com/api/v2"


class DeribitPublicClient:
    """
    Minimal wrapper around Deribit's public endpoints used for backtesting:
    - get_tradingview_chart_data
    - get_instruments
    - ticker
    """

    def __init__(
        self,
        base_url: str = DEFAULT_DERIBIT_MAINNET,
        timeout: float = 15.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(base_url=self.base_url, timeout=timeout)

    def close(self) -> None:
        """Close the HTTP client."""
        self.client.close()

    def _get(self, method: str, params: Dict[str, Any]) -> Any:
        """Make a GET request to the public API."""
        resp = self.client.get("/public/" + method, params=params)
        resp.raise_for_status()
        data = resp.json()
        if data.get("error"):
            raise RuntimeError(f"Deribit API error: {data['error']}")
        return data["result"]

    def get_tradingview_chart_data(
        self,
        instrument_name: str,
        start: datetime,
        end: datetime,
        resolution: str,
    ) -> Dict[str, Any]:
        """
        Wrap public/get_tradingview_chart_data.
        resolution examples: '1','5','15','60','240','1D'.
        Returns dict with keys: ticks, open, high, low, close, volume
        """
        start_ms = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)
        return self._get(
            "get_tradingview_chart_data",
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
        kind: 'option', 'future', etc.
        """
        return self._get(
            "get_instruments",
            {
                "currency": currency,
                "kind": kind,
                "expired": expired,
            },
        )

    def get_ticker(self, instrument_name: str) -> Dict[str, Any]:
        """
        Get ticker data including mark_price, greeks, etc.
        """
        return self._get("ticker", {"instrument_name": instrument_name})

    def get_index_price(self, index_name: str) -> Dict[str, Any]:
        """
        Get current index price.
        index_name examples: 'btc_usd', 'eth_usd'
        """
        return self._get("get_index_price", {"index_name": index_name})

    def get_book_summary_by_currency(
        self, currency: str, kind: str = "option"
    ) -> List[Dict[str, Any]]:
        """
        Get book summary for all instruments of a currency in one bulk call.
        Returns mark_price, mark_iv, underlying_price, etc. for each instrument.
        Much more efficient than calling get_ticker for each instrument.
        """
        return self._get(
            "get_book_summary_by_currency",
            {"currency": currency, "kind": kind},
        )
