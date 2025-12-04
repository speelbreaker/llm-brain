"""
Deribit API client for testnet.
Provides typed methods for public and private API endpoints.
"""
from __future__ import annotations

import hashlib
import hmac
import time
from datetime import datetime
from typing import Any, Optional

import httpx

from src.config import settings


class DeribitAPIError(Exception):
    """Custom exception for Deribit API errors."""
    def __init__(self, code: int, message: str):
        self.code = code
        self.message = message
        super().__init__(f"Deribit API Error {code}: {message}")


class DeribitClient:
    """
    Minimal wrapper for Deribit testnet API.
    Supports both public and private endpoints.
    """
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
    ):
        self.base_url = base_url or settings.deribit_base_url
        self.client_id = client_id or settings.deribit_client_id
        self.client_secret = client_secret or settings.deribit_client_secret
        
        self._access_token: Optional[str] = None
        self._refresh_token: Optional[str] = None
        self._token_expiry: float = 0
        
        self._client = httpx.Client(timeout=30.0)
    
    def _make_request(
        self,
        method: str,
        params: Optional[dict[str, Any]] = None,
        private: bool = False,
    ) -> Any:
        """Make an API request to Deribit."""
        if private:
            self._ensure_authenticated()
        
        url = f"{self.base_url}/api/v2/{method}"
        
        headers = {}
        if private and self._access_token:
            headers["Authorization"] = f"Bearer {self._access_token}"
        
        try:
            response = self._client.get(url, params=params, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            if "error" in data:
                error = data["error"]
                raise DeribitAPIError(error.get("code", -1), error.get("message", "Unknown error"))
            
            return data.get("result")
        except httpx.HTTPStatusError as e:
            raise DeribitAPIError(-1, f"HTTP error: {e}")
        except httpx.RequestError as e:
            raise DeribitAPIError(-1, f"Request error: {e}")
    
    def _ensure_authenticated(self) -> None:
        """Ensure we have a valid access token."""
        if not self.client_id or not self.client_secret:
            raise DeribitAPIError(-1, "Client ID and secret are required for private endpoints")
        
        if self._access_token and time.time() < self._token_expiry - 60:
            return
        
        self._authenticate()
    
    def _authenticate(self) -> None:
        """Authenticate and get access token."""
        params = {
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        }
        
        result = self._make_request("public/auth", params)
        
        self._access_token = result["access_token"]
        self._refresh_token = result.get("refresh_token")
        self._token_expiry = time.time() + result.get("expires_in", 900)
    
    def get_index_price(self, currency: str) -> float:
        """Get the current index (spot) price for a currency."""
        params = {"index_name": f"{currency.lower()}_usd"}
        result = self._make_request("public/get_index_price", params)
        return result["index_price"]
    
    def get_ticker(self, instrument_name: str) -> dict[str, Any]:
        """Get ticker data for an instrument."""
        params = {"instrument_name": instrument_name}
        return self._make_request("public/ticker", params)
    
    def get_instruments(
        self,
        currency: str,
        kind: str = "option",
        expired: bool = False,
    ) -> list[dict[str, Any]]:
        """Get list of instruments for a currency."""
        params = {
            "currency": currency.upper(),
            "kind": kind,
            "expired": str(expired).lower(),
        }
        return self._make_request("public/get_instruments", params)
    
    def get_order_book(
        self,
        instrument_name: str,
        depth: int = 5,
    ) -> dict[str, Any]:
        """Get order book for an instrument."""
        params = {
            "instrument_name": instrument_name,
            "depth": depth,
        }
        return self._make_request("public/get_order_book", params)
    
    def get_account_summary(self, currency: str) -> dict[str, Any]:
        """Get account summary (private endpoint)."""
        params = {"currency": currency.upper()}
        return self._make_request("private/get_account_summary", params, private=True)
    
    def get_positions(
        self,
        currency: str,
        kind: str = "option",
    ) -> list[dict[str, Any]]:
        """Get open positions (private endpoint)."""
        params = {
            "currency": currency.upper(),
            "kind": kind,
        }
        return self._make_request("private/get_positions", params, private=True)
    
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
    ) -> dict[str, Any]:
        """
        Place an order (private endpoint).
        
        Args:
            instrument_name: Instrument to trade
            side: 'buy' or 'sell'
            amount: Order size
            order_type: 'limit' or 'market'
            price: Limit price (required for limit orders)
            post_only: If True, order will be cancelled if it would fill immediately
            reduce_only: If True, order will only reduce position
            label: Optional order label
        """
        params = {
            "instrument_name": instrument_name,
            "amount": amount,
            "type": order_type,
            "post_only": str(post_only).lower(),
            "reduce_only": str(reduce_only).lower(),
        }
        
        if price is not None:
            params["price"] = price
        
        if label:
            params["label"] = label
        
        method = f"private/{side}"
        return self._make_request(method, params, private=True)
    
    def cancel_order(self, order_id: str) -> dict[str, Any]:
        """Cancel an order by ID (private endpoint)."""
        params = {"order_id": order_id}
        return self._make_request("private/cancel", params, private=True)
    
    def cancel_all_by_instrument(self, instrument_name: str) -> int:
        """Cancel all orders for an instrument (private endpoint)."""
        params = {"instrument_name": instrument_name}
        return self._make_request("private/cancel_all_by_instrument", params, private=True)
    
    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()
    
    def __enter__(self) -> "DeribitClient":
        return self
    
    def __exit__(self, *args: Any) -> None:
        self.close()
