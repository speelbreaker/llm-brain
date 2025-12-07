"""
Base Deribit client with shared HTTP/JSON-RPC logic.

This module provides the foundation for both the live trading client
and the backtest/public data client. It implements:
- HTTP request handling with httpx
- JSON-RPC error parsing
- Common response validation

Subclasses add authentication (trading) or remain public-only (backtest).
"""
from __future__ import annotations

import time
from typing import Any, Dict, Optional

import httpx


class DeribitAPIError(Exception):
    """Custom exception for Deribit API errors."""
    def __init__(self, code: int, message: str):
        self.code = code
        self.message = message
        super().__init__(f"Deribit API Error {code}: {message}")


class DeribitBaseClient:
    """
    Base Deribit API client with shared HTTP logic.
    
    Subclasses:
    - DeribitClient (src/deribit_client.py): Adds authentication for trading
    - DeribitPublicClient (src/backtest/deribit_client.py): Public endpoints only
    """
    
    DEFAULT_MAINNET = "https://www.deribit.com"
    DEFAULT_TESTNET = "https://test.deribit.com"
    
    def __init__(
        self,
        base_url: str,
        timeout: float = 30.0,
    ):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=timeout)
    
    def _make_public_request(
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Make a public API request using GET.
        
        Args:
            method: API method name (e.g., "public/get_index_price")
            params: Query parameters
            
        Returns:
            Result from Deribit API response
            
        Raises:
            DeribitAPIError: If API returns an error
        """
        url = f"{self.base_url}/api/v2/{method}"
        
        try:
            response = self._client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            self._check_error(data)
            return data.get("result")
            
        except httpx.HTTPStatusError as e:
            self._handle_http_error(e)
        except httpx.RequestError as e:
            raise DeribitAPIError(-1, f"Request error: {e}")
    
    def _make_jsonrpc_request(
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Any:
        """
        Make a JSON-RPC 2.0 request using POST.
        
        Used for private endpoints that require authentication.
        
        Args:
            method: RPC method name (e.g., "private/get_positions")
            params: RPC parameters
            headers: Additional headers (e.g., Authorization)
            
        Returns:
            Result from Deribit API response
            
        Raises:
            DeribitAPIError: If API returns an error
        """
        url = f"{self.base_url}/api/v2/"
        
        json_rpc = {
            "jsonrpc": "2.0",
            "id": int(time.time() * 1000),
            "method": method,
            "params": params or {},
        }
        
        request_headers = {"Content-Type": "application/json"}
        if headers:
            request_headers.update(headers)
        
        try:
            response = self._client.post(url, json=json_rpc, headers=request_headers)
            response.raise_for_status()
            data = response.json()
            
            self._check_error(data)
            return data.get("result")
            
        except httpx.HTTPStatusError as e:
            self._handle_http_error(e)
        except httpx.RequestError as e:
            raise DeribitAPIError(-1, f"Request error: {e}")
    
    def _check_error(self, data: Dict[str, Any]) -> None:
        """Check API response for errors and raise if found."""
        if "error" in data:
            error = data["error"]
            raise DeribitAPIError(
                error.get("code", -1),
                error.get("message", "Unknown error"),
            )
    
    def _handle_http_error(self, e: httpx.HTTPStatusError) -> None:
        """Handle HTTP status errors, extracting API error details if available."""
        try:
            error_body = e.response.json() if e.response else None
            if error_body and "error" in error_body:
                err = error_body["error"]
                raise DeribitAPIError(err.get("code", -1), err.get("message", str(e)))
        except Exception:
            pass
        raise DeribitAPIError(-1, f"HTTP error: {e}")
    
    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()
    
    def __enter__(self) -> "DeribitBaseClient":
        return self
    
    def __exit__(self, *args: Any) -> None:
        self.close()
