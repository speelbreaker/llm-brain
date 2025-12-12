"""
Base Deribit client with shared HTTP/JSON-RPC logic.

This module provides the foundation for both the live trading client
and the backtest/public data client. It implements:
- HTTP request handling with httpx
- JSON-RPC error parsing
- Common response validation
- Error classification for consistent failure handling

Subclasses add authentication (trading) or remain public-only (backtest).
"""
from __future__ import annotations

import time
from enum import Enum
from typing import Any, Dict, Optional

import httpx


DEFAULT_TIMEOUT_SECONDS = 10.0
HEALTHCHECK_TIMEOUT_SECONDS = 5.0


class DeribitErrorCode(str, Enum):
    """Classification of Deribit API errors for consistent handling."""
    NETWORK = "network_error"
    TIMEOUT = "timeout_error"
    AUTH = "auth_error"
    RATE_LIMIT = "rate_limit"
    FORBIDDEN = "forbidden"
    NOT_FOUND = "not_found"
    BAD_REQUEST = "bad_request"
    SERVER_ERROR = "server_error"
    UNKNOWN = "unknown"


class HealthSeverity(str, Enum):
    """Severity classification for health guard decisions.
    
    TRANSIENT: Temporary issues that will likely resolve on retry (network glitch, rate limit)
    DEGRADED: System is impaired but can continue with reduced functionality (non-critical endpoints down)
    FATAL: Critical failure requiring agent pause (auth failure, mainnet safety violation)
    """
    TRANSIENT = "transient"
    DEGRADED = "degraded"
    FATAL = "fatal"


ERROR_SEVERITY_MAP: Dict[DeribitErrorCode, HealthSeverity] = {
    DeribitErrorCode.NETWORK: HealthSeverity.TRANSIENT,
    DeribitErrorCode.TIMEOUT: HealthSeverity.TRANSIENT,
    DeribitErrorCode.RATE_LIMIT: HealthSeverity.TRANSIENT,
    DeribitErrorCode.SERVER_ERROR: HealthSeverity.TRANSIENT,
    DeribitErrorCode.AUTH: HealthSeverity.FATAL,
    DeribitErrorCode.FORBIDDEN: HealthSeverity.FATAL,
    DeribitErrorCode.NOT_FOUND: HealthSeverity.DEGRADED,
    DeribitErrorCode.BAD_REQUEST: HealthSeverity.DEGRADED,
    DeribitErrorCode.UNKNOWN: HealthSeverity.DEGRADED,
}


def get_error_severity(error_code: DeribitErrorCode) -> HealthSeverity:
    """Get the severity classification for an error code."""
    return ERROR_SEVERITY_MAP.get(error_code, HealthSeverity.DEGRADED)


def classify_http_status(status_code: int) -> DeribitErrorCode:
    """Classify HTTP status code to error code."""
    if status_code == 401:
        return DeribitErrorCode.AUTH
    elif status_code == 403:
        return DeribitErrorCode.FORBIDDEN
    elif status_code == 404:
        return DeribitErrorCode.NOT_FOUND
    elif status_code == 429:
        return DeribitErrorCode.RATE_LIMIT
    elif status_code == 400:
        return DeribitErrorCode.BAD_REQUEST
    elif 500 <= status_code < 600:
        return DeribitErrorCode.SERVER_ERROR
    else:
        return DeribitErrorCode.UNKNOWN


class DeribitAPIError(Exception):
    """
    Custom exception for Deribit API errors.
    
    Attributes:
        code: Numeric error code from Deribit (-1 for local errors)
        message: Human-readable error message
        error_code: Classified error type for programmatic handling
        http_status: HTTP status code if applicable
    """
    def __init__(
        self,
        code: int,
        message: str,
        error_code: DeribitErrorCode = DeribitErrorCode.UNKNOWN,
        http_status: Optional[int] = None,
    ):
        self.code = code
        self.message = message
        self.error_code = error_code
        self.http_status = http_status
        super().__init__(f"Deribit API Error {code}: {message} [{error_code.value}]")


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
        except httpx.TimeoutException as e:
            raise DeribitAPIError(
                -1, f"Request timeout: {e}",
                error_code=DeribitErrorCode.TIMEOUT,
            )
        except httpx.RequestError as e:
            raise DeribitAPIError(
                -1, f"Network error: {e}",
                error_code=DeribitErrorCode.NETWORK,
            )
    
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
        except httpx.TimeoutException as e:
            raise DeribitAPIError(
                -1, f"Request timeout: {e}",
                error_code=DeribitErrorCode.TIMEOUT,
            )
        except httpx.RequestError as e:
            raise DeribitAPIError(
                -1, f"Network error: {e}",
                error_code=DeribitErrorCode.NETWORK,
            )
    
    def _check_error(self, data: Dict[str, Any]) -> None:
        """Check API response for errors and raise if found."""
        if "error" in data:
            error = data["error"]
            err_code = error.get("code", -1)
            err_msg = error.get("message", "Unknown error")
            
            error_type = DeribitErrorCode.UNKNOWN
            msg_lower = err_msg.lower()
            if "unauthorized" in msg_lower or "invalid" in msg_lower and "token" in msg_lower:
                error_type = DeribitErrorCode.AUTH
            elif "forbidden" in msg_lower or "access denied" in msg_lower:
                error_type = DeribitErrorCode.FORBIDDEN
            elif "rate limit" in msg_lower or "too many" in msg_lower:
                error_type = DeribitErrorCode.RATE_LIMIT
            
            raise DeribitAPIError(err_code, err_msg, error_code=error_type)
    
    def _handle_http_error(self, e: httpx.HTTPStatusError) -> None:
        """Handle HTTP status errors, extracting API error details if available."""
        http_status = e.response.status_code if e.response else None
        error_type = classify_http_status(http_status) if http_status else DeribitErrorCode.UNKNOWN
        
        try:
            error_body = e.response.json() if e.response else None
            if error_body and "error" in error_body:
                err = error_body["error"]
                raise DeribitAPIError(
                    err.get("code", -1),
                    err.get("message", str(e)),
                    error_code=error_type,
                    http_status=http_status,
                )
        except DeribitAPIError:
            raise
        except Exception:
            pass
        
        status_msg = f" (HTTP {http_status})" if http_status else ""
        raise DeribitAPIError(
            -1,
            f"HTTP error{status_msg}: {e}",
            error_code=error_type,
            http_status=http_status,
        )
    
    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()
    
    def __enter__(self) -> "DeribitBaseClient":
        return self
    
    def __exit__(self, *args: Any) -> None:
        self.close()
