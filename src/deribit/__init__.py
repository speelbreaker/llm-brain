"""
Deribit API client package.

Provides shared base client and error types for both live trading
and backtest/public data clients.
"""
from src.deribit.base_client import DeribitBaseClient, DeribitAPIError

__all__ = ["DeribitBaseClient", "DeribitAPIError"]
