"""
Generic interface for market data used by the backtest simulator.
Implementations can be live Deribit API, offline CSV, Tardis, etc.
"""
from __future__ import annotations

from datetime import datetime
from typing import Protocol, Literal, List, TYPE_CHECKING
import pandas as pd

if TYPE_CHECKING:
    from .types import OptionSnapshot

Timeframe = Literal["1m", "5m", "15m", "1h", "4h", "1d"]


class MarketDataSource(Protocol):
    """
    Generic interface for market data used by the backtest simulator.
    Implementations can be live Deribit API, offline CSV, Tardis, etc.
    """

    def get_spot_ohlc(
        self,
        underlying: str,
        start: datetime,
        end: datetime,
        timeframe: Timeframe,
    ) -> pd.DataFrame:
        """
        Return OHLCV for the underlying index/future used as 'spot' for options.
        Index: timestamp.
        Columns: ['open', 'high', 'low', 'close', 'volume'].
        """
        ...

    def list_option_chain(
        self,
        underlying: str,
        as_of: datetime,
    ) -> "List[OptionSnapshot]":
        """
        Return option chain snapshot for underlying at (or near) 'as_of'.
        Returns List[OptionSnapshot] - the generic option snapshot type.
        """
        ...

    def get_option_ohlc(
        self,
        instrument_name: str,
        start: datetime,
        end: datetime,
        timeframe: Timeframe,
    ) -> pd.DataFrame:
        """
        Return OHLCV for a specific option instrument over the requested period.
        Same structure as get_spot_ohlc.
        """
        ...
