"""
LIVE_DERIBIT data source for backtesting using captured Deribit snapshots.

This data source loads exam datasets built from harvested Deribit options data
and provides the MarketDataSource interface for the backtester.
"""
from __future__ import annotations

from datetime import datetime, date, timezone, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Any, TYPE_CHECKING

import pandas as pd
import numpy as np

from .data_source import Timeframe
from .types import OptionSnapshot

if TYPE_CHECKING:
    pass


class LiveDeribitDataSource:
    """
    Data source that uses captured Deribit snapshots (exam dataset format).
    
    Implements the MarketDataSource protocol for backtesting with real
    market data captured by the data harvester.
    """
    
    def __init__(
        self,
        underlying: str,
        start_date: date,
        end_date: date,
        base_dir: str | Path = "data/live_deribit",
        canonical_underlying: Optional[str] = None,
    ):
        """
        Initialize the data source.
        
        Args:
            underlying: Asset symbol for directory lookup (e.g., "BTC_USDC", "ETH_USDC")
            start_date: Start date for data
            end_date: End date for data
            base_dir: Root directory for harvester data
            canonical_underlying: The canonical underlying symbol (e.g., "BTC", "ETH") 
                                  used in OptionSnapshots. If None, derived from underlying.
        """
        self.underlying = underlying.upper()
        self.start_date = start_date
        self.end_date = end_date
        self.base_dir = Path(base_dir)
        
        if canonical_underlying:
            self._canonical_underlying = canonical_underlying.upper()
        else:
            self._canonical_underlying = self.underlying.replace("_USDC", "").upper()
        
        self._df: Optional[pd.DataFrame] = None
        self._summary: Optional[Dict[str, Any]] = None
        self._spot_ohlc_cache: Optional[pd.DataFrame] = None
        self._snapshot_times: Optional[List[datetime]] = None
        
    def _ensure_loaded(self) -> None:
        """Load data if not already loaded."""
        if self._df is not None:
            return
            
        from src.data.live_deribit_exam import build_live_deribit_exam_dataset
        
        self._df, self._summary = build_live_deribit_exam_dataset(
            underlying=self.underlying,
            start_date=self.start_date,
            end_date=self.end_date,
            base_dir=self.base_dir,
            write_files=False,
        )
        
        if "harvest_time" in self._df.columns:
            self._df["harvest_time"] = pd.to_datetime(self._df["harvest_time"], utc=True)
            self._snapshot_times = sorted(self._df["harvest_time"].unique().tolist())
        else:
            self._snapshot_times = []
    
    def _find_closest_snapshot_time(self, as_of: datetime, tolerance_minutes: int = 60) -> Optional[datetime]:
        """Find the closest snapshot time to the requested time."""
        self._ensure_loaded()
        
        if not self._snapshot_times:
            return None
        
        as_of_ts = as_of.timestamp() if as_of.tzinfo else as_of.replace(tzinfo=timezone.utc).timestamp()
        
        best_time = None
        best_diff = float("inf")
        
        for snap_time in self._snapshot_times:
            snap_ts = snap_time.timestamp() if hasattr(snap_time, 'timestamp') else pd.Timestamp(snap_time).timestamp()
            diff = abs(snap_ts - as_of_ts)
            if diff < best_diff:
                best_diff = diff
                best_time = snap_time
        
        if best_diff > tolerance_minutes * 60:
            return None
        
        return best_time
    
    def _build_spot_ohlc_cache(self) -> None:
        """Build the spot OHLC cache from the full dataset."""
        if self._spot_ohlc_cache is not None:
            return
        
        if self._df is None or self._df.empty:
            self._spot_ohlc_cache = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
            return
        
        if "harvest_time" not in self._df.columns or "underlying_price" not in self._df.columns:
            self._spot_ohlc_cache = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
            return
        
        spot_df = self._df.groupby("harvest_time")["underlying_price"].agg(
            open="first",
            high="max",
            low="min",
            close="last",
        ).reset_index()
        
        spot_df["volume"] = 0.0
        spot_df = spot_df.set_index("harvest_time")
        spot_df.index.name = None
        
        self._spot_ohlc_cache = spot_df[["open", "high", "low", "close", "volume"]]
    
    def get_spot_ohlc(
        self,
        underlying: str,
        start: datetime,
        end: datetime,
        timeframe: Timeframe,
    ) -> pd.DataFrame:
        """
        Return OHLCV for the underlying index/future used as 'spot' for options.
        
        Since we have snapshot data (not continuous), we synthesize OHLC from
        the underlying_price column at each snapshot time. Uses cached aggregation
        for performance.
        """
        self._ensure_loaded()
        self._build_spot_ohlc_cache()
        
        if self._spot_ohlc_cache is None or self._spot_ohlc_cache.empty:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        
        start_utc = start if start.tzinfo else start.replace(tzinfo=timezone.utc)
        end_utc = end if end.tzinfo else end.replace(tzinfo=timezone.utc)
        
        mask = (self._spot_ohlc_cache.index >= start_utc) & (self._spot_ohlc_cache.index <= end_utc)
        return self._spot_ohlc_cache[mask].copy()
    
    def list_option_chain(
        self,
        underlying: str,
        as_of: datetime,
        settlement_ccy: str = "USDC",
        margin_type: str = "linear",
    ) -> List[OptionSnapshot]:
        """
        Return option chain snapshot for underlying at (or near) 'as_of'.
        
        Note: settlement_ccy and margin_type parameters are accepted for interface
        compatibility but not used for filtering since harvested data is already
        filtered to USDC linear options.
        """
        self._ensure_loaded()
        
        if self._df is None or self._df.empty:
            return []
        
        snap_time = self._find_closest_snapshot_time(as_of, tolerance_minutes=120)
        if snap_time is None:
            return []
        
        df = self._df[self._df["harvest_time"] == snap_time].copy()
        
        if df.empty:
            return []
        
        options: List[OptionSnapshot] = []
        
        for _, row in df.iterrows():
            try:
                expiry_ts = row.get("expiry_timestamp")
                if pd.notna(expiry_ts):
                    expiry_dt = datetime.fromtimestamp(float(expiry_ts), tz=timezone.utc)
                elif pd.notna(row.get("expiry")):
                    expiry_val = row["expiry"]
                    if hasattr(expiry_val, "to_pydatetime"):
                        expiry_dt = expiry_val.to_pydatetime()
                    else:
                        expiry_dt = pd.to_datetime(expiry_val).to_pydatetime()
                    if expiry_dt.tzinfo is None:
                        expiry_dt = expiry_dt.replace(tzinfo=timezone.utc)
                else:
                    continue
                
                delta = row.get("greek_delta")
                if pd.isna(delta):
                    delta = None
                else:
                    delta = float(delta)
                
                iv = row.get("mark_iv")
                if pd.isna(iv):
                    iv = None
                else:
                    iv = float(iv)
                    if iv > 1:
                        iv = iv / 100.0
                
                mark_price = row.get("mark_price")
                if pd.isna(mark_price):
                    mark_price = None
                else:
                    mark_price = float(mark_price)
                
                option_type = row.get("option_type", "C")
                kind = "call" if str(option_type).upper().startswith("C") else "put"
                
                opt = OptionSnapshot(
                    instrument_name=str(row["instrument_name"]),
                    underlying=self._canonical_underlying,
                    kind=kind,
                    strike=float(row["strike"]),
                    expiry=expiry_dt,
                    delta=delta,
                    iv=iv,
                    mark_price=mark_price,
                    settlement_ccy="USDC",
                    margin_type="linear",
                )
                options.append(opt)
                
            except Exception:
                continue
        
        return options
    
    def get_option_ohlc(
        self,
        instrument_name: str,
        start: datetime,
        end: datetime,
        timeframe: Timeframe,
    ) -> pd.DataFrame:
        """
        Return OHLCV for a specific option instrument over the requested period.
        
        Synthesizes OHLC from snapshot mark_price data.
        """
        self._ensure_loaded()
        
        if self._df is None or self._df.empty:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        
        start_utc = start if start.tzinfo else start.replace(tzinfo=timezone.utc)
        end_utc = end if end.tzinfo else end.replace(tzinfo=timezone.utc)
        
        mask = (
            (self._df["instrument_name"] == instrument_name) &
            (self._df["harvest_time"] >= start_utc) &
            (self._df["harvest_time"] <= end_utc)
        )
        df = self._df[mask].copy()
        
        if df.empty:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        
        df = df.sort_values("harvest_time")
        
        ohlc = df.groupby("harvest_time")["mark_price"].agg(
            open="first",
            high="max",
            low="min",
            close="last",
        ).reset_index()
        
        ohlc["volume"] = 0.0
        ohlc = ohlc.set_index("harvest_time")
        ohlc.index.name = None
        
        return ohlc[["open", "high", "low", "close", "volume"]]
    
    def get_summary(self) -> Dict[str, Any]:
        """Return the summary dict for the loaded data."""
        self._ensure_loaded()
        return self._summary or {}
    
    def get_dataframe(self) -> pd.DataFrame:
        """Return the full DataFrame."""
        self._ensure_loaded()
        return self._df if self._df is not None else pd.DataFrame()
    
    def close(self) -> None:
        """Close/cleanup resources (no-op for file-based source)."""
        pass
