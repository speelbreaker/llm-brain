"""
Real Scraper Data Source

MarketDataSource implementation that reads from normalized Real Scraper data.
Used for backtesting with historical option data from Kaggle/Real Scraper datasets.
"""

from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
import pandas as pd

from .data_source import MarketDataSource, Timeframe
from .types import OptionSnapshot
from .real_scraper_loader import load_real_scraper_data, get_real_scraper_snapshot


class RealScraperDataSource(MarketDataSource):
    """
    MarketDataSource implementation using Real Scraper historical data.
    
    Loads data from data/real_scraper/<UNDERLYING>/<DATE>/*.parquet files
    produced by scripts/import_real_scraper_deribit.py.
    """
    
    def __init__(
        self,
        underlying: str,
        start_ts: datetime,
        end_ts: datetime,
        data_root: Path = Path("data/real_scraper"),
    ):
        self.underlying = underlying.upper()
        self.start_ts = start_ts
        self.end_ts = end_ts
        self.data_root = data_root
        
        self._df: Optional[pd.DataFrame] = None
        self._spot_cache: Dict[datetime, float] = {}
        
    def _ensure_loaded(self) -> pd.DataFrame:
        """Ensure data is loaded, loading it if necessary."""
        if self._df is None:
            self._df = load_real_scraper_data(
                underlying=self.underlying,
                start_ts=self.start_ts,
                end_ts=self.end_ts,
                data_root=self.data_root,
            )
            
            if not self._df.empty and "harvest_time" in self._df.columns and "underlying_price" in self._df.columns:
                for _, row in self._df.drop_duplicates(subset=["harvest_time"]).iterrows():
                    if pd.notna(row["harvest_time"]) and pd.notna(row["underlying_price"]):
                        self._spot_cache[row["harvest_time"]] = float(row["underlying_price"])
        
        return self._df
    
    def close(self) -> None:
        """Clean up resources."""
        self._df = None
        self._spot_cache.clear()
    
    def get_spot_ohlc(
        self,
        underlying: str,
        start: datetime,
        end: datetime,
        timeframe: Timeframe,
    ) -> pd.DataFrame:
        """
        Return OHLCV for the underlying.
        
        Since Real Scraper data contains snapshots rather than OHLC candles,
        we synthesize OHLC from the underlying_price in each snapshot.
        """
        df = self._ensure_loaded()
        
        if df.empty or "underlying_price" not in df.columns:
            return pd.DataFrame()
        
        start_utc = start if start.tzinfo else start.replace(tzinfo=timezone.utc)
        end_utc = end if end.tzinfo else end.replace(tzinfo=timezone.utc)
        
        mask = (df["harvest_time"] >= start_utc) & (df["harvest_time"] <= end_utc)
        filtered = df[mask].copy()
        
        if filtered.empty:
            return pd.DataFrame()
        
        unique_times = filtered.drop_duplicates(subset=["harvest_time"])[["harvest_time", "underlying_price"]]
        unique_times = unique_times.dropna()
        unique_times = unique_times.sort_values("harvest_time")
        
        result = pd.DataFrame({
            "open": unique_times["underlying_price"].values,
            "high": unique_times["underlying_price"].values,
            "low": unique_times["underlying_price"].values,
            "close": unique_times["underlying_price"].values,
            "volume": [0.0] * len(unique_times),
        }, index=pd.DatetimeIndex(unique_times["harvest_time"].values, name="timestamp"))
        
        return result
    
    def list_option_chain(
        self,
        underlying: str,
        as_of: datetime,
    ) -> List[OptionSnapshot]:
        """
        Return option chain snapshot for underlying at (or near) 'as_of'.
        """
        df = self._ensure_loaded()
        
        if df.empty:
            return []
        
        snapshot_df = get_real_scraper_snapshot(df, as_of, tolerance_minutes=120)
        
        if snapshot_df.empty:
            return []
        
        calls_only = snapshot_df[snapshot_df["option_type"].str.upper() == "C"].copy()
        
        if calls_only.empty:
            return []
        
        options: List[OptionSnapshot] = []
        
        for _, row in calls_only.iterrows():
            try:
                expiry_ts = row.get("expiry_timestamp")
                if pd.notna(expiry_ts):
                    expiry_dt = datetime.fromtimestamp(float(expiry_ts), tz=timezone.utc)
                else:
                    expiry_str = row.get("expiry")
                    if pd.notna(expiry_str):
                        expiry_dt = datetime.strptime(str(expiry_str), "%Y-%m-%d").replace(tzinfo=timezone.utc)
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
                
                mark_price = row.get("mark_price")
                if pd.isna(mark_price):
                    mark_price = None
                else:
                    mark_price = float(mark_price)
                
                opt = OptionSnapshot(
                    instrument_name=str(row["instrument_name"]),
                    underlying=underlying.upper(),
                    kind="call",
                    strike=float(row["strike"]),
                    expiry=expiry_dt,
                    delta=delta,
                    iv=iv,
                    mark_price=mark_price,
                    settlement_ccy="USDC",
                    margin_type="linear",
                )
                options.append(opt)
                
            except Exception as e:
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
        Return OHLC data for a specific option instrument.
        
        Synthesizes OHLC from mark_price snapshots.
        """
        df = self._ensure_loaded()
        
        if df.empty or "instrument_name" not in df.columns:
            return pd.DataFrame()
        
        start_utc = start if start.tzinfo else start.replace(tzinfo=timezone.utc)
        end_utc = end if end.tzinfo else end.replace(tzinfo=timezone.utc)
        
        mask = (
            (df["instrument_name"] == instrument_name) &
            (df["harvest_time"] >= start_utc) &
            (df["harvest_time"] <= end_utc)
        )
        filtered = df[mask].copy()
        
        if filtered.empty:
            return pd.DataFrame()
        
        filtered = filtered.sort_values("harvest_time")
        
        result = pd.DataFrame({
            "open": filtered["mark_price"].values,
            "high": filtered["mark_price"].values,
            "low": filtered["mark_price"].values,
            "close": filtered["mark_price"].values,
            "volume": filtered.get("volume", pd.Series([0.0] * len(filtered))).values,
        }, index=pd.DatetimeIndex(filtered["harvest_time"].values, name="timestamp"))
        
        return result
    
    def get_spot_price(self, as_of: datetime) -> Optional[float]:
        """
        Get spot price at a specific time.
        
        Useful for quick lookups without going through full OHLC.
        """
        df = self._ensure_loaded()
        
        if df.empty or "underlying_price" not in df.columns:
            return None
        
        if self._spot_cache:
            cache_times = list(self._spot_cache.keys())
            as_of_utc = as_of if as_of.tzinfo else as_of.replace(tzinfo=timezone.utc)
            
            closest_time = min(cache_times, key=lambda t: abs((t - as_of_utc).total_seconds()))
            diff_minutes = abs((closest_time - as_of_utc).total_seconds()) / 60
            
            if diff_minutes <= 120:
                return self._spot_cache[closest_time]
        
        snapshot = get_real_scraper_snapshot(df, as_of, tolerance_minutes=120)
        if not snapshot.empty and "underlying_price" in snapshot.columns:
            prices = snapshot["underlying_price"].dropna()
            if len(prices) > 0:
                return float(prices.iloc[0])
        
        return None
