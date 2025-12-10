"""
Intraday data status reporting for Deribit data harvester.
Provides read-only inspection of the data store to report simple stats.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd


def _parse_timestamp(val) -> Optional[datetime]:
    """Parse a timestamp value from parquet into a datetime."""
    if val is None:
        return None
    if isinstance(val, pd.Timestamp):
        return val.to_pydatetime()
    if isinstance(val, datetime):
        return val
    if isinstance(val, str):
        try:
            return datetime.fromisoformat(val.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


@dataclass
class IntradayDataStatus:
    """Status of the intraday data scraper / storage."""
    ok: bool
    source: str
    backend: str
    rows_total: int
    days_covered: int
    first_timestamp: Optional[datetime]
    last_timestamp: Optional[datetime]
    approx_size_mb: float
    target_interval_sec: int
    is_running: bool
    error: Optional[str] = None


def get_intraday_data_status(settings=None) -> IntradayDataStatus:
    """
    Inspect the intraday data store and report simple stats.
    This is read-only and safe if no data exists yet.
    
    Args:
        settings: Optional settings object (unused for now, but available for future config)
    
    Returns:
        IntradayDataStatus with information about the data store
    """
    data_root = os.getenv("HARVESTER_DATA_ROOT", "data/live_deribit")
    interval_minutes = int(os.getenv("HARVESTER_INTERVAL_MINUTES", "15"))
    target_interval_sec = interval_minutes * 60
    
    root_path = Path(data_root)
    
    if not root_path.exists():
        return IntradayDataStatus(
            ok=True,
            source="Deribit intraday",
            backend="parquet",
            rows_total=0,
            days_covered=0,
            first_timestamp=None,
            last_timestamp=None,
            approx_size_mb=0.0,
            target_interval_sec=target_interval_sec,
            is_running=False,
            error="No intraday data directory found yet",
        )
    
    parquet_files = list(root_path.glob("**/*.parquet"))
    
    if not parquet_files:
        return IntradayDataStatus(
            ok=True,
            source="Deribit intraday",
            backend="parquet",
            rows_total=0,
            days_covered=0,
            first_timestamp=None,
            last_timestamp=None,
            approx_size_mb=0.0,
            target_interval_sec=target_interval_sec,
            is_running=False,
            error="No parquet files found yet",
        )
    
    try:
        total_bytes = sum(f.stat().st_size for f in parquet_files)
        approx_size_mb = round(total_bytes / (1024 * 1024), 1)
        
        files_sorted = sorted(parquet_files, key=lambda f: f.stat().st_mtime)
        
        oldest_files = files_sorted[:5]
        newest_files = files_sorted[-5:]
        sample_files = list(set(oldest_files + newest_files))
        
        rows_per_file_samples: list[int] = []
        oldest_timestamp: Optional[datetime] = None
        newest_timestamp: Optional[datetime] = None
        
        for pf in oldest_files:
            try:
                df = pd.read_parquet(pf, columns=["harvest_time"])
                rows_per_file_samples.append(len(df))
                
                if not df.empty and "harvest_time" in df.columns:
                    ts_col = df["harvest_time"]
                    if len(ts_col) > 0:
                        first_val = ts_col.iloc[0]
                        ts = _parse_timestamp(first_val)
                        if ts and (oldest_timestamp is None or ts < oldest_timestamp):
                            oldest_timestamp = ts
            except Exception:
                continue
        
        for pf in newest_files:
            try:
                df = pd.read_parquet(pf, columns=["harvest_time"])
                rows_per_file_samples.append(len(df))
                
                if not df.empty and "harvest_time" in df.columns:
                    ts_col = df["harvest_time"]
                    if len(ts_col) > 0:
                        last_val = ts_col.iloc[-1]
                        ts = _parse_timestamp(last_val)
                        if ts and (newest_timestamp is None or ts > newest_timestamp):
                            newest_timestamp = ts
            except Exception:
                continue
        
        avg_rows_per_file = sum(rows_per_file_samples) / len(rows_per_file_samples) if rows_per_file_samples else 0
        rows_total = int(avg_rows_per_file * len(parquet_files))
        
        first_timestamp = oldest_timestamp
        last_timestamp = newest_timestamp
        days_covered = 0
        
        if first_timestamp and first_timestamp.tzinfo is None:
            first_timestamp = first_timestamp.replace(tzinfo=timezone.utc)
        if last_timestamp and last_timestamp.tzinfo is None:
            last_timestamp = last_timestamp.replace(tzinfo=timezone.utc)
        
        if first_timestamp and last_timestamp:
            days_covered = (last_timestamp.date() - first_timestamp.date()).days + 1
        
        is_running = False
        if last_timestamp:
            now = datetime.now(timezone.utc)
            if last_timestamp.tzinfo is None:
                last_timestamp = last_timestamp.replace(tzinfo=timezone.utc)
            
            time_since_last = (now - last_timestamp).total_seconds()
            is_running = time_since_last < (2 * target_interval_sec)
        
        return IntradayDataStatus(
            ok=True,
            source="Deribit intraday",
            backend="parquet",
            rows_total=rows_total,
            days_covered=days_covered,
            first_timestamp=first_timestamp,
            last_timestamp=last_timestamp,
            approx_size_mb=approx_size_mb,
            target_interval_sec=target_interval_sec,
            is_running=is_running,
            error=None,
        )
        
    except Exception as e:
        return IntradayDataStatus(
            ok=False,
            source="Deribit intraday",
            backend="parquet",
            rows_total=0,
            days_covered=0,
            first_timestamp=None,
            last_timestamp=None,
            approx_size_mb=0.0,
            target_interval_sec=target_interval_sec,
            is_running=False,
            error=str(e),
        )
