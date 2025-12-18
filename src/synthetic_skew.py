"""
Synthetic skew engine for IV smile modeling.

Derives skew factors from live Deribit IV vs delta data and applies them
to the synthetic RV-based pricing universe.

Supports three modes:
- "live": Fetch current skew from Deribit API (for live trading)
- "harvested": Compute skew from historical harvested data (for backtests)
- "none": Return flat skew (1.0) - deterministic fallback
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, date, timezone, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Literal

import httpx

logger = logging.getLogger(__name__)

DERIBIT_API = "https://www.deribit.com/api/v2"

_LIVE_SKEW_CACHE: Dict[Tuple[str, str], List["SkewAnchor"]] = {}

_HARVESTED_SKEW_CACHE: Dict[Tuple[str, str, date], List["SkewAnchor"]] = {}

SkewSource = Literal["live", "harvested", "none"]


@dataclass
class SkewAnchor:
    """A single anchor point in the skew curve."""
    delta: float
    ratio: float


def _deribit_get(path: str, params: dict) -> dict:
    """Make a GET request to Deribit public API."""
    url = f"{DERIBIT_API}/{path}"
    with httpx.Client(timeout=15.0) as client:
        resp = client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        if "result" not in data:
            raise RuntimeError(f"Unexpected Deribit response: {data}")
        return data["result"]


def compute_live_skew_anchors(
    underlying: str,
    option_type: str = "call",
    min_dte: float = 3.0,
    max_dte: float = 14.0,
    max_quotes: int = 80,
    as_of: Optional[datetime] = None,
) -> List[SkewAnchor]:
    """
    Build a simple skew template from current Deribit data.

    Steps:
      1) Fetch non-expired option instruments for the given underlying.
      2) Filter by option_type (call/put) and DTE range.
      3) For each instrument, fetch ticker to get mark_iv and delta.
      4) Find ATM IV (instrument with delta closest to 0.5 for calls).
      5) For anchor deltas [0.15, 0.25, 0.35, 0.50], compute average IV / ATM_IV.
      6) Return list of SkewAnchor sorted by delta ascending.
      
    Args:
        underlying: Asset symbol (e.g., "BTC", "ETH")
        option_type: "call" or "put"
        min_dte: Minimum days to expiry
        max_dte: Maximum days to expiry
        max_quotes: Maximum number of quotes to fetch
        as_of: Reference time for DTE calculation (default: now)
    """
    try:
        instruments = _deribit_get(
            "public/get_instruments",
            {
                "currency": underlying,
                "kind": "option",
                "expired": "false",
            },
        )
    except Exception:
        return _flat_anchors()

    now = as_of if as_of else datetime.now(timezone.utc)
    
    filtered_instruments: List[dict] = []
    for inst in instruments:
        if inst.get("option_type") != option_type:
            continue

        expiration_ts_ms = inst.get("expiration_timestamp", 0)
        expiration = datetime.fromtimestamp(expiration_ts_ms / 1000.0, tz=timezone.utc)
        dte_days = (expiration - now).total_seconds() / 86400.0

        if dte_days < min_dte or dte_days > max_dte:
            continue
        
        filtered_instruments.append({
            "instrument_name": inst["instrument_name"],
            "dte": dte_days,
        })
    
    filtered_instruments.sort(key=lambda x: x["dte"])
    filtered_instruments = filtered_instruments[:max_quotes]
    
    quotes: List[dict] = []
    for fi in filtered_instruments:
        try:
            ticker = _deribit_get("public/ticker", {"instrument_name": fi["instrument_name"]})
            mark_iv = ticker.get("mark_iv")
            
            greeks = ticker.get("greeks") or {}
            delta = greeks.get("delta")

            if mark_iv is None or mark_iv <= 0:
                continue
            if delta is None:
                continue

            quotes.append({
                "instrument": fi["instrument_name"],
                "mark_iv": float(mark_iv),
                "delta": float(delta),
                "dte": fi["dte"],
            })
        except Exception:
            continue

    return _compute_anchors_from_quotes(quotes, option_type)


def compute_harvested_skew_anchors(
    underlying: str,
    option_type: str = "call",
    as_of: datetime = None,
    min_dte: float = 3.0,
    max_dte: float = 14.0,
    data_root: Optional[Path] = None,
) -> List[SkewAnchor]:
    """
    Compute skew anchors from harvested Deribit snapshot data.
    
    Loads options data from the closest snapshot at/before as_of and
    computes the same anchor ratios as the live version.
    
    Args:
        underlying: Asset symbol (e.g., "BTC", "ETH")
        option_type: "call" or "put"
        as_of: Reference time for data lookup
        min_dte: Minimum days to expiry
        max_dte: Maximum days to expiry
        data_root: Root directory for harvested data (default: data/live_deribit)
        
    Returns:
        List of SkewAnchor, or flat anchors if data is unavailable.
    """
    if as_of is None:
        return _flat_anchors()
    
    if data_root is None:
        data_root = Path("data/live_deribit")
    
    try:
        quotes = _load_harvested_quotes(
            underlying=underlying,
            option_type=option_type,
            as_of=as_of,
            min_dte=min_dte,
            max_dte=max_dte,
            data_root=data_root,
        )
        
        if not quotes:
            logger.debug(f"No harvested quotes found for {underlying} {option_type} at {as_of}")
            return _flat_anchors()
            
        return _compute_anchors_from_quotes(quotes, option_type)
        
    except Exception as e:
        logger.debug(f"Failed to compute harvested skew anchors: {e}")
        return _flat_anchors()


def _load_harvested_quotes(
    underlying: str,
    option_type: str,
    as_of: datetime,
    min_dte: float,
    max_dte: float,
    data_root: Path,
) -> List[dict]:
    """
    Load option quotes from harvested data closest to as_of.
    
    Returns list of dicts with mark_iv, delta, dte keys.
    """
    import pandas as pd
    
    underlying_upper = underlying.upper()
    
    search_dirs = [
        data_root / underlying_upper,
        data_root / f"{underlying_upper}_USDC",
    ]
    
    parquet_files: List[Tuple[Path, datetime]] = []
    
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
            
        for pq_file in search_dir.glob("**/*.parquet"):
            file_ts = _parse_file_timestamp(pq_file)
            if file_ts:
                parquet_files.append((pq_file, file_ts))
    
    if not parquet_files:
        return []
    
    parquet_files.sort(key=lambda x: x[1])
    
    as_of_ts = as_of.timestamp() if as_of.tzinfo else as_of.replace(tzinfo=timezone.utc).timestamp()
    
    best_file = None
    best_diff = float("inf")
    
    for pq_file, file_ts in parquet_files:
        file_ts_val = file_ts.timestamp() if hasattr(file_ts, 'timestamp') else file_ts
        diff = as_of_ts - file_ts_val
        
        if diff >= 0 and diff < best_diff:
            best_diff = diff
            best_file = pq_file
    
    if best_file is None:
        if parquet_files:
            best_file = min(parquet_files, key=lambda x: abs(x[1].timestamp() - as_of_ts))[0]
        else:
            return []
    
    try:
        df = pd.read_parquet(best_file)
    except Exception:
        return []
    
    option_type_lower = option_type.lower()
    if "option_type" in df.columns:
        df = df[df["option_type"].str.lower() == option_type_lower]
    
    if df.empty:
        return []
    
    if "harvest_time" in df.columns:
        harvest_time = pd.to_datetime(df["harvest_time"].iloc[0], utc=True)
    else:
        harvest_time = as_of if as_of.tzinfo else as_of.replace(tzinfo=timezone.utc)
    
    if "expiry_timestamp" in df.columns:
        expiry_ts = pd.to_numeric(df["expiry_timestamp"], errors="coerce")
        harvest_ts = harvest_time.timestamp()
        df["dte_days"] = (expiry_ts - harvest_ts) / 86400.0
    elif "dte_days" not in df.columns:
        return []
    
    df = df[(df["dte_days"] >= min_dte) & (df["dte_days"] <= max_dte)]
    
    if df.empty:
        return []
    
    iv_col = "mark_iv" if "mark_iv" in df.columns else None
    delta_col = "greek_delta" if "greek_delta" in df.columns else ("delta" if "delta" in df.columns else None)
    
    if iv_col is None or delta_col is None:
        return []
    
    df = df.dropna(subset=[iv_col, delta_col])
    df = df[df[iv_col] > 0]
    
    quotes = []
    for _, row in df.iterrows():
        quotes.append({
            "mark_iv": float(row[iv_col]),
            "delta": float(row[delta_col]),
            "dte": float(row["dte_days"]),
        })
    
    return quotes


def _parse_file_timestamp(filepath: Path) -> Optional[datetime]:
    """Parse timestamp from parquet filename."""
    import re
    
    name = filepath.stem
    
    patterns = [
        r"(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})",
        r"(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2}:\d{2})",
        r"(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, name)
        if match:
            try:
                groups = match.groups()
                if len(groups) == 2:
                    date_str = groups[0]
                    time_str = groups[1].replace("-", ":")
                    dt_str = f"{date_str}T{time_str}"
                    return datetime.fromisoformat(dt_str).replace(tzinfo=timezone.utc)
                elif len(groups) == 6:
                    dt = datetime(
                        int(groups[0]), int(groups[1]), int(groups[2]),
                        int(groups[3]), int(groups[4]), int(groups[5]),
                        tzinfo=timezone.utc
                    )
                    return dt
            except (ValueError, IndexError):
                continue
    
    try:
        stat = filepath.stat()
        return datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
    except Exception:
        return None


def _compute_anchors_from_quotes(quotes: List[dict], option_type: str) -> List[SkewAnchor]:
    """Compute skew anchors from a list of quote dicts."""
    if len(quotes) < 3:
        return _flat_anchors()
    
    if option_type == "call":
        atm_quote = min(quotes, key=lambda q: abs(q["delta"] - 0.5))
    else:
        atm_quote = min(quotes, key=lambda q: abs(q["delta"] + 0.5))

    iv_atm = atm_quote["mark_iv"]
    if iv_atm <= 0:
        return _flat_anchors()

    anchor_deltas = [0.15, 0.25, 0.35, 0.50]
    anchors: List[SkewAnchor] = []

    for anchor_d in anchor_deltas:
        nearby = [
            q for q in quotes
            if abs(abs(q["delta"]) - anchor_d) <= 0.05
        ]

        if nearby:
            avg_iv = sum(q["mark_iv"] for q in nearby) / len(nearby)
            ratio = avg_iv / iv_atm
            ratio = max(0.4, min(ratio, 1.4))
        else:
            ratio = 1.0

        anchors.append(SkewAnchor(delta=anchor_d, ratio=ratio))

    anchors.sort(key=lambda a: a.delta)
    return anchors


def _flat_anchors() -> List[SkewAnchor]:
    """Return flat skew anchors (ratio = 1.0) as fallback."""
    return [
        SkewAnchor(delta=0.15, ratio=1.0),
        SkewAnchor(delta=0.25, ratio=1.0),
        SkewAnchor(delta=0.35, ratio=1.0),
        SkewAnchor(delta=0.50, ratio=1.0),
    ]


def get_skew_factor(
    underlying: str,
    option_type: str,
    abs_delta: float,
    skew_enabled: bool,
    min_dte: float = 3.0,
    max_dte: float = 14.0,
    as_of: Optional[datetime] = None,
    source: SkewSource = "none",
    data_root: Optional[Path] = None,
) -> float:
    """
    Return a skew factor for a given absolute delta in [0, 1].

    Args:
        underlying: Asset symbol (e.g., "BTC", "ETH")
        option_type: "call" or "put"
        abs_delta: Absolute delta value in [0, 1]
        skew_enabled: If False, always return 1.0
        min_dte: Minimum DTE for anchor computation
        max_dte: Maximum DTE for anchor computation
        as_of: Reference time (used for harvested mode and DTE calculation)
        source: Skew source mode (default: "none" for safety):
            - "none": Return 1.0 (flat skew, default - safe for backtests)
            - "harvested": Compute from historical harvested data (for backtests)
            - "live": Fetch from live Deribit API (ONLY for live trading)
        data_root: Root directory for harvested data (only used if source="harvested")
    
    Returns:
        Skew factor multiplier (typically 0.8 - 1.2)
    
    IMPORTANT: Default is "none" to prevent look-ahead bias. Only use source="live"
    in actual live trading code, never in backtests or historical simulations.
    """
    if not skew_enabled or source == "none":
        return 1.0

    underlying_key = underlying.upper()
    option_type_key = option_type.lower()
    
    if source == "live":
        cache_key = (underlying_key, option_type_key)
        
        if cache_key not in _LIVE_SKEW_CACHE:
            try:
                anchors = compute_live_skew_anchors(
                    underlying=underlying,
                    option_type=option_type,
                    min_dte=min_dte,
                    max_dte=max_dte,
                    as_of=as_of,
                )
                _LIVE_SKEW_CACHE[cache_key] = anchors
            except Exception:
                _LIVE_SKEW_CACHE[cache_key] = _flat_anchors()
        
        anchors = _LIVE_SKEW_CACHE[cache_key]
        
    elif source == "harvested":
        if as_of is None:
            return 1.0
            
        cache_date = as_of.date() if hasattr(as_of, 'date') else as_of
        cache_key = (underlying_key, option_type_key, cache_date)
        
        if cache_key not in _HARVESTED_SKEW_CACHE:
            try:
                anchors = compute_harvested_skew_anchors(
                    underlying=underlying,
                    option_type=option_type,
                    as_of=as_of,
                    min_dte=min_dte,
                    max_dte=max_dte,
                    data_root=data_root,
                )
                _HARVESTED_SKEW_CACHE[cache_key] = anchors
            except Exception:
                _HARVESTED_SKEW_CACHE[cache_key] = _flat_anchors()
        
        anchors = _HARVESTED_SKEW_CACHE[cache_key]
    else:
        return 1.0

    if not anchors:
        return 1.0

    abs_delta = max(0.0, min(1.0, abs_delta))

    if abs_delta <= anchors[0].delta:
        return anchors[0].ratio

    if abs_delta >= anchors[-1].delta:
        return anchors[-1].ratio

    for i in range(len(anchors) - 1):
        if anchors[i].delta <= abs_delta <= anchors[i + 1].delta:
            d1, r1 = anchors[i].delta, anchors[i].ratio
            d2, r2 = anchors[i + 1].delta, anchors[i + 1].ratio
            if d2 - d1 < 1e-9:
                return r1
            t = (abs_delta - d1) / (d2 - d1)
            return r1 + t * (r2 - r1)

    return 1.0


def clear_skew_cache() -> None:
    """Clear all skew anchor caches (useful for testing or refresh)."""
    _LIVE_SKEW_CACHE.clear()
    _HARVESTED_SKEW_CACHE.clear()


def clear_live_skew_cache() -> None:
    """Clear only the live skew cache."""
    _LIVE_SKEW_CACHE.clear()


def clear_harvested_skew_cache() -> None:
    """Clear only the harvested skew cache."""
    _HARVESTED_SKEW_CACHE.clear()


def get_cached_anchors(underlying: str, option_type: str) -> Optional[List[SkewAnchor]]:
    """Return cached live anchors if available, else None."""
    cache_key = (underlying.upper(), option_type.lower())
    return _LIVE_SKEW_CACHE.get(cache_key)


def get_cached_harvested_anchors(
    underlying: str, 
    option_type: str, 
    as_of_date: date
) -> Optional[List[SkewAnchor]]:
    """Return cached harvested anchors for a specific date if available, else None."""
    cache_key = (underlying.upper(), option_type.lower(), as_of_date)
    return _HARVESTED_SKEW_CACHE.get(cache_key)
