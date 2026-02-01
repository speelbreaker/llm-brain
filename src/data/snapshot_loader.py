"""
Snapshot Loader - Reads the latest harvested market data from disk.

Used as a fallback when live API calls fail or for backtesting.
Reads from: data/live_deribit/{underlying}/{year}/{month}/{day}/*.parquet
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# Default path matching scripts/data_harvester.py
DATA_ROOT = Path(os.getenv("HARVESTER_DATA_ROOT", "data/live_deribit"))


def get_latest_snapshot_path(underlying: str) -> Optional[Path]:
    """
    Find the most recent parquet snapshot for the given underlying.
    Scans today and yesterday to handle boundary conditions.
    """
    now = datetime.now(timezone.utc)
    
    # Check today and yesterday (in case we just crossed midnight UTC)
    dates_to_check = [now, now.replace(day=now.day-1) if now.day > 1 else None]
    dates_to_check = [d for d in dates_to_check if d is not None]
    
    # Also check explicitly yesterday if near start of month
    if now.day == 1:
        # Simple fallback for month boundary logic: just scan the dir if needed
        # For now, let's keep it simple. If today is empty, we might miss data 
        # from 23:59 yesterday until first harvest. 
        # Robust implementation would walk back dates properly.
        pass

    candidates = []
    
    base_dir = DATA_ROOT / underlying.upper()
    
    # DEBUG LOGGING
    try:
        with open("/tmp/snapshot_debug.log", "a") as f:
            f.write(f"Scanning {base_dir} (exists={base_dir.exists()}) CWD={os.getcwd()}\n")
    except Exception:
        pass

    if not base_dir.exists():
        return None

    # Walk the directory tree to find all parquet files
    # Structure: {underlying}/{year}/{month}/{day}/{filename}
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".parquet"):
                full_path = Path(root) / file
                candidates.append(full_path)
    
    if not candidates:
        return None
    
    # Sort by modification time (or filename timestamp)
    # Filename format: {currency}_{date}_{time}.parquet
    # We can just sort by string name as ISO dates sort naturally, 
    # but os.path.getmtime is safer for actual creation time.
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    
    return candidates[0]


def load_latest_snapshot(underlying: str) -> List[Dict[str, Any]]:
    """
    Load the latest snapshot and convert to list-of-dicts format
    compatible with sensor computation (mimicking Deribit API response).
    
    Returns:
        List of dicts with keys like 'instrument_name', 'strike', 'mark_iv', etc.
    """
    path = get_latest_snapshot_path(underlying)
    if not path:
        return []
    
    try:
        df = pd.read_parquet(path)
        
        # Map harvester columns back to API-like structure expected by sensors.py
        # Harvester: best_bid_price, greek_delta, ...
        # API (sensors.py usage): bid_price, delta, ...
        
        records = df.to_dict("records")
        
        normalized = []
        for row in records:
            item = {
                "instrument_name": row.get("instrument_name"),
                "strike": row.get("strike"),
                "option_type": row.get("option_type"), # 'C' or 'P' usually
                "expiry_timestamp": row.get("expiry_timestamp"),
                # sensors.py uses 'mark_iv', 'mark_price' which are present
                "mark_iv": row.get("mark_iv"),
                "mark_price": row.get("mark_price"),
                "underlying_price": row.get("underlying_price"), # Added mapping
                "bid_price": row.get("best_bid_price"),
                "ask_price": row.get("best_ask_price"),
                "greeks": {
                    "delta": row.get("greek_delta"),
                    "gamma": row.get("greek_gamma"),
                    "theta": row.get("greek_theta"),
                    "vega": row.get("greek_vega"),
                }
            }
            
            # Helper for sensors.py which sometimes looks for 'option_type' 
            # as "call"/"put" lower/upper case depending on context.
            # Harvester saves 'C'/'P'. Let's normalize to what parsing expects if needed.
            # actually _parse_option_expiry handles the instrument name.
            # compute_skew_25d checks inst.get("option_type") which harvester might save as C/P.
            
            # Add explicit 'call'/'put' for safety if needed, 
            # though sensors.py largely re-parses instrument_name.
            if row.get("option_type") == "C":
                item["option_type"] = "call"
            elif row.get("option_type") == "P":
                item["option_type"] = "put"
                
            normalized.append(item)
            
        return normalized
        
    except Exception as e:
        print(f"[SnapshotLoader] Failed to load {path}: {e}")
        return []
