"""
Synthetic skew engine for IV smile modeling.

Derives skew factors from live Deribit IV vs delta data and applies them
to the synthetic RV-based pricing universe.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional

import httpx


DERIBIT_API = "https://www.deribit.com/api/v2"

_SKEW_CACHE: Dict[Tuple[str, str], List["SkewAnchor"]] = {}


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

    now = datetime.now(timezone.utc)
    
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
) -> float:
    """
    Return a skew factor for a given absolute delta in [0, 1].

    - If skew_enabled is False, always return 1.0.
    - Otherwise:
        - Lazily initialize anchors for (underlying, option_type) in _SKEW_CACHE.
        - Interpolate linearly between the two nearest anchors by delta.
        - If abs_delta < smallest anchor delta -> use smallest anchor ratio.
        - If abs_delta > largest anchor delta -> use largest anchor ratio.
    """
    if not skew_enabled:
        return 1.0

    cache_key = (underlying.upper(), option_type.lower())

    if cache_key not in _SKEW_CACHE:
        try:
            anchors = compute_live_skew_anchors(
                underlying=underlying,
                option_type=option_type,
                min_dte=min_dte,
                max_dte=max_dte,
            )
            _SKEW_CACHE[cache_key] = anchors
        except Exception:
            _SKEW_CACHE[cache_key] = _flat_anchors()

    anchors = _SKEW_CACHE[cache_key]
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
    """Clear the skew anchor cache (useful for testing or refresh)."""
    _SKEW_CACHE.clear()


def get_cached_anchors(underlying: str, option_type: str) -> Optional[List[SkewAnchor]]:
    """Return cached anchors if available, else None."""
    cache_key = (underlying.upper(), option_type.lower())
    return _SKEW_CACHE.get(cache_key)
