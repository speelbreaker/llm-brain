"""Shared expiry parsing utilities for Deribit instruments.

This module centralizes all expiry date parsing logic to ensure
consistent behavior across the live agent, backtests, and position tracking.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Optional

MONTH_MAP = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4,
    "MAY": 5, "JUN": 6, "JUL": 7, "AUG": 8,
    "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}


def parse_deribit_expiry(
    expiry_or_instrument: str | datetime | None,
) -> Optional[datetime]:
    """
    Convert a Deribit option expiry or instrument name into a timezone-aware datetime.

    This is the single source of truth for expiry parsing across the codebase.
    All expiries are normalized to 08:00:00 UTC (Deribit's settlement time).

    Accepts:
        - A datetime object (returned as-is, with UTC timezone added if naive)
        - An ISO-style instrument: "BTC-2025-01-03-90000-C"
        - A classic Deribit instrument: "BTC-6DEC24-90000-C" or "BTC-27DEC24-100000-C"
        - A bare expiry string: "6DEC24" or "27DEC24"

    Args:
        expiry_or_instrument: The expiry date, instrument name, or datetime to parse.

    Returns:
        A timezone-aware datetime in UTC at 08:00:00, or None if parsing fails.

    Examples:
        >>> parse_deribit_expiry("BTC-27DEC24-100000-C")
        datetime(2024, 12, 27, 8, 0, 0, tzinfo=timezone.utc)
        
        >>> parse_deribit_expiry("BTC-2025-01-03-90000-C")
        datetime(2025, 1, 3, 8, 0, 0, tzinfo=timezone.utc)
        
        >>> parse_deribit_expiry("6DEC24")
        datetime(2024, 12, 6, 8, 0, 0, tzinfo=timezone.utc)
    """
    if expiry_or_instrument is None:
        return None

    if isinstance(expiry_or_instrument, datetime):
        if expiry_or_instrument.tzinfo is None:
            return expiry_or_instrument.replace(tzinfo=timezone.utc)
        return expiry_or_instrument

    symbol = str(expiry_or_instrument).strip()
    if not symbol:
        return None

    iso_match = re.match(r"^[A-Z]+-(\d{4})-(\d{2})-(\d{2})-", symbol)
    if iso_match:
        year, month, day = map(int, iso_match.groups())
        return datetime(year, month, day, 8, 0, 0, tzinfo=timezone.utc)

    classic_match = re.match(r"^[A-Z]+-(\d{1,2})([A-Z]{3})(\d{2})-", symbol)
    if classic_match:
        day_str, mon_str, year_2 = classic_match.groups()
        day = int(day_str)
        month = MONTH_MAP.get(mon_str.upper())
        if month is None:
            return None
        year = 2000 + int(year_2)
        return datetime(year, month, day, 8, 0, 0, tzinfo=timezone.utc)

    bare_match = re.match(r"^(\d{1,2})([A-Z]{3})(\d{2})$", symbol.upper())
    if bare_match:
        day_str, mon_str, year_2 = bare_match.groups()
        day = int(day_str)
        month = MONTH_MAP.get(mon_str)
        if month is None:
            return None
        year = 2000 + int(year_2)
        return datetime(year, month, day, 8, 0, 0, tzinfo=timezone.utc)

    return None
