"""Tests for src/utils/expiry.py - Deribit expiry parsing."""
import pytest
from datetime import datetime, timezone
from src.utils.expiry import parse_deribit_expiry


class TestParseDeribitExpiry:
    """Tests for the parse_deribit_expiry function."""

    def test_classic_deribit_format(self):
        """Parse classic Deribit instrument name like BTC-05DEC24-90000-C."""
        result = parse_deribit_expiry("BTC-05DEC24-90000-C")
        assert result is not None
        assert result.year == 2024
        assert result.month == 12
        assert result.day == 5
        assert result.hour == 8  # Deribit settlement time
        assert result.tzinfo == timezone.utc

    def test_classic_deribit_two_digit_day(self):
        """Parse instrument with two-digit day like BTC-27DEC24-100000-C."""
        result = parse_deribit_expiry("BTC-27DEC24-100000-C")
        assert result is not None
        assert result.year == 2024
        assert result.month == 12
        assert result.day == 27

    def test_iso_format(self):
        """Parse ISO-style instrument name like BTC-2025-01-03-90000-C."""
        result = parse_deribit_expiry("BTC-2025-01-03-90000-C")
        assert result is not None
        assert result.year == 2025
        assert result.month == 1
        assert result.day == 3
        assert result.hour == 8
        assert result.tzinfo == timezone.utc

    def test_eth_instrument(self):
        """Parse ETH instrument name."""
        result = parse_deribit_expiry("ETH-15JAN25-4000-C")
        assert result is not None
        assert result.year == 2025
        assert result.month == 1
        assert result.day == 15

    def test_bare_expiry_string(self):
        """Parse bare expiry string like 6DEC24."""
        result = parse_deribit_expiry("6DEC24")
        assert result is not None
        assert result.year == 2024
        assert result.month == 12
        assert result.day == 6

    def test_datetime_input_returned_as_is(self):
        """When given a datetime object, return it as-is."""
        dt = datetime(2024, 12, 5, 8, 0, 0, tzinfo=timezone.utc)
        result = parse_deribit_expiry(dt)
        assert result == dt

    def test_naive_datetime_gets_utc(self):
        """Naive datetime should get UTC timezone added."""
        dt = datetime(2024, 12, 5, 8, 0, 0)
        result = parse_deribit_expiry(dt)
        assert result is not None
        assert result.tzinfo == timezone.utc

    def test_invalid_string_returns_none(self):
        """Invalid strings should return None."""
        result = parse_deribit_expiry("invalid")
        assert result is None

    def test_empty_string_returns_none(self):
        """Empty string should return None."""
        result = parse_deribit_expiry("")
        assert result is None

    def test_none_input_returns_none(self):
        """None input should return None."""
        result = parse_deribit_expiry(None)
        assert result is None

    def test_all_months(self):
        """Test all month abbreviations are parsed correctly."""
        months = [
            ("JAN", 1), ("FEB", 2), ("MAR", 3), ("APR", 4),
            ("MAY", 5), ("JUN", 6), ("JUL", 7), ("AUG", 8),
            ("SEP", 9), ("OCT", 10), ("NOV", 11), ("DEC", 12),
        ]
        for abbr, expected_month in months:
            result = parse_deribit_expiry(f"BTC-15{abbr}24-90000-C")
            assert result is not None, f"Failed for month {abbr}"
            assert result.month == expected_month, f"Wrong month for {abbr}"

    def test_put_option(self):
        """Parse put option instrument name."""
        result = parse_deribit_expiry("BTC-05DEC24-90000-P")
        assert result is not None
        assert result.year == 2024
        assert result.month == 12
        assert result.day == 5
