"""
Tests for the IV sanity check module.

These tests use mocking to avoid running actual backtests.
"""

from unittest.mock import patch, MagicMock

from scripts.iv_sanity_check import (
    IVSanitySelectorResult,
    _check_selector,
    run_iv_sanity_check,
)


def make_backtest_result(
    num_trades: int, net_profit_pct: float, error: str | None = None
):
    """Create a mock backtest result."""
    return {
        "num_trades": num_trades,
        "net_profit_pct": net_profit_pct,
        "metrics": {"net_profit_pct": net_profit_pct, "num_trades": num_trades},
        "error": error,
    }


class TestIVSanitySelectorResult:
    """Tests for the IVSanitySelectorResult dataclass."""

    def test_to_dict_returns_all_fields(self):
        result = IVSanitySelectorResult(
            selector="generic",
            iv_low=0.8,
            iv_high=1.2,
            num_trades_low=5,
            num_trades_high=7,
            net_profit_pct_low=1.5,
            net_profit_pct_high=2.5,
            passed=True,
            reason="ok",
        )
        d = result.to_dict()

        assert d["selector"] == "generic"
        assert d["iv_low"] == 0.8
        assert d["iv_high"] == 1.2
        assert d["num_trades_low"] == 5
        assert d["num_trades_high"] == 7
        assert d["net_profit_pct_low"] == 1.5
        assert d["net_profit_pct_high"] == 2.5
        assert d["passed"] is True
        assert d["reason"] == "ok"


class TestCheckSelectorGeneric:
    """Tests for generic selector pass/fail logic."""

    @patch("scripts.iv_sanity_check._run_single_backtest")
    def test_generic_passes_when_profit_pct_differs(self, mock_backtest):
        mock_backtest.side_effect = [
            make_backtest_result(num_trades=5, net_profit_pct=1.0),
            make_backtest_result(num_trades=5, net_profit_pct=3.0),
        ]

        result = _check_selector("generic", 0.8, 1.2)

        assert result.passed is True
        assert result.reason == "ok"
        assert result.num_trades_low == 5
        assert result.num_trades_high == 5
        assert result.net_profit_pct_low == 1.0
        assert result.net_profit_pct_high == 3.0

    @patch("scripts.iv_sanity_check._run_single_backtest")
    def test_generic_passes_when_trades_differ(self, mock_backtest):
        mock_backtest.side_effect = [
            make_backtest_result(num_trades=3, net_profit_pct=1.0),
            make_backtest_result(num_trades=7, net_profit_pct=1.0),
        ]

        result = _check_selector("generic", 0.8, 1.2)

        assert result.passed is True
        assert result.num_trades_low == 3
        assert result.num_trades_high == 7

    @patch("scripts.iv_sanity_check._run_single_backtest")
    def test_generic_fails_when_no_differentiation(self, mock_backtest):
        mock_backtest.side_effect = [
            make_backtest_result(num_trades=5, net_profit_pct=1.0),
            make_backtest_result(num_trades=5, net_profit_pct=1.2),
        ]

        result = _check_selector("generic", 0.8, 1.2)

        assert result.passed is False
        assert "No differentiation" in result.reason

    @patch("scripts.iv_sanity_check._run_single_backtest")
    def test_generic_fails_when_no_trades(self, mock_backtest):
        mock_backtest.side_effect = [
            make_backtest_result(num_trades=0, net_profit_pct=0.0),
            make_backtest_result(num_trades=0, net_profit_pct=0.0),
        ]

        result = _check_selector("generic", 0.8, 1.2)

        assert result.passed is False
        assert "No trades" in result.reason

    @patch("scripts.iv_sanity_check._run_single_backtest")
    def test_generic_fails_on_backtest_error(self, mock_backtest):
        mock_backtest.side_effect = [
            make_backtest_result(num_trades=0, net_profit_pct=0.0, error="API timeout"),
            make_backtest_result(num_trades=5, net_profit_pct=1.0),
        ]

        result = _check_selector("generic", 0.8, 1.2)

        assert result.passed is False
        assert "low IV backtest error" in result.reason


class TestCheckSelectorGregBot:
    """Tests for gregbot selector pass/fail logic."""

    @patch("scripts.iv_sanity_check._run_single_backtest")
    def test_gregbot_passes_when_high_iv_better(self, mock_backtest):
        mock_backtest.side_effect = [
            make_backtest_result(num_trades=4, net_profit_pct=0.8),
            make_backtest_result(num_trades=6, net_profit_pct=2.0),
        ]

        result = _check_selector("gregbot", 0.9, 1.1)

        assert result.passed is True
        assert result.reason == "ok"

    @patch("scripts.iv_sanity_check._run_single_backtest")
    def test_gregbot_fails_when_high_not_better(self, mock_backtest):
        mock_backtest.side_effect = [
            make_backtest_result(num_trades=5, net_profit_pct=1.0),
            make_backtest_result(num_trades=4, net_profit_pct=1.0),
        ]

        result = _check_selector("gregbot", 0.9, 1.1)

        assert result.passed is False
        assert "GregBot not responding" in result.reason


class TestRunIVSanityCheck:
    """Tests for the main run_iv_sanity_check function."""

    @patch("scripts.iv_sanity_check.DeribitDataSource")
    @patch("scripts.iv_sanity_check._check_selector")
    def test_all_pass_returns_ok(self, mock_check, mock_ds):
        mock_ds.return_value = MagicMock()
        mock_check.side_effect = [
            IVSanitySelectorResult(
                selector="generic",
                iv_low=0.8,
                iv_high=1.2,
                num_trades_low=5,
                num_trades_high=6,
                net_profit_pct_low=1.0,
                net_profit_pct_high=2.5,
                passed=True,
                reason="ok",
            ),
            IVSanitySelectorResult(
                selector="gregbot",
                iv_low=0.9,
                iv_high=1.1,
                num_trades_low=4,
                num_trades_high=5,
                net_profit_pct_low=0.8,
                net_profit_pct_high=1.8,
                passed=True,
                reason="ok",
            ),
        ]

        result = run_iv_sanity_check()

        assert result["status"] == "ok"
        assert "All IV sanity checks passed" in result["summary"]
        assert len(result["selectors"]) == 2
        assert "checked_at" in result

    @patch("scripts.iv_sanity_check.DeribitDataSource")
    @patch("scripts.iv_sanity_check._check_selector")
    def test_all_fail_returns_failed(self, mock_check, mock_ds):
        mock_ds.return_value = MagicMock()
        mock_check.side_effect = [
            IVSanitySelectorResult(
                selector="generic",
                iv_low=0.8,
                iv_high=1.2,
                num_trades_low=0,
                num_trades_high=0,
                net_profit_pct_low=0.0,
                net_profit_pct_high=0.0,
                passed=False,
                reason="No trades",
            ),
            IVSanitySelectorResult(
                selector="gregbot",
                iv_low=0.9,
                iv_high=1.1,
                num_trades_low=0,
                num_trades_high=0,
                net_profit_pct_low=0.0,
                net_profit_pct_high=0.0,
                passed=False,
                reason="No trades",
            ),
        ]

        result = run_iv_sanity_check()

        assert result["status"] == "failed"
        assert "All checks failed" in result["summary"]

    @patch("scripts.iv_sanity_check.DeribitDataSource")
    @patch("scripts.iv_sanity_check._check_selector")
    def test_partial_pass_returns_degraded(self, mock_check, mock_ds):
        mock_ds.return_value = MagicMock()
        mock_check.side_effect = [
            IVSanitySelectorResult(
                selector="generic",
                iv_low=0.8,
                iv_high=1.2,
                num_trades_low=5,
                num_trades_high=6,
                net_profit_pct_low=1.0,
                net_profit_pct_high=2.5,
                passed=True,
                reason="ok",
            ),
            IVSanitySelectorResult(
                selector="gregbot",
                iv_low=0.9,
                iv_high=1.1,
                num_trades_low=0,
                num_trades_high=0,
                net_profit_pct_low=0.0,
                net_profit_pct_high=0.0,
                passed=False,
                reason="No trades",
            ),
        ]

        result = run_iv_sanity_check()

        assert result["status"] == "degraded"
        assert "Partial pass" in result["summary"]

    @patch("scripts.iv_sanity_check.DeribitDataSource")
    def test_data_source_error_returns_failed(self, mock_ds):
        mock_ds.side_effect = Exception("Connection refused")

        result = run_iv_sanity_check()

        assert result["status"] == "failed"
        assert "Failed to initialize data source" in result["summary"]
