"""
Tests for the healthcheck module.

These tests use mocking to avoid real network calls.
"""
import pytest
from unittest.mock import MagicMock, patch

from src.healthcheck import (
    CheckStatus,
    HealthCheckResult,
    check_config,
    check_deribit_public,
    check_deribit_private,
    check_state_builder,
    run_agent_healthcheck,
)
from src.config import Settings
from src.deribit_client import DeribitAPIError


class FakeDeribitClient:
    """Fake Deribit client for testing."""

    def __init__(
        self,
        btc_price: float = 95000.0,
        eth_price: float = 3500.0,
        equity: float = 100000.0,
        fail_public: bool = False,
        fail_private: bool = False,
    ):
        self.btc_price = btc_price
        self.eth_price = eth_price
        self.equity = equity
        self.fail_public = fail_public
        self.fail_private = fail_private

    def get_index_price(self, underlying: str) -> float:
        if self.fail_public:
            raise DeribitAPIError(-1, "public API unavailable")
        if underlying == "BTC":
            return self.btc_price
        elif underlying == "ETH":
            return self.eth_price
        return 0.0

    def get_account_summary(self, currency: str) -> dict:
        if self.fail_private:
            raise DeribitAPIError(-1, "authentication failed")
        return {"equity": self.equity, "currency": currency}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class TestCheckConfig:
    """Tests for check_config function."""

    def test_valid_config_returns_ok(self):
        cfg = Settings()
        result = check_config(cfg)
        assert result.status == CheckStatus.OK
        assert "mode=" in result.detail
        assert "env=" in result.detail

    def test_invalid_loop_interval_returns_fail(self):
        cfg = Settings(loop_interval_sec=0)
        result = check_config(cfg)
        assert result.status == CheckStatus.FAIL
        assert "loop_interval_sec" in result.detail

    def test_invalid_margin_pct_returns_fail(self):
        cfg = Settings(max_margin_used_pct=150)
        result = check_config(cfg)
        assert result.status == CheckStatus.FAIL
        assert "max_margin_used_pct" in result.detail

    def test_negative_delta_returns_fail(self):
        cfg = Settings(max_net_delta_abs=-1)
        result = check_config(cfg)
        assert result.status == CheckStatus.FAIL
        assert "max_net_delta_abs" in result.detail


class TestCheckDeribitPublic:
    """Tests for check_deribit_public function."""

    def test_successful_connection_returns_ok(self):
        client = FakeDeribitClient(btc_price=95000, eth_price=3500)
        result = check_deribit_public(client)
        assert result.status == CheckStatus.OK
        assert "BTC=$95,000" in result.detail
        assert "ETH=$3,500" in result.detail

    def test_api_error_returns_fail(self):
        client = FakeDeribitClient(fail_public=True)
        result = check_deribit_public(client)
        assert result.status == CheckStatus.FAIL
        assert "API error" in result.detail or "unavailable" in result.detail


class TestCheckDeribitPrivate:
    """Tests for check_deribit_private function."""

    def test_no_credentials_returns_skipped(self):
        client = FakeDeribitClient()
        cfg = Settings(deribit_client_id="", deribit_client_secret="")
        result = check_deribit_private(client, cfg)
        assert result.status == CheckStatus.SKIPPED
        assert "no private API credentials" in result.detail

    def test_valid_credentials_returns_ok(self):
        client = FakeDeribitClient(equity=150000)
        cfg = Settings(deribit_client_id="test_id", deribit_client_secret="test_secret")
        result = check_deribit_private(client, cfg)
        assert result.status == CheckStatus.OK
        assert "150,000" in result.detail

    def test_auth_error_returns_fail(self):
        client = FakeDeribitClient(fail_private=True)
        cfg = Settings(deribit_client_id="test_id", deribit_client_secret="test_secret")
        result = check_deribit_private(client, cfg)
        assert result.status == CheckStatus.FAIL
        assert "authentication" in result.detail.lower() or "error" in result.detail.lower()


class TestCheckStateBuilder:
    """Tests for check_state_builder function."""

    def test_successful_build_returns_ok(self):
        mock_state = MagicMock()
        mock_state.portfolio.equity_usd = 100000.0
        mock_state.positions = [MagicMock(), MagicMock()]
        mock_state.candidates = [MagicMock()]

        client = FakeDeribitClient()
        cfg = Settings()

        with patch("src.state_builder.build_agent_state", return_value=mock_state):
            result = check_state_builder(client, cfg)
            assert result.status == CheckStatus.OK
            assert "equity=$100,000" in result.detail
            assert "positions=2" in result.detail
            assert "candidates=1" in result.detail

    def test_api_error_returns_fail(self):
        client = FakeDeribitClient()
        cfg = Settings()

        with patch("src.state_builder.build_agent_state", side_effect=DeribitAPIError(-1, "connection failed")):
            result = check_state_builder(client, cfg)
            assert result.status == CheckStatus.FAIL
            assert "failed" in result.detail.lower()


class TestRunAgentHealthcheck:
    """Tests for run_agent_healthcheck aggregator."""

    def test_all_ok_returns_overall_ok(self):
        mock_state = MagicMock()
        mock_state.portfolio.equity_usd = 50000.0
        mock_state.positions = []
        mock_state.candidates = []

        cfg = Settings(deribit_client_id="", deribit_client_secret="")

        with patch("src.healthcheck.DeribitClient") as MockClient:
            mock_client = FakeDeribitClient()
            MockClient.return_value.__enter__ = MagicMock(return_value=mock_client)
            MockClient.return_value.__exit__ = MagicMock(return_value=False)

            with patch("src.state_builder.build_agent_state", return_value=mock_state):
                result = run_agent_healthcheck(cfg)

        assert result["overall_status"] in ("OK", "WARN")
        assert len(result["results"]) >= 3

    def test_any_fail_returns_overall_fail(self):
        cfg = Settings(loop_interval_sec=0)

        with patch("src.healthcheck.DeribitClient") as MockClient:
            mock_client = FakeDeribitClient()
            MockClient.return_value.__enter__ = MagicMock(return_value=mock_client)
            MockClient.return_value.__exit__ = MagicMock(return_value=False)

            result = run_agent_healthcheck(cfg)

        assert result["overall_status"] == "FAIL"
        config_result = next(r for r in result["results"] if r["name"] == "config")
        assert config_result["status"] == "fail"


class TestHealthCheckResult:
    """Tests for HealthCheckResult dataclass."""

    def test_result_creation(self):
        result = HealthCheckResult(
            name="test",
            status=CheckStatus.OK,
            detail="all good"
        )
        assert result.name == "test"
        assert result.status == CheckStatus.OK
        assert result.detail == "all good"


class TestCheckStatus:
    """Tests for CheckStatus enum."""

    def test_status_values(self):
        assert CheckStatus.OK.value == "ok"
        assert CheckStatus.WARN.value == "warn"
        assert CheckStatus.FAIL.value == "fail"
        assert CheckStatus.SKIPPED.value == "skipped"

    def test_status_is_string_enum(self):
        assert isinstance(CheckStatus.OK, str)
        assert CheckStatus.OK == "ok"
