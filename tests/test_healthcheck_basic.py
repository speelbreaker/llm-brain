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
from src.deribit.base_client import DeribitErrorCode


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
            raise DeribitAPIError(-1, "public API unavailable", error_code=DeribitErrorCode.NETWORK)
        if underlying == "BTC":
            return self.btc_price
        elif underlying == "ETH":
            return self.eth_price
        return 0.0

    def get_account_summary(self, currency: str) -> dict:
        if self.fail_private:
            raise DeribitAPIError(-1, "authentication failed", error_code=DeribitErrorCode.AUTH)
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

    def test_no_credentials_in_production_returns_fail(self):
        client = FakeDeribitClient()
        cfg = Settings(mode="production", deribit_client_id="", deribit_client_secret="")
        result = check_deribit_private(client, cfg)
        assert result.status == CheckStatus.FAIL
        assert result.error_code == "DERIBIT_PRIVATE_MISSING_CREDS"

    def test_valid_credentials_returns_ok(self):
        client = FakeDeribitClient(equity=150000)
        cfg = Settings(deribit_client_id="test_id", deribit_client_secret="test_secret")
        result = check_deribit_private(client, cfg)
        assert result.status == CheckStatus.OK
        assert "150,000.00" in result.detail

    def test_auth_error_returns_fail(self):
        client = FakeDeribitClient(fail_private=True)
        cfg = Settings(deribit_client_id="test_id", deribit_client_secret="test_secret")
        result = check_deribit_private(client, cfg)
        assert result.status == CheckStatus.FAIL
        assert "authentication" in result.detail.lower() or "error" in result.detail.lower()


class TestCheckFidelityGateModes:
    """Tests for fidelity gate behavior across modes."""

    def _mock_facts(self, *, available: bool, gate_label: str | None = None):
        return {
            "available": available,
            "gate_label": gate_label,
            "overall_score": 50.0,
            "run_id": "test_run",
        }

    def test_fidelity_gate_off_never_blocks(self, monkeypatch):
        from src import healthcheck

        monkeypatch.setenv("FIDELITY_GATE_MODE", "off")
        monkeypatch.delenv("HEALTH_STRICT_SYNTHETIC_GATE", raising=False)

        monkeypatch.setattr(
            "src.ops.fidelity_status.get_fidelity_facts",
            lambda underlying, base_dir=None: self._mock_facts(available=True, gate_label="UNTRUSTED"),
        )
        result = healthcheck.check_fidelity_gate(Settings())
        assert result.status == CheckStatus.WARN
        assert result.can_trade is True

    def test_fidelity_gate_warn_allows_trade(self, monkeypatch):
        from src import healthcheck

        monkeypatch.setenv("FIDELITY_GATE_MODE", "warn")
        monkeypatch.delenv("HEALTH_STRICT_SYNTHETIC_GATE", raising=False)

        monkeypatch.setattr(
            "src.ops.fidelity_status.get_fidelity_facts",
            lambda underlying, base_dir=None: self._mock_facts(available=False, gate_label=None),
        )
        result = healthcheck.check_fidelity_gate(Settings())
        assert result.status == CheckStatus.WARN
        assert result.can_trade is True

    def test_fidelity_gate_block_blocks_untrusted(self, monkeypatch):
        from src import healthcheck

        monkeypatch.setenv("FIDELITY_GATE_MODE", "block")
        monkeypatch.delenv("HEALTH_STRICT_SYNTHETIC_GATE", raising=False)

        monkeypatch.setattr(
            "src.ops.fidelity_status.get_fidelity_facts",
            lambda underlying, base_dir=None: self._mock_facts(available=True, gate_label="UNTRUSTED"),
        )
        result = healthcheck.check_fidelity_gate(Settings())
        assert result.status == CheckStatus.FAIL
        assert result.can_trade is False

    def test_fidelity_gate_block_blocks_missing(self, monkeypatch):
        from src import healthcheck

        monkeypatch.setenv("FIDELITY_GATE_MODE", "block")
        monkeypatch.delenv("HEALTH_STRICT_SYNTHETIC_GATE", raising=False)

        monkeypatch.setattr(
            "src.ops.fidelity_status.get_fidelity_facts",
            lambda underlying, base_dir=None: self._mock_facts(available=False, gate_label=None),
        )
        result = healthcheck.check_fidelity_gate(Settings())
        assert result.status == CheckStatus.FAIL
        assert result.can_trade is False

class TestCheckStateBuilder:
    """Tests for check_state_builder function."""

    def test_successful_build_returns_ok(self):
        mock_state = MagicMock()
        mock_state.portfolio.equity_usd = 100000.0
        mock_state.portfolio.option_positions = [MagicMock(), MagicMock()]
        mock_state.candidate_options = [MagicMock()]

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
        mock_state.portfolio.option_positions = []
        mock_state.candidate_options = []

        cfg = Settings(deribit_client_id="", deribit_client_secret="")

        with patch("src.healthcheck.DeribitClient") as MockClient:
            mock_client = FakeDeribitClient()
            MockClient.return_value.__enter__ = MagicMock(return_value=mock_client)
            MockClient.return_value.__exit__ = MagicMock(return_value=False)

            ok_harvest = HealthCheckResult(
                name="harvest_freshness",
                status=CheckStatus.OK,
                detail="harvest ok",
                severity="OK",
                can_trade=True,
            )
            ok_calibration = HealthCheckResult(
                name="calibration_freshness",
                status=CheckStatus.OK,
                detail="calibration ok",
                severity="OK",
                can_trade=True,
            )
            ok_fidelity = HealthCheckResult(
                name="fidelity_gate",
                status=CheckStatus.OK,
                detail="fidelity ok",
                severity="OK",
                can_trade=True,
            )

            with patch("src.state_builder.build_agent_state", return_value=mock_state), \
                patch("src.healthcheck.check_harvest_freshness", return_value=ok_harvest), \
                patch("src.healthcheck.check_calibration_freshness", return_value=ok_calibration), \
                patch("src.healthcheck.check_fidelity_gate", return_value=ok_fidelity):
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

        assert result["checks_overall"] == "FAIL"
        config_result = next(r for r in result["results"] if r["name"] == "config")
        assert config_result["status"] == "FAIL"


class TestRunAgentHealthcheckGateDecision:
    """Ensure gate_overall drives trade readiness and summary overrides checks."""

    def _patch_common(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src import healthcheck

        monkeypatch.setattr(healthcheck, "DeribitClient", FakeDeribitClient)
        for name in ("check_deribit_public", "check_deribit_private", "check_state_builder"):
            monkeypatch.setattr(
                healthcheck,
                name,
                lambda *args, **kwargs: HealthCheckResult(
                    name=name,
                    status=CheckStatus.OK,
                    detail="ok",
                    severity="OK",
                    can_trade=True,
                ),
            )
        for name in ("check_harvest_freshness", "check_calibration_freshness", "check_fidelity_gate"):
            monkeypatch.setattr(
                healthcheck,
                name,
                lambda *args, **kwargs: HealthCheckResult(
                    name=name,
                    status=CheckStatus.OK,
                    detail="ok",
                    severity="OK",
                    can_trade=True,
                ),
            )
        monkeypatch.setattr(
            healthcheck,
            "check_risk_config",
            lambda cfg: HealthCheckResult(
                name="risk_config",
                status=CheckStatus.OK,
                detail="risk config ok",
                severity="OK",
                can_trade=True,
            ),
        )

        def resolve_ops_facts(cfg, *, now=None):
            return {
                "now": "2025-01-01T00:00:00+00:00",
                "underlyings_active": ["BTC"],
                "paths": {
                    "live_deribit_data_dir": "data/live_deribit",
                    "calibration_dir": "data/calibration_runs",
                    "fidelity_dir": "data/fidelity_runs",
                },
                "harvest": {"BTC": {}},
                "calibration": {"BTC": {}},
                "fidelity": {"BTC": {}},
            }

        monkeypatch.setattr(
            "src.ops.facts_resolver.resolve_ops_facts",
            resolve_ops_facts,
        )
        monkeypatch.setattr(
            "src.ops.gate_factories.build_underlying_gate_fns",
            lambda **kwargs: [],
        )

    def test_gate_failure_precedes_checks(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src import healthcheck
        from src.ops.gates import GateRunner

        self._patch_common(monkeypatch)
        monkeypatch.setenv("FIDELITY_GATE_MODE", "block")
        monkeypatch.setenv("CALIBRATION_GATE_MODE", "warn")

        def _run_gates(self, gate_fns):
            return {
                "gates": [],
                "gate_overall": {
                    "global": {
                        "status": "FAIL",
                        "severity": "FATAL",
                        "can_trade": False,
                        "message": "gate blocked",
                        "code": "GATE_BLOCKED",
                    },
                    "by_underlying": {"BTC": {"status": "FAIL", "severity": "FATAL", "can_trade": False}},
                },
            }

        monkeypatch.setattr(GateRunner, "run", _run_gates)

        result = run_agent_healthcheck(Settings())
        assert result["overall_status"] == "FAIL"
        assert result["checks_overall"] == "OK"
        assert "gates FAIL" in result["summary"]
        assert "code=GATE_BLOCKED" in result["summary"]
        assert result["can_trade"] is False

    def test_check_failure_trumps_gates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src import healthcheck
        from src.ops.gates import GateRunner

        self._patch_common(monkeypatch)
        monkeypatch.setenv("FIDELITY_GATE_MODE", "off")
        monkeypatch.setenv("CALIBRATION_GATE_MODE", "off")

        monkeypatch.setattr(
            healthcheck,
            "check_config",
            lambda cfg: HealthCheckResult(
                name="check_config",
                status=CheckStatus.FAIL,
                detail="config missing",
                severity="FATAL",
                can_trade=False,
            ),
        )

        def _run_gates(self, gate_fns):
            return {
                "gates": [],
                "gate_overall": {
                    "global": {"status": "PASS", "severity": "OK", "can_trade": True},
                    "by_underlying": {},
                },
            }

        monkeypatch.setattr(GateRunner, "run", _run_gates)

        result = run_agent_healthcheck(Settings())
        assert result["overall_status"] == "FAIL"
        assert result["checks_overall"] == "FAIL"
        assert "check_config FAIL" in result["summary"]

    def test_gate_eval_error_fails_closed_when_gates_enabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.ops.gates import GateRunner

        self._patch_common(monkeypatch)
        monkeypatch.setenv("FIDELITY_GATE_MODE", "block")
        monkeypatch.setenv("CALIBRATION_GATE_MODE", "off")

        def _boom(self, gate_fns):
            raise RuntimeError("boom")

        monkeypatch.setattr(GateRunner, "run", _boom)

        result = run_agent_healthcheck(Settings())
        assert result["overall_status"] == "FAIL"
        assert result["can_trade"] is False
        gates_runner = next((c for c in (result.get("checks") or []) if c.get("name") == "gates_runner"), None)
        assert gates_runner is not None
        assert gates_runner.get("error_code") == "GATES_EVAL_ERROR"


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
        assert result.severity == "OK"
        assert result.can_trade is True


class TestCheckStatus:
    """Tests for CheckStatus enum."""

    def test_status_values(self):
        assert CheckStatus.OK.value == "OK"
        assert CheckStatus.WARN.value == "WARN"
        assert CheckStatus.FAIL.value == "FAIL"
        assert CheckStatus.SKIPPED.value == "SKIPPED"

    def test_status_is_string_enum(self):
        assert isinstance(CheckStatus.OK, str)
        assert CheckStatus.OK == "OK"
