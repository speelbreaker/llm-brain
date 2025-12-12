"""
Tests for healthcheck config validation.

Validates that check_config, check_risk_config, and check_llm_config
properly detect misconfigurations and return appropriate status codes.
"""
import os
import pytest
from unittest.mock import patch

from src.config import Settings
from src.healthcheck import (
    CheckStatus,
    check_config,
    check_risk_config,
    check_llm_config,
    run_agent_healthcheck,
    get_llm_readiness,
)


class TestCheckConfig:
    """Tests for basic config validation."""
    
    def test_valid_config_returns_ok(self):
        """Valid basic config should return OK status."""
        cfg = Settings(
            deribit_env="testnet",
            loop_interval_sec=60,
            max_margin_used_pct=80.0,
            max_net_delta_abs=5.0,
        )
        result = check_config(cfg)
        assert result.status == CheckStatus.OK
        assert "testnet" in result.detail
    
    def test_invalid_deribit_env_rejected_by_pydantic(self):
        """Invalid deribit_env should be rejected by Pydantic at Settings construction."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            Settings(deribit_env="invalid_env")
    
    def test_zero_loop_interval_returns_fail(self):
        """Zero loop_interval_sec should return FAIL."""
        cfg = Settings(loop_interval_sec=0)
        result = check_config(cfg)
        assert result.status == CheckStatus.FAIL
        assert "loop_interval_sec" in result.detail
    
    def test_negative_loop_interval_returns_fail(self):
        """Negative loop_interval_sec should return FAIL."""
        cfg = Settings(loop_interval_sec=-10)
        result = check_config(cfg)
        assert result.status == CheckStatus.FAIL
        assert "loop_interval_sec" in result.detail
    
    def test_invalid_margin_pct_returns_fail(self):
        """Invalid max_margin_used_pct should return FAIL."""
        cfg = Settings(max_margin_used_pct=0)
        result = check_config(cfg)
        assert result.status == CheckStatus.FAIL
        assert "max_margin_used_pct" in result.detail
        
        cfg = Settings(max_margin_used_pct=150)
        result = check_config(cfg)
        assert result.status == CheckStatus.FAIL
    
    def test_negative_delta_returns_fail(self):
        """Negative max_net_delta_abs should return FAIL."""
        cfg = Settings(max_net_delta_abs=-1)
        result = check_config(cfg)
        assert result.status == CheckStatus.FAIL
        assert "max_net_delta_abs" in result.detail


class TestCheckRiskConfig:
    """Tests for risk config validation."""
    
    def test_valid_risk_config_returns_ok(self):
        """Valid risk config should return OK."""
        cfg = Settings(
            kill_switch_enabled=False,
            daily_drawdown_limit_pct=5.0,
            max_expiry_exposure=0.5,
        )
        result = check_risk_config(cfg)
        assert result.status == CheckStatus.OK
    
    def test_kill_switch_on_returns_warn(self):
        """Kill switch ON should return WARN."""
        cfg = Settings(kill_switch_enabled=True)
        result = check_risk_config(cfg)
        assert result.status == CheckStatus.WARN
        assert "kill_switch is ON" in result.detail
    
    def test_zero_drawdown_limit_returns_warn(self):
        """Zero drawdown limit (disabled) should return WARN."""
        cfg = Settings(daily_drawdown_limit_pct=0.0, kill_switch_enabled=False)
        result = check_risk_config(cfg)
        assert result.status == CheckStatus.WARN
        assert "daily_drawdown_limit_pct is 0" in result.detail
    
    def test_negative_drawdown_limit_returns_fail(self):
        """Negative drawdown limit should return FAIL."""
        cfg = Settings(daily_drawdown_limit_pct=-5.0)
        result = check_risk_config(cfg)
        assert result.status == CheckStatus.FAIL
        assert "daily_drawdown_limit_pct must be >= 0" in result.detail
    
    def test_zero_expiry_exposure_returns_fail(self):
        """Zero expiry exposure should return FAIL."""
        cfg = Settings(max_expiry_exposure=0)
        result = check_risk_config(cfg)
        assert result.status == CheckStatus.FAIL
        assert "max_expiry_exposure must be > 0" in result.detail
    
    def test_negative_expiry_exposure_returns_fail(self):
        """Negative expiry exposure should return FAIL."""
        cfg = Settings(max_expiry_exposure=-0.5)
        result = check_risk_config(cfg)
        assert result.status == CheckStatus.FAIL


class TestCheckLLMConfig:
    """Tests for LLM config validation."""
    
    def test_llm_disabled_returns_ok(self):
        """LLM disabled should return OK (no validation needed)."""
        cfg = Settings(llm_enabled=False)
        result = check_llm_config(cfg)
        assert result.status == CheckStatus.OK
        assert "LLM disabled" in result.detail
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key"}, clear=False)
    def test_llm_enabled_with_key_returns_ok(self):
        """LLM enabled with API key should return OK."""
        cfg = Settings(
            llm_enabled=True,
            llm_model_name="gpt-4.1-mini",
        )
        result = check_llm_config(cfg)
        assert result.status == CheckStatus.OK
        assert "LLM enabled" in result.detail
    
    @patch.dict(os.environ, {}, clear=True)
    def test_llm_enabled_without_key_returns_fail(self):
        """LLM enabled without API key should return FAIL."""
        for key in ["OPENAI_API_KEY", "AI_INTEGRATIONS_OPENAI_API_KEY"]:
            if key in os.environ:
                del os.environ[key]
        
        cfg = Settings(llm_enabled=True)
        result = check_llm_config(cfg)
        assert result.status == CheckStatus.FAIL
        assert "OPENAI_API_KEY not set" in result.detail
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key"}, clear=False)
    def test_llm_enabled_without_model_returns_fail(self):
        """LLM enabled without model name should return FAIL."""
        cfg = Settings(
            llm_enabled=True,
            llm_model_name="",
        )
        result = check_llm_config(cfg)
        assert result.status == CheckStatus.FAIL
        assert "llm_model_name is empty" in result.detail


class TestGetLLMReadiness:
    """Tests for LLM readiness check."""
    
    def test_llm_disabled_not_ready(self):
        """LLM disabled should return not ready."""
        cfg = Settings(llm_enabled=False)
        result = get_llm_readiness(cfg)
        assert result["ready"] is False
        assert "disabled" in result["reason"].lower()
    
    @patch.dict(os.environ, {}, clear=True)
    def test_llm_no_key_not_ready(self):
        """LLM enabled without API key should return not ready."""
        for key in ["OPENAI_API_KEY", "AI_INTEGRATIONS_OPENAI_API_KEY"]:
            if key in os.environ:
                del os.environ[key]
        
        cfg = Settings(llm_enabled=True)
        result = get_llm_readiness(cfg)
        assert result["ready"] is False
        assert "api key" in result["reason"].lower()
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key"}, clear=False)
    def test_llm_ready(self):
        """LLM enabled with API key should return ready."""
        cfg = Settings(
            llm_enabled=True,
            llm_model_name="gpt-4.1-mini",
        )
        result = get_llm_readiness(cfg)
        assert result["ready"] is True
        assert result["has_api_key"] is True


class TestRunAgentHealthcheck:
    """Tests for the aggregated healthcheck runner."""
    
    @patch("src.healthcheck.DeribitClient")
    def test_healthcheck_includes_config_checks(self, mock_client):
        """Healthcheck should include config, risk, and LLM checks."""
        mock_client.return_value.__enter__ = lambda s: s
        mock_client.return_value.__exit__ = lambda s, *args: None
        mock_client.return_value.get_index_price.return_value = 50000
        
        cfg = Settings(
            deribit_env="testnet",
            llm_enabled=False,
        )
        result = run_agent_healthcheck(cfg)
        
        check_names = [r["name"] for r in result["results"]]
        assert "config" in check_names
        assert "risk_config" in check_names
        assert "llm_config" in check_names
    
    @patch("src.healthcheck.DeribitClient")
    def test_healthcheck_fail_propagates(self, mock_client):
        """A FAIL in any check should make overall status FAIL."""
        mock_client.side_effect = Exception("Connection failed")
        
        cfg = Settings(
            deribit_env="testnet",
            max_expiry_exposure=-1,
        )
        result = run_agent_healthcheck(cfg)
        
        assert result["overall_status"] == "FAIL"
    
    @patch("src.healthcheck.DeribitClient")
    def test_healthcheck_warn_propagates(self, mock_client):
        """A WARN should make overall status WARN if no FAILs."""
        mock_client.return_value.__enter__ = lambda s: s
        mock_client.return_value.__exit__ = lambda s, *args: None
        mock_client.return_value.get_index_price.return_value = 50000
        mock_client.return_value.get_account_summary.return_value = {"equity": 10000}
        
        with patch("src.healthcheck.check_state_builder") as mock_state:
            mock_state.return_value = type("R", (), {"name": "state_builder", "status": CheckStatus.OK, "detail": "ok"})()
            
            cfg = Settings(
                deribit_env="testnet",
                kill_switch_enabled=True,
                llm_enabled=False,
            )
            result = run_agent_healthcheck(cfg)
            
            has_warn = any(r["status"] == "warn" for r in result["results"])
            assert has_warn
            if result["overall_status"] != "FAIL":
                assert result["overall_status"] == "WARN"
    
    def test_healthcheck_returns_summary(self):
        """Healthcheck should return a summary string."""
        cfg = Settings(max_expiry_exposure=-1)
        result = run_agent_healthcheck(cfg)
        
        assert "summary" in result
        assert isinstance(result["summary"], str)
