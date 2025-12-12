"""
Agent Healthcheck Module

Provides health check functionality for the Options Trading Agent.
Exercises the critical pipeline: config → Deribit → state builder.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Any

from src.config import Settings, settings
from src.deribit_client import DeribitClient, DeribitAPIError


class CheckStatus(str, Enum):
    OK = "ok"
    WARN = "warn"
    FAIL = "fail"
    SKIPPED = "skipped"


@dataclass
class HealthCheckResult:
    name: str
    status: CheckStatus
    detail: str


def _validate_basic_config(cfg: Settings) -> list[str]:
    """Validate basic config settings. Returns list of issues (critical failures)."""
    issues = []

    if cfg.deribit_env not in ("testnet", "mainnet"):
        issues.append(f"invalid deribit_env: {cfg.deribit_env}")

    if cfg.loop_interval_sec <= 0:
        issues.append(f"loop_interval_sec must be > 0, got {cfg.loop_interval_sec}")

    if not (0 < cfg.max_margin_used_pct <= 100):
        issues.append(f"max_margin_used_pct must be in (0, 100], got {cfg.max_margin_used_pct}")

    if cfg.max_net_delta_abs < 0:
        issues.append(f"max_net_delta_abs must be >= 0, got {cfg.max_net_delta_abs}")

    return issues


def _validate_risk_settings(cfg: Settings) -> tuple[list[str], list[str]]:
    """Validate risk-related settings. Returns (warnings, failures)."""
    warnings = []
    failures = []

    if cfg.kill_switch_enabled:
        warnings.append("kill_switch is ON (trading disabled)")

    if cfg.daily_drawdown_limit_pct == 0.0:
        warnings.append("daily_drawdown_limit_pct is 0 (disabled)")
    elif cfg.daily_drawdown_limit_pct < 0:
        failures.append(f"daily_drawdown_limit_pct must be >= 0, got {cfg.daily_drawdown_limit_pct}")

    if cfg.max_expiry_exposure <= 0:
        failures.append(f"max_expiry_exposure must be > 0, got {cfg.max_expiry_exposure}")

    if hasattr(cfg, 'research_max_expiry_exposure') and cfg.research_max_expiry_exposure <= 0:
        warnings.append(f"research_max_expiry_exposure is <= 0: {cfg.research_max_expiry_exposure}")

    return warnings, failures


def _validate_llm_settings(cfg: Settings) -> tuple[list[str], list[str]]:
    """Validate LLM-related settings. Returns (warnings, failures)."""
    warnings = []
    failures = []

    if not cfg.llm_enabled:
        return warnings, failures

    if not cfg.llm_model_name:
        failures.append("llm_enabled=True but llm_model_name is empty")

    openai_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("AI_INTEGRATIONS_OPENAI_API_KEY")
    if not openai_key:
        failures.append("llm_enabled=True but OPENAI_API_KEY not set")

    if cfg.llm_timeout_seconds <= 0:
        warnings.append(f"llm_timeout_seconds should be > 0, got {cfg.llm_timeout_seconds}")

    if cfg.llm_max_decision_tokens <= 0:
        warnings.append(f"llm_max_decision_tokens should be > 0, got {cfg.llm_max_decision_tokens}")

    return warnings, failures


def check_config(cfg: Settings) -> HealthCheckResult:
    """Validate basic configuration sanity."""
    try:
        basic_issues = _validate_basic_config(cfg)
        if basic_issues:
            return HealthCheckResult(
                name="config",
                status=CheckStatus.FAIL,
                detail="; ".join(basic_issues)
            )

        mode = "training" if cfg.training_mode else ("research" if cfg.is_research else "live")
        detail = f"mode={mode}, env={cfg.deribit_env}, loop_interval={cfg.loop_interval_sec}s"
        return HealthCheckResult(
            name="config",
            status=CheckStatus.OK,
            detail=detail
        )

    except Exception as e:
        return HealthCheckResult(
            name="config",
            status=CheckStatus.FAIL,
            detail=f"config validation error: {str(e)}"
        )


def check_risk_config(cfg: Settings) -> HealthCheckResult:
    """Validate risk management configuration."""
    try:
        warnings, failures = _validate_risk_settings(cfg)

        if failures:
            return HealthCheckResult(
                name="risk_config",
                status=CheckStatus.FAIL,
                detail="; ".join(failures)
            )

        if warnings:
            return HealthCheckResult(
                name="risk_config",
                status=CheckStatus.WARN,
                detail="; ".join(warnings)
            )

        ks_status = "ON" if cfg.kill_switch_enabled else "OFF"
        dd_limit = cfg.daily_drawdown_limit_pct
        return HealthCheckResult(
            name="risk_config",
            status=CheckStatus.OK,
            detail=f"kill_switch={ks_status}, drawdown_limit={dd_limit}%, expiry_exposure={cfg.max_expiry_exposure}"
        )

    except Exception as e:
        return HealthCheckResult(
            name="risk_config",
            status=CheckStatus.FAIL,
            detail=f"risk config validation error: {str(e)}"
        )


def check_llm_config(cfg: Settings) -> HealthCheckResult:
    """Validate LLM configuration."""
    try:
        if not cfg.llm_enabled:
            return HealthCheckResult(
                name="llm_config",
                status=CheckStatus.OK,
                detail="LLM disabled (llm_enabled=False)"
            )

        warnings, failures = _validate_llm_settings(cfg)

        if failures:
            return HealthCheckResult(
                name="llm_config",
                status=CheckStatus.FAIL,
                detail="; ".join(failures)
            )

        if warnings:
            return HealthCheckResult(
                name="llm_config",
                status=CheckStatus.WARN,
                detail="; ".join(warnings)
            )

        return HealthCheckResult(
            name="llm_config",
            status=CheckStatus.OK,
            detail=f"LLM enabled, model={cfg.llm_model_name}, mode={cfg.decision_mode}"
        )

    except Exception as e:
        return HealthCheckResult(
            name="llm_config",
            status=CheckStatus.FAIL,
            detail=f"LLM config validation error: {str(e)}"
        )


def check_deribit_public(client: DeribitClient) -> HealthCheckResult:
    """Check public Deribit API connectivity."""
    try:
        btc_price = client.get_index_price("BTC")
        eth_price = client.get_index_price("ETH")

        return HealthCheckResult(
            name="deribit_public",
            status=CheckStatus.OK,
            detail=f"public API OK, BTC=${btc_price:,.0f}, ETH=${eth_price:,.0f}"
        )

    except DeribitAPIError as e:
        return HealthCheckResult(
            name="deribit_public",
            status=CheckStatus.FAIL,
            detail=f"Deribit API error: {str(e)}"
        )
    except Exception as e:
        return HealthCheckResult(
            name="deribit_public",
            status=CheckStatus.FAIL,
            detail=f"network error: {str(e)}"
        )


def check_deribit_private(client: DeribitClient, cfg: Settings) -> HealthCheckResult:
    """Check private Deribit API connectivity (requires credentials)."""
    try:
        if not cfg.deribit_client_id or not cfg.deribit_client_secret:
            return HealthCheckResult(
                name="deribit_private",
                status=CheckStatus.SKIPPED,
                detail="no private API credentials configured"
            )

        settlement_ccy = cfg.option_settlement_ccy or "USDC"
        account = client.get_account_summary(settlement_ccy)
        equity = account.get("equity", 0)

        return HealthCheckResult(
            name="deribit_private",
            status=CheckStatus.OK,
            detail=f"private API OK, equity={equity:,.2f} {settlement_ccy}"
        )

    except DeribitAPIError as e:
        return HealthCheckResult(
            name="deribit_private",
            status=CheckStatus.FAIL,
            detail=f"Deribit private API error: {str(e)}"
        )
    except Exception as e:
        return HealthCheckResult(
            name="deribit_private",
            status=CheckStatus.FAIL,
            detail=f"private API error: {str(e)}"
        )


def check_state_builder(client: DeribitClient, cfg: Settings) -> HealthCheckResult:
    """Check that the state builder pipeline works."""
    try:
        from src.state_builder import build_agent_state

        state = build_agent_state(client, cfg)

        equity = state.portfolio.equity_usd
        positions = len(state.portfolio.option_positions)
        candidates = len(state.candidate_options)

        return HealthCheckResult(
            name="state_builder",
            status=CheckStatus.OK,
            detail=f"built AgentState: equity=${equity:,.2f}, positions={positions}, candidates={candidates}"
        )

    except DeribitAPIError as e:
        if "authentication" in str(e).lower() or "unauthorized" in str(e).lower():
            return HealthCheckResult(
                name="state_builder",
                status=CheckStatus.WARN,
                detail=f"state built partially (private API auth failed): {str(e)}"
            )
        return HealthCheckResult(
            name="state_builder",
            status=CheckStatus.FAIL,
            detail=f"state builder failed: {str(e)}"
        )
    except Exception as e:
        error_msg = str(e).lower()
        if "credentials" in error_msg or "authentication" in error_msg:
            return HealthCheckResult(
                name="state_builder",
                status=CheckStatus.WARN,
                detail=f"state built with partial portfolio (no private credentials)"
            )
        return HealthCheckResult(
            name="state_builder",
            status=CheckStatus.FAIL,
            detail=f"state builder error: {str(e)}"
        )


def run_agent_healthcheck(cfg: Settings | None = None) -> dict[str, Any]:
    """
    Run all health checks and return aggregated results.
    
    Returns:
        dict with 'overall_status', 'results' list, and 'summary' string
    """
    cfg = cfg or settings

    results: list[HealthCheckResult] = []

    results.append(check_config(cfg))
    results.append(check_risk_config(cfg))
    results.append(check_llm_config(cfg))

    try:
        with DeribitClient() as client:
            results.append(check_deribit_public(client))
            results.append(check_deribit_private(client, cfg))
            results.append(check_state_builder(client, cfg))
    except Exception as e:
        results.append(HealthCheckResult(
            name="deribit_client",
            status=CheckStatus.FAIL,
            detail=f"failed to create Deribit client: {str(e)}"
        ))

    has_fail = any(r.status == CheckStatus.FAIL for r in results)
    has_warn = any(r.status == CheckStatus.WARN for r in results)

    if has_fail:
        overall_status = "FAIL"
    elif has_warn:
        overall_status = "WARN"
    else:
        overall_status = "OK"

    summary_parts = []
    for r in results:
        if r.status == CheckStatus.FAIL:
            summary_parts.append(f"{r.name} FAIL")
        elif r.status == CheckStatus.WARN:
            summary_parts.append(f"{r.name} WARN")
    
    if not summary_parts:
        summary = "All checks passed"
    else:
        summary = ", ".join(summary_parts)

    return {
        "overall_status": overall_status,
        "summary": summary,
        "results": [
            {"name": r.name, "status": r.status.value, "detail": r.detail}
            for r in results
        ]
    }


def get_llm_readiness(cfg: Settings | None = None) -> dict[str, Any]:
    """
    Check if LLM is ready to use for diagnostic tests.
    
    Returns:
        dict with 'ready' bool, 'reason' string, and config info
    """
    cfg = cfg or settings

    if not cfg.llm_enabled:
        return {
            "ready": False,
            "reason": "LLM is disabled (llm_enabled=False)",
            "llm_enabled": False,
            "model_name": cfg.llm_model_name,
            "has_api_key": False,
        }

    openai_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("AI_INTEGRATIONS_OPENAI_API_KEY")
    if not openai_key:
        return {
            "ready": False,
            "reason": "OpenAI API key not configured",
            "llm_enabled": True,
            "model_name": cfg.llm_model_name,
            "has_api_key": False,
        }

    if not cfg.llm_model_name:
        return {
            "ready": False,
            "reason": "LLM model name is empty",
            "llm_enabled": True,
            "model_name": "",
            "has_api_key": True,
        }

    return {
        "ready": True,
        "reason": "LLM is ready",
        "llm_enabled": True,
        "model_name": cfg.llm_model_name,
        "has_api_key": True,
    }


def format_healthcheck_banner(result: dict[str, Any]) -> str:
    """Format healthcheck result as a startup banner string."""
    lines = [f"Healthcheck: {result['overall_status']}"]

    for r in result["results"]:
        status = r["status"].upper()
        name = r["name"]
        detail = r["detail"]
        lines.append(f"  - {name}: {status} – {detail}")

    return "\n".join(lines)
