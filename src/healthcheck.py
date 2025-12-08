"""
Agent Healthcheck Module

Provides health check functionality for the Options Trading Agent.
Exercises the critical pipeline: config → Deribit → state builder.
"""
from __future__ import annotations

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


def check_config(cfg: Settings) -> HealthCheckResult:
    """Validate basic configuration sanity."""
    try:
        issues = []

        if cfg.deribit_env not in ("testnet", "mainnet"):
            issues.append(f"invalid deribit_env: {cfg.deribit_env}")

        if cfg.loop_interval_sec <= 0:
            issues.append(f"loop_interval_sec must be > 0, got {cfg.loop_interval_sec}")

        if not (0 < cfg.max_margin_used_pct <= 100):
            issues.append(f"max_margin_used_pct must be in (0, 100], got {cfg.max_margin_used_pct}")

        if cfg.max_net_delta_abs < 0:
            issues.append(f"max_net_delta_abs must be >= 0, got {cfg.max_net_delta_abs}")

        if issues:
            return HealthCheckResult(
                name="config",
                status=CheckStatus.FAIL,
                detail="; ".join(issues)
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
        dict with 'overall_status' and 'results' list
    """
    cfg = cfg or settings

    results: list[HealthCheckResult] = []

    results.append(check_config(cfg))

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

    return {
        "overall_status": overall_status,
        "results": [
            {"name": r.name, "status": r.status.value, "detail": r.detail}
            for r in results
        ]
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
