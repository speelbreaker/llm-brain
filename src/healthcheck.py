"""
Agent Healthcheck Module

Provides health check functionality for the Options Trading Agent.
Exercises the critical pipeline: config → Deribit → state builder.

Features:
- Comprehensive config validation (basic, risk, LLM settings)
- Deribit public/private API connectivity checks
- State builder pipeline validation
- Cached health status for runtime guard integration
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from src.config import Settings, settings
from src.deribit_client import DeribitClient, DeribitAPIError
from src.deribit.base_client import DeribitErrorCode


class CheckStatus(str, Enum):
    OK = "OK"
    WARN = "WARN"
    FAIL = "FAIL"
    SKIPPED = "SKIPPED"


@dataclass
class HealthCheckResult:
    name: str
    status: CheckStatus
    detail: str
    error_code: Optional[str] = None
    severity: str = "OK"
    can_trade: bool = True
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class CachedHealthStatus:
    """Cached result of a healthcheck for runtime guard integration.
    
    Attributes:
        overall_status: "OK" | "WARN" | "FAIL"
        worst_severity: Highest severity classification from any checks
        can_trade: Whether trading should be allowed based on checks
        last_run_at: Timestamp of the healthcheck
        summary: Short description of health status
        details: Full healthcheck result dict
        agent_paused_due_to_health: Whether agent is currently paused
    """
    overall_status: str  # "OK" | "WARN" | "FAIL"
    worst_severity: Optional[str]
    can_trade: bool
    last_run_at: datetime
    summary: str
    details: dict = field(default_factory=dict)
    agent_paused_due_to_health: bool = False


_cached_health_status: Optional[CachedHealthStatus] = None
_health_cache_lock = threading.Lock()
_agent_paused_due_to_health: bool = False


def _compute_worst_severity(result: dict) -> tuple[str, bool]:
    """Compute worst severity and can_trade from structured healthcheck results."""
    checks = result.get("checks") or result.get("results") or []
    worst = "OK"
    can_trade = True

    for check in checks:
        severity = (check.get("severity") or "OK").upper()
        if severity == "FATAL":
            worst = "FATAL"
        elif severity == "DEGRADED" and worst != "FATAL":
            worst = "DEGRADED"

        if check.get("can_trade") is False:
            can_trade = False

    return worst, can_trade


def run_and_cache_healthcheck(cfg: Settings | None = None) -> CachedHealthStatus:
    """
    Run full healthcheck and cache the result.
    
    This is the primary entry point for runtime health guards.
    Thread-safe: uses a lock to prevent concurrent writes.
    
    Returns:
        CachedHealthStatus with the healthcheck results
    """
    global _cached_health_status
    
    result = run_agent_healthcheck(cfg)
    worst_severity = result.get("worst_severity")
    can_trade = result.get("can_trade")
    if worst_severity is None or can_trade is None:
        worst_severity_fallback, can_trade_fallback = _compute_worst_severity(result)
        if worst_severity is None:
            worst_severity = worst_severity_fallback
        if can_trade is None:
            can_trade = can_trade_fallback
        result.setdefault("worst_severity", worst_severity)
        result.setdefault("can_trade", can_trade)
    result.setdefault("checked_at", datetime.now(timezone.utc).isoformat())
    
    status = CachedHealthStatus(
        overall_status=result["overall_status"],
        worst_severity=worst_severity or "OK",
        can_trade=bool(can_trade),
        last_run_at=datetime.now(timezone.utc),
        summary=result["summary"],
        details=result,
        agent_paused_due_to_health=_agent_paused_due_to_health,
    )
    
    with _health_cache_lock:
        _cached_health_status = status
    
    return status


def get_cached_health_status() -> Optional[CachedHealthStatus]:
    """
    Get the last cached healthcheck result.
    
    Returns:
        CachedHealthStatus if available, None if healthcheck hasn't been run yet.
    """
    with _health_cache_lock:
        return _cached_health_status


def set_agent_paused_due_to_health(paused: bool) -> None:
    """Set the agent paused state for health guard."""
    global _agent_paused_due_to_health, _cached_health_status
    
    with _health_cache_lock:
        _agent_paused_due_to_health = paused
        if _cached_health_status:
            _cached_health_status.agent_paused_due_to_health = paused


def is_agent_paused_due_to_health() -> bool:
    """Check if agent is paused due to health failure."""
    return _agent_paused_due_to_health


def get_health_status_for_api() -> dict:
    """
    Get health status formatted for API response.
    
    CONTRACT (always enforced):
    - overall_status: one of {"OK", "WARN", "FAIL"} - NEVER null
    - can_trade: boolean ALWAYS (False on any error/missing data)
    - worst_severity: non-null ALWAYS ("CRITICAL" on error, "OK" otherwise)
    - summary: non-empty string ALWAYS
    - checks: list ALWAYS (empty list OK)
    - gates: list ALWAYS (empty list OK)
    - gate_overall: dict ALWAYS ({"status": "OFF"} if no gates active)
    
    Additional fields:
    - checked_at: ISO timestamp or null
    - cache_age_seconds: seconds since last check or null
    - checks_overall: OK/WARN/FAIL or null
    - can_trade_by_underlying: dict or null
    - agent_paused_due_to_health: bool
    """
    cached = get_cached_health_status()
    
    # Fail-closed defaults when healthcheck hasn't run
    if cached is None:
        return {
            "checked_at": None,
            "cache_age_seconds": None,
            "last_run_at": None,
            "overall_status": "FAIL",
            "checks_overall": "FAIL",
            "checks_summary": "",
            "worst_severity": "CRITICAL",
            "can_trade": False,
            "summary": "HEALTHCHECK_NOT_RUN: Healthcheck has not been executed yet",
            "checks": [],
            "gates": [],
            "gate_overall": {"status": "FAIL", "message": "healthcheck not run"},
            "can_trade_by_underlying": None,
            "agent_paused_due_to_health": _agent_paused_due_to_health,
        }

    now = datetime.now(timezone.utc)
    age_seconds = max(0.0, (now - cached.last_run_at).total_seconds())
    details = cached.details or {}
    checks = details.get("checks") or details.get("results") or []
    gates = details.get("gates") or []
    gate_overall_raw = details.get("gate_overall")
    can_trade_by_underlying = details.get("can_trade_by_underlying")
    ops_facts = details.get("ops_facts")

    checks_summary = details.get("checks_summary")
    if not isinstance(checks_summary, str):
        checks_summary = ", ".join(
            f"{c.get('name')}:{c.get('status')}" for c in checks if isinstance(c, dict)
        )

    # Enforce non-null overall_status
    overall_status = cached.overall_status
    if overall_status not in ("OK", "WARN", "FAIL"):
        overall_status = "FAIL"

    # Enforce non-null worst_severity
    worst_severity = cached.worst_severity
    if not worst_severity or not isinstance(worst_severity, str):
        worst_severity = "CRITICAL" if overall_status == "FAIL" else "OK"

    # Enforce boolean can_trade
    can_trade = cached.can_trade
    if not isinstance(can_trade, bool):
        can_trade = False

    # SINGLE SOURCE OF TRUTH: overall_status=FAIL must imply can_trade=False
    # This ensures consistency - if the system is in FAIL state, trading is blocked.
    if overall_status == "FAIL":
        can_trade = False

    # Enforce non-empty summary
    summary = cached.summary
    if not summary or not isinstance(summary, str):
        summary = f"STATUS_{overall_status}"

    # Enforce gate_overall is always a dict
    if isinstance(gate_overall_raw, dict):
        gate_overall = gate_overall_raw
    elif gates:
        # Gates exist but no gate_overall - derive status
        gate_overall = {"status": "UNKNOWN", "message": "gate_overall missing"}
    else:
        # No gates active
        gate_overall = {"status": "OFF", "message": "no gates configured"}

    return {
        "checked_at": cached.last_run_at.isoformat(),
        "cache_age_seconds": age_seconds,
        "last_run_at": cached.last_run_at.isoformat(),
        "overall_status": overall_status,
        "checks_overall": details.get("checks_overall"),
        "checks_overall_status": details.get("checks_overall"),
        "checks_summary": checks_summary,
        "worst_severity": worst_severity,
        "can_trade": can_trade,
        "summary": summary,
        "checks": checks,
        "gates": gates,
        "gate_overall": gate_overall,
        "can_trade_by_underlying": can_trade_by_underlying,
        "ops_facts": ops_facts,
        "agent_paused_due_to_health": cached.agent_paused_due_to_health,
    }


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
        warnings.append("llm_enabled=True but OPENAI_API_KEY not set")

    if cfg.llm_timeout_seconds <= 0:
        warnings.append(f"llm_timeout_seconds should be > 0, got {cfg.llm_timeout_seconds}")

    if cfg.llm_max_decision_tokens <= 0:
        warnings.append(f"llm_max_decision_tokens should be > 0, got {cfg.llm_max_decision_tokens}")

    return warnings, failures


def _env_flag(name: str) -> bool:
    value = os.environ.get(name) or ""
    return value.strip().lower() in ("1", "true", "yes")


def _is_live_mode(cfg: Settings) -> bool:
    return cfg.mode == "production"


def _resolve_fidelity_gate_mode(cfg: Settings) -> str:
    # Prefer env var for testability/runtime overrides; fall back to cfg.
    env = (os.getenv("FIDELITY_GATE_MODE") or "").strip().lower()
    mode = env or (getattr(cfg, "fidelity_gate_mode", None) or "off").strip().lower()
    if mode not in ("off", "warn", "block"):
        mode = "off"
    return mode


def check_config(cfg: Settings) -> HealthCheckResult:
    """Validate basic config settings."""
    try:
        issues = _validate_basic_config(cfg)
        if issues:
            return HealthCheckResult(
                name="config",
                status=CheckStatus.FAIL,
                detail="; ".join(issues),
                error_code="CONFIG_INVALID",
                severity="FATAL",
                can_trade=False,
            )
        return HealthCheckResult(
            name="config",
            status=CheckStatus.OK,
            detail=f"mode={cfg.mode} deribit_env={cfg.deribit_env}",
            severity="OK",
            can_trade=True,
        )
    except Exception as e:
        return HealthCheckResult(
            name="config",
            status=CheckStatus.FAIL,
            detail=f"config validation error: {str(e)}",
            error_code="CONFIG_INVALID",
            severity="FATAL",
            can_trade=False,
        )


def check_risk_config(cfg: Settings) -> HealthCheckResult:
    """Validate risk configuration (kill switch, drawdown, exposure limits)."""
    try:
        warnings, failures = _validate_risk_settings(cfg)

        if failures:
            return HealthCheckResult(
                name="risk_config",
                status=CheckStatus.FAIL,
                detail="; ".join(failures),
                error_code="RISK_CONFIG_INVALID",
                severity="FATAL",
                can_trade=False,
            )

        if warnings:
            can_trade = not bool(cfg.kill_switch_enabled)
            return HealthCheckResult(
                name="risk_config",
                status=CheckStatus.WARN,
                detail="; ".join(warnings),
                error_code="RISK_CONFIG_WARN",
                severity="DEGRADED",
                can_trade=can_trade,
            )

        return HealthCheckResult(
            name="risk_config",
            status=CheckStatus.OK,
            detail="risk config OK",
            severity="OK",
            can_trade=True,
        )
    except Exception as e:
        return HealthCheckResult(
            name="risk_config",
            status=CheckStatus.FAIL,
            detail=f"risk config validation error: {str(e)}",
            error_code="RISK_CONFIG_INVALID",
            severity="FATAL",
            can_trade=False,
        )


def check_llm_config(cfg: Settings) -> HealthCheckResult:
    """Validate LLM configuration."""
    try:
        llm_required = cfg.decision_mode == "llm_only"
        if not cfg.llm_enabled:
            if llm_required:
                return HealthCheckResult(
                    name="llm_config",
                    status=CheckStatus.FAIL,
                    detail="LLM disabled but decision_mode=llm_only",
                    error_code="LLM_DISABLED",
                    severity="FATAL",
                    can_trade=False,
                )
            return HealthCheckResult(
                name="llm_config",
                status=CheckStatus.OK,
                detail="LLM disabled (llm_enabled=False)",
                severity="OK",
                can_trade=True,
            )

        warnings, failures = _validate_llm_settings(cfg)

        if failures:
            return HealthCheckResult(
                name="llm_config",
                status=CheckStatus.FAIL,
                detail="; ".join(failures),
                error_code="LLM_CONFIG_INVALID",
                severity="FATAL",
                can_trade=False,
            )

        if warnings:
            missing_key = any("OPENAI_API_KEY" in w for w in warnings)
            if missing_key:
                return HealthCheckResult(
                    name="llm_config",
                    status=CheckStatus.WARN,
                    detail="; ".join(warnings),
                    error_code="LLM_MISSING_KEY",
                    severity="DEGRADED",
                    can_trade=not llm_required,
                )
            return HealthCheckResult(
                name="llm_config",
                status=CheckStatus.WARN,
                detail="; ".join(warnings),
                error_code="LLM_CONFIG_WARN",
                severity="DEGRADED",
                can_trade=True,
            )

        return HealthCheckResult(
            name="llm_config",
            status=CheckStatus.OK,
            detail=f"LLM enabled, model={cfg.llm_model_name}, mode={cfg.decision_mode}",
            severity="OK",
            can_trade=True,
        )

    except Exception as e:
        return HealthCheckResult(
            name="llm_config",
            status=CheckStatus.FAIL,
            detail=f"LLM config validation error: {str(e)}",
            error_code="LLM_CONFIG_INVALID",
            severity="FATAL",
            can_trade=False,
        )


def check_harvest_freshness(
    cfg: Settings,
    base_dir: str | Path | None = None,
) -> HealthCheckResult:
    """Check freshness of harvested live snapshots."""
    from src.harvest_status import harvest_freshness_for_underlying
    from src.harvest_status import get_harvest_root

    now = datetime.now(timezone.utc)
    base = get_harvest_root(base_dir)
    per_underlying: dict[str, Any] = {}
    statuses: list[str] = []
    detail_parts: list[str] = []

    for underlying in cfg.underlyings:
        u = underlying.upper()
        freshness = harvest_freshness_for_underlying(
            underlying=u,
            base_dir=base,
            now=now,
        )

        status = str(freshness.get("status") or "FAIL").upper()
        age_minutes = freshness.get("age_minutes")
        last_snapshot_at = freshness.get("last_snapshot_at")

        if status == "FAIL":
            detail_parts.append(f"{u}=missing")
        else:
            detail_parts.append(f"{u}={int(round(float(age_minutes or 0.0)))}m {status}")

        per_underlying[u] = {
            "status": status,
            "age_minutes": age_minutes,
            "last_snapshot_at": last_snapshot_at,
            "latest_file": freshness.get("latest_file"),
            "harvest_dir": freshness.get("harvest_dir"),
            "dirs_checked": freshness.get("dirs_checked"),
        }
        statuses.append(status)

    if not statuses:
        return HealthCheckResult(
            name="harvest_freshness",
            status=CheckStatus.SKIPPED,
            detail="no underlyings configured",
            severity="OK",
            can_trade=True,
        )

    if "FAIL" in statuses:
        overall_status = CheckStatus.FAIL
        error_code = "HARVEST_STALE"
        severity = "FATAL"
        can_trade = True
    elif "WARN" in statuses:
        overall_status = CheckStatus.WARN
        error_code = "HARVEST_LAG"
        severity = "DEGRADED"
        can_trade = True
    else:
        overall_status = CheckStatus.OK
        error_code = None
        severity = "OK"
        can_trade = True

    return HealthCheckResult(
        name="harvest_freshness",
        status=overall_status,
        detail=", ".join(detail_parts) if detail_parts else "no harvest data",
        error_code=error_code,
        severity=severity,
        can_trade=can_trade,
        meta={
            "base_dir": str(base),
            "per_underlying": per_underlying,
        },
    )


def check_calibration_freshness(cfg: Settings, base_dir: str | Path | None = None) -> HealthCheckResult:
    """Check freshness and quality of the latest calibration run."""
    now = datetime.now(timezone.utc)
    per_underlying: dict[str, Any] = {}
    detail_parts: list[str] = []
    flags = {"failed": False, "stale": False, "blocked": False, "lag": False}

    from src.ops.calibration_status import get_calibration_facts

    for underlying in cfg.underlyings:
        u = underlying.upper()
        facts = get_calibration_facts(underlying=u, base_dir=base_dir, now=now)

        if not bool(facts.get("available")):
            flags["stale"] = True
            per_underlying[u] = {"status": "FAIL", **facts}
            detail_parts.append(f"{u}=missing")
            continue

        last_status = str(facts.get("last_status") or "unknown").lower()
        age_hours = float(facts.get("age_hours") or 0.0)

        if last_status == "failed":
            flags["failed"] = True
            status = "FAIL"
        elif age_hours > 72:
            flags["stale"] = True
            status = "FAIL"
        elif age_hours > 36:
            flags["lag"] = True
            status = "WARN"
        elif last_status == "blocked":
            flags["blocked"] = True
            status = "WARN"
        else:
            status = "OK"

        per_underlying[u] = {"status": status, **facts}
        detail_parts.append(f"{u}={int(round(age_hours))}h {status}")

    if flags["failed"]:
        overall_status = CheckStatus.FAIL
        error_code = "CALIBRATION_FAILED"
        severity = "FATAL"
        can_trade = True
    elif flags["stale"]:
        overall_status = CheckStatus.FAIL
        error_code = "CALIBRATION_STALE"
        severity = "FATAL"
        can_trade = True
    elif flags["blocked"]:
        overall_status = CheckStatus.WARN
        error_code = "CALIBRATION_BLOCKED"
        severity = "DEGRADED"
        can_trade = True
    elif flags["lag"]:
        overall_status = CheckStatus.WARN
        error_code = "CALIBRATION_LAG"
        severity = "DEGRADED"
        can_trade = True
    else:
        overall_status = CheckStatus.OK
        error_code = None
        severity = "OK"
        can_trade = True

    return HealthCheckResult(
        name="calibration_freshness",
        status=overall_status,
        detail=", ".join(detail_parts) if detail_parts else "no calibration data",
        error_code=error_code,
        severity=severity,
        can_trade=can_trade,
        meta={
            "per_underlying": per_underlying,
        },
    )


def check_fidelity_gate(cfg: Settings, base_dir: str | Path | None = None) -> HealthCheckResult:
    """Check latest Synthetic Fidelity gate status."""
    gate_mode = _resolve_fidelity_gate_mode(cfg)
    from src.ops.fidelity_status import get_fidelity_facts

    per_underlying: dict[str, dict[str, Any]] = {}
    detail_parts: list[str] = []

    worst_label = "TRUSTED"

    def _rank(label: str) -> int:
        if label == "MISSING":
            return 3
        if label == "UNTRUSTED":
            return 2
        if label == "WARNING":
            return 1
        return 0

    for underlying in cfg.underlyings:
        u = (underlying or "").upper().strip()
        if not u:
            continue
        facts = get_fidelity_facts(underlying=u, base_dir=base_dir)
        per_underlying[u] = dict(facts)

        if not facts.get("available"):
            label = "MISSING"
        else:
            label = str(facts.get("gate_label") or facts.get("gate") or "").upper() or "UNKNOWN"

        score = facts.get("overall_score")
        run_id = facts.get("run_id")
        part = f"{u}={label}"
        if score is not None:
            part += f"({score})"
        if run_id:
            part += f"[{run_id}]"
        detail_parts.append(part)

        if _rank(label) > _rank(worst_label):
            worst_label = label

    if not per_underlying:
        worst_label = "MISSING"

    detail = ", ".join(detail_parts) if detail_parts else "no fidelity runs found"
    meta = {
        "available": worst_label != "MISSING",
        "gate_label": worst_label,
        "per_underlying": per_underlying,
        "base_dir": str(base_dir) if base_dir is not None else None,
    }

    if worst_label == "MISSING":
        # Fail-closed only when gate_mode=block; otherwise advisory.
        can_trade = (gate_mode != "block")
        return HealthCheckResult(
            name="fidelity_gate",
            status=CheckStatus.FAIL if gate_mode == "block" else CheckStatus.WARN,
            detail=detail,
            error_code="FIDELITY_MISSING",
            severity="FATAL" if gate_mode == "block" else "DEGRADED",
            can_trade=can_trade,
            meta=meta,
        )

    if worst_label == "UNTRUSTED":
        # Fail-closed only when gate_mode=block; otherwise advisory.
        can_trade = (gate_mode != "block")
        return HealthCheckResult(
            name="fidelity_gate",
            status=CheckStatus.FAIL if gate_mode == "block" else CheckStatus.WARN,
            detail=detail,
            error_code="FIDELITY_UNTRUSTED",
            severity="FATAL" if gate_mode == "block" else "DEGRADED",
            can_trade=can_trade,
            meta=meta,
        )

    if worst_label == "WARNING":
        return HealthCheckResult(
            name="fidelity_gate",
            status=CheckStatus.WARN,
            detail=detail,
            error_code="FIDELITY_WARNING",
            severity="DEGRADED",
            can_trade=True,
            meta=meta,
        )

    return HealthCheckResult(
        name="fidelity_gate",
        status=CheckStatus.OK,
        detail=detail,
        error_code=None,
        severity="OK",
        can_trade=True,
        meta=meta,
    )


def _classify_deribit_error(
    e: DeribitAPIError,
) -> tuple[CheckStatus, str, str, str, bool]:
    """Classify Deribit errors into status, detail, error_code, severity, can_trade."""
    error_code = getattr(e, "error_code", DeribitErrorCode.UNKNOWN)
    http_status = getattr(e, "http_status", None)

    if error_code == DeribitErrorCode.AUTH:
        return (
            CheckStatus.FAIL,
            f"Authentication error (401): {e.message}",
            "DERIBIT_AUTH",
            "FATAL",
            False,
        )
    if error_code == DeribitErrorCode.FORBIDDEN:
        return (
            CheckStatus.FAIL,
            f"Access forbidden (403): {e.message}",
            "DERIBIT_AUTH",
            "FATAL",
            False,
        )
    if error_code == DeribitErrorCode.RATE_LIMIT:
        return (
            CheckStatus.WARN,
            f"Rate limited (429): {e.message}",
            "DERIBIT_RATE_LIMIT",
            "DEGRADED",
            True,
        )
    if error_code == DeribitErrorCode.TIMEOUT:
        return (
            CheckStatus.WARN,
            f"Request timeout: {e.message}",
            "DERIBIT_TIMEOUT",
            "DEGRADED",
            True,
        )
    if error_code == DeribitErrorCode.NETWORK:
        return (
            CheckStatus.FAIL,
            f"Network error: {e.message}",
            "DERIBIT_NETWORK",
            "DEGRADED",
            False,
        )
    if error_code == DeribitErrorCode.SERVER_ERROR:
        return (
            CheckStatus.WARN,
            f"Server error ({http_status or '5xx'}): {e.message}",
            "DERIBIT_SERVER_ERROR",
            "DEGRADED",
            True,
        )
    return (
        CheckStatus.FAIL,
        f"API error [{error_code.value}]: {e.message}",
        "DERIBIT_ERROR",
        "DEGRADED",
        False,
    )


def check_deribit_public(client: DeribitClient) -> HealthCheckResult:
    """Check public Deribit API connectivity."""
    try:
        btc_price = client.get_index_price("BTC")
        eth_price = client.get_index_price("ETH")

        return HealthCheckResult(
            name="deribit_public",
            status=CheckStatus.OK,
            detail=f"public API OK, BTC=${btc_price:,.0f}, ETH=${eth_price:,.0f}",
            severity="OK",
            can_trade=True,
        )

    except DeribitAPIError as e:
        status, detail, error_code, severity, can_trade = _classify_deribit_error(e)
        return HealthCheckResult(
            name="deribit_public",
            status=status,
            detail=detail,
            error_code=error_code,
            severity=severity,
            can_trade=can_trade,
        )
    except Exception as e:
        return HealthCheckResult(
            name="deribit_public",
            status=CheckStatus.FAIL,
            detail=f"network error: {str(e)}",
            error_code="DERIBIT_NETWORK",
            severity="DEGRADED",
            can_trade=False,
        )


def check_deribit_private(client: DeribitClient, cfg: Settings) -> HealthCheckResult:
    """Check private Deribit API connectivity (requires credentials)."""
    has_creds = bool(getattr(cfg, "deribit_client_id", "")) and bool(getattr(cfg, "deribit_client_secret", ""))
    if not has_creds:
        if _is_live_mode(cfg):
            return HealthCheckResult(
                name="deribit_private",
                status=CheckStatus.FAIL,
                detail="missing private API credentials in production mode",
                error_code="DERIBIT_PRIVATE_MISSING_CREDS",
                severity="FATAL",
                can_trade=False,
            )
        return HealthCheckResult(
            name="deribit_private",
            status=CheckStatus.SKIPPED,
            detail="no private API credentials",
            severity="OK",
            can_trade=True,
        )

    try:
        settlement_ccy = str(getattr(cfg, "option_settlement_ccy", "") or "USDC").upper()
        summary = client.get_account_summary(settlement_ccy)
        equity = summary.get("equity")
        detail = f"private API OK, equity=${float(equity):,.2f}" if equity is not None else "private API OK"
        return HealthCheckResult(
            name="deribit_private",
            status=CheckStatus.OK,
            detail=detail,
            severity="OK",
            can_trade=True,
            meta={"currency": summary.get("currency")},
        )
    except DeribitAPIError as e:
        status, detail, error_code, severity, can_trade = _classify_deribit_error(e)
        return HealthCheckResult(
            name="deribit_private",
            status=status,
            detail=detail,
            error_code=error_code,
            severity=severity,
            can_trade=can_trade,
        )
    except Exception as e:
        return HealthCheckResult(
            name="deribit_private",
            status=CheckStatus.FAIL,
            detail=f"private API error: {str(e)}",
            error_code="DERIBIT_ERROR",
            severity="DEGRADED",
            can_trade=False,
        )


def check_state_builder(client: DeribitClient, cfg: Settings) -> HealthCheckResult:
    """Check agent state builder pipeline (Deribit -> portfolio -> candidates)."""
    try:
        from src.state_builder import build_agent_state

        state = build_agent_state(client, cfg)
        equity = float(getattr(getattr(state, "portfolio", None), "equity_usd", 0.0) or 0.0)
        positions = len(getattr(getattr(state, "portfolio", None), "option_positions", []) or [])
        candidates = len(getattr(state, "candidate_options", []) or [])
        return HealthCheckResult(
            name="state_builder",
            status=CheckStatus.OK,
            detail=f"state build OK, equity=${equity:,.0f}, positions={positions}, candidates={candidates}",
            severity="OK",
            can_trade=True,
        )
    except DeribitAPIError as e:
        return HealthCheckResult(
            name="state_builder",
            status=CheckStatus.FAIL,
            detail=f"failed to build state: {e.message}",
            error_code="STATE_BUILD_FAILED",
            severity="DEGRADED",
            can_trade=False,
        )
    except Exception as e:
        return HealthCheckResult(
            name="state_builder",
            status=CheckStatus.FAIL,
            detail=f"failed to build state: {str(e)}",
            error_code="STATE_BUILD_FAILED",
            severity="DEGRADED",
            can_trade=False,
        )


def _severity_from_gate_status(status: str | None) -> str:
    if not status:
        return "OK"
    value = status.upper()
    if value == "FAIL":
        return "FATAL"
    if value == "WARN":
        return "DEGRADED"
    return "OK"


def _derive_gate_can_trade(gate_overall: dict[str, Any] | None, gates: list[dict[str, Any]] | None) -> bool | None:
    """Return can_trade derived from gates only, or None if unavailable."""
    if isinstance(gate_overall, dict):
        global_gate = gate_overall.get("global") if isinstance(gate_overall.get("global"), dict) else gate_overall
        if isinstance(global_gate, dict) and "can_trade" in global_gate:
            return bool(global_gate.get("can_trade"))

    if isinstance(gates, list) and gates:
        # Conservative: any gate explicitly forbidding trade blocks trade.
        return not any((g or {}).get("can_trade") is False for g in gates if isinstance(g, dict))

    return None


def _derive_gate_worst_severity(
    gate_overall: dict[str, Any] | None,
    gates: list[dict[str, Any]] | None,
) -> str | None:
    """Return worst severity derived from gates only, or None if unavailable."""
    if isinstance(gate_overall, dict):
        global_gate = gate_overall.get("global") if isinstance(gate_overall.get("global"), dict) else gate_overall
        if isinstance(global_gate, dict):
            sev = global_gate.get("severity")
            if isinstance(sev, str) and sev.strip():
                return sev.strip().upper()
            status = global_gate.get("status")
            if isinstance(status, str) and status.strip():
                # GateRunner uses PASS/WARN/FAIL.
                s = status.strip().upper()
                if s == "FAIL":
                    return "FATAL"
                if s == "WARN":
                    return "DEGRADED"
                if s == "PASS":
                    return "OK"

    if isinstance(gates, list) and gates:
        worst = "OK"
        for g in gates:
            if not isinstance(g, dict):
                continue
            sev = (g.get("severity") or "").upper()
            status = (g.get("status") or "").upper()
            # If severity is missing, infer from status.
            if not sev:
                if status == "FAIL":
                    sev = "FATAL"
                elif status == "WARN":
                    sev = "DEGRADED"
                else:
                    sev = "OK"
            if sev == "FATAL":
                return "FATAL"
            if sev == "DEGRADED" and worst != "FATAL":
                worst = "DEGRADED"
        return worst

    return None


def _result_to_dict(result: HealthCheckResult) -> dict[str, Any]:
    status_value = result.status.value if isinstance(result.status, CheckStatus) else str(result.status)
    payload = {
        "name": result.name,
        "status": status_value,
        "detail": result.detail,
        "error_code": getattr(result, "error_code", None),
        "severity": getattr(result, "severity", "OK"),
        "can_trade": getattr(result, "can_trade", True),
    }
    meta = getattr(result, "meta", None)
    if meta:
        payload["meta"] = meta
    return payload


def _normalize_gate_status(status: str | None, severity: str | None) -> str | None:
    if isinstance(status, str):
        value = status.upper()
        if value == "PASS":
            return "OK"
        if value == "WARN":
            return "WARN"
        if value == "FAIL":
            return "FAIL"
    if isinstance(severity, str):
        value = severity.upper()
        if value == "FATAL":
            return "FAIL"
        if value == "DEGRADED":
            return "WARN"
    return None


def _extract_gate_overall_info(payload: dict[str, Any] | None) -> dict[str, str] | None:
    if not isinstance(payload, dict):
        return None
    global_gate = payload.get("global") if isinstance(payload.get("global"), dict) else payload
    gate_status = (global_gate or {}).get("status")
    gate_severity = (global_gate or {}).get("severity")
    summary_status = _normalize_gate_status(gate_status, gate_severity)
    if summary_status is None:
        return None
    info: dict[str, str] = {"status": summary_status}
    if isinstance(gate_severity, str):
        info["severity"] = gate_severity
    message = (global_gate or {}).get("message")
    if isinstance(message, str) and message.strip():
        info["message"] = message.strip()
    code = (global_gate or {}).get("code") or (global_gate or {}).get("error_code")
    if isinstance(code, str) and code.strip():
        info["code"] = code.strip()
    return info


def _format_gate_summary(info: dict[str, str]) -> str:
    parts = [f"gates {info.get('status', 'UNKNOWN')}"]
    if severity := info.get("severity"):
        parts.append(f"({severity})")
    if code := info.get("code"):
        parts.append(f"code={code}")
    if message := info.get("message"):
        parts.append(message)
    return " ".join(parts)


def _format_check_summary(result: HealthCheckResult) -> str:
    status = result.status.value if isinstance(result.status, CheckStatus) else str(result.status)
    detail = result.detail or ""
    if detail:
        return f"{result.name} {status}: {detail}"
    return f"{result.name} {status}"


def run_agent_healthcheck(cfg: Settings | None = None) -> dict[str, Any]:
    """
    Run all health checks and return aggregated results.
    
    Returns:
        dict with 'overall_status', 'results' list, and 'summary' string
    """
    cfg = cfg or settings
    checked_at = datetime.now(timezone.utc).isoformat()

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
            detail=f"failed to create Deribit client: {str(e)}",
            error_code="DERIBIT_CLIENT_INIT",
            severity="FATAL",
            can_trade=False,
        ))

    from src.ops.facts_resolver import resolve_ops_facts

    ops_facts = resolve_ops_facts(cfg)
    resolved_paths = ops_facts.get("paths") or {}

    def _call_check_with_optional_base_dir(check_fn: Any, *, base_dir: str | None) -> HealthCheckResult:
        try:
            return check_fn(cfg, base_dir=base_dir)
        except TypeError as e:
            msg = str(e)
            if "base_dir" in msg and "unexpected keyword argument" in msg:
                return check_fn(cfg)
            raise

    results.append(
        _call_check_with_optional_base_dir(
            check_harvest_freshness,
            base_dir=resolved_paths.get("live_deribit_data_dir"),
        )
    )
    results.append(
        _call_check_with_optional_base_dir(
            check_calibration_freshness,
            base_dir=resolved_paths.get("calibration_dir"),
        )
    )
    results.append(
        _call_check_with_optional_base_dir(
            check_fidelity_gate,
            base_dir=resolved_paths.get("fidelity_dir"),
        )
    )

    # Unified data-readiness gates (Truth -> Trust -> Trade)
    gates: list[dict[str, Any]] = []
    gate_overall: dict[str, Any] | None = None
    can_trade_by_underlying: dict[str, Any] | None = None
    gates_check_added = False

    # Gate modes determine whether gate evaluation errors must fail closed.
    fidelity_mode_env = _resolve_fidelity_gate_mode(cfg)
    calibration_mode_env = (os.getenv("CALIBRATION_GATE_MODE") or "off").strip().lower()
    gates_required = fidelity_mode_env != "off" or calibration_mode_env != "off"
    try:
        from src.ops.gates import GateMode, GateRunner
        from src.ops.gate_factories import build_underlying_gate_fns

        harvest_mode = GateMode.WARN
        fidelity_mode = GateMode(fidelity_mode_env)
        calibration_mode = GateMode(calibration_mode_env)
        harvest_required = fidelity_mode != GateMode.OFF or calibration_mode != GateMode.OFF

        require_usdc = bool(getattr(cfg, "option_margin_type", "linear") == "linear") or (
            str(getattr(cfg, "option_settlement_ccy", "USDC") or "").upper() == "USDC"
        )

        gate_fns = []
        for underlying in (ops_facts.get("underlyings_active") or []):
            u = str(underlying).upper().strip()
            harvest_facts = (ops_facts.get("harvest") or {}).get(u) or {}
            fidelity_facts = (ops_facts.get("fidelity") or {}).get(u) or {}
            calibration_facts = (ops_facts.get("calibration") or {}).get(u) or {}

            required_dir = (harvest_facts.get("expected_dir") or f"{u}_USDC") if require_usdc else None

            gate_fns.extend(
                build_underlying_gate_fns(
                    underlying=u,
                    harvest_mode=harvest_mode,
                    harvest_required=harvest_required,
                    harvest_facts=harvest_facts,
                    require_harvest_dir=required_dir,
                    fidelity_mode=fidelity_mode,
                    fidelity_facts=fidelity_facts,
                    calibration_mode=calibration_mode,
                    calibration_facts=calibration_facts,
                )
            )

        out = GateRunner().run(gate_fns)
        gates = out.get("gates") or []
        gate_overall = out.get("gate_overall") or None
        by_u = (gate_overall or {}).get("by_underlying") if isinstance(gate_overall, dict) else None
        if isinstance(by_u, dict):
            # Preserve the by_underlying payload shape (dicts with status/severity/can_trade)
            # so consumers can use more than just the boolean.
            can_trade_by_underlying = dict(by_u)
    except Exception as e:
        detail = f"gates eval error: {type(e).__name__}: {e}"
        results.append(
            HealthCheckResult(
                name="gates_runner",
                status=CheckStatus.FAIL if gates_required else CheckStatus.WARN,
                detail=detail,
                error_code="GATES_EVAL_ERROR",
                severity="FATAL" if gates_required else "DEGRADED",
                can_trade=False if gates_required else True,
            )
        )
        gates_check_added = True
        gates = []
        gate_overall = (
            {
                "status": "FAIL",
                "severity": "FATAL",
                "can_trade": False,
                "message": "gates_eval_error",
                "code": "GATES_EVAL_ERROR",
            }
            if gates_required
            else None
        )
        can_trade_by_underlying = None

    fail_check = next((r for r in results if r.status == CheckStatus.FAIL), None)
    warn_check = next((r for r in results if r.status == CheckStatus.WARN), None)
    if fail_check:
        checks_overall_status = "FAIL"
    elif warn_check:
        checks_overall_status = "WARN"
    else:
        checks_overall_status = "OK"

    gate_info = _extract_gate_overall_info(gate_overall)

    def _worst_status(a: str, b: str) -> str:
        order = {"OK": 0, "WARN": 1, "FAIL": 2}
        return a if order.get(a, 0) >= order.get(b, 0) else b

    gates_present = bool(gates) or gate_overall is not None
    gate_status = (gate_info or {}).get("status") if isinstance(gate_info, dict) else None
    if not gate_status:
        gate_status = "WARN" if gates_present else "OK"

    # Core trading blockers (can_trade=False) must always dominate.
    blocking_check = next(
        (r for r in results if r.status in (CheckStatus.FAIL, CheckStatus.WARN) and r.can_trade is False),
        None,
    )

    if blocking_check is not None:
        overall_status = "FAIL"
        summary = _format_check_summary(blocking_check)
    elif gates_present:
        # Single truth: when gates are present, overall status/summary is gate-authoritative.
        overall_status = gate_status
        summary = _format_gate_summary(gate_info) if gate_info else f"gates {gate_status}"
        
        # If gate status is FAIL, it must be FAIL even if no individual blocking_check was found above
        if gate_status == "FAIL":
            overall_status = "FAIL"
    else:
        overall_status = checks_overall_status
        if overall_status == "FAIL" and fail_check is not None:
            summary = _format_check_summary(fail_check)
        elif overall_status == "WARN" and warn_check is not None:
            summary = _format_check_summary(warn_check)
        else:
            summary = "All checks passed"

    if can_trade_by_underlying is None and not gates_check_added:
        results.append(
            HealthCheckResult(
                name="gates_runner",
                status=CheckStatus.WARN,
                detail="gates unavailable",
                error_code="GATES_UNAVAILABLE",
                severity="DEGRADED",
                can_trade=True,
            )
        )
        gates_check_added = True

    checks = [_result_to_dict(r) for r in results]
    global_can_trade_from_checks = all(check.get("can_trade") is not False for check in checks)

    can_trade_by_underlying_final: dict[str, bool] | None = None
    if isinstance(can_trade_by_underlying, dict):
        can_trade_by_underlying_final = {}
        for k, v in can_trade_by_underlying.items():
            if isinstance(v, dict):
                can_trade_by_underlying_final[str(k).upper()] = bool(v.get("can_trade"))
            else:
                can_trade_by_underlying_final[str(k).upper()] = bool(v)
    elif gates_check_added:
        derived_gate_can_trade = _derive_gate_can_trade(gate_overall if isinstance(gate_overall, dict) else None, gates)
        default_can_trade = bool(derived_gate_can_trade) if derived_gate_can_trade is not None else False
        can_trade_by_underlying_final = {
            str(u).upper(): default_can_trade for u in (cfg.underlyings or [])
        }

    # Combine global "can_trade" from checks with per-underlying gate decisions.
    if isinstance(can_trade_by_underlying_final, dict) and not global_can_trade_from_checks:
        for key in list(can_trade_by_underlying_final.keys()):
            can_trade_by_underlying_final[key] = False

    if isinstance(can_trade_by_underlying_final, dict):
        can_trade = bool(global_can_trade_from_checks) and any(can_trade_by_underlying_final.values())
    else:
        can_trade = bool(global_can_trade_from_checks)

    def _worst_severity_value(a: str | None, b: str | None) -> str:
        order = {"OK": 0, "DEGRADED": 1, "FATAL": 2}
        aa = (a or "OK").upper()
        bb = (b or "OK").upper()
        return aa if order.get(aa, 0) >= order.get(bb, 0) else bb

    checks_worst_severity, _checks_can_trade_unused = _compute_worst_severity({"checks": checks})
    gates_worst_severity = (
        _derive_gate_worst_severity(gate_overall if isinstance(gate_overall, dict) else None, gates)
        if gates_present
        else None
    )
    worst_severity = _worst_severity_value(checks_worst_severity, gates_worst_severity)

    checks_summary = ", ".join(f"{r.name}:{r.status.value}" for r in results)

    return {
        "overall_status": overall_status,
        "summary": summary,
        "checked_at": checked_at,
        "checks_summary": checks_summary,
        "checks_overall": checks_overall_status,
        "worst_severity": worst_severity,
        "can_trade": can_trade,
        "checks": checks,
        "results": checks,
        "gates": gates,
        "gate_overall": gate_overall,
        "can_trade_by_underlying": can_trade_by_underlying_final,
        "ops_facts": ops_facts,
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
