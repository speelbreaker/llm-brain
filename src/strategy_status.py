"""
Strategy & Safeguards status models and builder.
Provides a read-only view of current trading configuration.
"""
from __future__ import annotations

from typing import List, Optional, Literal, Dict, Any, Tuple

from pydantic import BaseModel, Field


class SafeguardStatus(BaseModel):
    """Status of a single safeguard."""
    name: str
    status: Literal["ON", "OFF"]
    details: str


class ModeRules(BaseModel):
    """Rules that apply in a specific mode (training or live)."""
    description: str
    delta_range: Optional[Tuple[float, float]] = None
    dte_range: Optional[Tuple[int, int]] = None
    ivrv_min: Optional[float] = None
    notes: List[str] = Field(default_factory=list)


class StrategyStatus(BaseModel):
    """Complete strategy and safeguards status for UI display."""
    mode: Literal["training", "research", "live"]
    network: Literal["testnet", "mainnet"]
    dry_run: bool
    llm_enabled: bool
    policy_version: str
    explore_prob: float
    explore_top_k: int

    effective_delta_range: Tuple[float, float]
    effective_dte_range: Tuple[int, int]

    max_margin_used_pct: float
    max_net_delta_abs: float
    per_expiry_exposure_limit: Optional[float] = None
    daily_drawdown_limit_pct: float = 0.0
    kill_switch_enabled: bool = False

    training_mode: bool
    training_on_testnet: bool
    training_strategies_enabled: List[str]
    training_max_calls_per_underlying: Optional[int] = None
    training_max_calls_per_expiry: Optional[int] = None
    ladders_enabled: bool

    training_risk_label: Optional[str] = None

    training_rules: ModeRules
    live_rules: ModeRules

    safeguards: List[SafeguardStatus]


def infer_training_risk_label(strategies: List[str]) -> str:
    """Infer a human-friendly risk label from enabled strategies."""
    if "aggressive" in strategies and "conservative" in strategies:
        return "Mixed (Conservative + Aggressive)"
    if "aggressive" in strategies:
        return "Aggressive"
    if "moderate" in strategies:
        return "Moderate"
    if "conservative" in strategies:
        return "Conservative"
    return "Unknown"


def build_strategy_status(config_snapshot: Optional[Dict[str, Any]] = None) -> StrategyStatus:
    """
    Build a StrategyStatus instance from the current config and optional snapshot.
    Uses src.config.settings as the source of truth.
    """
    from src.config import settings

    snapshot = config_snapshot or {}

    delta_range = (
        snapshot.get("effective_delta_range")
        or (settings.effective_delta_min, settings.effective_delta_max)
    )
    if isinstance(delta_range, list):
        delta_range = tuple(delta_range)

    dte_range = (
        snapshot.get("effective_dte_range")
        or (settings.effective_dte_min, settings.effective_dte_max)
    )
    if isinstance(dte_range, list):
        dte_range = tuple(dte_range)

    per_expiry_limit = settings.effective_max_expiry_exposure

    training_strategies = list(settings.training_strategies)
    training_risk_label = infer_training_risk_label(training_strategies)

    training_rules = ModeRules(
        description="Training mode on Deribit testnet: explore behavior and gather data.",
        delta_range=delta_range,
        dte_range=dte_range,
        ivrv_min=settings.effective_ivrv_min,
        notes=[
            "Runs on Deribit testnet only. No real capital is used.",
            "Risk checks for margin, per-expiry exposure, and position size are relaxed so the bot can open many trades and learn.",
            f"Can open multiple covered calls per underlying (up to {settings.max_calls_per_underlying_training}).",
            f"Per-expiry training cap: {settings.training_max_calls_per_expiry} calls.",
            "Ladder across conservative / moderate / aggressive profiles when enabled.",
            "Uses the same delta and DTE windows as live mode, but is allowed to operate near the edges of those ranges when exploring.",
            "IV/RV filters are applied but can be slightly loosened in training to see behavior in borderline conditions.",
            "Orders are sent with dry_run = False on TESTNET so trades show up as real positions in the Deribit testnet UI.",
        ],
    )

    live_rules = ModeRules(
        description="Normal / live mode: prioritize capital preservation, enforce full risk checks.",
        delta_range=(settings.delta_min, settings.delta_max),
        dte_range=(settings.dte_min, settings.dte_max),
        ivrv_min=settings.ivrv_min,
        notes=[
            "Trades only on the configured network (typically mainnet) and can use real capital when dry_run is disabled.",
            f"Enforces full risk engine: margin_used_pct must stay below {settings.max_margin_used_pct}%.",
            f"Net delta must remain within limits (max |delta|: {settings.max_net_delta_abs}).",
            f"Per-expiry exposure limit is enforced: {settings.max_expiry_exposure} notional.",
            "Restricts the number of open short calls per underlying and expiry to keep the position set simple and easy to manage.",
            f"Requires delta in [{settings.delta_min}, {settings.delta_max}] and DTE in [{settings.dte_min}, {settings.dte_max}].",
            f"Requires IV/RV >= {settings.ivrv_min} before selling premium.",
            "Chooses a single best action per decision tick (no aggressive laddering across multiple profiles in one go).",
            "If any critical risk check fails, the decision is downgraded to DO_NOTHING instead of forcing a trade.",
            "Dry run mode can be enabled for shadow trading on live data; when dry_run = True, all orders are simulated only.",
        ],
    )

    kill_switch = snapshot.get("kill_switch_enabled", settings.kill_switch_enabled)
    daily_dd_limit = snapshot.get("daily_drawdown_limit_pct", settings.daily_drawdown_limit_pct)

    safeguards = [
        SafeguardStatus(
            name="Global Kill Switch",
            status="ON" if kill_switch else "OFF",
            details=(
                "ACTIVE: All trading actions are blocked!"
                if kill_switch
                else "Inactive. Trading allowed subject to other checks."
            ),
        ),
        SafeguardStatus(
            name="Daily Drawdown Guard",
            status="ON" if daily_dd_limit > 0 else "OFF",
            details=(
                f"Active: blocks new risk after {daily_dd_limit:.1f}% daily peak-to-trough loss."
                if daily_dd_limit > 0
                else "Disabled (limit = 0)."
            ),
        ),
        SafeguardStatus(
            name="Per-expiry exposure limit",
            status="OFF" if settings.is_training_on_testnet else "ON",
            details=(
                "Skipped in training on testnet to allow many legs."
                if settings.is_training_on_testnet
                else f"Active: cap per expiry ~{per_expiry_limit} notional."
            ),
        ),
        SafeguardStatus(
            name="Margin usage cap",
            status="ON",
            details=f"Max margin_used_pct: {snapshot.get('max_margin_used_pct', settings.max_margin_used_pct)}%.",
        ),
        SafeguardStatus(
            name="Net delta cap",
            status="ON",
            details=f"Max |net_delta|: {snapshot.get('max_net_delta_abs', settings.max_net_delta_abs)}.",
        ),
        SafeguardStatus(
            name="LLM decision gating",
            status="ON" if snapshot.get("llm_enabled", settings.llm_enabled) else "OFF",
            details=f"LLM override enabled: {snapshot.get('llm_enabled', settings.llm_enabled)}.",
        ),
        SafeguardStatus(
            name="Dry run",
            status="ON" if snapshot.get("dry_run", settings.dry_run) else "OFF",
            details="If ON, orders are simulated. If OFF, orders hit the exchange.",
        ),
    ]

    mode: Literal["training", "research", "live"]
    if settings.is_training_enabled:
        mode = "training"
    elif settings.is_research:
        mode = "research"
    else:
        mode = "live"

    return StrategyStatus(
        mode=mode,
        network="testnet" if settings.is_testnet else "mainnet",
        dry_run=snapshot.get("dry_run", settings.dry_run),
        llm_enabled=snapshot.get("llm_enabled", settings.llm_enabled),
        policy_version=snapshot.get("policy_version", settings.policy_version),
        explore_prob=snapshot.get("explore_prob", settings.explore_prob),
        explore_top_k=snapshot.get("explore_top_k", settings.explore_top_k),
        effective_delta_range=delta_range,
        effective_dte_range=dte_range,
        max_margin_used_pct=snapshot.get("max_margin_used_pct", settings.max_margin_used_pct),
        max_net_delta_abs=snapshot.get("max_net_delta_abs", settings.max_net_delta_abs),
        per_expiry_exposure_limit=per_expiry_limit,
        daily_drawdown_limit_pct=daily_dd_limit,
        kill_switch_enabled=kill_switch,
        training_mode=settings.training_mode,
        training_on_testnet=settings.is_training_on_testnet,
        training_strategies_enabled=training_strategies,
        training_max_calls_per_underlying=settings.max_calls_per_underlying_training,
        training_max_calls_per_expiry=settings.training_max_calls_per_expiry,
        ladders_enabled=settings.training_profile_mode == "ladder",
        training_risk_label=training_risk_label,
        training_rules=training_rules,
        live_rules=live_rules,
        safeguards=safeguards,
    )
