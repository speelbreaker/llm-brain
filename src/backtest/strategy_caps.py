"""
Strategy capabilities metadata for backtest configuration.

Defines what configuration fields each selector/strategy supports,
enabling the UI to show only applicable controls and the engine
to apply correct overrides.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class FieldHint:
    """UX hint for a configuration field."""

    field_name: str
    disabled: bool = False
    readonly: bool = False
    hidden: bool = False
    tooltip: str = ""
    display_value: Optional[str] = None


@dataclass
class StrategyCapabilities:
    """Defines what configuration options a strategy supports."""

    selector_name: str
    display_name: str
    description: str

    supports_exit_style: bool = True
    supports_dte_range: bool = True
    supports_delta_range: bool = True
    supports_targets: bool = True

    config_owner_fields: List[str] = field(default_factory=list)
    user_fields: List[str] = field(default_factory=list)

    strategy_defaults_summary: Dict[str, Any] = field(default_factory=dict)

    field_hints: List[FieldHint] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["field_hints"] = [asdict(h) for h in self.field_hints]
        return d


GENERIC_COVERED_CALL_CAPS = StrategyCapabilities(
    selector_name="generic_covered_call",
    display_name="Generic Covered Call",
    description="Standard covered call strategy with user-configurable DTE, delta, and exit style.",
    supports_exit_style=True,
    supports_dte_range=True,
    supports_delta_range=True,
    supports_targets=True,
    config_owner_fields=[],
    user_fields=[
        "exit_style",
        "target_dte",
        "target_delta",
        "min_dte",
        "max_dte",
        "delta_min",
        "delta_max",
    ],
    strategy_defaults_summary={},
)


GREGBOT_CAPS = StrategyCapabilities(
    selector_name="gregbot",
    display_name="GregBot VRP Harvester",
    description="Greg Mandolini's VRP harvesting strategy with internal entry/exit rules. DTE, delta, and exit logic are managed by GregBot.",
    supports_exit_style=False,
    supports_dte_range=False,
    supports_delta_range=False,
    supports_targets=False,
    config_owner_fields=[
        "exit_style",
        "target_dte",
        "target_delta",
        "min_dte",
        "max_dte",
        "delta_min",
        "delta_max",
    ],
    user_fields=[
        "underlying",
        "start",
        "end",
        "timeframe",
        "decision_interval_hours",
        "synthetic_iv_multiplier",
    ],
    strategy_defaults_summary={
        "exit_logic": "GregBot-managed (take-profit, roll, expiry rules)",
        "dte_targeting": "Dynamic per strategy (7-21 days typical)",
        "delta_targeting": "Strategy-dependent (0.15-0.40 typical)",
        "entry_rules": "VRP waterfall with sensor-based signals",
        "position_management": "Rolling and profit-taking built-in",
    },
    field_hints=[
        FieldHint(
            field_name="exit_style",
            disabled=True,
            hidden=True,
            tooltip="GregBot always exits on profit targets; expiry holds and rolls are disabled.",
            display_value="GregBot-Managed",
        ),
        FieldHint(
            field_name="target_dte",
            readonly=True,
            tooltip="Values controlled by GregBot strategy.",
            display_value="7-21 days (dynamic)",
        ),
        FieldHint(
            field_name="target_delta",
            readonly=True,
            tooltip="Values controlled by GregBot strategy.",
            display_value="0.15-0.40 (dynamic)",
        ),
        FieldHint(
            field_name="min_dte",
            readonly=True,
            tooltip="Values controlled by GregBot strategy.",
            display_value="3",
        ),
        FieldHint(
            field_name="max_dte",
            readonly=True,
            tooltip="Values controlled by GregBot strategy.",
            display_value="30",
        ),
        FieldHint(
            field_name="delta_min",
            readonly=True,
            tooltip="Values controlled by GregBot strategy.",
            display_value="0.10",
        ),
        FieldHint(
            field_name="delta_max",
            readonly=True,
            tooltip="Values controlled by GregBot strategy.",
            display_value="0.45",
        ),
    ],
)


STRATEGY_CAPS_REGISTRY: Dict[str, StrategyCapabilities] = {
    "generic_covered_call": GENERIC_COVERED_CALL_CAPS,
    "gregbot": GREGBOT_CAPS,
}


def get_strategy_caps(selector_name: str) -> Optional[StrategyCapabilities]:
    """Get capabilities for a selector/strategy by name."""
    return STRATEGY_CAPS_REGISTRY.get(selector_name)


def list_available_strategies() -> List[Dict[str, Any]]:
    """List all available strategies with their capabilities."""
    return [caps.to_dict() for caps in STRATEGY_CAPS_REGISTRY.values()]


@dataclass
class ConfigValidationResult:
    """Result of validating and applying strategy overrides to config."""

    effective_config: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)
    overrides_applied: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "effective_config": self.effective_config,
            "warnings": self.warnings,
            "overrides_applied": self.overrides_applied,
        }


GREGBOT_DEFAULTS = {
    "exit_style": "gregbot_managed",
    "target_dte": 14,
    "target_delta": 0.25,
    "min_dte": 3,
    "max_dte": 30,
    "delta_min": 0.10,
    "delta_max": 0.45,
}


def apply_strategy_overrides(
    selector_name: str,
    user_config: Dict[str, Any],
) -> ConfigValidationResult:
    """
    Apply strategy-specific overrides to user config.

    For strategies that own certain fields (like GregBot), this will:
    1. Override those fields with strategy defaults
    2. Generate warnings if user provided conflicting values
    3. Return the effective config that will actually be used

    Args:
        selector_name: The strategy/selector name
        user_config: User-provided configuration dict

    Returns:
        ConfigValidationResult with effective_config, warnings, and overrides_applied
    """
    caps = get_strategy_caps(selector_name)
    effective_config = user_config.copy()
    warnings: List[str] = []
    overrides_applied: Dict[str, Any] = {}

    if caps is None:
        warnings.append(f"Unknown selector '{selector_name}', using as-is")
        return ConfigValidationResult(
            effective_config=effective_config,
            warnings=warnings,
            overrides_applied=overrides_applied,
        )

    if selector_name == "gregbot":
        for field_name, default_value in GREGBOT_DEFAULTS.items():
            user_value = user_config.get(field_name)

            if user_value is not None and user_value != default_value:
                if field_name == "exit_style":
                    warnings.append(
                        f"'{field_name}' is managed by GregBot. "
                        f"Ignoring user value '{user_value}'; using GregBot exit logic."
                    )
                elif field_name in ("target_dte", "target_delta"):
                    warnings.append(
                        f"'{field_name}' is dynamically determined by GregBot strategy signals. "
                        f"Ignoring user value '{user_value}'."
                    )
                elif field_name in ("min_dte", "max_dte", "delta_min", "delta_max"):
                    warnings.append(
                        f"'{field_name}' range is controlled by GregBot. "
                        f"Ignoring user value '{user_value}'."
                    )

                overrides_applied[field_name] = {
                    "user_value": user_value,
                    "effective_value": default_value,
                }

            effective_config[field_name] = default_value

    return ConfigValidationResult(
        effective_config=effective_config,
        warnings=warnings,
        overrides_applied=overrides_applied,
    )
