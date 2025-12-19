from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Dict, Any


StrategyName = Literal[
    "covered_call",
    "cash_secured_put",
    "short_strangle",
    "put_spread_credit",
    "call_spread_debit",
    "calendar",
]


@dataclass(frozen=True)
class StrategySpec:
    """Frozen strategy spec used as a measurement instrument.

    P0: We store the spec but do not implement full execution logic for all six yet.
    """

    name: StrategyName
    description: str

    entry_schedule: str
    selection_rule: Dict[str, Any]
    exit_rule: Dict[str, Any]
    sizing_rule: Dict[str, Any]


def canonical_strategy_specs() -> list[StrategySpec]:
    return [
        StrategySpec(
            name="covered_call",
            description="Short call against spot.",
            entry_schedule="daily_00_utc",
            selection_rule={"kind": "call", "target_abs_delta": 0.25, "target_dte_days": 7},
            exit_rule={"type": "hold_to_expiry"},
            sizing_rule={"type": "fixed_underlying", "underlying_units": 1.0},
        ),
        StrategySpec(
            name="cash_secured_put",
            description="Short put with cash collateral.",
            entry_schedule="daily_00_utc",
            selection_rule={"kind": "put", "target_abs_delta": 0.25, "target_dte_days": 7},
            exit_rule={"type": "hold_to_expiry"},
            sizing_rule={"type": "fixed_notional_usd", "notional_usd": 10_000},
        ),
        StrategySpec(
            name="short_strangle",
            description="Short put + short call.",
            entry_schedule="daily_00_utc",
            selection_rule={"kind": "strangle", "put_abs_delta": 0.25, "call_abs_delta": 0.25, "target_dte_days": 7},
            exit_rule={"type": "hold_to_expiry"},
            sizing_rule={"type": "fixed_notional_usd", "notional_usd": 10_000},
        ),
        StrategySpec(
            name="put_spread_credit",
            description="Short put + long lower put (credit spread).",
            entry_schedule="daily_00_utc",
            selection_rule={"kind": "put_spread_credit", "short_abs_delta": 0.25, "width_pct": 0.05, "target_dte_days": 14},
            exit_rule={"type": "hold_to_expiry"},
            sizing_rule={"type": "fixed_notional_usd", "notional_usd": 10_000},
        ),
        StrategySpec(
            name="call_spread_debit",
            description="Long call + short higher call (debit spread).",
            entry_schedule="daily_00_utc",
            selection_rule={"kind": "call_spread_debit", "long_abs_delta": 0.50, "width_pct": 0.05, "target_dte_days": 14},
            exit_rule={"type": "hold_to_expiry"},
            sizing_rule={"type": "fixed_notional_usd", "notional_usd": 10_000},
        ),
        StrategySpec(
            name="calendar",
            description="Long farther-dated, short nearer-dated (same strike/delta bucket).",
            entry_schedule="daily_00_utc",
            selection_rule={"kind": "calendar", "abs_delta": 0.25, "short_dte_days": 7, "long_dte_days": 30},
            exit_rule={"type": "roll_short_leg"},
            sizing_rule={"type": "fixed_notional_usd", "notional_usd": 10_000},
        ),
    ]
