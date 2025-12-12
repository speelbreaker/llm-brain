"""
Greg sandbox position entry helper.

Creates sandbox positions for testing the Greg strategy pipeline,
position management, and hedge engine without using real market data.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

from src.position_tracker import PositionChain, PositionLeg, PositionTracker


GREG_STRATEGIES = [
    "STRATEGY_A_STRADDLE",
    "STRATEGY_A_STRANGLE",
    "STRATEGY_B_CALENDAR",
    "STRATEGY_C_SHORT_PUT",
    "STRATEGY_D_IRON_BUTTERFLY",
    "STRATEGY_F_BULL_PUT_SPREAD",
    "STRATEGY_F_BEAR_CALL_SPREAD",
]


STRATEGY_LEG_TEMPLATES: Dict[str, List[Dict[str, Any]]] = {
    "STRATEGY_A_STRADDLE": [
        {"option_type": "CALL", "side": "SHORT", "strike_offset": 0, "dte_offset": 7},
        {"option_type": "PUT", "side": "SHORT", "strike_offset": 0, "dte_offset": 7},
    ],
    "STRATEGY_A_STRANGLE": [
        {"option_type": "CALL", "side": "SHORT", "strike_offset": 0.05, "dte_offset": 7},
        {"option_type": "PUT", "side": "SHORT", "strike_offset": -0.05, "dte_offset": 7},
    ],
    "STRATEGY_B_CALENDAR": [
        {"option_type": "CALL", "side": "SHORT", "strike_offset": 0, "dte_offset": 7},
        {"option_type": "CALL", "side": "LONG", "strike_offset": 0, "dte_offset": 30},
    ],
    "STRATEGY_C_SHORT_PUT": [
        {"option_type": "PUT", "side": "SHORT", "strike_offset": -0.05, "dte_offset": 7},
    ],
    "STRATEGY_D_IRON_BUTTERFLY": [
        {"option_type": "PUT", "side": "LONG", "strike_offset": -0.05, "dte_offset": 7},
        {"option_type": "PUT", "side": "SHORT", "strike_offset": 0, "dte_offset": 7},
        {"option_type": "CALL", "side": "SHORT", "strike_offset": 0, "dte_offset": 7},
        {"option_type": "CALL", "side": "LONG", "strike_offset": 0.05, "dte_offset": 7},
    ],
    "STRATEGY_F_BULL_PUT_SPREAD": [
        {"option_type": "PUT", "side": "SHORT", "strike_offset": -0.03, "dte_offset": 7},
        {"option_type": "PUT", "side": "LONG", "strike_offset": -0.08, "dte_offset": 7},
    ],
    "STRATEGY_F_BEAR_CALL_SPREAD": [
        {"option_type": "CALL", "side": "SHORT", "strike_offset": 0.03, "dte_offset": 7},
        {"option_type": "CALL", "side": "LONG", "strike_offset": 0.08, "dte_offset": 7},
    ],
}


SYNTHETIC_SPOT_PRICES = {
    "BTC": 100000.0,
    "ETH": 4000.0,
}


def _generate_synthetic_symbol(
    underlying: str,
    option_type: str,
    strike: float,
    expiry: datetime,
) -> str:
    """Generate a synthetic Deribit-style option symbol."""
    expiry_str = expiry.strftime("%d%b%y").upper()
    strike_str = f"{int(strike)}"
    type_char = option_type[0]
    return f"{underlying}-{expiry_str}-{strike_str}-{type_char}"


def _round_strike(strike: float, underlying: str) -> float:
    """Round strike to nearest valid Deribit strike increment."""
    if underlying == "BTC":
        increment = 1000 if strike < 50000 else 2500
    else:
        increment = 50 if strike < 2000 else 100
    return round(strike / increment) * increment


def build_synthetic_legs(
    underlying: str,
    strategy_type: str,
    size: float,
) -> Tuple[List[PositionLeg], datetime]:
    """
    Build synthetic option legs for a sandbox position.
    
    Returns:
        (legs, expiry) tuple
    """
    now = datetime.now(timezone.utc)
    spot = SYNTHETIC_SPOT_PRICES.get(underlying, 50000.0)
    
    templates = STRATEGY_LEG_TEMPLATES.get(strategy_type, [])
    if not templates:
        templates = [{"option_type": "CALL", "side": "SHORT", "strike_offset": 0, "dte_offset": 7}]
    
    legs: List[PositionLeg] = []
    min_expiry = now + timedelta(days=365)
    
    for template in templates:
        strike_offset = template["strike_offset"]
        dte_offset = template["dte_offset"]
        option_type = template["option_type"]
        side = template["side"]
        
        strike = _round_strike(spot * (1 + strike_offset), underlying)
        expiry = now + timedelta(days=dte_offset)
        if expiry < min_expiry:
            min_expiry = expiry
        
        symbol = _generate_synthetic_symbol(underlying, option_type, strike, expiry)
        synthetic_price = spot * 0.02
        
        leg = PositionLeg(
            symbol=symbol,
            underlying=underlying,
            option_type=option_type,
            side=side,
            quantity=size,
            entry_price=synthetic_price,
            entry_time=now,
            mark_price=synthetic_price,
        )
        legs.append(leg)
    
    return legs, min_expiry


@dataclass
class SandboxPositionResult:
    """Result of creating a sandbox position."""
    success: bool
    position_id: Optional[str] = None
    strategy_type: Optional[str] = None
    underlying: Optional[str] = None
    num_legs: int = 0
    error: Optional[str] = None


def open_greg_strategy_position(
    tracker: PositionTracker,
    underlying: str,
    strategy_type: str,
    size: float,
    sandbox: bool = True,
    origin: str = "GREG_SANDBOX",
    run_id: Optional[str] = None,
) -> SandboxPositionResult:
    """
    Open a sandbox Greg strategy position for testing.
    
    Creates a synthetic position in the position tracker that can be used
    to test the Greg dashboard, position management, and hedge engine.
    
    Args:
        tracker: Position tracker instance
        underlying: 'BTC' or 'ETH'
        strategy_type: One of the Greg strategies (e.g., STRATEGY_A_STRADDLE)
        size: Position size (e.g., 0.01 for BTC, 0.1 for ETH)
        sandbox: Whether this is a sandbox position (default True)
        origin: Origin identifier (default 'GREG_SANDBOX')
        run_id: Run identifier for batch tracking
        
    Returns:
        SandboxPositionResult with success status and position details
    """
    try:
        if strategy_type not in GREG_STRATEGIES:
            return SandboxPositionResult(
                success=False,
                error=f"Unknown strategy type: {strategy_type}",
            )
        
        if underlying not in ["BTC", "ETH"]:
            return SandboxPositionResult(
                success=False,
                error=f"Unknown underlying: {underlying}",
            )
        
        legs, expiry = build_synthetic_legs(underlying, strategy_type, size)
        
        position_id = f"sandbox_{underlying}_{strategy_type}_{uuid.uuid4().hex[:8]}"
        
        first_leg = legs[0] if legs else None
        option_type = first_leg.option_type if first_leg else "CALL"
        
        chain = PositionChain(
            position_id=position_id,
            underlying=underlying,
            option_type=option_type,
            strategy_type=strategy_type,  # type: ignore[arg-type]
            mode="DRY_RUN",
            exit_style="hold_to_expiry",
            legs=legs,
            expiry=expiry,
            sandbox=sandbox,
            origin=origin,
            run_id=run_id,
        )
        
        with tracker._lock:
            tracker._chains[position_id] = chain
            tracker._save_to_disk()
        
        return SandboxPositionResult(
            success=True,
            position_id=position_id,
            strategy_type=strategy_type,
            underlying=underlying,
            num_legs=len(legs),
        )
        
    except Exception as e:
        return SandboxPositionResult(
            success=False,
            strategy_type=strategy_type,
            underlying=underlying,
            error=str(e),
        )


def get_sandbox_positions(tracker: PositionTracker) -> List[PositionChain]:
    """Get all sandbox positions from the tracker."""
    with tracker._lock:
        return [
            chain for chain in tracker._chains.values()
            if chain.is_sandbox()
        ]


def clear_sandbox_positions(tracker: PositionTracker) -> int:
    """Clear all sandbox positions from the tracker. Returns count cleared."""
    with tracker._lock:
        sandbox_ids = [
            position_id for position_id, chain in tracker._chains.items()
            if chain.is_sandbox()
        ]
        for position_id in sandbox_ids:
            del tracker._chains[position_id]
        tracker._save_to_disk()
        return len(sandbox_ids)
