"""
Delta-hedging engine for Greg's neutral short-vol strategies.

This module computes net delta across options and perps, and proposes
hedge orders to maintain delta-neutrality per the strategy rules.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Literal, Optional

logger = logging.getLogger(__name__)

GREG_POSITION_RULES_PATH = Path("docs/greg_mandolini/GREG_POSITION_RULES_V1.json")

HedgeMode = Literal["DYNAMIC_DELTA", "LIGHT_DELTA", "LOOSE_DELTA", "NONE"]


@dataclass
class HedgeRules:
    """Hedge configuration for a strategy type."""
    mode: HedgeMode
    delta_abs_threshold: float
    target_delta: float = 0.0
    check_frequency: str = "60s"
    action: str = ""
    description: str = ""

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "HedgeRules":
        if data is None:
            return cls(mode="NONE", delta_abs_threshold=999.0)
        
        mode_raw = data.get("mode", "NONE")
        if mode_raw == "delta_hedge_perp":
            mode = "DYNAMIC_DELTA"
        elif mode_raw in ("DYNAMIC_DELTA", "LIGHT_DELTA", "LOOSE_DELTA", "NONE"):
            mode = mode_raw
        else:
            mode = "NONE"
        
        return cls(
            mode=mode,
            delta_abs_threshold=data.get("delta_abs_threshold", 0.15),
            target_delta=data.get("target_delta", 0.0),
            check_frequency=data.get("check_frequency", "60s"),
            action=data.get("action", ""),
            description=data.get("description", ""),
        )


@dataclass
class HedgeOrder:
    """A proposed hedge order."""
    underlying: str
    instrument: str
    side: Literal["buy", "sell"]
    size: float
    net_delta_before: float
    net_delta_after: float
    strategy_position_id: str
    strategy_type: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "underlying": self.underlying,
            "instrument": self.instrument,
            "side": self.side,
            "size": self.size,
            "net_delta_before": self.net_delta_before,
            "net_delta_after": self.net_delta_after,
            "strategy_position_id": self.strategy_position_id,
            "strategy_type": self.strategy_type,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class HedgeResult:
    """Result of a hedge execution."""
    success: bool
    order: Optional[HedgeOrder]
    executed: bool
    dry_run: bool
    message: str
    order_id: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "order": self.order.to_dict() if self.order else None,
            "executed": self.executed,
            "dry_run": self.dry_run,
            "message": self.message,
            "order_id": self.order_id,
            "error": self.error,
        }


@dataclass
class GregPosition:
    """Representation of a Greg strategy position for hedging."""
    position_id: str
    strategy_type: str
    underlying: str
    option_legs: List[Dict[str, Any]]
    hedge_perp_size: float = 0.0
    net_delta: float = 0.0
    entry_price: float = 0.0
    current_value: float = 0.0

    @property
    def is_hedgeable(self) -> bool:
        return self.strategy_type in (
            "STRATEGY_A_STRADDLE",
            "STRATEGY_A_STRANGLE",
            "STRATEGY_B_CALENDAR",
            "STRATEGY_D_IRON_BUTTERFLY",
        )


def load_greg_hedge_rules() -> Dict[str, Any]:
    """Load Greg position rules from JSON file."""
    if not GREG_POSITION_RULES_PATH.exists():
        logger.warning(f"Greg rules file not found: {GREG_POSITION_RULES_PATH}")
        return {}
    
    try:
        with open(GREG_POSITION_RULES_PATH, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load Greg rules: {e}")
        return {}


class HedgeEngine:
    """
    Delta-hedging engine for Greg's neutral short-vol strategies.
    
    Computes net delta and proposes/executes hedge orders using perpetuals.
    """
    
    def __init__(
        self,
        dry_run: bool = True,
        deribit_client: Any = None,
        greg_rules: Optional[Dict[str, Any]] = None,
    ):
        self._dry_run = dry_run
        self._client = deribit_client
        self._rules = greg_rules or load_greg_hedge_rules()
        self._lock = Lock()
        self._last_hedge_time: Dict[str, datetime] = {}
        self._hedge_history: List[HedgeResult] = []
        
        self._global_defs = self._rules.get("global_definitions", {})
        self._hedge_instruments = self._global_defs.get("hedge_instrument", {
            "BTC": "BTC-PERPETUAL",
            "ETH": "ETH-PERPETUAL",
        })
        self._perp_delta = self._global_defs.get("perp_delta_per_contract", 1.0)
        
        logger.info(f"HedgeEngine initialized (dry_run={dry_run})")

    def get_hedge_rules(self, strategy_type: str) -> HedgeRules:
        """Get hedge rules for a strategy type."""
        strategies = self._rules.get("strategies", {})
        strategy_config = strategies.get(strategy_type, {})
        hedge_config = strategy_config.get("hedge")
        return HedgeRules.from_dict(hedge_config)

    def compute_net_delta(
        self,
        option_deltas: List[float],
        perp_delta: float = 0.0,
    ) -> float:
        """
        Compute net delta for a position.
        
        Args:
            option_deltas: List of delta values for each option leg
            perp_delta: Delta from existing perp hedge (positive = long, negative = short)
        
        Returns:
            Net delta of the position
        """
        options_delta = sum(option_deltas)
        return options_delta + perp_delta

    def compute_net_delta_for_position(self, position: GregPosition) -> float:
        """Compute net delta from a GregPosition object."""
        option_deltas = [leg.get("delta", 0.0) for leg in position.option_legs]
        perp_delta = position.hedge_perp_size * self._perp_delta
        return self.compute_net_delta(option_deltas, perp_delta)

    def build_hedge_order(
        self,
        position: GregPosition,
        hedge_rules: HedgeRules,
    ) -> Optional[HedgeOrder]:
        """
        Build a hedge order if net delta exceeds threshold.
        
        For DYNAMIC/LIGHT hedging:
          - If abs(net_delta) <= trigger threshold: return None (no hedge needed)
          - Else: compute hedge size to bring delta to target (usually 0)
        """
        if hedge_rules.mode == "NONE":
            return None
        
        net_delta = self.compute_net_delta_for_position(position)
        threshold = hedge_rules.delta_abs_threshold
        target = hedge_rules.target_delta
        
        if abs(net_delta) <= threshold:
            logger.debug(
                f"[{position.position_id}] No hedge needed: |{net_delta:.4f}| <= {threshold}"
            )
            return None
        
        hedge_delta_needed = target - net_delta
        
        if abs(hedge_delta_needed) < 0.001:
            return None
        
        underlying = position.underlying.upper()
        instrument = self._hedge_instruments.get(underlying, f"{underlying}-PERPETUAL")
        
        side: Literal["buy", "sell"] = "buy" if hedge_delta_needed > 0 else "sell"
        size = abs(hedge_delta_needed)
        
        net_delta_after = net_delta + hedge_delta_needed
        
        order = HedgeOrder(
            underlying=underlying,
            instrument=instrument,
            side=side,
            size=round(size, 4),
            net_delta_before=round(net_delta, 4),
            net_delta_after=round(net_delta_after, 4),
            strategy_position_id=position.position_id,
            strategy_type=position.strategy_type,
        )
        
        logger.info(
            f"[HEDGE] strategy={position.strategy_type} underlying={underlying} "
            f"net_delta_before={net_delta:+.4f}, hedge_size={'-' if side == 'sell' else '+'}{size:.4f} {instrument}, "
            f"net_delta_after≈{net_delta_after:.4f}"
        )
        
        return order

    def apply_hedge(self, order: HedgeOrder) -> HedgeResult:
        """
        Apply a hedge order.
        
        In DRY_RUN mode: logs the action without sending real orders.
        In LIVE mode: places the order via deribit_client and tags it.
        """
        if self._dry_run:
            msg = (
                f"[DRY-RUN HEDGE] {order.side.upper()} {order.size:.4f} {order.instrument} "
                f"for {order.strategy_position_id} (delta: {order.net_delta_before:+.4f} -> {order.net_delta_after:+.4f})"
            )
            print(msg)
            logger.info(msg)
            
            result = HedgeResult(
                success=True,
                order=order,
                executed=False,
                dry_run=True,
                message=msg,
            )
            with self._lock:
                self._hedge_history.append(result)
            return result
        
        if self._client is None:
            error_msg = "Cannot execute hedge: no deribit_client configured"
            logger.error(error_msg)
            return HedgeResult(
                success=False,
                order=order,
                executed=False,
                dry_run=False,
                message=error_msg,
                error=error_msg,
            )
        
        try:
            order_result = self._client.place_order(
                instrument_name=order.instrument,
                side=order.side,
                amount=order.size,
                order_type="market",
                label=f"hedge_{order.strategy_position_id[:20]}",
            )
            
            order_id = order_result.get("order", {}).get("order_id")
            msg = (
                f"[HEDGE EXECUTED] {order.side.upper()} {order.size:.4f} {order.instrument} "
                f"order_id={order_id}"
            )
            print(msg)
            logger.info(msg)
            
            result = HedgeResult(
                success=True,
                order=order,
                executed=True,
                dry_run=False,
                message=msg,
                order_id=order_id,
            )
            with self._lock:
                self._hedge_history.append(result)
                self._last_hedge_time[order.strategy_position_id] = datetime.now(timezone.utc)
            return result
            
        except Exception as e:
            error_msg = f"Hedge order failed: {e}"
            logger.error(error_msg)
            return HedgeResult(
                success=False,
                order=order,
                executed=False,
                dry_run=False,
                message=error_msg,
                error=str(e),
            )

    def step(self, position: GregPosition) -> Optional[HedgeResult]:
        """
        Main entry point for hedging a single position.
        
        Determines strategy type, loads hedge rules, computes net delta,
        and possibly constructs and sends hedge order.
        """
        if not position.is_hedgeable:
            logger.debug(f"Position {position.position_id} is not hedgeable (type={position.strategy_type})")
            return None
        
        hedge_rules = self.get_hedge_rules(position.strategy_type)
        
        if hedge_rules.mode == "NONE":
            return None
        
        order = self.build_hedge_order(position, hedge_rules)
        
        if order is None:
            return None
        
        return self.apply_hedge(order)

    def hedge_all_positions(
        self,
        positions: List[GregPosition],
    ) -> List[HedgeResult]:
        """
        Process all open Greg positions and apply hedges where needed.
        
        Returns list of HedgeResults for any hedges that were applied.
        """
        results = []
        for pos in positions:
            result = self.step(pos)
            if result is not None:
                results.append(result)
        return results

    def get_hedge_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent hedge history."""
        with self._lock:
            history = self._hedge_history[-limit:]
            return [h.to_dict() for h in reversed(history)]

    @property
    def dry_run(self) -> bool:
        return self._dry_run

    def set_dry_run(self, value: bool) -> None:
        self._dry_run = value
        logger.info(f"HedgeEngine dry_run set to {value}")


_hedge_engine: Optional[HedgeEngine] = None


def get_hedge_engine(
    dry_run: bool = True,
    deribit_client: Any = None,
    force_new: bool = False,
) -> HedgeEngine:
    """Get or create the global HedgeEngine instance."""
    global _hedge_engine
    if _hedge_engine is None or force_new:
        _hedge_engine = HedgeEngine(
            dry_run=dry_run,
            deribit_client=deribit_client,
        )
    return _hedge_engine
