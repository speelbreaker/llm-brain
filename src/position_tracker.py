"""
Position tracking for bot-managed options positions.

Keeps an in-memory view of open and closed position chains so the web UI
can display PnL without scraping logs. Designed for *approximate* PnL;
if you need exchange-perfect numbers, extend this module to read full
trade history from Deribit.

This module is deliberately self-contained and has no side effects on import.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
from typing import Dict, List, Optional, Literal, Any

Side = Literal["SHORT", "LONG"]
OptionType = Literal["CALL", "PUT"]
StrategyType = Literal["COVERED_CALL", "CASH_SECURED_PUT"]
ModeType = Literal["LIVE", "DRY_RUN"]

MONTH_MAP = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4,
    "MAY": 5, "JUN": 6, "JUL": 7, "AUG": 8,
    "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def parse_deribit_expiry(symbol: str) -> Optional[datetime]:
    """
    Try to parse expiry from a Deribit-style option symbol.

    Supported examples:
      BTC-2025-01-03-90000-C  (ISO style)
      BTC-6DEC24-90000-C      (Deribit classic style)

    Returns a timezone-aware UTC datetime at 08:00:00 if possible,
    otherwise None.
    """
    if not symbol:
        return None

    m = re.match(r"^[A-Z]+-(\d{4})-(\d{2})-(\d{2})-", symbol)
    if m:
        year, month, day = map(int, m.groups())
        return datetime(year, month, day, 8, 0, 0, tzinfo=timezone.utc)

    m = re.match(r"^[A-Z]+-(\d{1,2})([A-Z]{3})(\d{2})-", symbol)
    if m:
        day_str, mon_str, year_2 = m.groups()
        day = int(day_str)
        mon_str = mon_str.upper()
        month = MONTH_MAP.get(mon_str)
        if month is None:
            return None
        year = 2000 + int(year_2)
        return datetime(year, month, day, 8, 0, 0, tzinfo=timezone.utc)

    return None


@dataclass
class PositionLeg:
    """Single executed leg within a position chain."""
    symbol: str
    underlying: str
    option_type: OptionType
    side: Side
    quantity: float
    entry_price: float
    entry_time: datetime
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    mark_price: Optional[float] = None

    def is_open(self) -> bool:
        return self.exit_time is None


@dataclass
class PositionChain:
    """
    A chain of legs for a single bot-managed position.

    Example covered-call chain:
      - leg 1: short BTC 90k call (open)
      - leg 2: buy back 90k call, sell 95k call (roll)
      - leg 3: buy back 95k call (close)
    """
    position_id: str
    underlying: str
    option_type: OptionType
    strategy_type: StrategyType
    mode: ModeType
    exit_style: Optional[str] = None

    legs: List[PositionLeg] = field(default_factory=list)
    open_time: datetime = field(default_factory=_utc_now)
    close_time: Optional[datetime] = None
    realized_pnl: float = 0.0
    realized_pnl_pct: float = 0.0
    max_drawdown_pct: float = 0.0

    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    expiry: Optional[datetime] = None

    def is_open(self) -> bool:
        return self.close_time is None

    @property
    def num_legs(self) -> int:
        return len(self.legs)

    @property
    def num_rolls(self) -> int:
        return max(0, self.num_legs - 1)

    @property
    def symbol(self) -> str:
        return self.legs[-1].symbol if self.legs else ""


class PositionTracker:
    """
    Thread-safe tracker of bot-managed positions.

    Assumes linear USDC-settled options with contract_size=1.0.
    For inverse / other contract sizes, adjust `_pnl_for_leg`.
    """

    def __init__(self, notional_multiplier: float = 1.0) -> None:
        self._lock = Lock()
        self._chains: Dict[str, PositionChain] = {}
        self._notional_multiplier = float(notional_multiplier)

    def _find_open_chain_for(self, underlying: str, strategy_type: StrategyType) -> Optional[PositionChain]:
        """Find an open chain for the given underlying and strategy (must be called with lock held)."""
        for chain in self._chains.values():
            if chain.is_open() and chain.underlying == underlying and chain.strategy_type == strategy_type:
                return chain
        return None

    def process_execution_result(self, result: Dict[str, Any]) -> None:
        """
        Update chains based on a single execution result.

        Expected shape (what execute_action/execute_actions already return):
          - result["status"]: "executed" | "simulated" | ...
          - result["action"]: "OPEN_COVERED_CALL" | "ROLL_COVERED_CALL" | "CLOSE_COVERED_CALL"
          - result["params"]: dict with symbol/from_symbol/to_symbol, size, underlying, etc.
          - result["orders"]: list with order dicts {symbol, size, price, ...}
          - result["underlying"] (optional)
          - result["strategy"]  (optional)
          - result["dry_run"]: bool
        """
        status = result.get("status")
        if status in {"error", "skipped"}:
            return

        action = str(result.get("action", ""))
        params = result.get("params", {}) or {}
        orders = result.get("orders", []) or []

        underlying = params.get("underlying") or result.get("underlying") or "?"
        strategy_type: StrategyType = "COVERED_CALL"
        option_type: OptionType = "CALL"
        mode: ModeType = "DRY_RUN" if result.get("dry_run", False) else "LIVE"
        exit_style = params.get("exit_style") or result.get("exit_style")

        def _extract_price(symbol_key: str) -> float:
            sym = params.get(symbol_key) or params.get("symbol")
            if sym:
                for o in orders:
                    if o.get("symbol") == sym and o.get("price") is not None:
                        return float(o["price"])
            for o in orders:
                if o.get("price") is not None:
                    return float(o["price"])
            return float(params.get("price", 0.0))

        now = _utc_now()

        with self._lock:
            if action == "OPEN_COVERED_CALL":
                symbol = params.get("symbol") or (orders[0].get("symbol") if orders else "")
                size = float(params.get("size") or (orders[0].get("size", 0.0) if orders else 0.0))
                price = _extract_price("symbol")
                
                position_id = f"{underlying}-{strategy_type}-{now.strftime('%Y%m%d%H%M%S')}"

                leg = PositionLeg(
                    symbol=symbol,
                    underlying=underlying,
                    option_type=option_type,
                    side="SHORT",
                    quantity=size,
                    entry_price=price,
                    entry_time=now,
                    mark_price=price,
                )
                expiry = parse_deribit_expiry(symbol)
                chain = PositionChain(
                    position_id=position_id,
                    underlying=underlying,
                    option_type=option_type,
                    strategy_type=strategy_type,
                    mode=mode,
                    exit_style=exit_style,
                    legs=[leg],
                    open_time=now,
                    expiry=expiry,
                )
                self._chains[position_id] = chain

            elif action == "ROLL_COVERED_CALL":
                chain = self._find_open_chain_for(underlying, strategy_type)
                if chain is None:
                    return

                from_symbol = params.get("from_symbol") or ""
                to_symbol = params.get("to_symbol") or ""
                size = float(params.get("size") or (orders[0].get("size", 0.0) if orders else 0.0))

                close_price = _extract_price("from_symbol")
                for leg in chain.legs:
                    if leg.is_open() and (not from_symbol or leg.symbol == from_symbol):
                        leg.exit_price = close_price
                        leg.exit_time = now
                        chain.realized_pnl += self._pnl_for_leg(leg)

                open_price = _extract_price("to_symbol") or close_price
                new_symbol = to_symbol or from_symbol
                new_leg = PositionLeg(
                    symbol=new_symbol,
                    underlying=underlying,
                    option_type=option_type,
                    side="SHORT",
                    quantity=size,
                    entry_price=open_price,
                    entry_time=now,
                    mark_price=open_price,
                )
                chain.legs.append(new_leg)
                
                new_expiry = parse_deribit_expiry(new_symbol)
                if new_expiry is not None:
                    chain.expiry = new_expiry

            elif action == "CLOSE_COVERED_CALL":
                chain = self._find_open_chain_for(underlying, strategy_type)
                if chain is None:
                    return
                symbol = params.get("symbol") or chain.symbol
                close_price = _extract_price("symbol")
                for leg in chain.legs:
                    if leg.is_open() and (not symbol or leg.symbol == symbol):
                        leg.exit_price = close_price
                        leg.exit_time = now
                        chain.realized_pnl += self._pnl_for_leg(leg)

                chain.close_time = now

            if chain is not None:
                self._update_chain_unrealized(chain)

    def refresh_marks(self, client: Any) -> None:
        """
        Update mark prices/unrealized PnL for all open chains.

        `client` must have get_ticker(symbol) -> {"mark_price": float, ...}.
        """
        with self._lock:
            for chain in self._chains.values():
                if not chain.is_open():
                    continue
                for leg in chain.legs:
                    if leg.is_open() and leg.symbol:
                        try:
                            ticker = client.get_ticker(leg.symbol)
                            if ticker and "mark_price" in ticker:
                                leg.mark_price = float(ticker["mark_price"])
                        except Exception:
                            pass
                self._update_chain_unrealized(chain)

    def get_open_positions_payload(self) -> Dict[str, Any]:
        with self._lock:
            positions: List[Dict[str, Any]] = []
            for chain in self._chains.values():
                if not chain.is_open():
                    continue
                positions.append(self._chain_to_open_summary(chain))

            totals = {
                "positions_count": len(positions),
                "unrealized_pnl": float(sum(p["unrealized_pnl"] for p in positions)) if positions else 0.0,
                "unrealized_pnl_pct": float(sum(p["unrealized_pnl_pct"] for p in positions)) if positions else 0.0,
            }
            return {"positions": positions, "totals": totals}

    def get_closed_positions_payload(self) -> Dict[str, Any]:
        with self._lock:
            chains: List[Dict[str, Any]] = []
            for chain in self._chains.values():
                if chain.is_open():
                    continue
                chains.append(self._chain_to_closed_summary(chain))

            totals = {
                "chains_count": len(chains),
                "realized_pnl": float(sum(c["realized_pnl"] for c in chains)) if chains else 0.0,
                "realized_pnl_pct": float(sum(c["realized_pnl_pct"] for c in chains)) if chains else 0.0,
            }
            return {"chains": chains, "totals": totals}

    def _pnl_for_leg(self, leg: PositionLeg, mark_price: Optional[float] = None) -> float:
        """
        Compute PnL in USD for a single leg.

        Assumes linear USDC-settled options with contract_size=1.0:
          SHORT: (entry - exit_or_mark) * qty  (profit when bought back cheaper)
          LONG:  (exit_or_mark - entry) * qty  (profit when sold higher)
        """
        if mark_price is None:
            if leg.exit_price is None:
                return 0.0
            px2 = float(leg.exit_price)
        else:
            px2 = float(mark_price)

        px1 = float(leg.entry_price)
        qty = float(leg.quantity)

        if leg.side == "SHORT":
            return (px1 - px2) * qty * self._notional_multiplier
        else:
            return (px2 - px1) * qty * self._notional_multiplier

    def _update_chain_unrealized(self, chain: PositionChain) -> None:
        realized = chain.realized_pnl
        unrealized = 0.0

        for leg in chain.legs:
            if leg.is_open():
                mark = leg.mark_price if leg.mark_price is not None else leg.entry_price
                unrealized += self._pnl_for_leg(leg, mark_price=mark)

        chain.unrealized_pnl = unrealized
        if chain.legs:
            base = abs(chain.legs[0].entry_price * chain.legs[0].quantity * self._notional_multiplier) or 1.0
            chain.unrealized_pnl_pct = (realized + unrealized) / base * 100.0

    def _chain_to_open_summary(self, chain: PositionChain) -> Dict[str, Any]:
        dte = 0.0
        return {
            "position_id": chain.position_id,
            "underlying": chain.underlying,
            "symbol": chain.symbol,
            "option_type": chain.option_type,
            "strategy_type": chain.strategy_type,
            "side": "SHORT",
            "quantity": chain.legs[-1].quantity if chain.legs else 0.0,
            "entry_price": chain.legs[0].entry_price if chain.legs else 0.0,
            "mark_price": chain.legs[-1].entry_price if chain.legs else 0.0,
            "unrealized_pnl": chain.unrealized_pnl,
            "unrealized_pnl_pct": chain.unrealized_pnl_pct,
            "entry_time": chain.open_time.isoformat(),
            "expiry": None,
            "dte": dte,
            "num_rolls": chain.num_rolls,
            "mode": chain.mode,
            "exit_style": chain.exit_style or "hold_to_expiry",
        }

    def _chain_to_closed_summary(self, chain: PositionChain) -> Dict[str, Any]:
        if chain.close_time is None:
            holding_days = 0.0
        else:
            holding_days = (chain.close_time - chain.open_time).total_seconds() / 86400.0

        return {
            "position_id": chain.position_id,
            "underlying": chain.underlying,
            "symbol": chain.symbol,
            "option_type": chain.option_type,
            "strategy_type": chain.strategy_type,
            "open_time": chain.open_time.isoformat(),
            "close_time": (chain.close_time or _utc_now()).isoformat(),
            "holding_days": holding_days,
            "num_legs": chain.num_legs,
            "num_rolls": chain.num_rolls,
            "realized_pnl": chain.realized_pnl,
            "realized_pnl_pct": chain.realized_pnl_pct,
            "max_drawdown_pct": chain.max_drawdown_pct,
            "mode": chain.mode,
            "exit_style": chain.exit_style or "hold_to_expiry",
            "note": None,
        }


position_tracker = PositionTracker(notional_multiplier=1.0)
