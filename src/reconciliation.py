"""
Position reconciliation between local tracker and Deribit exchange.

Compares bot-managed positions with live exchange positions and either
halts trading on divergence or auto-heals local state to match exchange.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence, Dict, List, Tuple, Any, Optional

DivergenceAction = Literal["halt", "auto_heal"]


@dataclass
class PositionSizeMismatch:
    """Represents a size mismatch between local and exchange positions."""
    instrument_name: str
    side: str
    size_tracker: float
    size_exchange: float
    diff_usd: float = 0.0
    
    @property
    def size_diff(self) -> float:
        """Absolute difference in size."""
        return abs(self.size_tracker - self.size_exchange)


@dataclass
class PositionReconciliationDiff:
    """Result of comparing local tracker positions vs exchange positions."""
    untracked_on_exchange: List[Dict[str, Any]] = field(default_factory=list)
    missing_on_exchange: List[Dict[str, Any]] = field(default_factory=list)
    size_mismatches: List[PositionSizeMismatch] = field(default_factory=list)
    exchange_count: int = 0
    local_count: int = 0
    tolerance_usd: float = 10.0
    
    @property
    def is_clean(self) -> bool:
        """True only when no mismatches are found."""
        return (
            len(self.untracked_on_exchange) == 0
            and len(self.missing_on_exchange) == 0
            and len(self.size_mismatches) == 0
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "is_clean": self.is_clean,
            "divergent": not self.is_clean,
            "untracked_on_exchange": [
                p.get("symbol", p.get("instrument_name", "unknown"))
                for p in self.untracked_on_exchange
            ],
            "missing_on_exchange": [
                p.get("symbol", "unknown") for p in self.missing_on_exchange
            ],
            "size_mismatches": [
                {
                    "symbol": m.instrument_name,
                    "side": m.side,
                    "local": m.size_tracker,
                    "exchange": m.size_exchange,
                    "diff_usd": m.diff_usd,
                }
                for m in self.size_mismatches
            ],
            "exchange_count": self.exchange_count,
            "local_count": self.local_count,
            "tolerance_usd": self.tolerance_usd,
        }


def normalize_symbol(symbol: str) -> str:
    """Normalize symbol for comparison."""
    return symbol.strip().upper()


def _extract_underlying(symbol: str) -> str:
    """Extract underlying from Deribit symbol (e.g., BTC-20DEC24-100000-C -> BTC)."""
    parts = symbol.split("-")
    if parts:
        return parts[0].upper()
    return "BTC"


def _get_position_key(symbol: str, side: str) -> str:
    """Create a unique key for position matching (symbol + direction)."""
    normalized_side = "SHORT" if side.lower() in ("sell", "short") else "LONG"
    return f"{normalize_symbol(symbol)}:{normalized_side}"


def diff_positions(
    tracked: Sequence[Dict[str, Any]],
    exchange: Sequence[Dict[str, Any]],
    tolerance_usd: float = 10.0,
    spot_prices: Optional[Dict[str, float]] = None,
) -> PositionReconciliationDiff:
    """
    Compare tracked (local) positions against exchange positions.
    
    Args:
        tracked: Local position tracker positions.
            Expected: [{"symbol": str, "quantity": float, "side": str, ...}, ...]
        exchange: Positions from Deribit exchange.
            Expected: [{"symbol": str, "size": float, "direction": str, ...}, ...]
        tolerance_usd: Size mismatch tolerance in USD equivalent.
        spot_prices: Optional spot prices for USD conversion (e.g., {"BTC": 100000}).
    
    Returns:
        PositionReconciliationDiff with categorized mismatches.
    """
    spot_prices = spot_prices or {"BTC": 100000.0, "ETH": 3500.0}
    
    exchange_by_key: Dict[str, Dict[str, Any]] = {}
    for pos in exchange:
        symbol = pos.get("symbol", "") or pos.get("instrument_name", "")
        if not symbol:
            continue
        size = abs(float(pos.get("size", 0)))
        if size <= 0:
            continue
        direction = pos.get("direction", "") or pos.get("side", "sell")
        key = _get_position_key(symbol, direction)
        exchange_by_key[key] = {**pos, "symbol": normalize_symbol(symbol)}
    
    tracked_by_key: Dict[str, Dict[str, Any]] = {}
    for pos in tracked:
        symbol = pos.get("symbol", "")
        if not symbol:
            continue
        quantity = abs(float(pos.get("quantity", 0) or pos.get("size", 0)))
        if quantity <= 0:
            continue
        side = pos.get("side", "SHORT")
        key = _get_position_key(symbol, side)
        tracked_by_key[key] = {**pos, "symbol": normalize_symbol(symbol)}
    
    diff = PositionReconciliationDiff(
        exchange_count=len(exchange_by_key),
        local_count=len(tracked_by_key),
        tolerance_usd=tolerance_usd,
    )
    
    for key, ex_pos in exchange_by_key.items():
        if key not in tracked_by_key:
            diff.untracked_on_exchange.append(ex_pos)
    
    for key, tr_pos in tracked_by_key.items():
        if key not in exchange_by_key:
            diff.missing_on_exchange.append(tr_pos)
    
    for key in set(exchange_by_key.keys()) & set(tracked_by_key.keys()):
        ex_pos = exchange_by_key[key]
        tr_pos = tracked_by_key[key]
        
        exchange_size = abs(float(ex_pos.get("size", 0)))
        tracker_size = abs(float(tr_pos.get("quantity", 0) or tr_pos.get("size", 0)))
        size_diff = abs(exchange_size - tracker_size)
        
        symbol = ex_pos.get("symbol", "")
        underlying = _extract_underlying(symbol)
        spot = spot_prices.get(underlying, 100000.0)
        diff_usd = size_diff * spot
        
        if diff_usd > tolerance_usd:
            direction = ex_pos.get("direction", "") or ex_pos.get("side", "sell")
            side = "SHORT" if direction.lower() in ("sell", "short") else "LONG"
            diff.size_mismatches.append(
                PositionSizeMismatch(
                    instrument_name=symbol,
                    side=side,
                    size_tracker=tracker_size,
                    size_exchange=exchange_size,
                    diff_usd=diff_usd,
                )
            )
    
    return diff


def reconcile_positions(
    exchange_positions: Sequence[Dict[str, Any]],
    local_positions: Sequence[Dict[str, Any]],
    action: DivergenceAction,
    tolerance_usd: float = 10.0,
    spot_prices: Optional[Dict[str, float]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Compare Deribit positions with our local position tracker.

    Args:
        exchange_positions: Raw positions from Deribit private API.
            Expected shape: [{"symbol": str, "size": float, "side": str, ...}, ...]
        local_positions: Positions from position_tracker.
            Expected shape: [{"symbol": str, "quantity": float, "side": str, ...}, ...]
        action: "halt" to just report divergence, "auto_heal" to rebuild local state.
        tolerance_usd: Tolerance for size mismatches in USD.
        spot_prices: Optional spot prices for USD conversion.

    Returns:
        (new_local_positions, stats)

        Where stats includes:
            - "divergent": bool - True if any mismatch detected
            - "is_clean": bool - True if no mismatches
            - "missing_in_local": list of symbols present on exchange, not in local
            - "missing_in_exchange": list of symbols present in local, not on exchange
            - "size_mismatches": list of (symbol, local_size, exchange_size)
            - "exchange_count": int
            - "local_count": int
    """
    diff = diff_positions(
        tracked=local_positions,
        exchange=exchange_positions,
        tolerance_usd=tolerance_usd,
        spot_prices=spot_prices,
    )
    
    stats: Dict[str, Any] = {
        "divergent": not diff.is_clean,
        "is_clean": diff.is_clean,
        "missing_in_local": [p.get("symbol", "") for p in diff.untracked_on_exchange],
        "missing_in_exchange": [p.get("symbol", "") for p in diff.missing_on_exchange],
        "size_mismatches": [
            (m.instrument_name, m.size_tracker, m.size_exchange)
            for m in diff.size_mismatches
        ],
        "exchange_count": diff.exchange_count,
        "local_count": diff.local_count,
        "tolerance_usd": tolerance_usd,
    }

    if action == "halt" or diff.is_clean:
        return list(local_positions), stats

    new_local_positions: List[Dict[str, Any]] = []
    for pos in exchange_positions:
        symbol = pos.get("symbol", "") or pos.get("instrument_name", "")
        if not symbol or abs(float(pos.get("size", 0))) <= 0:
            continue
        
        new_pos = {
            "symbol": normalize_symbol(symbol),
            "underlying": pos.get("underlying", _extract_underlying(symbol)),
            "side": "SHORT" if pos.get("direction") == "sell" else "LONG",
            "quantity": abs(float(pos.get("size", 0))),
            "entry_price": float(pos.get("average_price", 0) or pos.get("avg_price", 0) or 0),
            "mark_price": float(pos.get("mark_price", 0) or 0),
            "unrealized_pnl": float(pos.get("unrealized_pnl", 0) or pos.get("total_profit_loss", 0) or 0),
            "delta": float(pos.get("delta", 0) or 0),
            "option_type": pos.get("option_type", "call").upper(),
            "strategy_type": "COVERED_CALL",
            "mode": "LIVE",
            "healed_from_exchange": True,
        }
        new_local_positions.append(new_pos)

    return new_local_positions, stats


def format_reconciliation_summary(stats: Dict[str, Any]) -> str:
    """Format reconciliation stats as human-readable summary."""
    lines = []
    is_clean = stats.get("is_clean", not stats.get("divergent", False))
    lines.append(f"Reconciliation: {'IN SYNC' if is_clean else 'DIVERGENT'}")
    lines.append(f"  Exchange positions: {stats.get('exchange_count', 0)}")
    lines.append(f"  Local positions: {stats.get('local_count', 0)}")

    missing_in_local = stats.get("missing_in_local", [])
    if missing_in_local:
        lines.append(f"  Missing in local ({len(missing_in_local)}): {', '.join(missing_in_local)}")

    missing_in_exchange = stats.get("missing_in_exchange", [])
    if missing_in_exchange:
        lines.append(f"  Missing on exchange ({len(missing_in_exchange)}): {', '.join(missing_in_exchange)}")

    size_mismatches = stats.get("size_mismatches", [])
    if size_mismatches:
        lines.append(f"  Size mismatches ({len(size_mismatches)}):")
        for item in size_mismatches:
            if isinstance(item, tuple):
                sym, local_sz, ex_sz = item
                lines.append(f"    {sym}: local={local_sz:.4f}, exchange={ex_sz:.4f}")
            elif isinstance(item, dict):
                lines.append(f"    {item['symbol']}: local={item['local']:.4f}, exchange={item['exchange']:.4f}")

    return "\n".join(lines)


def run_reconciliation_once(
    deribit_client: Any,
    position_tracker: Any,
    settings: Any,
    spot_prices: Optional[Dict[str, float]] = None,
) -> PositionReconciliationDiff:
    """
    Run a single reconciliation check between exchange and local tracker.
    
    Args:
        deribit_client: DeribitClient instance for fetching exchange positions.
        position_tracker: PositionTracker instance for local positions.
        settings: Settings instance with reconciliation config.
        spot_prices: Optional spot prices for USD conversion.
    
    Returns:
        PositionReconciliationDiff with comparison results.
    """
    exchange_positions_raw: List[Dict[str, Any]] = []
    for currency in settings.underlyings:
        try:
            positions = deribit_client.get_positions(currency=currency, kind="option")
            exchange_positions_raw.extend(positions)
        except Exception as e:
            print(f"[Reconciliation] Failed to fetch positions for {currency}: {e}")
    
    local_payload = position_tracker.get_open_positions_payload()
    local_positions = local_payload.get("positions", [])
    
    diff = diff_positions(
        tracked=local_positions,
        exchange=exchange_positions_raw,
        tolerance_usd=settings.position_reconcile_tolerance_usd,
        spot_prices=spot_prices,
    )
    
    return diff


def get_reconciliation_status(diff: PositionReconciliationDiff) -> str:
    """Get a simple status string for logging."""
    return "clean" if diff.is_clean else "out_of_sync"
