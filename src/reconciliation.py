"""
Position reconciliation between local tracker and Deribit exchange.

Compares bot-managed positions with live exchange positions and either
halts trading on divergence or auto-heals local state to match exchange.
"""
from __future__ import annotations

from typing import Literal, Sequence, Dict, List, Tuple, Any

DivergenceAction = Literal["halt", "auto_heal"]


def normalize_symbol(symbol: str) -> str:
    """Normalize symbol for comparison."""
    return symbol.strip().upper()


def reconcile_positions(
    exchange_positions: Sequence[Dict[str, Any]],
    local_positions: Sequence[Dict[str, Any]],
    action: DivergenceAction,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Compare Deribit positions with our local position tracker.

    Args:
        exchange_positions: Raw positions from Deribit private API.
            Expected shape: [{"symbol": str, "size": float, "side": str, ...}, ...]
        local_positions: Positions from position_tracker.
            Expected shape: [{"symbol": str, "quantity": float, "side": str, ...}, ...]
        action: "halt" to just report divergence, "auto_heal" to rebuild local state.

    Returns:
        (new_local_positions, stats)

        Where stats includes:
            - "divergent": bool - True if any mismatch detected
            - "missing_in_local": list of symbols present on exchange, not in local
            - "missing_in_exchange": list of symbols present in local, not on exchange
            - "size_mismatches": list of (symbol, local_size, exchange_size)
            - "exchange_count": int
            - "local_count": int
    """
    exchange_by_symbol: Dict[str, Dict[str, Any]] = {}
    for pos in exchange_positions:
        symbol = normalize_symbol(pos.get("symbol", "") or pos.get("instrument_name", ""))
        if symbol and abs(float(pos.get("size", 0))) > 0:
            exchange_by_symbol[symbol] = pos

    local_by_symbol: Dict[str, Dict[str, Any]] = {}
    for pos in local_positions:
        symbol = normalize_symbol(pos.get("symbol", ""))
        if symbol:
            local_by_symbol[symbol] = pos

    missing_in_local: List[str] = []
    missing_in_exchange: List[str] = []
    size_mismatches: List[Tuple[str, float, float]] = []

    for symbol in exchange_by_symbol:
        if symbol not in local_by_symbol:
            missing_in_local.append(symbol)

    for symbol in local_by_symbol:
        if symbol not in exchange_by_symbol:
            missing_in_exchange.append(symbol)

    for symbol in set(exchange_by_symbol.keys()) & set(local_by_symbol.keys()):
        exchange_size = abs(float(exchange_by_symbol[symbol].get("size", 0)))
        local_size = abs(float(local_by_symbol[symbol].get("quantity", 0)))
        if abs(exchange_size - local_size) > 1e-9:
            size_mismatches.append((symbol, local_size, exchange_size))

    divergent = bool(missing_in_local or missing_in_exchange or size_mismatches)

    stats: Dict[str, Any] = {
        "divergent": divergent,
        "missing_in_local": missing_in_local,
        "missing_in_exchange": missing_in_exchange,
        "size_mismatches": size_mismatches,
        "exchange_count": len(exchange_by_symbol),
        "local_count": len(local_by_symbol),
    }

    if action == "halt" or not divergent:
        return list(local_positions), stats

    new_local_positions: List[Dict[str, Any]] = []
    for symbol, ex_pos in exchange_by_symbol.items():
        new_pos = {
            "symbol": symbol,
            "underlying": ex_pos.get("underlying", _extract_underlying(symbol)),
            "side": "SHORT" if ex_pos.get("direction") == "sell" else "LONG",
            "quantity": abs(float(ex_pos.get("size", 0))),
            "entry_price": float(ex_pos.get("average_price", 0) or ex_pos.get("avg_price", 0) or 0),
            "mark_price": float(ex_pos.get("mark_price", 0) or 0),
            "unrealized_pnl": float(ex_pos.get("unrealized_pnl", 0) or ex_pos.get("total_profit_loss", 0) or 0),
            "delta": float(ex_pos.get("delta", 0) or 0),
            "option_type": ex_pos.get("option_type", "call").upper(),
            "strategy_type": "COVERED_CALL",
            "mode": "LIVE",
            "healed_from_exchange": True,
        }
        new_local_positions.append(new_pos)

    return new_local_positions, stats


def _extract_underlying(symbol: str) -> str:
    """Extract underlying from Deribit symbol (e.g., BTC-20DEC24-100000-C -> BTC)."""
    parts = symbol.split("-")
    if parts:
        return parts[0].upper()
    return "BTC"


def format_reconciliation_summary(stats: Dict[str, Any]) -> str:
    """Format reconciliation stats as human-readable summary."""
    lines = []
    lines.append(f"Reconciliation: {'DIVERGENT' if stats['divergent'] else 'IN SYNC'}")
    lines.append(f"  Exchange positions: {stats['exchange_count']}")
    lines.append(f"  Local positions: {stats['local_count']}")

    if stats["missing_in_local"]:
        lines.append(f"  Missing in local ({len(stats['missing_in_local'])}): {', '.join(stats['missing_in_local'])}")

    if stats["missing_in_exchange"]:
        lines.append(f"  Missing on exchange ({len(stats['missing_in_exchange'])}): {', '.join(stats['missing_in_exchange'])}")

    if stats["size_mismatches"]:
        lines.append(f"  Size mismatches ({len(stats['size_mismatches'])}):")
        for sym, local_sz, ex_sz in stats["size_mismatches"]:
            lines.append(f"    {sym}: local={local_sz:.4f}, exchange={ex_sz:.4f}")

    return "\n".join(lines)
