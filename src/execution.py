"""
Execution module.
Translates abstract actions into Deribit orders.
Supports batch execution for training mode experimentation.
"""
from __future__ import annotations

import time
from typing import Any, Optional, Tuple

from src.config import Settings, settings
from src.deribit_client import DeribitClient, DeribitAPIError
from src.models import ActionType
from src.position_tracker import position_tracker


def _round_price(price: float) -> float:
    """
    Round price to valid tick size for Deribit BTC options.
    
    Tick size steps (from Deribit):
    - Below 0.005: tick_size = 0.0001
    - At or above 0.005: tick_size = 0.0005
    """
    if price < 0.005:
        tick_size = 0.0001
    else:
        tick_size = 0.0005
    
    rounded = round(price / tick_size) * tick_size
    return round(rounded, 4)


def _extract_underlying(symbol: str) -> str:
    """Extract underlying (BTC/ETH) from symbol like BTC-26DEC25-96000-C."""
    if symbol.startswith("BTC"):
        return "BTC"
    elif symbol.startswith("ETH"):
        return "ETH"
    return "?"


def _wait_for_fill(
    client: DeribitClient,
    order_id: str,
    *,
    timeout_seconds: float = 20.0,
    poll_seconds: float = 0.5,
) -> Tuple[bool, dict[str, Any]]:
    """Poll Deribit order state until filled/cancelled/timeout.

    Returns (filled, order_state_dict).
    """
    deadline = time.time() + max(timeout_seconds, 0.5)
    last: dict[str, Any] = {}

    while time.time() < deadline:
        try:
            st = client.get_order_state(order_id)
            last = st.get("order") if isinstance(st, dict) and "order" in st else st
        except Exception:
            last = last or {}

        state = str((last or {}).get("order_state") or "").lower()
        filled_amount = float((last or {}).get("filled_amount") or 0.0)
        amount = float((last or {}).get("amount") or 0.0)

        if state in ("filled", "cancelled", "rejected"):
            return state == "filled", last

        # Some venues report 'open' but filled_amount==amount.
        if amount > 0 and filled_amount >= amount - 1e-9:
            return True, last

        time.sleep(poll_seconds)

    return False, last


def _get_mid_price(client: DeribitClient, symbol: str) -> float:
    """Get the mid price for an instrument, rounded to valid tick size."""
    try:
        ticker = client.get_ticker(symbol)
        bid = ticker.get("best_bid_price", 0.0) or 0.0
        ask = ticker.get("best_ask_price", 0.0) or 0.0
        
        if bid > 0 and ask > 0:
            mid = (bid + ask) / 2
        elif bid > 0:
            mid = bid
        elif ask > 0:
            mid = ask
        else:
            mid = ticker.get("mark_price", 0.0) or 0.0
        
        return _round_price(mid)
    except DeribitAPIError:
        return 0.0


def execute_action(
    client: DeribitClient,
    action_dict: dict[str, Any],
    config: Settings | None = None,
) -> dict[str, Any]:
    """
    Execute a proposed action by placing orders on Deribit.
    
    Args:
        client: Deribit API client
        action_dict: Action dict with keys: action, params, reasoning
        config: Settings configuration
    
    Returns:
        Dict with execution results including order IDs, prices, and any errors
    """
    cfg = config or settings
    
    action_str = action_dict.get("action", "DO_NOTHING")
    params = action_dict.get("params", {})
    
    if isinstance(action_str, ActionType):
        action_type = action_str
    else:
        try:
            action_type = ActionType(action_str)
        except ValueError:
            return {
                "status": "error",
                "message": f"Invalid action type: {action_str}",
            }
    
    if action_type == ActionType.DO_NOTHING:
        return {
            "status": "skipped",
            "message": "Action is DO_NOTHING, no orders placed",
        }
    
    # Extract underlying from action_dict or infer from symbol
    underlying = action_dict.get("underlying") or _extract_underlying(params.get("symbol", ""))
    
    if cfg.dry_run:
        return _simulate_execution(action_type, params, client, cfg, underlying)
    
    return _execute_real(action_type, params, client, cfg, underlying)


def _simulate_execution(
    action_type: ActionType,
    params: dict[str, Any],
    client: DeribitClient,
    config: Settings,
    underlying: str = "?",
) -> dict[str, Any]:
    """Simulate execution without placing real orders."""
    result = {
        "status": "simulated",
        "dry_run": True,
        "action": action_type.value,
        "params": params,
        "orders": [],
        "underlying": underlying,
    }
    
    if action_type == ActionType.OPEN_COVERED_CALL:
        symbol = params.get("symbol", "")
        size = params.get("size", config.default_order_size)
        
        mid_price = _get_mid_price(client, symbol) if symbol else 0.0
        
        result["orders"].append({
            "type": "SELL",
            "symbol": symbol,
            "size": size,
            "price": mid_price,
            "simulated": True,
        })
        result["message"] = f"Would sell {size} {symbol} at ~{mid_price:.6f}"
    
    elif action_type == ActionType.CLOSE_COVERED_CALL:
        symbol = params.get("symbol", "")
        size = params.get("size", config.default_order_size)
        
        mid_price = _get_mid_price(client, symbol) if symbol else 0.0
        
        result["orders"].append({
            "type": "BUY",
            "symbol": symbol,
            "size": size,
            "price": mid_price,
            "simulated": True,
        })
        result["message"] = f"Would buy {size} {symbol} at ~{mid_price:.6f}"
    
    elif action_type == ActionType.ROLL_COVERED_CALL:
        from_symbol = params.get("from_symbol", "")
        to_symbol = params.get("to_symbol", "")
        size = params.get("size", config.default_order_size)
        
        from_mid = _get_mid_price(client, from_symbol) if from_symbol else 0.0
        to_mid = _get_mid_price(client, to_symbol) if to_symbol else 0.0
        
        result["orders"].append({
            "type": "BUY",
            "symbol": from_symbol,
            "size": size,
            "price": from_mid,
            "simulated": True,
            "leg": "close",
        })
        result["orders"].append({
            "type": "SELL",
            "symbol": to_symbol,
            "size": size,
            "price": to_mid,
            "simulated": True,
            "leg": "open",
        })
        result["message"] = (
            f"Would roll: close {from_symbol} at ~{from_mid:.6f}, "
            f"open {to_symbol} at ~{to_mid:.6f}"
        )
    
    print(f"[DRY-RUN] {result.get('message', 'Simulated execution')}")
    try:
        position_tracker.process_execution_result(result)
    except Exception as e:
        print(f"[PositionTracker] error (simulated): {e}")
    return result


def _execute_real(
    action_type: ActionType,
    params: dict[str, Any],
    client: DeribitClient,
    config: Settings,
    underlying: str = "?",
) -> dict[str, Any]:
    """Execute real orders on Deribit testnet."""
    result = {
        "status": "executed",
        "dry_run": False,
        "action": action_type.value,
        "params": params,
        "orders": [],
        "errors": [],
        "underlying": underlying,
    }
    
    if action_type == ActionType.OPEN_COVERED_CALL:
        symbol = params.get("symbol", "")
        size = params.get("size", config.default_order_size)
        
        mid_price = _get_mid_price(client, symbol)
        if mid_price <= 0:
            result["status"] = "error"
            result["errors"].append(f"Could not get price for {symbol}")
            return result
        
        try:
            order_result = client.place_order(
                instrument_name=symbol,
                side="sell",
                amount=size,
                order_type="limit",
                price=mid_price,
                post_only=True,
                label="agent_covered_call",
            )
            
            oid = order_result.get("order", {}).get("order_id")
            filled = False
            avg_px = mid_price
            filled_amt = 0.0
            if oid:
                filled, st = _wait_for_fill(client, oid, timeout_seconds=30.0, poll_seconds=0.5)
                avg_px = float((st or {}).get("average_price") or mid_price)
                filled_amt = float((st or {}).get("filled_amount") or 0.0)

            if not filled:
                result["status"] = "error"
                result["errors"].append(f"OPEN not filled within timeout for {symbol} (order_id={oid})")
                # best-effort cancel
                if oid:
                    try:
                        client.cancel_order(oid)
                    except Exception:
                        pass
                return result

            result["orders"].append({
                "type": "SELL",
                "symbol": symbol,
                "size": filled_amt or size,
                "price": avg_px,
                "order_id": oid,
                "order_state": "filled",
                "filled_amount": filled_amt,
                "average_price": avg_px,
            })
            result["message"] = f"Sold {filled_amt or size} {symbol} at {avg_px:.6f}"
            print(f"[EXECUTED] {result['message']}")
            
        except DeribitAPIError as e:
            result["status"] = "error"
            result["errors"].append(str(e))
    
    elif action_type == ActionType.CLOSE_COVERED_CALL:
        symbol = params.get("symbol", "")
        size = params.get("size", config.default_order_size)
        
        mid_price = _get_mid_price(client, symbol)
        if mid_price <= 0:
            result["status"] = "error"
            result["errors"].append(f"Could not get price for {symbol}")
            return result
        
        try:
            order_result = client.place_order(
                instrument_name=symbol,
                side="buy",
                amount=size,
                order_type="limit",
                price=mid_price,
                reduce_only=True,
                label="agent_close_cc",
            )
            
            oid = order_result.get("order", {}).get("order_id")
            filled = False
            avg_px = mid_price
            filled_amt = 0.0
            if oid:
                filled, st = _wait_for_fill(client, oid, timeout_seconds=30.0, poll_seconds=0.5)
                avg_px = float((st or {}).get("average_price") or mid_price)
                filled_amt = float((st or {}).get("filled_amount") or 0.0)

            if not filled:
                result["status"] = "error"
                result["errors"].append(f"CLOSE not filled within timeout for {symbol} (order_id={oid})")
                if oid:
                    try:
                        client.cancel_order(oid)
                    except Exception:
                        pass
                return result

            result["orders"].append({
                "type": "BUY",
                "symbol": symbol,
                "size": filled_amt or size,
                "price": avg_px,
                "order_id": oid,
                "order_state": "filled",
                "filled_amount": filled_amt,
                "average_price": avg_px,
            })
            result["message"] = f"Bought {filled_amt or size} {symbol} at {avg_px:.6f}"
            print(f"[EXECUTED] {result['message']}")
            
        except DeribitAPIError as e:
            result["status"] = "error"
            result["errors"].append(str(e))
    
    elif action_type == ActionType.ROLL_COVERED_CALL:
        from_symbol = params.get("from_symbol", "")
        to_symbol = params.get("to_symbol", "")
        size = params.get("size", config.default_order_size)
        
        from_mid = _get_mid_price(client, from_symbol)
        to_mid = _get_mid_price(client, to_symbol)
        
        if from_mid <= 0:
            result["status"] = "error"
            result["errors"].append(f"Could not get price for {from_symbol}")
            return result
        
        if to_mid <= 0:
            result["status"] = "error"
            result["errors"].append(f"Could not get price for {to_symbol}")
            return result
        
        try:
            close_result = client.place_order(
                instrument_name=from_symbol,
                side="buy",
                amount=size,
                order_type="limit",
                price=from_mid,
                reduce_only=True,
                label="agent_roll_close",
            )

            close_oid = close_result.get("order", {}).get("order_id")
            if not close_oid:
                result["status"] = "error"
                result["errors"].append("Close leg missing order_id")
                return result

            close_filled, close_state = _wait_for_fill(client, close_oid, timeout_seconds=30.0, poll_seconds=0.5)
            close_avg = float((close_state or {}).get("average_price") or from_mid)
            close_filled_amt = float((close_state or {}).get("filled_amount") or 0.0)

            if not close_filled:
                result["status"] = "error"
                result["errors"].append(f"Close leg not filled within timeout (order_id={close_oid})")
                try:
                    client.cancel_order(close_oid)
                except Exception:
                    pass
                result["orders"].append({
                    "type": "BUY",
                    "symbol": from_symbol,
                    "size": close_filled_amt or size,
                    "price": close_avg,
                    "order_id": close_oid,
                    "order_state": str((close_state or {}).get("order_state") or "unknown"),
                    "leg": "close",
                })
                return result

            result["orders"].append({
                "type": "BUY",
                "symbol": from_symbol,
                "size": close_filled_amt or size,
                "price": close_avg,
                "order_id": close_oid,
                "order_state": "filled",
                "filled_amount": close_filled_amt,
                "average_price": close_avg,
                "leg": "close",
            })

        except DeribitAPIError as e:
            result["status"] = "error"
            result["errors"].append(f"Close leg failed: {e}")
            return result

        # NOTE: open leg is NOT executed here anymore.
        # Agent loop must re-check latches + recompute eligibility after close fill.
        result["status"] = "close_filled"
        result["message"] = f"Close leg filled for roll ({from_symbol}); open leg deferred"
    
    try:
        position_tracker.process_execution_result(result)
    except Exception as e:
        print(f"[PositionTracker] error (real): {e}")
    return result


def execute_actions(
    client: DeribitClient,
    actions: list[dict[str, Any]],
    config: Settings | None = None,
) -> dict[str, Any]:
    """
    Execute multiple actions (for training mode batch execution).
    
    Args:
        client: Deribit API client
        actions: List of action dicts with keys: action, params, reasoning
        config: Settings configuration
    
    Returns:
        Dict with batch execution results
    """
    cfg = config or settings
    
    if not actions:
        return {
            "status": "skipped",
            "message": "No actions to execute",
            "results": [],
        }
    
    if len(actions) == 1:
        result = execute_action(client, actions[0], cfg)
        if "strategy" in actions[0]:
            result["strategy"] = actions[0]["strategy"]
        return {
            "status": result.get("status", "unknown"),
            "dry_run": result.get("dry_run", cfg.dry_run),
            "results": [result],
        }
    
    results = []
    for action in actions:
        result = execute_action(client, action, cfg)
        
        if "strategy" in action:
            result["strategy"] = action["strategy"]
        if "underlying" in action:
            result["underlying"] = action["underlying"]
        
        results.append(result)
    
    all_simulated = all(r.get("status") == "simulated" for r in results)
    any_error = any(r.get("status") == "error" for r in results)
    
    if all_simulated:
        status = "simulated_batch"
    elif any_error:
        status = "partial_error"
    else:
        status = "executed_batch"
    
    by_strategy = {}
    by_underlying = {}
    for r in results:
        s = r.get("strategy", "unknown")
        u = r.get("underlying", "unknown")
        by_strategy[s] = by_strategy.get(s, 0) + 1
        by_underlying[u] = by_underlying.get(u, 0) + 1
    
    return {
        "status": status,
        "dry_run": cfg.dry_run,
        "total_actions": len(results),
        "by_strategy": by_strategy,
        "by_underlying": by_underlying,
        "results": results,
    }
