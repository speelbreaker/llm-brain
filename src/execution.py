"""
Execution module.
Translates abstract actions into Deribit orders.
"""
from __future__ import annotations

from typing import Any

from src.config import Settings, settings
from src.deribit_client import DeribitClient, DeribitAPIError
from src.models import ActionType


def _get_mid_price(client: DeribitClient, symbol: str) -> float:
    """Get the mid price for an instrument."""
    try:
        ticker = client.get_ticker(symbol)
        bid = ticker.get("best_bid_price", 0.0) or 0.0
        ask = ticker.get("best_ask_price", 0.0) or 0.0
        
        if bid > 0 and ask > 0:
            return (bid + ask) / 2
        elif bid > 0:
            return bid
        elif ask > 0:
            return ask
        else:
            return ticker.get("mark_price", 0.0) or 0.0
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
    
    if cfg.dry_run:
        return _simulate_execution(action_type, params, client, cfg)
    
    return _execute_real(action_type, params, client, cfg)


def _simulate_execution(
    action_type: ActionType,
    params: dict[str, Any],
    client: DeribitClient,
    config: Settings,
) -> dict[str, Any]:
    """Simulate execution without placing real orders."""
    result = {
        "status": "simulated",
        "dry_run": True,
        "action": action_type.value,
        "params": params,
        "orders": [],
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
    return result


def _execute_real(
    action_type: ActionType,
    params: dict[str, Any],
    client: DeribitClient,
    config: Settings,
) -> dict[str, Any]:
    """Execute real orders on Deribit testnet."""
    result = {
        "status": "executed",
        "dry_run": False,
        "action": action_type.value,
        "params": params,
        "orders": [],
        "errors": [],
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
            
            result["orders"].append({
                "type": "SELL",
                "symbol": symbol,
                "size": size,
                "price": mid_price,
                "order_id": order_result.get("order", {}).get("order_id"),
                "order_state": order_result.get("order", {}).get("order_state"),
            })
            result["message"] = f"Sold {size} {symbol} at {mid_price:.6f}"
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
            
            result["orders"].append({
                "type": "BUY",
                "symbol": symbol,
                "size": size,
                "price": mid_price,
                "order_id": order_result.get("order", {}).get("order_id"),
                "order_state": order_result.get("order", {}).get("order_state"),
            })
            result["message"] = f"Bought {size} {symbol} at {mid_price:.6f}"
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
            
            result["orders"].append({
                "type": "BUY",
                "symbol": from_symbol,
                "size": size,
                "price": from_mid,
                "order_id": close_result.get("order", {}).get("order_id"),
                "order_state": close_result.get("order", {}).get("order_state"),
                "leg": "close",
            })
            
        except DeribitAPIError as e:
            result["status"] = "partial_error"
            result["errors"].append(f"Close leg failed: {e}")
            return result
        
        try:
            open_result = client.place_order(
                instrument_name=to_symbol,
                side="sell",
                amount=size,
                order_type="limit",
                price=to_mid,
                post_only=True,
                label="agent_roll_open",
            )
            
            result["orders"].append({
                "type": "SELL",
                "symbol": to_symbol,
                "size": size,
                "price": to_mid,
                "order_id": open_result.get("order", {}).get("order_id"),
                "order_state": open_result.get("order", {}).get("order_state"),
                "leg": "open",
            })
            
            result["message"] = (
                f"Rolled: closed {from_symbol} at {from_mid:.6f}, "
                f"opened {to_symbol} at {to_mid:.6f}"
            )
            print(f"[EXECUTED] {result['message']}")
            
        except DeribitAPIError as e:
            result["status"] = "partial_error"
            result["errors"].append(f"Open leg failed: {e}")
    
    return result
