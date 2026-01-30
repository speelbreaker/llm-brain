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
from src.execution_ledger import ExecutionLedger, OrderPlan


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


class OrderPollStatus:
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    OPEN_TIMEOUT_UNFILLED = "OPEN_TIMEOUT_UNFILLED"
    OPEN_TIMEOUT_PARTIAL = "OPEN_TIMEOUT_PARTIAL"


def _normalize_order_state_payload(st: dict[str, Any]) -> dict[str, Any]:
    """Normalize Deribit order state payload to include fill metrics always."""
    order = st.get("order") if isinstance(st, dict) and "order" in st else st
    order = order or {}
    return {
        "order_id": order.get("order_id"),
        "instrument_name": order.get("instrument_name") or order.get("instrument"),
        "order_state": order.get("order_state"),
        "amount": float(order.get("amount") or 0.0),
        "filled_amount": float(order.get("filled_amount") or 0.0),
        "average_price": (float(order.get("average_price")) if order.get("average_price") is not None else None),
        "last_update_timestamp_ms": order.get("last_update_timestamp"),
    }


def _poll_order_until_terminal_or_timeout(
    client: DeribitClient,
    order_id: str,
    *,
    timeout_seconds: float = 20.0,
    poll_seconds: float = 0.5,
) -> Tuple[str, dict[str, Any]]:
    """Poll Deribit order state until terminal or timeout.

    Returns (status_enum, normalized_payload).
    """
    eps = 1e-9
    deadline = time.time() + max(timeout_seconds, 0.5)
    last: dict[str, Any] = {}

    while time.time() < deadline:
        try:
            st = client.get_order_state(order_id)
            last = _normalize_order_state_payload(st)
        except Exception:
            last = last or {}

        state = str((last or {}).get("order_state") or "").lower()
        filled_amount = float((last or {}).get("filled_amount") or 0.0)
        amount = float((last or {}).get("amount") or 0.0)

        if state == "filled" or (amount > 0 and filled_amount >= amount - eps):
            return OrderPollStatus.FILLED, last
        if state == "cancelled":
            return OrderPollStatus.CANCELLED, last
        if state == "rejected":
            return OrderPollStatus.REJECTED, last

        time.sleep(poll_seconds)

    # Timeout: re-fetch once (do not trust cached last)
    try:
        st = client.get_order_state(order_id)
        last = _normalize_order_state_payload(st)
    except Exception:
        last = last or {}

    filled_amount = float((last or {}).get("filled_amount") or 0.0)
    amount = float((last or {}).get("amount") or 0.0)

    if filled_amount > eps and amount > 0 and filled_amount < amount - eps:
        return OrderPollStatus.OPEN_TIMEOUT_PARTIAL, last
    return OrderPollStatus.OPEN_TIMEOUT_UNFILLED, last


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


_ledger = ExecutionLedger()


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
            status = None
            avg_px = mid_price
            filled_amt = 0.0
            if oid:
                status, st = _poll_order_until_terminal_or_timeout(client, oid, timeout_seconds=30.0, poll_seconds=0.5)
                avg_px = float((st or {}).get("average_price") or mid_price)
                filled_amt = float((st or {}).get("filled_amount") or 0.0)

            if (not oid) or status != OrderPollStatus.FILLED:
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
            status = None
            avg_px = mid_price
            filled_amt = 0.0
            if oid:
                status, st = _poll_order_until_terminal_or_timeout(client, oid, timeout_seconds=30.0, poll_seconds=0.5)
                avg_px = float((st or {}).get("average_price") or mid_price)
                filled_amt = float((st or {}).get("filled_amount") or 0.0)

            if (not oid) or status != OrderPollStatus.FILLED:
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
            # WAL + idempotent close-leg dispatch (narrow PR#30 scope: ROLL close leg only)
            import uuid

            # Best-effort position_id for linkage
            position_id = position_tracker.get_open_position_id_for_symbol(from_symbol) or from_symbol

            # Enforce at most one ACTIVE ROLL_CC intent per position_id (I*):
            try:
                existing_intent_id = _ledger.get_active_intent_id(position_id=str(position_id), intent_type="ROLL_CC")
            except Exception as e:
                # Fail-closed on ledger corruption; do NOT mint a new intent.
                result["status"] = "error"
                result["errors"].append(f"Execution ledger corruption: {e}")
                return result

            intent_id = str(existing_intent_id or params.get("intent_id") or uuid.uuid4().hex)

            # Currency resolver: prefer explicit underlying param, else derive from instrument.
            currency = (params.get("currency") or params.get("underlying") or "")
            if not currency:
                if from_symbol and "-" in from_symbol:
                    currency = from_symbol.split("-")[0]
            currency = str(currency).upper()
            if not currency:
                result["status"] = "error"
                result["errors"].append("Currency resolution failed for roll close")
                return result

            plan = OrderPlan(
                instrument_name=from_symbol,
                side="buy",
                amount=float(size),
                order_type="limit",
                price=float(from_mid),
                post_only=False,
                reduce_only=True,
            )

            attempt = 0

            # I4: do not dispatch again while existence is uncertain (SUBMIT_UNKNOWN)
            # or already acknowledged (ACKED). In those cases return in-flight and let
            # reconcile loop advance the intent.
            latest = _ledger.get_latest_attempt(intent_id=intent_id, leg="CLOSE")
            if latest is not None:
                ds = str(latest.get("dispatch_state") or "")
                if ds in {"ACKED", "SUBMIT_UNKNOWN"}:
                    result["status"] = "in_flight"
                    result["intent_id"] = intent_id
                    result["close_label"] = latest.get("label")
                    result["message"] = f"ROLL close leg in-flight (dispatch_state={ds}); awaiting reconcile"
                    return result

            label, created_now = _ledger.prewrite_attempt(
                intent_id=intent_id,
                position_id=str(position_id),
                intent_type="ROLL_CC",
                currency=currency,
                leg="CLOSE",
                attempt=attempt,
                plan=plan,
            )

            # PREWRITTEN is only dispatchable by the call that created it. If we did not
            # create it now, treat it as in-flight and let reconcile recover by label.
            if not created_now:
                result["status"] = "in_flight"
                result["intent_id"] = intent_id
                result["close_label"] = label
                result["message"] = "ROLL close leg attempt already PREWRITTEN; reconcile before any dispatch"
                return result

            close_oid = None
            try:
                close_result = client.place_order(
                    instrument_name=from_symbol,
                    side="buy",
                    amount=size,
                    order_type="limit",
                    price=from_mid,
                    post_only=bool(plan.post_only),
                    reduce_only=bool(plan.reduce_only),
                    label=label,
                )
                close_oid = close_result.get("order", {}).get("order_id")
                _ledger.commit_dispatch_result(
                    intent_id=intent_id,
                    leg="CLOSE",
                    attempt=attempt,
                    ok=True,
                    order_id=close_oid,
                    error=None,
                )
            except Exception as e:
                _ledger.commit_dispatch_result(
                    intent_id=intent_id,
                    leg="CLOSE",
                    attempt=attempt,
                    ok=False,
                    order_id=None,
                    error=str(e),
                )
                # Do not retry; allow reconcile loop to adopt by label.
                result["status"] = "error"
                result["errors"].append(f"Close leg dispatch failed (SUBMIT_UNKNOWN): {e}")
                result["intent_id"] = intent_id
                result["close_label"] = label
                return result

            if not close_oid:
                # Ack returned without order_id; treat as submit-unknown.
                _ledger.commit_dispatch_result(
                    intent_id=intent_id,
                    leg="CLOSE",
                    attempt=attempt,
                    ok=False,
                    order_id=None,
                    error="Missing order_id in Deribit response",
                )
                result["status"] = "error"
                result["errors"].append("Close leg missing order_id (SUBMIT_UNKNOWN)")
                result["intent_id"] = intent_id
                result["close_label"] = label
                return result

            poll_status, close_state = _poll_order_until_terminal_or_timeout(client, close_oid, timeout_seconds=30.0, poll_seconds=0.5)

            # On timeout: best-effort cancel, then re-fetch to record final fill facts.
            if poll_status in (OrderPollStatus.OPEN_TIMEOUT_PARTIAL, OrderPollStatus.OPEN_TIMEOUT_UNFILLED):
                try:
                    client.cancel_order(close_oid)
                except Exception:
                    pass
                try:
                    st2 = client.get_order_state(close_oid)
                    close_state = _normalize_order_state_payload(st2)
                except Exception:
                    close_state = close_state or {}

            _ledger.update_attempt_from_truth(
                intent_id=intent_id,
                leg="CLOSE",
                attempt=attempt,
                truth={
                    **(close_state or {}),
                    "order_id": close_oid,
                    "instrument_name": from_symbol,
                },
            )

            close_avg = float((close_state or {}).get("average_price") or from_mid)
            close_filled_amt = float((close_state or {}).get("filled_amount") or 0.0)

            if poll_status != OrderPollStatus.FILLED:
                result["status"] = "error"
                result["errors"].append(f"Close leg not filled (status={poll_status}, order_id={close_oid})")
                result["orders"].append({
                    "type": "BUY",
                    "symbol": from_symbol,
                    "size": close_filled_amt or size,
                    "price": close_avg,
                    "order_id": close_oid,
                    "order_state": str((close_state or {}).get("order_state") or "unknown"),
                    "leg": "close",
                    "label": label,
                    "poll_status": poll_status,
                })
                result["intent_id"] = intent_id
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
                "label": label,
            })
            result["intent_id"] = intent_id

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
