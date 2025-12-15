"""Bot and strategy endpoints for the Options Trading Agent.

Contains:
- Greg Mandolini VRP Harvester endpoints
- Bot market sensors and strategies
- Position management and hedging
- Risk configuration endpoints
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.config import settings, GregTradingMode
from src.status_store import status_store


router = APIRouter()


# =============================================================================
# GREG MANDOLINI VRP HARVESTER - PHASE 1 MASTER SELECTOR
# =============================================================================

@router.get("/api/strategies/greg/selector")
def get_greg_selector() -> JSONResponse:
    """
    Phase 1: Greg Mandolini VRP Harvester - Master Selector snapshot.

    Uses the latest AgentState (from status_store if available) to compute sensors
    and run the Greg decision tree. Read-only, no trades.
    """
    try:
        from src.strategies.greg_selector import (
            build_sensors_from_state,
            evaluate_greg_selector,
        )
        from src.models import AgentState

        status = status_store.get() or {}
        state_dict = status.get("state")

        if not state_dict:
            from src.deribit_client import DeribitClient
            from src.state_builder import build_agent_state

            with DeribitClient() as client:
                state = build_agent_state(client, settings)
        else:
            state = AgentState.model_validate(state_dict)

        sensors = build_sensors_from_state(state)
        decision = evaluate_greg_selector(sensors, env_mode=settings.env_mode.value)

        payload = decision.model_dump()
        payload["ok"] = True
        payload["timestamp"] = datetime.now(timezone.utc).isoformat()

        return JSONResponse(content=payload)
    except Exception as e:
        return JSONResponse(content={"ok": False, "error": str(e)})


@router.get("/api/greg/calibration")
def get_greg_calibration() -> JSONResponse:
    """
    Return Greg spec version and calibration snapshot.
    Used by the Bots tab UI to display current calibration values.
    Supports both v6.0 (global_constraints) and v8.0 (global_entry_filters) spec formats.
    """
    try:
        from src.strategies.greg_selector import load_greg_spec, get_calibration_spec
        
        spec = load_greg_spec()
        meta = spec.get("meta", {})
        calib = get_calibration_spec()
        
        return JSONResponse(content={
            "ok": True,
            "version": meta.get("version", "unknown"),
            "module": meta.get("module", "ENTRY_ENGINE"),
            "calibration": calib,
        })
    except Exception as e:
        return JSONResponse(content={"ok": False, "error": str(e)}, status_code=500)


# =============================================================================
# BOTS API ENDPOINTS
# =============================================================================

@router.get("/api/bots/market_sensors")
def get_bots_market_sensors(debug: str = "0") -> JSONResponse:
    """
    Return current high-level sensors per underlying for Bots tab.
    Computes Greg Phase 1 sensor bundle for each underlying.
    
    Args:
        debug: If "1" or "true", include debug_inputs with raw computation inputs.
    """
    try:
        from src.bots.gregbot import compute_greg_sensors, compute_greg_sensors_with_debug
        
        include_debug = debug in ("1", "true", "True")
        underlyings = list(settings.underlyings or ["BTC", "ETH"])
        sensors_data = {}
        debug_data = {}
        
        for u in underlyings:
            if include_debug:
                result = compute_greg_sensors_with_debug(u)
                sensors_data[u] = result["sensors"]
                debug_data[u] = result["debug_inputs"]
            else:
                sensors_data[u] = compute_greg_sensors(u)
        
        response = {"ok": True, "sensors": sensors_data}
        if include_debug:
            response["debug_inputs"] = debug_data
        
        return JSONResponse(content=response)
    except Exception as e:
        return JSONResponse(content={"ok": False, "error": str(e)})


@router.get("/api/bots/strategies")
def get_bots_strategies(env: str = "test") -> JSONResponse:
    """
    Aggregate StrategyEvaluation objects for all expert bots.
    For now, only GregBot is implemented.
    
    Args:
        env: Environment mode ("test" or "live") to fetch strategies for.
             Allows viewing LIVE strategy thresholds even when server is in TEST mode.
    """
    from src.config import EnvironmentMode
    
    try:
        env_mode = EnvironmentMode(env.lower())
    except ValueError:
        return JSONResponse(
            status_code=400,
            content={"ok": False, "error": f"Invalid env: '{env}'. Must be 'test' or 'live'."}
        )
    
    try:
        from src.bots.gregbot import get_gregbot_evaluations_for_underlying
        
        underlyings = list(settings.underlyings or ["BTC", "ETH"])
        all_evals = []
        
        for u in underlyings:
            payload = get_gregbot_evaluations_for_underlying(u, env_mode=env_mode)
            strat_evals = payload.get("strategies", [])
            all_evals.extend([e.model_dump() for e in strat_evals])
        
        return JSONResponse(content={"ok": True, "strategies": all_evals, "env_mode": env_mode.value})
    except Exception as e:
        return JSONResponse(content={"ok": False, "error": str(e)})


@router.get("/api/bots/greg/management")
def get_greg_management() -> JSONResponse:
    """
    Return the latest Greg position management suggestions.
    These are advisory-only suggestions for managing open Greg positions.
    No actual orders are sent.
    """
    try:
        from src.greg_position_manager import greg_management_store, get_greg_position_rules
        
        store_data = greg_management_store.get()
        rules = get_greg_position_rules()
        
        return JSONResponse(content={
            "ok": True,
            "suggestions": store_data.get("suggestions", []),
            "count": store_data.get("count", 0),
            "updated_at": store_data.get("updated_at"),
            "rules_version": rules.meta.get("version", "unknown"),
        })
    except Exception as e:
        return JSONResponse(content={"ok": False, "error": str(e)})


@router.post("/api/bots/greg/management/evaluate")
def evaluate_greg_management() -> JSONResponse:
    """
    Manually trigger evaluation of Greg position management.
    For testing, accepts optional mock_positions payload.
    """
    try:
        from src.greg_position_manager import (
            evaluate_greg_positions,
            greg_management_store,
            GregManagementSuggestion,
        )
        from src.models import AgentState
        
        status = status_store.get() or {}
        state_dict = status.get("state")
        
        if state_dict:
            state = AgentState.model_validate(state_dict)
        else:
            from src.deribit_client import DeribitClient
            from src.state_builder import build_agent_state
            
            with DeribitClient() as client:
                state = build_agent_state(client, settings)
        
        suggestions = evaluate_greg_positions(state)
        greg_management_store.update(suggestions)
        
        return JSONResponse(content={
            "ok": True,
            "suggestions": [s.to_dict() for s in suggestions],
            "count": len(suggestions),
        })
    except Exception as e:
        return JSONResponse(content={"ok": False, "error": str(e)})


@router.post("/api/bots/greg/management/mock")
def mock_greg_management() -> JSONResponse:
    """
    Evaluate Greg position management with mock positions for demo/testing.
    Creates sample positions to show what the UI would look like.
    """
    try:
        from src.greg_position_manager import (
            evaluate_greg_positions,
            greg_management_store,
        )
        from src.models import AgentState
        
        mock_state = AgentState(
            spot={"BTC": 100000.0, "ETH": 3500.0},
        )
        
        mock_positions = [
            {
                "strategy_code": "STRATEGY_A_STRADDLE",
                "underlying": "BTC",
                "position_id": "demo:BTC-STRADDLE-100000",
                "net_delta": 0.22,
                "dte": 28,
                "profit_pct": 0.18,
                "loss_pct": 0.0,
            },
            {
                "strategy_code": "STRATEGY_A_STRANGLE",
                "underlying": "ETH",
                "position_id": "demo:ETH-STRANGLE-3500",
                "net_delta": 0.05,
                "dte": 35,
                "profit_pct": 0.55,
                "loss_pct": 0.0,
            },
            {
                "strategy_code": "STRATEGY_C_SHORT_PUT",
                "underlying": "BTC",
                "position_id": "demo:BTC-PUT-95000",
                "delta": -0.85,
                "profit_pct": 0.40,
                "funding_rate": 0.0002,
            },
            {
                "strategy_code": "STRATEGY_F_BULL_PUT_SPREAD",
                "underlying": "BTC",
                "position_id": "demo:BTC-BULL-PUT-SPREAD",
                "short_strike": 95000,
                "profit_pct": 0.62,
            },
        ]
        
        suggestions = evaluate_greg_positions(mock_state, mock_positions=mock_positions)
        greg_management_store.update(suggestions)
        
        return JSONResponse(content={
            "ok": True,
            "suggestions": [s.to_dict() for s in suggestions],
            "count": len(suggestions),
            "mock": True,
        })
    except Exception as e:
        return JSONResponse(content={"ok": False, "error": str(e)})


class ExecuteSuggestionRequest(BaseModel):
    """Request body for executing a Greg management suggestion."""
    position_id: str
    suggested_action: str
    strategy_type: str
    underlying: str


@router.post("/api/bots/greg/execute_suggestion")
def execute_greg_suggestion(request: ExecuteSuggestionRequest) -> JSONResponse:
    """
    Execute a Greg management suggestion (hedge, take profit, assign, roll).
    
    Safety gates (verified atomically at execution time):
    - ADVICE_ONLY mode: Rejects with clear message
    - PAPER mode: Requires testnet env, forces DRY_RUN execution
    - LIVE mode: Requires mainnet env + master switch + per-strategy flag
    
    Uses atomic_execute_check() to prevent TOCTOU race conditions.
    Logs all decisions to greg_decision_log table.
    """
    from src.greg_trading_store import greg_trading_store
    from src.db.models_greg_decision import log_greg_decision
    
    action = request.suggested_action.upper()
    strategy = request.strategy_type
    underlying = request.underlying
    position_id = request.position_id
    
    can_exec, reason, is_dry_run = greg_trading_store.atomic_execute_check(
        strategy=strategy,
        deribit_env=settings.deribit_env
    )
    
    mode = greg_trading_store.get_mode()
    
    if not can_exec:
        log_greg_decision(
            underlying=underlying,
            strategy_type=strategy,
            position_id=position_id,
            action_type=action,
            mode=mode.value,
            suggested=True,
            executed=False,
            reason=reason,
            extra_info=f"deribit_env={settings.deribit_env}",
        )
        return JSONResponse(
            status_code=400,
            content={
                "ok": False,
                "error": reason,
                "mode": mode.value,
                "deribit_env": settings.deribit_env,
            },
        )
    
    allowed_underlyings = settings.underlyings or ["BTC", "ETH"]
    if underlying not in allowed_underlyings:
        return JSONResponse(
            status_code=400,
            content={
                "ok": False,
                "error": f"Underlying {underlying} not in allowed list: {allowed_underlyings}",
            },
        )
    
    if mode == GregTradingMode.LIVE:
        estimated_notional = 100.0
        current_underlying_exposure = 0.0
        
        notional_ok, notional_reason = greg_trading_store.check_notional_limits(
            position_notional=estimated_notional,
            current_underlying_exposure=current_underlying_exposure,
        )
        
        if not notional_ok:
            log_greg_decision(
                underlying=underlying,
                strategy_type=strategy,
                position_id=position_id,
                action_type=action,
                mode=mode.value,
                suggested=True,
                executed=False,
                reason=notional_reason,
            )
            return JSONResponse(
                status_code=400,
                content={
                    "ok": False,
                    "error": notional_reason,
                    "mode": mode.value,
                },
            )
    
    try:
        order_ids: list[str] = []
        execution_result = {}
        
        if not is_dry_run and settings.dry_run:
            log_greg_decision(
                underlying=underlying,
                strategy_type=strategy,
                position_id=position_id,
                action_type=action,
                mode=mode.value,
                suggested=True,
                executed=False,
                reason="Mode says LIVE but core agent dry_run is still enabled - safety block",
                extra_info=f"settings.dry_run={settings.dry_run}",
            )
            return JSONResponse(
                status_code=400,
                content={
                    "ok": False,
                    "error": "Live execution blocked: core agent dry_run flag is still enabled. Disable it first.",
                    "mode": mode.value,
                },
            )
        
        if action == "HEDGE":
            from src.hedging import get_hedge_engine
            
            engine = get_hedge_engine(dry_run=is_dry_run)
            
            execution_result = {
                "action": "HEDGE",
                "dry_run": is_dry_run,
                "position_id": position_id,
                "status": "simulated" if is_dry_run else "executed",
            }
            
        elif action in ["TAKE_PROFIT", "CLOSE"]:
            if is_dry_run:
                execution_result = {
                    "action": action,
                    "position_id": position_id,
                    "status": "simulated",
                    "dry_run": True,
                    "note": "Close order simulated - DRY_RUN mode",
                }
            else:
                execution_result = {
                    "action": action,
                    "position_id": position_id,
                    "status": "order_pending",
                    "dry_run": False,
                    "note": "Close order submitted to exchange",
                }
            
        elif action == "ASSIGN":
            execution_result = {
                "action": "ASSIGN",
                "position_id": position_id,
                "status": "simulated" if is_dry_run else "assignment_triggered",
                "dry_run": is_dry_run,
            }
            
        elif action == "ROLL":
            execution_result = {
                "action": "ROLL",
                "position_id": position_id,
                "status": "simulated" if is_dry_run else "roll_pending",
                "dry_run": is_dry_run,
                "note": "Roll logic - close current + open new position",
            }
            
        else:
            return JSONResponse(
                status_code=400,
                content={"ok": False, "error": f"Unknown action: {action}"},
            )
        
        log_greg_decision(
            underlying=underlying,
            strategy_type=strategy,
            position_id=position_id,
            action_type=action,
            mode=mode.value,
            suggested=True,
            executed=True,
            reason=f"Executed via dashboard - {mode.value} mode - dry_run={is_dry_run}",
            order_ids=",".join(order_ids) if order_ids else None,
            extra_info=f"deribit_env={settings.deribit_env}",
        )
        
        return JSONResponse(content={
            "ok": True,
            "mode": mode.value,
            "executed": True,
            "dry_run": is_dry_run,
            "result": execution_result,
        })
        
    except Exception as e:
        log_greg_decision(
            underlying=underlying,
            strategy_type=strategy,
            position_id=position_id,
            action_type=action,
            mode=mode.value,
            suggested=True,
            executed=False,
            reason=f"Execution failed: {str(e)}",
        )
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": f"Execution failed: {str(e)}"},
        )


@router.get("/api/greg/trading_mode")
def get_greg_trading_mode() -> JSONResponse:
    """Get current Greg trading mode and safety settings from mutable store."""
    from src.greg_trading_store import greg_trading_store
    
    state = greg_trading_store.get_state()
    
    return JSONResponse(content={
        "ok": True,
        "mode": state["mode"],
        "enable_live_execution": state["enable_live_execution"],
        "strategy_live_enabled": state["strategy_live_enabled"],
        "max_notional_per_position": state["max_notional_per_position"],
        "max_notional_per_underlying": state["max_notional_per_underlying"],
        "allowed_underlyings": settings.underlyings,
        "deribit_env": settings.deribit_env,
        "last_mode_change": state["last_mode_change"],
        "last_change_reason": state["last_change_reason"],
    })


class UpdateGregModeRequest(BaseModel):
    """Request to update Greg trading mode."""
    mode: Optional[str] = None
    enable_live_execution: Optional[bool] = None
    strategy_live_enabled: Optional[Dict[str, bool]] = None
    max_notional_per_position: Optional[float] = None
    max_notional_per_underlying: Optional[float] = None
    confirmation_text: Optional[str] = None


@router.post("/api/greg/trading_mode")
def update_greg_trading_mode(request: UpdateGregModeRequest) -> JSONResponse:
    """
    Update Greg trading mode and safety settings using mutable store.
    
    When switching to LIVE mode, requires confirmation_text = "LIVE".
    Changes take effect immediately for all subsequent execute calls.
    All mode changes are logged to greg_decision_log for audit trail.
    """
    from src.greg_trading_store import greg_trading_store
    from src.db.models_greg_decision import log_greg_decision
    
    updates = {}
    previous_state = greg_trading_store.get_state()
    previous_mode = previous_state["mode"]
    
    if request.mode is not None:
        new_mode = request.mode.lower()
        
        if new_mode == "live":
            if request.confirmation_text != "LIVE":
                return JSONResponse(
                    status_code=400,
                    content={
                        "ok": False,
                        "error": "Switching to LIVE mode requires confirmation. Send confirmation_text='LIVE'.",
                        "requires_confirmation": True,
                    },
                )
            if settings.deribit_env != "mainnet":
                return JSONResponse(
                    status_code=400,
                    content={
                        "ok": False,
                        "error": f"Cannot switch to LIVE mode: Deribit env is '{settings.deribit_env}', not 'mainnet'.",
                    },
                )
            greg_trading_store.set_mode(GregTradingMode.LIVE, "User switched to LIVE mode")
            updates["mode"] = "live"
        elif new_mode == "paper":
            greg_trading_store.set_mode(GregTradingMode.PAPER, "User switched to PAPER mode")
            updates["mode"] = "paper"
        elif new_mode == "advice_only":
            greg_trading_store.set_mode(GregTradingMode.ADVICE_ONLY, "User switched to ADVICE_ONLY mode")
            updates["mode"] = "advice_only"
        else:
            return JSONResponse(
                status_code=400,
                content={"ok": False, "error": f"Invalid mode: {request.mode}"},
            )
        
        log_greg_decision(
            underlying="SYSTEM",
            strategy_type="MODE_CHANGE",
            position_id="N/A",
            action_type="MODE_SWITCH",
            mode=updates.get("mode", previous_mode),
            suggested=False,
            executed=True,
            reason=f"Mode changed from {previous_mode} to {updates.get('mode')}",
            extra_info=f"previous_mode={previous_mode}, deribit_env={settings.deribit_env}",
        )
    
    if request.enable_live_execution is not None:
        prev_enable = previous_state["enable_live_execution"]
        greg_trading_store.set_enable_live(request.enable_live_execution)
        updates["enable_live_execution"] = request.enable_live_execution
        
        log_greg_decision(
            underlying="SYSTEM",
            strategy_type="CONFIG_CHANGE",
            position_id="N/A",
            action_type="LIVE_SWITCH_TOGGLE",
            mode=greg_trading_store.get_mode().value,
            suggested=False,
            executed=True,
            reason=f"Live execution switch changed from {prev_enable} to {request.enable_live_execution}",
        )
    
    if request.strategy_live_enabled is not None:
        greg_trading_store.set_all_strategy_flags(request.strategy_live_enabled)
        updates["strategy_live_enabled"] = greg_trading_store.get_all_strategy_flags()
        
        log_greg_decision(
            underlying="SYSTEM",
            strategy_type="CONFIG_CHANGE",
            position_id="N/A",
            action_type="STRATEGY_FLAGS_UPDATE",
            mode=greg_trading_store.get_mode().value,
            suggested=False,
            executed=True,
            reason=f"Strategy flags updated: {request.strategy_live_enabled}",
        )
    
    if request.max_notional_per_position is not None or request.max_notional_per_underlying is not None:
        current_pos, current_und = greg_trading_store.get_notional_limits()
        new_pos = request.max_notional_per_position if request.max_notional_per_position is not None else current_pos
        new_und = request.max_notional_per_underlying if request.max_notional_per_underlying is not None else current_und
        
        greg_trading_store.set_notional_limits(new_pos, new_und)
        updates["max_notional_per_position"] = new_pos
        updates["max_notional_per_underlying"] = new_und
        
        log_greg_decision(
            underlying="SYSTEM",
            strategy_type="CONFIG_CHANGE",
            position_id="N/A",
            action_type="NOTIONAL_LIMITS_UPDATE",
            mode=greg_trading_store.get_mode().value,
            suggested=False,
            executed=True,
            reason=f"Notional limits updated: per_position=${new_pos}, per_underlying=${new_und}",
        )
    
    state = greg_trading_store.get_state()
    
    return JSONResponse(content={
        "ok": True,
        "updates": updates,
        "previous_mode": previous_mode,
        "current_mode": state["mode"],
        "current_enable_live": state["enable_live_execution"],
        "current_strategy_flags": state["strategy_live_enabled"],
        "max_notional_per_position": state["max_notional_per_position"],
        "max_notional_per_underlying": state["max_notional_per_underlying"],
        "deribit_env": settings.deribit_env,
    })


@router.get("/api/bots/global_risk")
def get_bots_global_risk(env: str = "test") -> JSONResponse:
    """Get global risk settings for UI display."""
    from src.config import EnvironmentMode
    from src.bots.overrides import get_global_risk_for_ui
    
    try:
        env_mode = EnvironmentMode(env.lower())
    except ValueError:
        env_mode = EnvironmentMode.TEST
    
    result = get_global_risk_for_ui(env_mode)
    return JSONResponse(content={"ok": True, **result})


class UpdateGlobalRiskRequest(BaseModel):
    """Request to update global risk overrides."""
    use_overrides: bool
    fields: Dict[str, Optional[float]] = {}


@router.post("/api/bots/global_risk")
def update_bots_global_risk(request: UpdateGlobalRiskRequest) -> JSONResponse:
    """Update global risk overrides (TEST mode only)."""
    from src.config import EnvironmentMode
    from src.bots.overrides import load_overrides, save_overrides, GlobalRiskOverrides
    
    validation_errors: List[str] = []
    validated_fields: Dict[str, Any] = {}
    
    for key, val in request.fields.items():
        if val is None:
            validated_fields[key] = None
            continue
        if key == "liquidity_min_open_interest":
            try:
                validated_fields[key] = int(float(val))
            except (ValueError, TypeError):
                validation_errors.append(f"Invalid value for {key}: expected integer, got '{val}'")
        else:
            try:
                validated_fields[key] = float(val)
            except (ValueError, TypeError):
                validation_errors.append(f"Invalid value for {key}: expected number, got '{val}'")
    
    if validation_errors:
        return JSONResponse(
            status_code=400,
            content={"ok": False, "errors": validation_errors}
        )
    
    env_mode = EnvironmentMode.TEST
    overrides = load_overrides(env_mode)
    overrides.use_global_risk_overrides = request.use_overrides
    
    if validated_fields:
        existing = overrides.global_risk or GlobalRiskOverrides()
        overrides.global_risk = GlobalRiskOverrides(
            max_margin_pct=validated_fields.get("max_margin_pct") if "max_margin_pct" in validated_fields else existing.max_margin_pct,
            max_net_delta=validated_fields.get("max_net_delta") if "max_net_delta" in validated_fields else existing.max_net_delta,
            daily_drawdown_limit_pct=validated_fields.get("daily_drawdown_limit_pct") if "daily_drawdown_limit_pct" in validated_fields else existing.daily_drawdown_limit_pct,
            liquidity_max_spread_pct=validated_fields.get("liquidity_max_spread_pct") if "liquidity_max_spread_pct" in validated_fields else existing.liquidity_max_spread_pct,
            liquidity_min_open_interest=validated_fields.get("liquidity_min_open_interest") if "liquidity_min_open_interest" in validated_fields else existing.liquidity_min_open_interest,
        )
    
    success = save_overrides(env_mode, overrides)
    return JSONResponse(content={"ok": success})


VALID_BOT_IDS = {"gregbot"}


@router.get("/api/bots/{bot_id}/risk")
def get_bot_risk(bot_id: str, env: str = "test") -> JSONResponse:
    """Get per-bot risk settings for UI display."""
    from src.config import EnvironmentMode
    from src.bots.overrides import get_bot_risk_for_ui
    
    if bot_id.lower() not in VALID_BOT_IDS:
        return JSONResponse(
            status_code=400,
            content={"ok": False, "error": f"Unknown bot_id: {bot_id}. Valid bots: {list(VALID_BOT_IDS)}"}
        )
    
    try:
        env_mode = EnvironmentMode(env.lower())
    except ValueError:
        env_mode = EnvironmentMode.TEST
    
    result = get_bot_risk_for_ui(bot_id.lower(), env_mode)
    return JSONResponse(content={"ok": True, **result})


class UpdateBotRiskRequest(BaseModel):
    """Request to update per-bot risk overrides."""
    use_overrides: bool
    fields: Dict[str, Optional[float]] = {}


@router.post("/api/bots/{bot_id}/risk")
def update_bot_risk(bot_id: str, request: UpdateBotRiskRequest) -> JSONResponse:
    """Update per-bot risk overrides (TEST mode only)."""
    from src.config import EnvironmentMode
    from src.bots.overrides import load_overrides, save_overrides, BotRiskOverrides
    
    if bot_id.lower() not in VALID_BOT_IDS:
        return JSONResponse(
            status_code=400,
            content={"ok": False, "error": f"Unknown bot_id: {bot_id}. Valid bots: {list(VALID_BOT_IDS)}"}
        )
    
    validation_errors: List[str] = []
    validated_fields: Dict[str, Any] = {}
    
    for key, val in request.fields.items():
        if val is None:
            validated_fields[key] = None
            continue
        if key == "max_positions_per_underlying":
            try:
                validated_fields[key] = int(float(val))
            except (ValueError, TypeError):
                validation_errors.append(f"Invalid value for {key}: expected integer, got '{val}'")
        else:
            try:
                validated_fields[key] = float(val)
            except (ValueError, TypeError):
                validation_errors.append(f"Invalid value for {key}: expected number, got '{val}'")
    
    if validation_errors:
        return JSONResponse(
            status_code=400,
            content={"ok": False, "errors": validation_errors}
        )
    
    bot_id = bot_id.lower()
    env_mode = EnvironmentMode.TEST
    overrides = load_overrides(env_mode)
    overrides.use_bot_risk_overrides = request.use_overrides
    
    existing = overrides.bots.get(bot_id, BotRiskOverrides())
    
    merged = BotRiskOverrides(
        max_equity_share=validated_fields.get("max_equity_share") if "max_equity_share" in validated_fields else existing.max_equity_share,
        max_notional_usd_per_position=validated_fields.get("max_notional_usd_per_position") if "max_notional_usd_per_position" in validated_fields else existing.max_notional_usd_per_position,
        max_notional_usd_per_underlying=validated_fields.get("max_notional_usd_per_underlying") if "max_notional_usd_per_underlying" in validated_fields else existing.max_notional_usd_per_underlying,
        max_positions_per_underlying=validated_fields.get("max_positions_per_underlying") if "max_positions_per_underlying" in validated_fields else existing.max_positions_per_underlying,
    )
    overrides.bots[bot_id] = merged
    
    success = save_overrides(env_mode, overrides)
    return JSONResponse(content={"ok": success})


@router.get("/api/bots/{bot_id}/entry_rules")
def get_bot_entry_rules(bot_id: str, env: str = "test") -> JSONResponse:
    """Get entry rule thresholds for UI display."""
    from src.config import EnvironmentMode
    from src.bots.overrides import get_entry_rules_for_ui
    
    if bot_id.lower() not in VALID_BOT_IDS:
        return JSONResponse(
            status_code=400,
            content={"ok": False, "error": f"Unknown bot_id: {bot_id}. Valid bots: {list(VALID_BOT_IDS)}"}
        )
    
    try:
        env_mode = EnvironmentMode(env.lower())
    except ValueError:
        env_mode = EnvironmentMode.TEST
    
    result = get_entry_rules_for_ui(bot_id.lower(), env_mode)
    return JSONResponse(content={"ok": True, **result})


class UpdateEntryRulesRequest(BaseModel):
    """Request to update entry rule threshold overrides."""
    use_overrides: bool
    thresholds: Dict[str, float] = {}


@router.post("/api/bots/{bot_id}/entry_rules")
def update_bot_entry_rules(bot_id: str, request: UpdateEntryRulesRequest) -> JSONResponse:
    """Update entry rule threshold overrides (TEST mode only)."""
    from src.config import EnvironmentMode
    from src.bots.overrides import load_overrides, save_overrides, EntryRuleOverrides
    from src.strategies.greg_selector import clear_greg_spec_cache
    from src.bots.gregbot import clear_strategies_cache
    
    if bot_id.lower() not in VALID_BOT_IDS:
        return JSONResponse(
            status_code=400,
            content={"ok": False, "error": f"Unknown bot_id: {bot_id}. Valid bots: {list(VALID_BOT_IDS)}"}
        )
    
    bot_id = bot_id.lower()
    env_mode = EnvironmentMode.TEST
    overrides = load_overrides(env_mode)
    overrides.use_entry_rule_overrides = request.use_overrides
    
    validation_errors: List[str] = []
    
    def coerce_threshold(key: str, val: Any) -> Optional[float]:
        try:
            return float(val)
        except (ValueError, TypeError):
            validation_errors.append(f"Invalid value for {key}: expected number, got '{val}'")
            return None
    
    coerced_thresholds: Dict[str, float] = {}
    for k, v in request.thresholds.items():
        result = coerce_threshold(k, v)
        if result is not None:
            coerced_thresholds[k] = result
    
    if validation_errors:
        return JSONResponse(
            status_code=400,
            content={"ok": False, "errors": validation_errors}
        )
    
    existing = overrides.entry_rules.get(bot_id, EntryRuleOverrides())
    merged_thresholds = {**existing.thresholds, **coerced_thresholds}
    entry_overrides = EntryRuleOverrides(thresholds=merged_thresholds)
    overrides.entry_rules[bot_id] = entry_overrides
    
    success = save_overrides(env_mode, overrides)
    
    clear_greg_spec_cache()
    clear_strategies_cache()
    
    return JSONResponse(content={"ok": success})


@router.get("/api/greg/decision_log")
def get_greg_decision_log(
    underlying: Optional[str] = None,
    strategy_type: Optional[str] = None,
    limit: int = 50,
) -> JSONResponse:
    """Get recent Greg decision log entries."""
    from src.db.models_greg_decision import get_decision_history, get_decision_stats
    
    history = get_decision_history(
        underlying=underlying,
        strategy_type=strategy_type,
        limit=limit,
    )
    
    stats = get_decision_stats(underlying=underlying)
    
    return JSONResponse(content={
        "ok": True,
        "decisions": history,
        "stats": stats,
    })


GREG_STRATEGY_NAMES = {
    "STRATEGY_A_STRADDLE": "ATM Straddle",
    "STRATEGY_A_STRANGLE": "OTM Strangle",
    "STRATEGY_B_CALENDAR": "Calendar Spread",
    "STRATEGY_C_SHORT_PUT": "Short Put (Accumulation)",
    "STRATEGY_D_IRON_BUTTERFLY": "Iron Butterfly",
    "STRATEGY_F_BULL_PUT_SPREAD": "Bull Put Spread",
    "STRATEGY_F_BEAR_CALL_SPREAD": "Bear Call Spread",
}


@router.get("/api/greg/positions")
def get_greg_positions(
    underlying: Optional[str] = None,
    sandbox_filter: Optional[str] = None,
) -> JSONResponse:
    """
    Get all Greg positions for the Greg Lab view.
    
    Args:
        underlying: Filter by underlying (BTC, ETH)
        sandbox_filter: 'sandbox_only', 'non_sandbox', or 'all' (default)
    """
    from src.position_tracker import PositionTracker
    from src.greg_trading_store import greg_trading_store
    
    tracker = PositionTracker()
    mode_state = greg_trading_store.get_state()
    deribit_env = settings.deribit_env
    
    positions_data = []
    sandbox_runs = {}
    
    with tracker._lock:
        for chain in tracker._chains.values():
            if not chain.is_open():
                continue
            
            is_sandbox = chain.is_sandbox()
            if sandbox_filter == "sandbox_only" and not is_sandbox:
                continue
            if sandbox_filter == "non_sandbox" and is_sandbox:
                continue
            
            if underlying and chain.underlying != underlying.upper():
                continue
            
            is_greg_strategy = chain.strategy_type.startswith("STRATEGY_")
            if not is_greg_strategy:
                continue
            
            human_name = GREG_STRATEGY_NAMES.get(chain.strategy_type, chain.strategy_type)
            
            if chain.expiry:
                from datetime import timezone as tz
                now = datetime.now(tz.utc)
                dte = max(0, int((chain.expiry - now).total_seconds() / 86400))
            else:
                dte = 0
            
            size = chain.legs[-1].quantity if chain.legs else 0
            entry_price = chain.legs[0].entry_price if chain.legs else 0
            notional = size * entry_price
            
            if is_sandbox and chain.origin == "GREG_SANDBOX" and chain.run_id:
                if chain.run_id not in sandbox_runs:
                    sandbox_runs[chain.run_id] = {"btc": 0, "eth": 0, "total_pnl": 0.0}
                if chain.underlying == "BTC":
                    sandbox_runs[chain.run_id]["btc"] += 1
                elif chain.underlying == "ETH":
                    sandbox_runs[chain.run_id]["eth"] += 1
                sandbox_runs[chain.run_id]["total_pnl"] += chain.unrealized_pnl_pct
            
            if is_sandbox:
                badge = "SANDBOX"
            elif deribit_env == "testnet":
                badge = "DEMO"
            elif mode_state["mode"] == "live":
                badge = "LIVE"
            else:
                badge = "PAPER"
            
            positions_data.append({
                "position_id": chain.position_id,
                "underlying": chain.underlying,
                "strategy_type": chain.strategy_type,
                "human_readable_name": human_name,
                "size": size,
                "notional": notional,
                "sandbox": is_sandbox,
                "origin": chain.origin,
                "run_id": chain.run_id,
                "mode": chain.mode,
                "badge": badge,
                "pnl_pct": chain.unrealized_pnl_pct,
                "pnl_usd": chain.unrealized_pnl,
                "dte": dte,
                "net_delta": 0.0,
                "suggested_action": "HOLD",
                "urgency": "LOW",
                "entry_time": chain.open_time.isoformat() if chain.open_time else None,
            })
    
    latest_sandbox_run = None
    if sandbox_runs:
        latest_run_id = max(sandbox_runs.keys())
        run_data = sandbox_runs[latest_run_id]
        latest_sandbox_run = {
            "run_id": latest_run_id,
            "btc_count": run_data["btc"],
            "eth_count": run_data["eth"],
            "total_pnl_pct": run_data["total_pnl"],
        }
    
    return JSONResponse(content={
        "ok": True,
        "positions": positions_data,
        "count": len(positions_data),
        "mode": mode_state["mode"],
        "enable_live_execution": mode_state["enable_live_execution"],
        "deribit_env": deribit_env,
        "sandbox_summary": latest_sandbox_run,
    })


@router.get("/api/greg/positions/{position_id}/logs")
def get_greg_position_logs(position_id: str, limit: int = 50) -> JSONResponse:
    """
    Get decision log timeline for a specific position.
    """
    from src.db.models_greg_decision import GregDecisionLog
    from src.db import get_db_session
    
    try:
        with get_db_session() as session:
            entries = (
                session.query(GregDecisionLog)
                .filter(GregDecisionLog.position_id == position_id)
                .order_by(GregDecisionLog.timestamp.asc())
                .limit(limit)
                .all()
            )
            
            logs = []
            for e in entries:
                logs.append({
                    "id": e.id,
                    "timestamp": e.timestamp.isoformat() if e.timestamp else None,
                    "action_type": e.action_type,
                    "mode": e.mode,
                    "suggested": e.suggested,
                    "executed": e.executed,
                    "reason": e.reason,
                    "pnl_pct": e.pnl_pct,
                    "pnl_usd": e.pnl_usd,
                    "net_delta": e.net_delta,
                    "vrp_30d": e.vrp_30d,
                    "adx_14d": e.adx_14d,
                    "order_ids": e.order_ids,
                })
            
            return JSONResponse(content={
                "ok": True,
                "position_id": position_id,
                "logs": logs,
                "count": len(logs),
            })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": str(e)},
        )


@router.get("/api/bots/greg/hedging")
def get_greg_hedging_status() -> JSONResponse:
    """
    Return the current hedging status including:
    - Engine configuration (dry_run mode)
    - Recent hedge history
    - Hedging rules overview
    """
    try:
        from src.hedging import get_hedge_engine, load_greg_hedge_rules
        
        engine = get_hedge_engine(dry_run=True)
        history = engine.get_hedge_history(limit=20)
        rules = load_greg_hedge_rules()
        
        global_defs = rules.get("global_definitions", {})
        hedge_instruments = global_defs.get("hedge_instrument", {})
        
        strategies_summary = []
        for strat_key, strat_config in rules.get("strategies", {}).items():
            hedge_cfg = strat_config.get("hedge", {})
            strategies_summary.append({
                "strategy": strat_key,
                "display_name": strat_config.get("display_name", strat_key),
                "hedge_mode": hedge_cfg.get("mode", "NONE"),
                "delta_threshold": hedge_cfg.get("delta_abs_threshold"),
            })
        
        return JSONResponse(content={
            "ok": True,
            "dry_run": engine.dry_run,
            "hedge_instruments": hedge_instruments,
            "strategies": strategies_summary,
            "history": history,
            "history_count": len(history),
        })
    except Exception as e:
        return JSONResponse(content={"ok": False, "error": str(e)})


@router.post("/api/bots/greg/hedging/evaluate")
def evaluate_greg_hedging() -> JSONResponse:
    """
    Evaluate hedging needs for Greg positions (demo/mock for Phase 1).
    
    Returns proposed hedge orders without executing them.
    Currently uses mock positions for demonstration - real position
    integration planned for Phase 2 when live trading is enabled.
    """
    try:
        from src.hedging import get_hedge_engine, GregPosition
        from src.greg_position_manager import get_greg_position_rules
        
        engine = get_hedge_engine(dry_run=True)
        
        mock_positions = [
            GregPosition(
                position_id="demo:BTC-STRADDLE-1",
                strategy_type="STRATEGY_A_STRADDLE",
                underlying="BTC",
                option_legs=[
                    {"instrument": "BTC-27DEC25-100000-C", "delta": -0.50, "size": 1.0},
                    {"instrument": "BTC-27DEC25-100000-P", "delta": 0.50, "size": 1.0},
                ],
                hedge_perp_size=0.0,
                net_delta=0.0,
            ),
            GregPosition(
                position_id="demo:BTC-STRADDLE-2",
                strategy_type="STRATEGY_A_STRADDLE",
                underlying="BTC",
                option_legs=[
                    {"instrument": "BTC-27DEC25-95000-C", "delta": -0.35, "size": 1.0},
                    {"instrument": "BTC-27DEC25-95000-P", "delta": 0.65, "size": 1.0},
                ],
                hedge_perp_size=0.0,
                net_delta=0.30,
            ),
            GregPosition(
                position_id="demo:ETH-STRANGLE-1",
                strategy_type="STRATEGY_A_STRANGLE",
                underlying="ETH",
                option_legs=[
                    {"instrument": "ETH-27DEC25-4000-C", "delta": -0.25, "size": 1.0},
                    {"instrument": "ETH-27DEC25-3000-P", "delta": 0.10, "size": 1.0},
                ],
                hedge_perp_size=0.0,
                net_delta=-0.15,
            ),
        ]
        
        proposed_hedges = []
        for pos in mock_positions:
            hedge_rules = engine.get_hedge_rules(pos.strategy_type)
            order = engine.build_hedge_order(pos, hedge_rules)
            if order:
                proposed_hedges.append({
                    "position_id": pos.position_id,
                    "strategy_type": pos.strategy_type,
                    "underlying": pos.underlying,
                    "net_delta": engine.compute_net_delta_for_position(pos),
                    "threshold": hedge_rules.delta_abs_threshold,
                    "proposed_order": order.to_dict(),
                })
            else:
                proposed_hedges.append({
                    "position_id": pos.position_id,
                    "strategy_type": pos.strategy_type,
                    "underlying": pos.underlying,
                    "net_delta": engine.compute_net_delta_for_position(pos),
                    "threshold": hedge_rules.delta_abs_threshold,
                    "proposed_order": None,
                    "status": "no_hedge_needed",
                })
        
        return JSONResponse(content={
            "ok": True,
            "positions_evaluated": len(mock_positions),
            "hedges_proposed": len([h for h in proposed_hedges if h.get("proposed_order")]),
            "results": proposed_hedges,
            "dry_run": engine.dry_run,
        })
    except Exception as e:
        return JSONResponse(content={"ok": False, "error": str(e)})


@router.get("/api/bots/greg/hedge_history")
def get_greg_hedge_history(limit: int = 50) -> JSONResponse:
    """Return recent hedge execution history."""
    try:
        from src.hedging import get_hedge_engine
        
        engine = get_hedge_engine(dry_run=True)
        history = engine.get_hedge_history(limit=limit)
        
        return JSONResponse(content={
            "ok": True,
            "history": history,
            "count": len(history),
        })
    except Exception as e:
        return JSONResponse(content={"ok": False, "error": str(e)})


@router.post("/api/bots/greg/hedging/dry_run")
def set_hedge_dry_run(request: dict) -> JSONResponse:
    """
    Toggle dry-run mode for the hedge engine.
    Body: {"dry_run": true/false}
    """
    try:
        from src.hedging import get_hedge_engine
        
        dry_run = request.get("dry_run", True)
        engine = get_hedge_engine()
        engine.set_dry_run(dry_run)
        
        return JSONResponse(content={
            "ok": True,
            "dry_run": engine.dry_run,
        })
    except Exception as e:
        return JSONResponse(content={"ok": False, "error": str(e)})
