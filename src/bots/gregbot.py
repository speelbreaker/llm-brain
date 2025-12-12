"""
GregBot - Greg Mandolini VRP Harvester implementation (ENTRY_ENGINE v8.0).

Produces StrategyEvaluation objects for each strategy based on current sensors.
Based on the JSON spec in docs/greg_mandolini/GREG_SELECTOR_RULES_FINAL.json
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from src.bots.types import StrategyCriterion, StrategyEvaluation
from src.bots.sensors import SensorBundle, compute_sensors_for_underlying
from src.config import settings, EnvironmentMode
from src.strategies.greg_selector import (
    GregSelectorSensors,
    load_greg_spec,
    evaluate_greg_selector,
    get_calibration_spec,
    get_calibration_spec_with_overrides,
)


def _get_calibration_value(key: str, default: float, env_mode: str | None = None) -> float:
    """
    Get a calibration value from the Greg spec, with TEST environment overrides applied.
    
    Args:
        key: Calibration key (e.g., "straddle_vrp_min" or "rsi_thresholds.lower")
        default: Fallback value if key not found
        env_mode: Environment mode ("test" or "live"). Uses settings.env_mode if None.
    
    Returns:
        Calibration value with any TEST overrides applied.
    """
    cal = get_calibration_spec_with_overrides(env_mode)
    if "." in key:
        parts = key.split(".")
        val = cal
        for part in parts:
            if isinstance(val, dict):
                val = val.get(part)
            else:
                return default
        return float(val) if isinstance(val, (int, float)) else default
    return float(cal.get(key, default))


_STRATEGIES_CACHE: Dict[str, List[Dict[str, Any]]] = {}


def _build_greg_strategies(env_mode: str | None = None) -> List[Dict[str, Any]]:
    """
    Build strategy definitions dynamically from the calibration spec.
    Thresholds are pulled from the JSON rather than hardcoded.
    
    Args:
        env_mode: Environment mode ("test" or "live"). Uses settings.env_mode if None.
                  In TEST mode, applies any overrides from bot_overrides_test.json.
    """
    skew_thresh = _get_calibration_value("skew_neutral_threshold", 4.0, env_mode)
    vrp_min = _get_calibration_value("min_vrp_floor", 0.0, env_mode)
    vrp_directional = _get_calibration_value("min_vrp_directional", 2.0, env_mode)
    rsi_lower = _get_calibration_value("rsi_thresholds.lower", 30.0, env_mode)
    rsi_upper = _get_calibration_value("rsi_thresholds.upper", 70.0, env_mode)
    
    straddle_vrp = _get_calibration_value("straddle_vrp_min", 15.0, env_mode)
    straddle_adx = _get_calibration_value("straddle_adx_max", 20.0, env_mode)
    straddle_chop = _get_calibration_value("straddle_chop_max", 0.6, env_mode)
    
    strangle_vrp = _get_calibration_value("strangle_vrp_min", 10.0, env_mode)
    strangle_adx = _get_calibration_value("strangle_adx_max", 30.0, env_mode)
    strangle_chop = _get_calibration_value("strangle_chop_max", 0.8, env_mode)
    
    calendar_term = _get_calibration_value("calendar_term_spread_min", 5.0, env_mode)
    calendar_rv_iv = _get_calibration_value("calendar_front_rv_iv_ratio_max", 0.8, env_mode)
    calendar_vrp = _get_calibration_value("calendar_vrp_7d_min", 5.0, env_mode)
    
    iron_fly_iv_rank = _get_calibration_value("iron_fly_iv_rank_min", 0.80, env_mode)
    iron_fly_vrp = _get_calibration_value("iron_fly_vrp_min", 10.0, env_mode)
    
    safety_adx = _get_calibration_value("safety_adx_high", 35.0, env_mode)
    safety_chop = _get_calibration_value("safety_chop_high", 0.85, env_mode)
    
    return [
        {
            "key": "STRATEGY_A_STRADDLE",
            "label": "Strategy A: ATM Straddle",
            "description": "High VRP, Low Movement, Neutral Skew",
            "criteria_defs": [
                {"metric": "vrp_30d", "min": straddle_vrp, "description": f"VRP > {straddle_vrp}"},
                {"metric": "chop_factor_7d", "max": straddle_chop, "description": f"Chop < {straddle_chop}"},
                {"metric": "adx_14d", "max": straddle_adx, "description": f"ADX < {straddle_adx}"},
                {"metric": "skew_25d", "abs_max": skew_thresh, "description": f"|Skew| < {skew_thresh}"},
            ],
        },
        {
            "key": "STRATEGY_A_STRANGLE",
            "label": "Strategy A: OTM Strangle",
            "description": "Good VRP, Market Drifting, Neutral Skew",
            "criteria_defs": [
                {"metric": "vrp_30d", "min": strangle_vrp, "description": f"VRP >= {strangle_vrp}"},
                {"metric": "chop_factor_7d", "max": strangle_chop, "description": f"Chop < {strangle_chop}"},
                {"metric": "adx_14d", "max": strangle_adx, "description": f"ADX < {strangle_adx}"},
                {"metric": "skew_25d", "abs_max": skew_thresh, "description": f"|Skew| < {skew_thresh}"},
            ],
        },
        {
            "key": "STRATEGY_B_CALENDAR",
            "label": "Strategy B: Calendar Spread",
            "description": "Term Structure Play",
            "criteria_defs": [
                {"metric": "term_structure_spread", "min": calendar_term, "description": f"Term spread > {calendar_term}"},
                {"metric": "front_rv_iv_ratio", "max": calendar_rv_iv, "description": f"Front RV/IV < {calendar_rv_iv}"},
                {"metric": "vrp_7d", "min": calendar_vrp, "description": f"VRP 7d > {calendar_vrp}"},
            ],
        },
        {
            "key": "STRATEGY_C_SHORT_PUT",
            "label": "Strategy C: Short Put",
            "description": "Bullish Accumulation",
            "criteria_defs": [
                {"metric": "skew_25d", "min": skew_thresh, "description": f"Skew > {skew_thresh} (puts expensive)"},
                {"metric": "price_vs_ma200", "min": 0, "description": "Price > MA200"},
                {"metric": "vrp_30d", "min": vrp_directional, "description": f"VRP > {vrp_directional}"},
            ],
        },
        {
            "key": "STRATEGY_D_IRON_BUTTERFLY",
            "label": "Strategy D: Iron Butterfly",
            "description": "Defined Risk, High Vol",
            "criteria_defs": [
                {"metric": "iv_rank_6m", "min": iron_fly_iv_rank, "description": f"IV Rank > {iron_fly_iv_rank*100:.0f}%"},
                {"metric": "vrp_30d", "min": iron_fly_vrp, "description": f"VRP > {iron_fly_vrp}"},
            ],
        },
        {
            "key": "STRATEGY_F_BULL_PUT_SPREAD",
            "label": "Strategy F: Bull Put Spread",
            "description": "Oversold + Fear Skew",
            "criteria_defs": [
                {"metric": "skew_25d", "min": skew_thresh, "description": f"Skew > {skew_thresh} (puts expensive)"},
                {"metric": "rsi_14d", "max": rsi_lower, "description": f"RSI < {rsi_lower} (oversold)"},
                {"metric": "vrp_30d", "min": vrp_directional, "description": f"VRP > {vrp_directional}"},
            ],
        },
        {
            "key": "STRATEGY_F_BEAR_CALL_SPREAD",
            "label": "Strategy F: Bear Call Spread",
            "description": "Overbought + FOMO Skew",
            "criteria_defs": [
                {"metric": "skew_25d", "max": -skew_thresh, "description": f"Skew < -{skew_thresh} (calls expensive)"},
                {"metric": "rsi_14d", "min": rsi_upper, "description": f"RSI > {rsi_upper} (overbought)"},
                {"metric": "vrp_30d", "min": vrp_directional, "description": f"VRP > {vrp_directional}"},
            ],
        },
        {
            "key": "NO_TRADE",
            "label": "No Trade",
            "description": "Conditions Unfavorable or Safety Filter Triggered",
            "criteria_defs": [
                {"metric": "adx_14d", "max": safety_adx, "description": f"ADX <= {safety_adx} (safety)"},
                {"metric": "chop_factor_7d", "max": safety_chop, "description": f"Chop <= {safety_chop} (safety)"},
            ],
        },
    ]


def get_greg_strategies(env_mode: str | None = None) -> List[Dict[str, Any]]:
    """
    Get strategy definitions for the given environment mode.
    
    Uses caching to avoid rebuilding on every call, but cache can be cleared
    when overrides change via clear_strategies_cache().
    
    Args:
        env_mode: Environment mode ("test" or "live"). Uses settings.env_mode if None.
    
    Returns:
        List of strategy definition dicts with calibration values applied.
    """
    cache_key = env_mode or "default"
    if cache_key not in _STRATEGIES_CACHE:
        _STRATEGIES_CACHE[cache_key] = _build_greg_strategies(env_mode)
    return _STRATEGIES_CACHE[cache_key]


def clear_strategies_cache() -> None:
    """Clear the strategies cache so it rebuilds with fresh calibration values."""
    _STRATEGIES_CACHE.clear()


def _evaluate_criterion(
    metric: str,
    value: Optional[float],
    min_val: Optional[float],
    max_val: Optional[float],
    abs_max: Optional[float] = None,
) -> StrategyCriterion:
    """Evaluate a single criterion and return a StrategyCriterion object."""
    note = None
    ok = False
    
    if value is None:
        note = "missing_data"
        ok = False
    else:
        ok = True
        if abs_max is not None:
            if abs(value) > abs_max:
                note = f"above_abs_max (|{value:.2f}| > {abs_max})"
                ok = False
            else:
                note = "ok"
        elif min_val is not None and value < min_val:
            note = f"below_min ({value:.2f} < {min_val})"
            ok = False
        elif max_val is not None and value > max_val:
            note = f"above_max ({value:.2f} > {max_val})"
            ok = False
        else:
            note = "ok"
    
    return StrategyCriterion(
        metric=metric,
        value=value,
        min=min_val,
        max=max_val,
        ok=ok,
        note=note,
    )


def _build_strategy_evaluation(
    strategy_def: Dict[str, Any],
    sensors: Dict[str, Optional[float]],
    underlying: str,
    selected_strategy: str,
) -> StrategyEvaluation:
    """Build a StrategyEvaluation for a single strategy."""
    criteria: List[StrategyCriterion] = []
    all_ok = True
    has_missing = False
    
    for cdef in strategy_def["criteria_defs"]:
        metric = cdef["metric"]
        value = sensors.get(metric)
        min_val = cdef.get("min")
        max_val = cdef.get("max")
        abs_max = cdef.get("abs_max")
        
        criterion = _evaluate_criterion(metric, value, min_val, max_val, abs_max)
        criteria.append(criterion)
        
        if not criterion.ok:
            all_ok = False
            if criterion.note == "missing_data":
                has_missing = True
    
    is_selected = strategy_def["key"] == selected_strategy
    
    if strategy_def["key"] == "NO_TRADE":
        status = "pass" if selected_strategy == "NO_TRADE" else "blocked"
    elif is_selected and all_ok:
        status = "pass"
    elif has_missing and not all_ok:
        any_non_missing_fail = any(
            not c.ok and c.note != "missing_data" for c in criteria
        )
        status = "no_data" if not any_non_missing_fail else "blocked"
    elif not all_ok:
        status = "blocked"
    else:
        status = "pass" if is_selected else "blocked"
    
    passing_criteria = [c for c in criteria if c.ok]
    failing_criteria = [c for c in criteria if not c.ok]
    
    if status == "pass":
        if criteria:
            details = "; ".join(
                f"{c.metric}={c.value:.2f} OK" if c.value else f"{c.metric}=? OK"
                for c in passing_criteria
            )
            summary = f"Pass: {details}" if details else "Pass: No criteria required."
        else:
            summary = "Pass: Default fallback - no favorable setup detected."
    elif status == "no_data":
        missing = [c.metric for c in criteria if c.note == "missing_data"]
        summary = f"No data: Missing {', '.join(missing)}."
    else:
        details = "; ".join(
            f"{c.metric}: {c.note}" for c in failing_criteria
        )
        summary = f"Blocked: {details}" if details else "Blocked: Criteria not met."
    
    return StrategyEvaluation(
        bot_name="GregBot",
        expert_id="greg_mandolini",
        underlying=underlying,
        strategy_key=strategy_def["key"],
        label=strategy_def["label"],
        status=status,
        summary=summary,
        criteria=criteria,
        debug={"selected_by_tree": is_selected, "description": strategy_def.get("description", "")},
    )


def _get_sensor_bundle(underlying: str) -> "SensorBundle":
    """
    Internal helper to get the SensorBundle for a given underlying.
    """
    from src.config import settings
    from src.status_store import status_store
    
    status = status_store.get() or {}
    state_dict = status.get("state", {})
    vol_state = state_dict.get("vol_state", {})
    
    if underlying.upper() == "BTC":
        iv_30d = vol_state.get("btc_iv")
        ivrv = vol_state.get("btc_ivrv", 1.0)
        skew = vol_state.get("btc_skew", 0)
    else:
        iv_30d = vol_state.get("eth_iv")
        ivrv = vol_state.get("eth_ivrv", 1.0)
        skew = vol_state.get("eth_skew", 0)
    
    if iv_30d and iv_30d > 0:
        pass
    else:
        iv_30d = None
    
    bundle = compute_sensors_for_underlying(
        underlying=underlying.upper(),
        iv_30d=iv_30d,
        iv_7d=None,
        skew=skew if skew != 0 else None,
    )
    
    return bundle


def compute_greg_sensors(underlying: str) -> Dict[str, Optional[float]]:
    """
    Compute Greg sensors for a given underlying.
    Fetches OHLC data and options chain data for all indicators.
    """
    bundle = _get_sensor_bundle(underlying)
    return bundle.to_dict()


def compute_greg_sensors_with_debug(underlying: str) -> Dict[str, Any]:
    """
    Compute Greg sensors with debug inputs for a given underlying.
    Returns both sensor values and debug inputs.
    """
    bundle = _get_sensor_bundle(underlying)
    return {
        "sensors": bundle.to_dict(),
        "debug_inputs": bundle.to_debug_dict(),
    }


def get_gregbot_evaluations_for_underlying(
    underlying: str,
    env_mode: Union[EnvironmentMode, str, None] = None,
) -> Dict[str, Any]:
    """
    Build current GregBot Phase 1 view for a single underlying.
    
    Args:
        underlying: The underlying asset (BTC or ETH)
        env_mode: Environment mode (EnvironmentMode enum, "test"/"live" string, or None).
                  If None, uses settings.env_mode.
                  This allows callers to request LIVE strategy data even when server is in TEST mode.
    
    Returns:
        {
            "underlying": "BTC",
            "sensors": {...},
            "strategies": [StrategyEvaluation, ...]
        }
    """
    if env_mode is None:
        effective_mode = settings.env_mode
    elif isinstance(env_mode, EnvironmentMode):
        effective_mode = env_mode
    else:
        try:
            effective_mode = EnvironmentMode(env_mode.lower())
        except ValueError:
            effective_mode = settings.env_mode
    
    mode_str = effective_mode.value
    
    sensors = compute_greg_sensors(underlying)
    
    greg_sensors = GregSelectorSensors(
        vrp_30d=sensors.get("vrp_30d"),
        vrp_7d=sensors.get("vrp_7d"),
        front_rv_iv_ratio=sensors.get("front_rv_iv_ratio"),
        chop_factor_7d=sensors.get("chop_factor_7d"),
        iv_rank_6m=sensors.get("iv_rank_6m"),
        term_structure_spread=sensors.get("term_structure_spread"),
        skew_25d=sensors.get("skew_25d"),
        adx_14d=sensors.get("adx_14d"),
        rsi_14d=sensors.get("rsi_14d"),
        price_vs_ma200=sensors.get("price_vs_ma200"),
        predicted_funding_rate=sensors.get("predicted_funding_rate"),
    )
    
    decision = evaluate_greg_selector(greg_sensors, env_mode=mode_str)
    selected_strategy = decision.selected_strategy
    
    strategies_list = get_greg_strategies(mode_str)
    
    evaluations: List[StrategyEvaluation] = []
    for strat_def in strategies_list:
        eval_obj = _build_strategy_evaluation(
            strat_def, sensors, underlying.upper(), selected_strategy
        )
        evaluations.append(eval_obj)
    
    return {
        "underlying": underlying.upper(),
        "sensors": sensors,
        "strategies": evaluations,
        "selected_strategy": selected_strategy,
        "decision_reasoning": decision.reasoning,
        "step_name": decision.step_name,
        "env_mode": mode_str,
    }
