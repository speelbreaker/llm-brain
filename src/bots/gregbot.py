"""
GregBot - Greg Mandolini VRP Harvester implementation.

Produces StrategyEvaluation objects for each strategy based on current sensors.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.bots.types import StrategyCriterion, StrategyEvaluation
from src.bots.sensors import SensorBundle, compute_sensors_for_underlying
from src.strategies.greg_selector import (
    GregSelectorSensors,
    load_greg_spec,
    evaluate_greg_selector,
)


GREG_STRATEGIES = [
    {
        "key": "STRATEGY_A_STRADDLE",
        "label": "ATM Straddle (High VRP, Calm)",
        "criteria_defs": [
            {"metric": "vrp_30d", "min": 5.0, "description": "VRP > 5"},
            {"metric": "chop_factor_7d", "max": 0.6, "description": "Chop < 0.6 (calm)"},
            {"metric": "adx_14d", "max": 20, "description": "ADX < 20 (choppy)"},
        ],
    },
    {
        "key": "STRATEGY_B_STRANGLE",
        "label": "OTM Strangle (Moderate VRP, Ranging)",
        "criteria_defs": [
            {"metric": "vrp_30d", "min": 3.0, "description": "VRP > 3"},
            {"metric": "chop_factor_7d", "min": 0.6, "max": 0.9, "description": "Chop 0.6-0.9"},
            {"metric": "rsi_14d", "min": 40, "max": 60, "description": "RSI 40-60 (neutral)"},
        ],
    },
    {
        "key": "STRATEGY_C_CALENDAR",
        "label": "Calendar Spread (Term Structure)",
        "criteria_defs": [
            {"metric": "term_structure_spread", "max": -2.0, "description": "Backwardation < -2"},
            {"metric": "iv_rank_6m", "min": 0.3, "description": "IV Rank > 30%"},
        ],
    },
    {
        "key": "STRATEGY_D_IRON_BUTTERFLY",
        "label": "Iron Butterfly (Low Vol, Pinned)",
        "criteria_defs": [
            {"metric": "vrp_30d", "min": 8.0, "description": "VRP > 8"},
            {"metric": "adx_14d", "max": 15, "description": "ADX < 15 (very choppy)"},
            {"metric": "rsi_14d", "min": 45, "max": 55, "description": "RSI 45-55 (pinned)"},
        ],
    },
    {
        "key": "STRATEGY_E_PUT_SPREAD",
        "label": "Put Credit Spread (Bullish Bias)",
        "criteria_defs": [
            {"metric": "price_vs_ma200", "min": 0, "description": "Price > MA200"},
            {"metric": "skew_25d", "min": 2.0, "description": "Skew > 2 (put premium)"},
            {"metric": "rsi_14d", "max": 70, "description": "RSI < 70 (not overbought)"},
        ],
    },
    {
        "key": "STRATEGY_F_CALL_SPREAD",
        "label": "Call Credit Spread (Bearish Bias)",
        "criteria_defs": [
            {"metric": "price_vs_ma200", "max": 0, "description": "Price < MA200"},
            {"metric": "skew_25d", "max": -2.0, "description": "Skew < -2 (call premium)"},
            {"metric": "rsi_14d", "min": 30, "description": "RSI > 30 (not oversold)"},
        ],
    },
    {
        "key": "STRATEGY_G_RATIO_SPREAD",
        "label": "Ratio Spread (Trending + Skewed)",
        "criteria_defs": [
            {"metric": "adx_14d", "min": 25, "description": "ADX > 25 (trending)"},
            {"metric": "skew_25d", "min": 3.0, "description": "Skew > 3"},
        ],
    },
    {
        "key": "STRATEGY_H_JADE_LIZARD",
        "label": "Jade Lizard (Bullish + High VRP)",
        "criteria_defs": [
            {"metric": "vrp_30d", "min": 6.0, "description": "VRP > 6"},
            {"metric": "price_vs_ma200", "min": 5, "description": "Price > 5% above MA200"},
            {"metric": "rsi_14d", "min": 50, "max": 70, "description": "RSI 50-70"},
        ],
    },
    {
        "key": "NO_TRADE",
        "label": "No Trade (Conditions Unfavorable)",
        "criteria_defs": [],
    },
]


def _evaluate_criterion(
    metric: str,
    value: Optional[float],
    min_val: Optional[float],
    max_val: Optional[float],
) -> StrategyCriterion:
    """Evaluate a single criterion and return a StrategyCriterion object."""
    note = None
    ok = False
    
    if value is None:
        note = "missing_data"
        ok = False
    else:
        ok = True
        if min_val is not None and value < min_val:
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
        
        criterion = _evaluate_criterion(metric, value, min_val, max_val)
        criteria.append(criterion)
        
        if not criterion.ok:
            all_ok = False
            if criterion.note == "missing_data":
                has_missing = True
    
    is_selected = strategy_def["key"] == selected_strategy
    
    if is_selected and all_ok:
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
    
    if strategy_def["key"] == "NO_TRADE":
        status = "pass" if selected_strategy == "NO_TRADE" else "blocked"
    
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
            summary = "Pass: Default fallback."
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
        debug={"selected_by_tree": is_selected},
    )


def compute_greg_sensors(underlying: str) -> Dict[str, Optional[float]]:
    """
    Compute Greg sensors for a given underlying.
    Fetches OHLC data and computes technical indicators.
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
    
    return bundle.to_dict()


def get_gregbot_evaluations_for_underlying(underlying: str) -> Dict[str, Any]:
    """
    Build current GregBot Phase 1 view for a single underlying.
    
    Returns:
        {
            "underlying": "BTC",
            "sensors": {...},
            "strategies": [StrategyEvaluation, ...]
        }
    """
    sensors = compute_greg_sensors(underlying)
    
    greg_sensors = GregSelectorSensors(
        vrp_30d=sensors.get("vrp_30d"),
        chop_factor_7d=sensors.get("chop_factor_7d"),
        iv_rank_6m=sensors.get("iv_rank_6m"),
        term_structure_spread=sensors.get("term_structure_spread"),
        skew_25d=sensors.get("skew_25d"),
        adx_14d=sensors.get("adx_14d"),
        rsi_14d=sensors.get("rsi_14d"),
        price_vs_ma200=sensors.get("price_vs_ma200"),
    )
    
    decision = evaluate_greg_selector(greg_sensors)
    selected_strategy = decision.selected_strategy
    
    evaluations: List[StrategyEvaluation] = []
    for strat_def in GREG_STRATEGIES:
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
    }
