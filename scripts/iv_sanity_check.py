"""
IV Sanity Check Module

Runs standardized backtests with different IV multipliers to validate that the
synthetic IV pricing layer is responding correctly to parameter changes.

Selectors tested:
- generic: Standard covered call strategy
- gregbot: GregBot VRP strategy

For each selector, runs two backtests with low and high IV multipliers,
then validates that results differ meaningfully (not stuck/broken).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from typing import Any, Callable, Dict, List, Literal, Optional

from src.backtest.covered_call_simulator import CoveredCallSimulator
from src.backtest.deribit_data_source import DeribitDataSource
from src.backtest.state_builder import build_historical_state
from src.backtest.types import CallSimulationConfig


SelectorName = Literal["generic", "gregbot"]


@dataclass
class IVSanitySelectorResult:
    """Result of IV sanity check for a single selector."""
    selector: SelectorName
    iv_low: float
    iv_high: float
    num_trades_low: int
    num_trades_high: int
    net_profit_pct_low: float
    net_profit_pct_high: float
    passed: bool
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _build_base_config() -> Dict[str, Any]:
    """
    Build the fixed backtest config for IV sanity checks.
    Uses a fixed 7-day window (2025-12-07 to 2025-12-13) with synthetic grid pricing.
    """
    start_date = datetime(2025, 12, 7, 0, 0, 0, tzinfo=timezone.utc)
    end_date = datetime(2025, 12, 13, 0, 0, 0, tzinfo=timezone.utc)
    
    return {
        "underlying": "BTC_USDC",
        "start": start_date,
        "end": end_date,
        "timeframe": "1d",
        "decision_interval_bars": 1,
        "initial_spot_position": 1.0,
        "contract_size": 1.0,
        "fee_rate": 0.0003,
        "target_dte": 7,
        "dte_tolerance": 4,
        "target_delta": 0.25,
        "delta_tolerance": 0.10,
        "min_dte": 3,
        "max_dte": 21,
        "delta_min": 0.15,
        "delta_max": 0.35,
        "hold_to_expiry": True,
        "initial_capital_usd": 10000.0,
        "position_size_underlying": 1.0,
        "pricing_mode": "synthetic_bs",
        "chain_mode": "synthetic_grid",
        "sigma_mode": "rv_x_multiplier",
        "option_settlement_ccy": "USDC",
    }


def _generate_decision_times(start: datetime, end: datetime, interval_hours: int = 24) -> List[datetime]:
    """Generate decision times between start and end at given hour intervals."""
    times: List[datetime] = []
    current = start
    while current <= end:
        times.append(current)
        current += timedelta(hours=interval_hours)
    return times


SELECTOR_PARAMS: Dict[SelectorName, Dict[str, float]] = {
    "generic": {
        "target_delta": 0.25,
        "delta_min": 0.15,
        "delta_max": 0.35,
    },
    "gregbot": {
        "target_delta": 0.35,
        "delta_min": 0.25,
        "delta_max": 0.45,
    },
}


def _run_single_backtest(
    selector: SelectorName,
    iv_multiplier: float,
    data_source: Optional[DeribitDataSource] = None,
) -> Dict[str, Any]:
    """
    Run a single backtest with the given IV multiplier.
    
    The selector controls scoring parameters:
    - generic: target_delta=0.25, delta range [0.15, 0.35] (conservative)
    - gregbot: target_delta=0.35, delta range [0.25, 0.45] (aggressive)
    
    Returns dict with:
    - num_trades: int
    - net_profit_pct: float
    - metrics: full metrics dict from simulator
    - error: str if any error occurred
    
    This function is designed to be easily mocked in tests.
    """
    try:
        ds = data_source or DeribitDataSource()
        base_cfg = _build_base_config()
        selector_params = SELECTOR_PARAMS[selector]
        
        cfg = CallSimulationConfig(
            underlying=base_cfg["underlying"],
            start=base_cfg["start"],
            end=base_cfg["end"],
            timeframe=base_cfg["timeframe"],
            decision_interval_bars=base_cfg["decision_interval_bars"],
            initial_spot_position=base_cfg["initial_spot_position"],
            contract_size=base_cfg["contract_size"],
            fee_rate=base_cfg["fee_rate"],
            target_dte=base_cfg["target_dte"],
            dte_tolerance=base_cfg["dte_tolerance"],
            target_delta=selector_params["target_delta"],
            delta_tolerance=base_cfg["delta_tolerance"],
            min_dte=base_cfg["min_dte"],
            max_dte=base_cfg["max_dte"],
            delta_min=selector_params["delta_min"],
            delta_max=selector_params["delta_max"],
            hold_to_expiry=base_cfg["hold_to_expiry"],
            initial_capital_usd=base_cfg["initial_capital_usd"],
            position_size_underlying=base_cfg["position_size_underlying"],
            pricing_mode=base_cfg["pricing_mode"],
            chain_mode=base_cfg["chain_mode"],
            sigma_mode=base_cfg["sigma_mode"],
            synthetic_iv_multiplier=iv_multiplier,
        )
        
        sim = CoveredCallSimulator(ds, cfg)
        
        decision_times = _generate_decision_times(
            base_cfg["start"], 
            base_cfg["end"], 
            interval_hours=24
        )
        
        def state_builder(t: datetime) -> Dict[str, Any]:
            return build_historical_state(ds, cfg, t)
        
        result = sim.simulate_policy_with_scoring(
            decision_times=decision_times,
            state_builder=state_builder,
            exit_style="hold_to_expiry",
        )
        
        return {
            "num_trades": len(result.trades),
            "net_profit_pct": result.metrics.get("net_profit_pct", 0.0),
            "metrics": result.metrics,
            "error": None,
        }
    except Exception as e:
        return {
            "num_trades": 0,
            "net_profit_pct": 0.0,
            "metrics": {},
            "error": str(e),
        }


def _check_selector(
    selector: SelectorName,
    iv_low: float,
    iv_high: float,
    data_source: Optional[DeribitDataSource] = None,
) -> IVSanitySelectorResult:
    """
    Run IV sanity check for a single selector.
    
    Pass/Fail logic (using net_profit_pct for comparison):
    - Generic: FAIL if num_trades_low == num_trades_high AND abs(net_profit_pct_diff) < 0.5
    - GregBot: FAIL if num_trades_high <= num_trades_low AND net_profit_pct_high <= net_profit_pct_low + 0.5
    """
    result_low = _run_single_backtest(selector, iv_low, data_source)
    result_high = _run_single_backtest(selector, iv_high, data_source)
    
    if result_low.get("error"):
        return IVSanitySelectorResult(
            selector=selector,
            iv_low=iv_low,
            iv_high=iv_high,
            num_trades_low=0,
            num_trades_high=0,
            net_profit_pct_low=0.0,
            net_profit_pct_high=0.0,
            passed=False,
            reason=f"low IV backtest error: {result_low['error']}",
        )
    
    if result_high.get("error"):
        return IVSanitySelectorResult(
            selector=selector,
            iv_low=iv_low,
            iv_high=iv_high,
            num_trades_low=result_low["num_trades"],
            num_trades_high=0,
            net_profit_pct_low=result_low["net_profit_pct"],
            net_profit_pct_high=0.0,
            passed=False,
            reason=f"high IV backtest error: {result_high['error']}",
        )
    
    num_trades_low = result_low["num_trades"]
    num_trades_high = result_high["num_trades"]
    net_profit_pct_low = result_low["net_profit_pct"]
    net_profit_pct_high = result_high["net_profit_pct"]
    profit_pct_diff = abs(net_profit_pct_high - net_profit_pct_low)
    
    passed = True
    reason = "ok"
    
    if selector == "generic":
        if num_trades_low == num_trades_high and profit_pct_diff < 0.5:
            passed = False
            reason = f"No differentiation: trades={num_trades_low}, net_profit_pct_diff={profit_pct_diff:.2f}%"
    else:
        if num_trades_high <= num_trades_low and net_profit_pct_high <= net_profit_pct_low + 0.5:
            passed = False
            reason = f"GregBot not responding to IV: trades_high={num_trades_high} <= trades_low={num_trades_low}, net_profit_pct_high={net_profit_pct_high:.2f}% <= net_profit_pct_low+0.5={net_profit_pct_low + 0.5:.2f}%"
    
    if num_trades_low == 0 and num_trades_high == 0:
        passed = False
        reason = "No trades executed in either scenario"
    
    return IVSanitySelectorResult(
        selector=selector,
        iv_low=iv_low,
        iv_high=iv_high,
        num_trades_low=num_trades_low,
        num_trades_high=num_trades_high,
        net_profit_pct_low=net_profit_pct_low,
        net_profit_pct_high=net_profit_pct_high,
        passed=passed,
        reason=reason,
    )


def run_iv_sanity_check() -> Dict[str, Any]:
    """
    Run IV sanity checks for all selectors and aggregate results.
    
    Returns dict with:
    - status: "ok" | "degraded" | "failed"
    - selectors: list of selector results
    - summary: human-readable summary
    - checked_at: ISO timestamp
    """
    checked_at = datetime.now(timezone.utc).isoformat()
    
    try:
        ds = DeribitDataSource()
    except Exception as e:
        return {
            "status": "failed",
            "selectors": [],
            "summary": f"Failed to initialize data source: {e}",
            "checked_at": checked_at,
        }
    
    selector_configs = [
        ("generic", 0.8, 1.2),
        ("gregbot", 0.9, 1.1),
    ]
    
    results = []
    for selector, iv_low, iv_high in selector_configs:
        result = _check_selector(selector, iv_low, iv_high, ds)
        results.append(result)
    
    all_passed = all(r.passed for r in results)
    none_passed = not any(r.passed for r in results)
    
    if all_passed:
        status = "ok"
        summary = "All IV sanity checks passed"
    elif none_passed:
        status = "failed"
        failed_reasons = [f"{r.selector}: {r.reason}" for r in results if not r.passed]
        summary = f"All checks failed: {'; '.join(failed_reasons)}"
    else:
        status = "degraded"
        passed_list = [r.selector for r in results if r.passed]
        failed_list = [r.selector for r in results if not r.passed]
        summary = f"Partial pass: {passed_list} ok, {failed_list} failed"
    
    return {
        "status": status,
        "selectors": [r.to_dict() for r in results],
        "summary": summary,
        "checked_at": checked_at,
    }


if __name__ == "__main__":
    import json
    result = run_iv_sanity_check()
    print(json.dumps(result, indent=2))
