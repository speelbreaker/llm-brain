#!/usr/bin/env python3
"""
Smoke Test Harness for Greg Strategies.

This script performs two types of testing:
1. Environment Matrix Tests - Verify selector picks expected strategy for synthetic regimes
2. Strategy Smoke Tests - Run all Greg strategies in DRY_RUN mode with simulated price moves

Usage:
    python scripts/smoke_greg_strategies.py --underlying BTC --dry-run
    python scripts/smoke_greg_strategies.py --underlying ETH --dry-run --env-only
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("smoke_greg")


TEST_STRATEGIES = [
    "STRATEGY_A_STRADDLE",
    "STRATEGY_A_STRANGLE",
    "STRATEGY_B_CALENDAR",
    "STRATEGY_C_SHORT_PUT",
    "STRATEGY_D_IRON_BUTTERFLY",
    "STRATEGY_F_BULL_PUT_SPREAD",
    "STRATEGY_F_BEAR_CALL_SPREAD",
]

ENV_TESTS = [
    {
        "name": "High VRP - Straddle regime",
        "env": {
            "vrp_30d": 18.0,
            "chop_factor_7d": 0.5,
            "adx_14d": 15.0,
            "term_structure_spread": 0.0,
            "skew_25d": 0.0,
        },
        "expected_strategy": "STRATEGY_A_STRADDLE",
    },
    {
        "name": "Contango - Calendar regime",
        "env": {
            "vrp_30d": 5.0,
            "chop_factor_7d": 0.7,
            "adx_14d": 20.0,
            "term_structure_spread": 7.0,
            "skew_25d": 0.0,
        },
        "expected_strategy": "STRATEGY_B_CALENDAR",
    },
    {
        "name": "Bullish skew - Accumulation put regime",
        "env": {
            "vrp_30d": 8.0,
            "chop_factor_7d": 0.7,
            "adx_14d": 25.0,
            "term_structure_spread": 0.0,
            "skew_25d": 6.0,
        },
        "expected_strategy": "STRATEGY_C_SHORT_PUT",
    },
    {
        "name": "Oversold + Bullish skew - Bull put spread regime",
        "env": {
            "vrp_30d": 6.0,
            "chop_factor_7d": 0.8,
            "adx_14d": 25.0,
            "term_structure_spread": 0.0,
            "skew_25d": 6.0,
            "rsi_14d": 28.0,
        },
        "expected_strategy": "STRATEGY_F_BULL_PUT_SPREAD",
    },
    {
        "name": "Overbought + Bearish skew - Bear call spread regime",
        "env": {
            "vrp_30d": 6.0,
            "chop_factor_7d": 0.8,
            "adx_14d": 25.0,
            "term_structure_spread": 0.0,
            "skew_25d": -6.0,
            "rsi_14d": 75.0,
        },
        "expected_strategy": "STRATEGY_F_BEAR_CALL_SPREAD",
    },
    {
        "name": "Moderate VRP + Low chop - Strangle regime",
        "env": {
            "vrp_30d": 12.0,
            "chop_factor_7d": 0.6,
            "adx_14d": 18.0,
            "term_structure_spread": 0.0,
            "skew_25d": 0.0,
        },
        "expected_strategy": "STRATEGY_A_STRANGLE",
    },
]

STRATEGY_MOCK_CONFIGS: Dict[str, Dict[str, Any]] = {
    "STRATEGY_A_STRADDLE": {
        "option_legs": [
            {"instrument": "BTC-CALL-ATM", "delta": 0.50, "size": 0.1},
            {"instrument": "BTC-PUT-ATM", "delta": -0.50, "size": 0.1},
        ],
        "base_profit_pct": 0.0,
        "base_loss_pct": 0.0,
    },
    "STRATEGY_A_STRANGLE": {
        "option_legs": [
            {"instrument": "BTC-CALL-OTM", "delta": 0.30, "size": 0.1},
            {"instrument": "BTC-PUT-OTM", "delta": -0.30, "size": 0.1},
        ],
        "base_profit_pct": 0.0,
        "base_loss_pct": 0.0,
    },
    "STRATEGY_B_CALENDAR": {
        "option_legs": [
            {"instrument": "BTC-CALL-FRONT", "delta": -0.50, "size": 0.1},
            {"instrument": "BTC-CALL-BACK", "delta": 0.48, "size": 0.1},
        ],
        "front_dte": 7,
        "back_dte": 30,
        "strike_pct": 1.0,
    },
    "STRATEGY_C_SHORT_PUT": {
        "option_legs": [
            {"instrument": "BTC-PUT-OTM", "delta": -0.25, "size": 0.1},
        ],
        "base_profit_pct": 0.0,
    },
    "STRATEGY_D_IRON_BUTTERFLY": {
        "option_legs": [
            {"instrument": "BTC-CALL-ATM", "delta": 0.50, "size": 0.1},
            {"instrument": "BTC-PUT-ATM", "delta": -0.50, "size": 0.1},
            {"instrument": "BTC-CALL-OTM", "delta": -0.20, "size": 0.1},
            {"instrument": "BTC-PUT-OTM", "delta": 0.20, "size": 0.1},
        ],
        "wing_spread_pct": 0.05,
    },
    "STRATEGY_F_BULL_PUT_SPREAD": {
        "option_legs": [
            {"instrument": "BTC-PUT-SHORT", "delta": -0.30, "size": 0.1},
            {"instrument": "BTC-PUT-LONG", "delta": 0.15, "size": 0.1},
        ],
        "is_bull_put": True,
    },
    "STRATEGY_F_BEAR_CALL_SPREAD": {
        "option_legs": [
            {"instrument": "BTC-CALL-SHORT", "delta": 0.30, "size": 0.1},
            {"instrument": "BTC-CALL-LONG", "delta": -0.15, "size": 0.1},
        ],
        "is_bull_put": False,
    },
}


@dataclass
class SmokeTestResult:
    """Result of a single strategy smoke test."""
    strategy_type: str
    underlying: str
    opened_ok: bool
    hedges_placed: int
    exits_triggered: List[str]
    errors: List[str] = field(default_factory=list)
    steps: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class EnvTestResult:
    """Result of a single environment matrix test."""
    name: str
    expected: str
    actual: str
    passed: bool


def build_synthetic_sensors(env: Dict[str, float]):
    """Build GregSelectorSensors from synthetic environment values."""
    from src.strategies.greg_selector import GregSelectorSensors
    
    return GregSelectorSensors(
        vrp_30d=env.get("vrp_30d"),
        vrp_7d=env.get("vrp_7d"),
        chop_factor_7d=env.get("chop_factor_7d"),
        iv_rank_6m=env.get("iv_rank_6m"),
        term_structure_spread=env.get("term_structure_spread"),
        skew_25d=env.get("skew_25d"),
        adx_14d=env.get("adx_14d"),
        rsi_14d=env.get("rsi_14d"),
        price_vs_ma200=env.get("price_vs_ma200"),
        predicted_funding_rate=env.get("predicted_funding_rate"),
    )


def build_synthetic_agent_state(
    underlying: str,
    spot_price: float,
    env: Optional[Dict[str, float]] = None,
):
    """Build a minimal AgentState for testing."""
    from src.models import AgentState, VolState, PortfolioState, MarketContext
    
    env = env or {}
    
    vrp = env.get("vrp_30d", 10.0)
    btc_iv = 60.0
    btc_rv = btc_iv - vrp
    
    vol_state = VolState(
        btc_iv=btc_iv,
        btc_rv=btc_rv,
        btc_ivrv=btc_iv / btc_rv if btc_rv > 0 else 1.0,
        btc_skew=env.get("skew_25d", 0.0),
        eth_iv=btc_iv * 1.1,
        eth_rv=btc_rv * 1.1,
        eth_ivrv=btc_iv / btc_rv if btc_rv > 0 else 1.0,
        eth_skew=env.get("skew_25d", 0.0),
    )
    
    market_context = MarketContext(
        underlying=underlying,
        time=datetime.now(timezone.utc),
        regime="sideways",
        realized_vol_7d=btc_rv,
        realized_vol_30d=btc_rv,
    )
    
    portfolio = PortfolioState(
        balances={"BTC": 1.0, "ETH": 10.0, "USD": 100000.0},
        equity_usd=200000.0,
        margin_available_usd=150000.0,
    )
    
    return AgentState(
        timestamp=datetime.now(timezone.utc),
        underlyings=[underlying],
        spot={"BTC": spot_price if underlying == "BTC" else 100000.0,
              "ETH": spot_price if underlying == "ETH" else 3500.0},
        portfolio=portfolio,
        vol_state=vol_state,
        market_context=market_context,
    )


def run_env_matrix_tests() -> List[EnvTestResult]:
    """Run the environment matrix tests for the Greg selector."""
    from src.strategies.greg_selector import evaluate_greg_selector
    
    results: List[EnvTestResult] = []
    
    logger.info("=" * 60)
    logger.info("ENVIRONMENT MATRIX TESTS")
    logger.info("=" * 60)
    
    for test in ENV_TESTS:
        name = test["name"]
        env = test["env"]
        expected = test["expected_strategy"]
        
        sensors = build_synthetic_sensors(env)
        decision = evaluate_greg_selector(sensors)
        actual = decision.selected_strategy
        passed = (actual == expected)
        
        status = "PASS" if passed else "FAIL"
        logger.info(
            f"[{status}] {name}: expected={expected}, got={actual}"
        )
        if not passed:
            logger.info(f"       Reasoning: {decision.reasoning}")
            logger.info(f"       Step: {decision.step_name} (rule {decision.rule_index})")
        
        results.append(EnvTestResult(
            name=name,
            expected=expected,
            actual=actual,
            passed=passed,
        ))
    
    passed_count = sum(1 for r in results if r.passed)
    total = len(results)
    logger.info("-" * 60)
    logger.info(f"Environment Matrix: {passed_count}/{total} tests passed")
    
    return results


def create_mock_position(
    strategy_type: str,
    underlying: str,
    spot_price: float,
    test_run_id: str,
) -> Dict[str, Any]:
    """Create a mock position for a given strategy type."""
    config = STRATEGY_MOCK_CONFIGS.get(strategy_type, {})
    option_legs = config.get("option_legs", [])
    
    for leg in option_legs:
        leg["underlying"] = underlying
    
    position = {
        "position_id": f"{test_run_id}:{strategy_type}",
        "strategy_code": strategy_type,
        "underlying": underlying,
        "option_legs": option_legs,
        "spot_price": spot_price,
        "entry_price": spot_price,
        "dte": 21,
        "profit_pct": config.get("base_profit_pct", 0.0),
        "loss_pct": config.get("base_loss_pct", 0.0),
    }
    
    if "CALENDAR" in strategy_type:
        position["front_dte"] = config.get("front_dte", 7)
        position["strike"] = spot_price * config.get("strike_pct", 1.0)
    
    if "IRON" in strategy_type:
        wing_spread = spot_price * config.get("wing_spread_pct", 0.05)
        position["center_strike"] = spot_price
        position["wing_spread"] = wing_spread
    
    if "SPREAD" in strategy_type:
        is_bull_put = config.get("is_bull_put", True)
        position["is_bull_put"] = is_bull_put
        if is_bull_put:
            position["short_strike"] = spot_price * 0.95
        else:
            position["short_strike"] = spot_price * 1.05
    
    if "SHORT_PUT" in strategy_type:
        position["delta"] = -0.25
        position["funding_rate"] = 0.0001
    
    net_delta = sum(leg.get("delta", 0.0) * leg.get("size", 1.0) for leg in option_legs)
    position["net_delta"] = net_delta
    
    return position


def simulate_price_move(
    position: Dict[str, Any],
    original_spot: float,
    new_spot: float,
    strategy_type: str,
) -> Dict[str, Any]:
    """
    Simulate a price move and update position metrics accordingly.
    Returns an updated copy of the position.
    """
    pos = position.copy()
    pos["spot_price"] = new_spot
    
    price_change_pct = (new_spot - original_spot) / original_spot
    
    if "STRADDLE" in strategy_type or "STRANGLE" in strategy_type:
        abs_change = abs(price_change_pct)
        pos["loss_pct"] = abs_change * 3.0
        pos["profit_pct"] = max(0, 0.1 - abs_change)
        
        legs = pos.get("option_legs", [])
        if price_change_pct > 0:
            for leg in legs:
                if leg.get("delta", 0) > 0:
                    leg["delta"] = min(0.9, leg["delta"] + abs_change * 2)
                else:
                    leg["delta"] = max(-0.1, leg["delta"] + abs_change * 2)
        else:
            for leg in legs:
                if leg.get("delta", 0) < 0:
                    leg["delta"] = max(-0.9, leg["delta"] - abs_change * 2)
                else:
                    leg["delta"] = min(0.1, leg["delta"] - abs_change * 2)
        
        net_delta = sum(leg.get("delta", 0.0) * leg.get("size", 1.0) for leg in legs)
        pos["net_delta"] = net_delta
    
    elif "CALENDAR" in strategy_type:
        pos["profit_pct"] = 0.15 if abs(price_change_pct) < 0.02 else -abs(price_change_pct)
    
    elif "SHORT_PUT" in strategy_type:
        if price_change_pct < 0:
            pos["delta"] = min(0.9, abs(pos.get("delta", 0.25)) + abs(price_change_pct) * 3)
            pos["loss_pct"] = abs(price_change_pct) * 2
        else:
            pos["delta"] = max(0.1, abs(pos.get("delta", 0.25)) - price_change_pct)
            pos["profit_pct"] = price_change_pct * 0.5
    
    elif "IRON" in strategy_type:
        abs_change = abs(price_change_pct)
        pos["profit_pct"] = max(0, 0.15 - abs_change * 2)
    
    elif "SPREAD" in strategy_type:
        is_bull_put = pos.get("is_bull_put", True)
        if is_bull_put:
            if price_change_pct < 0:
                pos["loss_pct"] = abs(price_change_pct) * 2
            else:
                pos["profit_pct"] = price_change_pct * 0.5
        else:
            if price_change_pct > 0:
                pos["loss_pct"] = abs(price_change_pct) * 2
            else:
                pos["profit_pct"] = abs(price_change_pct) * 0.5
    
    return pos


def run_strategy_smoke_test(
    strategy_type: str,
    underlying: str,
    base_spot: float,
    test_run_id: str,
    dry_run: bool = True,
) -> SmokeTestResult:
    """Run smoke test for a single strategy."""
    from src.greg_position_manager import evaluate_greg_positions
    
    result = SmokeTestResult(
        strategy_type=strategy_type,
        underlying=underlying,
        opened_ok=False,
        hedges_placed=0,
        exits_triggered=[],
    )
    
    logger.info("=" * 60)
    logger.info(f"Testing {strategy_type} on {underlying}")
    logger.info("=" * 60)
    
    try:
        position = create_mock_position(
            strategy_type=strategy_type,
            underlying=underlying,
            spot_price=base_spot,
            test_run_id=test_run_id,
        )
        result.opened_ok = True
        logger.info(f"Position opened: {position['position_id']}")
        logger.info(f"  Initial net_delta: {position.get('net_delta', 0.0):.4f}")
    except Exception as e:
        result.errors.append(f"Failed to open position: {e}")
        logger.error(f"Failed to open position: {e}")
        return result
    
    price_moves = [
        ("baseline", 0.0),
        ("up_3pct", 0.03),
        ("down_3pct", -0.03),
        ("up_5pct", 0.05),
        ("down_5pct", -0.05),
    ]
    
    hedge_engine = None
    try:
        from src.hedging import get_hedge_engine, GregPosition
        hedge_engine = get_hedge_engine()
        hedge_engine.set_dry_run(dry_run)
    except Exception as e:
        logger.warning(f"HedgeEngine not available: {e}")
    
    for move_name, move_pct in price_moves:
        new_spot = base_spot * (1 + move_pct)
        
        pos_updated = simulate_price_move(
            position=position,
            original_spot=base_spot,
            new_spot=new_spot,
            strategy_type=strategy_type,
        )
        
        state = build_synthetic_agent_state(underlying, new_spot)
        
        mock_positions = [pos_updated]
        suggestions = evaluate_greg_positions(state, mock_positions=mock_positions)
        
        step_info: Dict[str, Any] = {
            "move": move_name,
            "spot": new_spot,
            "net_delta": pos_updated.get("net_delta", 0.0),
            "suggestions": [],
            "hedge_order": None,
        }
        
        for suggestion in suggestions:
            step_info["suggestions"].append({
                "action": suggestion.action,
                "summary": suggestion.summary,
            })
            
            if suggestion.action in ("CLOSE", "TAKE_PROFIT", "ASSIGN"):
                result.exits_triggered.append(
                    f"{move_name}: {suggestion.action} - {suggestion.summary}"
                )
                logger.info(f"  [{move_name}] EXIT: {suggestion.action} - {suggestion.summary}")
            elif suggestion.action == "HEDGE":
                result.hedges_placed += 1
                logger.info(f"  [{move_name}] HEDGE suggested: {suggestion.summary}")
        
        if hedge_engine is not None:
            try:
                from src.hedging import GregPosition as HedgeGregPosition
                
                greg_pos = HedgeGregPosition(
                    position_id=pos_updated["position_id"],
                    strategy_type=strategy_type,
                    underlying=underlying,
                    option_legs=pos_updated.get("option_legs", []),
                    hedge_perp_size=0.0,
                )
                
                hedge_result = hedge_engine.step(greg_pos)
                
                if hedge_result is not None and hedge_result.order:
                    order = hedge_result.order
                    step_info["hedge_order"] = {
                        "side": order.side,
                        "size": order.size,
                        "instrument": order.instrument,
                    }
                    logger.info(
                        f"  [{move_name}] Hedge order: {order.side} {order.size:.4f} {order.instrument} "
                        f"(delta before={order.net_delta_before:.4f}, after={order.net_delta_after:.4f})"
                    )
            except Exception as e:
                logger.warning(f"  [{move_name}] Hedge step error: {e}")
        
        result.steps.append(step_info)
    
    return result


def run_all_strategy_tests(
    underlying: str,
    base_spot: float,
    dry_run: bool = True,
) -> List[SmokeTestResult]:
    """Run smoke tests for all strategies."""
    test_run_id = f"smoke_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("STRATEGY SMOKE TESTS")
    logger.info(f"Test Run ID: {test_run_id}")
    logger.info(f"Underlying: {underlying}")
    logger.info(f"Base Spot: ${base_spot:,.0f}")
    logger.info(f"DRY_RUN: {dry_run}")
    logger.info("=" * 60)
    
    results: List[SmokeTestResult] = []
    
    for strategy_type in TEST_STRATEGIES:
        try:
            result = run_strategy_smoke_test(
                strategy_type=strategy_type,
                underlying=underlying,
                base_spot=base_spot,
                test_run_id=test_run_id,
                dry_run=dry_run,
            )
            results.append(result)
        except Exception as e:
            logger.error(f"Strategy {strategy_type} failed with exception: {e}")
            results.append(SmokeTestResult(
                strategy_type=strategy_type,
                underlying=underlying,
                opened_ok=False,
                hedges_placed=0,
                exits_triggered=[],
                errors=[str(e)],
            ))
    
    return results


def print_summary(
    env_results: List[EnvTestResult],
    strategy_results: List[SmokeTestResult],
) -> None:
    """Print a summary of all test results."""
    print("\n" + "=" * 70)
    print("SMOKE TEST SUMMARY")
    print("=" * 70)
    
    if env_results:
        env_passed = sum(1 for r in env_results if r.passed)
        print(f"\nEnvironment Matrix Tests: {env_passed}/{len(env_results)} passed")
        for r in env_results:
            status = "PASS" if r.passed else "FAIL"
            print(f"  [{status}] {r.name}")
    
    if strategy_results:
        opened_count = sum(1 for r in strategy_results if r.opened_ok)
        print(f"\nStrategy Smoke Tests: {opened_count}/{len(strategy_results)} opened OK")
        
        print("\n{:<35} {:^10} {:^12} {:^15}".format(
            "Strategy", "Opened", "Hedges", "Exits"
        ))
        print("-" * 70)
        
        for r in strategy_results:
            opened = "OK" if r.opened_ok else "FAIL"
            hedges = str(r.hedges_placed)
            exits = str(len(r.exits_triggered))
            
            print(f"{r.strategy_type:<35} {opened:^10} {hedges:^12} {exits:^15}")
            
            if r.errors:
                for err in r.errors:
                    print(f"    ERROR: {err}")
            
            if r.exits_triggered:
                for exit_info in r.exits_triggered[:3]:
                    print(f"    - {exit_info}")
                if len(r.exits_triggered) > 3:
                    print(f"    ... and {len(r.exits_triggered) - 3} more exits")
    
    print("=" * 70)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Smoke test harness for Greg strategies",
    )
    parser.add_argument(
        "--underlying",
        type=str,
        choices=["BTC", "ETH"],
        default="BTC",
        help="Underlying asset to test (default: BTC)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Run in DRY_RUN mode (default: True)",
    )
    parser.add_argument(
        "--env-only",
        action="store_true",
        default=False,
        help="Only run environment matrix tests",
    )
    parser.add_argument(
        "--strategies-only",
        action="store_true",
        default=False,
        help="Only run strategy smoke tests",
    )
    parser.add_argument(
        "--spot",
        type=float,
        default=None,
        help="Override spot price for testing",
    )
    
    args = parser.parse_args()
    
    base_spot = args.spot
    if base_spot is None:
        base_spot = 100000.0 if args.underlying == "BTC" else 3500.0
    
    logger.info(f"Starting Greg Smoke Tests for {args.underlying}")
    logger.info(f"DRY_RUN mode: {args.dry_run}")
    
    env_results: List[EnvTestResult] = []
    strategy_results: List[SmokeTestResult] = []
    
    if not args.strategies_only:
        env_results = run_env_matrix_tests()
    
    if not args.env_only:
        strategy_results = run_all_strategy_tests(
            underlying=args.underlying,
            base_spot=base_spot,
            dry_run=args.dry_run,
        )
    
    print_summary(env_results, strategy_results)
    
    all_env_passed = all(r.passed for r in env_results) if env_results else True
    all_strategies_ok = all(r.opened_ok for r in strategy_results) if strategy_results else True
    
    if all_env_passed and all_strategies_ok:
        logger.info("All smoke tests completed successfully!")
        return 0
    else:
        logger.warning("Some smoke tests failed or had issues")
        return 1


if __name__ == "__main__":
    sys.exit(main())
