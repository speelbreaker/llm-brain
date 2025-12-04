#!/usr/bin/env python3
"""
Main agent loop script.
Runs the options trading agent with configurable decision mode (rule-based or LLM).
"""
from __future__ import annotations

import signal
import sys
import time
import traceback
from datetime import datetime
from typing import Any, Callable, Dict, Optional

from src.config import settings
from src.deribit_client import DeribitClient, DeribitAPIError
from src.state_builder import build_agent_state
from src.risk_engine import check_action_allowed
from src.policy_rule_based import decide_action as rule_decide_action
from src.agent_brain_llm import choose_action_with_llm
from src.execution import execute_action
from src.logging_utils import log_decision, log_error, print_decision_summary
from src.models import ActionType

StatusCallback = Callable[[Dict[str, Any]], None]

shutdown_requested = False


def signal_handler(signum: int, frame: object) -> None:
    """Handle shutdown signals gracefully."""
    global shutdown_requested
    print("\nShutdown requested...")
    shutdown_requested = True


def _build_status_snapshot(
    agent_state: Any,
    proposed_action: Dict[str, Any],
    final_action: Dict[str, Any],
    risk_allowed: bool,
    risk_reasons: list,
    execution_result: Dict[str, Any],
) -> Dict[str, Any]:
    """Build a compact status snapshot for the status store and logging."""
    return {
        "log_timestamp": datetime.utcnow().isoformat(),
        "state": {
            "timestamp": agent_state.timestamp.isoformat() if agent_state else None,
            "underlyings": agent_state.underlyings if agent_state else [],
            "spot": agent_state.spot if agent_state else {},
            "portfolio": {
                "balances": agent_state.portfolio.balances if agent_state else {},
                "equity_usd": agent_state.portfolio.equity_usd if agent_state else 0,
                "margin_used_pct": agent_state.portfolio.margin_used_pct if agent_state else 0,
                "margin_available_usd": agent_state.portfolio.margin_available_usd if agent_state else 0,
                "net_delta": agent_state.portfolio.net_delta if agent_state else 0,
                "positions_count": len(agent_state.portfolio.option_positions) if agent_state else 0,
                "positions": [
                    {
                        "symbol": p.symbol,
                        "side": p.side.value,
                        "size": p.size,
                        "strike": p.strike,
                        "expiry_dte": p.expiry_dte,
                    }
                    for p in (agent_state.portfolio.option_positions if agent_state else [])
                ],
            },
            "vol_state": {
                "btc_iv": agent_state.vol_state.btc_iv if agent_state else 0,
                "btc_ivrv": agent_state.vol_state.btc_ivrv if agent_state else 1,
                "eth_iv": agent_state.vol_state.eth_iv if agent_state else 0,
                "eth_ivrv": agent_state.vol_state.eth_ivrv if agent_state else 1,
            },
            "candidates_count": len(agent_state.candidate_options) if agent_state else 0,
            "top_candidates": [
                {
                    "symbol": c.symbol,
                    "dte": c.dte,
                    "delta": round(c.delta, 3),
                    "premium_usd": round(c.premium_usd, 2),
                    "ivrv": round(c.ivrv, 2),
                }
                for c in (agent_state.candidate_options[:3] if agent_state else [])
            ],
        },
        "proposed_action": proposed_action,
        "risk_check": {
            "allowed": risk_allowed,
            "reasons": risk_reasons,
        },
        "final_action": final_action,
        "execution": execution_result,
        "config_snapshot": {
            "dry_run": settings.dry_run,
            "llm_enabled": settings.llm_enabled,
            "max_margin_used_pct": settings.max_margin_used_pct,
            "max_net_delta_abs": settings.max_net_delta_abs,
        },
    }


def run_agent_loop_forever(
    status_callback: Optional[StatusCallback] = None,
) -> None:
    """
    Main agent loop. Runs forever (or until process dies).
    On each iteration:
      - builds AgentState,
      - decides action (rule-based or LLM),
      - checks risk,
      - executes or simulates,
      - logs to JSONL,
      - if status_callback is provided, calls it with a compact snapshot.
    
    Args:
        status_callback: Optional callback to receive status updates each iteration.
    """
    global shutdown_requested
    
    import threading
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    print("=" * 60)
    print("Options Trading Agent - Deribit Testnet")
    print("=" * 60)
    print(f"Mode: {'LLM-based' if settings.llm_enabled else 'Rule-based'}")
    print(f"Dry Run: {settings.dry_run}")
    print(f"Loop Interval: {settings.loop_interval_sec} seconds")
    print(f"Underlyings: {', '.join(settings.underlyings)}")
    print(f"Max Margin: {settings.max_margin_used_pct}%")
    print(f"Max Delta: {settings.max_net_delta_abs}")
    print("=" * 60)
    
    client = DeribitClient()
    
    print("\nStarting agent loop...\n")
    
    iteration = 0
    
    try:
        while not shutdown_requested:
            iteration += 1
            loop_start = time.time()
            
            print(f"\n{'='*60}")
            print(f"Iteration {iteration} - {datetime.utcnow().isoformat()}")
            print(f"{'='*60}")
            
            agent_state = None
            proposed_action = {
                "action": ActionType.DO_NOTHING.value,
                "params": {},
                "reasoning": "Initializing",
            }
            final_action = proposed_action.copy()
            allowed = False
            reasons: list = []
            execution_result: Dict[str, Any] = {"status": "pending"}
            
            try:
                print("Fetching agent state...")
                agent_state = build_agent_state(client, settings)
                
                print(f"Spot: BTC=${agent_state.spot.get('BTC', 0):,.0f}, "
                      f"ETH=${agent_state.spot.get('ETH', 0):,.0f}")
                print(f"Portfolio: ${agent_state.portfolio.equity_usd:,.2f} equity, "
                      f"{agent_state.portfolio.margin_used_pct:.1f}% margin used")
                print(f"Positions: {len(agent_state.portfolio.option_positions)}")
                print(f"Candidates: {len(agent_state.candidate_options)}")
                
                if agent_state.candidate_options:
                    print("\nTop Candidates:")
                    for i, c in enumerate(agent_state.candidate_options[:3], 1):
                        print(f"  {i}. {c.symbol} - DTE:{c.dte}, "
                              f"delta:{c.delta:.2f}, premium:${c.premium_usd:.2f}")
                
            except DeribitAPIError as e:
                error_msg = f"Failed to build agent state: {e}"
                print(f"ERROR: {error_msg}")
                log_error("state_build_error", error_msg)
                time.sleep(settings.loop_interval_sec)
                continue
            except Exception as e:
                error_msg = f"Unexpected error building state: {e}"
                print(f"ERROR: {error_msg}")
                traceback.print_exc()
                log_error("state_build_error", error_msg, {"traceback": traceback.format_exc()})
                time.sleep(settings.loop_interval_sec)
                continue
            
            try:
                print("\nMaking decision...")
                
                if settings.llm_enabled:
                    proposed_action = choose_action_with_llm(
                        agent_state,
                        agent_state.candidate_options,
                    )
                    print(f"LLM proposed: {proposed_action.get('action')}")
                else:
                    proposed_action = rule_decide_action(agent_state, settings)
                    print(f"Rule-based proposed: {proposed_action.get('action')}")
                
            except Exception as e:
                error_msg = f"Decision error: {e}"
                print(f"ERROR: {error_msg}")
                traceback.print_exc()
                log_error("decision_error", error_msg, {"traceback": traceback.format_exc()})
                proposed_action = {
                    "action": ActionType.DO_NOTHING.value,
                    "params": {},
                    "reasoning": f"Error during decision: {e}",
                }
            
            try:
                allowed, reasons = check_action_allowed(agent_state, proposed_action, settings)
                
                if not allowed:
                    print(f"Risk blocked: {', '.join(reasons)}")
                    final_action = {
                        "action": ActionType.DO_NOTHING.value,
                        "params": {},
                        "reasoning": f"Blocked by risk engine: {reasons}",
                    }
                else:
                    final_action = proposed_action
                
            except Exception as e:
                error_msg = f"Risk check error: {e}"
                print(f"ERROR: {error_msg}")
                log_error("risk_check_error", error_msg)
                allowed = False
                reasons = [f"Risk check failed: {e}"]
                final_action = {
                    "action": ActionType.DO_NOTHING.value,
                    "params": {},
                    "reasoning": f"Risk check error: {e}",
                }
            
            try:
                final_action_type = final_action.get("action", "DO_NOTHING")
                
                if final_action_type != ActionType.DO_NOTHING.value:
                    print(f"\nExecuting: {final_action_type}")
                    execution_result = execute_action(client, final_action, settings)
                else:
                    execution_result = {
                        "status": "skipped",
                        "message": "Action is DO_NOTHING",
                    }
                
            except Exception as e:
                error_msg = f"Execution error: {e}"
                print(f"ERROR: {error_msg}")
                traceback.print_exc()
                log_error("execution_error", error_msg, {"traceback": traceback.format_exc()})
                execution_result = {
                    "status": "error",
                    "message": error_msg,
                }
            
            snapshot = _build_status_snapshot(
                agent_state=agent_state,
                proposed_action=proposed_action,
                final_action=final_action,
                risk_allowed=allowed,
                risk_reasons=reasons,
                execution_result=execution_result,
            )
            
            try:
                log_decision(
                    agent_state=agent_state,
                    proposed_action=proposed_action,
                    final_action=final_action,
                    risk_allowed=allowed,
                    risk_reasons=reasons,
                    execution_result=execution_result,
                )
            except Exception as e:
                print(f"Warning: Failed to log decision: {e}")
            
            if status_callback is not None:
                try:
                    status_callback(snapshot)
                except Exception as e:
                    print(f"Warning: Status callback failed: {e}")
            
            print_decision_summary(
                proposed_action=proposed_action,
                risk_allowed=allowed,
                risk_reasons=reasons,
                execution_result=execution_result,
            )
            
            loop_duration = time.time() - loop_start
            sleep_time = max(0, settings.loop_interval_sec - loop_duration)
            
            if sleep_time > 0 and not shutdown_requested:
                print(f"Sleeping for {sleep_time:.0f} seconds...")
                
                sleep_chunk = min(sleep_time, 10)
                remaining = sleep_time
                while remaining > 0 and not shutdown_requested:
                    time.sleep(min(remaining, sleep_chunk))
                    remaining -= sleep_chunk
    
    except Exception as e:
        print(f"\nFatal error in agent loop: {e}")
        traceback.print_exc()
        log_error("fatal_error", str(e), {"traceback": traceback.format_exc()})
    
    finally:
        print("\nShutting down...")
        client.close()
        print("Agent stopped.")


def run_agent_loop() -> None:
    """Backward-compatible wrapper for the agent loop."""
    run_agent_loop_forever(status_callback=None)


def main() -> None:
    """Entry point for the agent."""
    print("\n" + "=" * 60)
    print("IMPORTANT: This is a RESEARCH/EXPERIMENTATION system")
    print("Running on Deribit TESTNET only")
    print("This is NOT financial advice")
    print("=" * 60 + "\n")
    
    run_agent_loop_forever()


if __name__ == "__main__":
    main()
