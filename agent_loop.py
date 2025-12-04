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

from src.config import settings
from src.deribit_client import DeribitClient, DeribitAPIError
from src.state_builder import build_agent_state
from src.risk_engine import check_action_allowed
from src.policy_rule_based import decide_action as rule_decide_action
from src.agent_brain_llm import choose_action_with_llm
from src.execution import execute_action
from src.logging_utils import log_decision, log_error, print_decision_summary
from src.models import ActionType


shutdown_requested = False


def signal_handler(signum: int, frame: object) -> None:
    """Handle shutdown signals gracefully."""
    global shutdown_requested
    print("\nShutdown requested...")
    shutdown_requested = True


def run_agent_loop() -> None:
    """
    Main agent loop.
    Continuously fetches state, makes decisions, and executes actions.
    """
    global shutdown_requested
    
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
                        "reasoning": f"Blocked by risk engine: {'; '.join(reasons)}",
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


def main() -> None:
    """Entry point for the agent."""
    print("\n" + "=" * 60)
    print("IMPORTANT: This is a RESEARCH/EXPERIMENTATION system")
    print("Running on Deribit TESTNET only")
    print("This is NOT financial advice")
    print("=" * 60 + "\n")
    
    run_agent_loop()


if __name__ == "__main__":
    main()
