"""
Logging utilities for structured JSON logs.
Captures agent state, decisions, and execution results for training.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from src.config import settings
from src.models import AgentState


def _ensure_log_dir() -> Path:
    """Ensure the log directory exists."""
    log_dir = Path(settings.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def _get_log_file_path() -> Path:
    """Get the path for today's log file."""
    log_dir = _ensure_log_dir()
    date_str = datetime.utcnow().strftime("%Y%m%d")
    return log_dir / f"agent_decisions_{date_str}.jsonl"


def _compress_agent_state(state: AgentState) -> dict[str, Any]:
    """Create a compact version of agent state for logging."""
    return {
        "timestamp": state.timestamp.isoformat(),
        "underlyings": state.underlyings,
        "spot": state.spot,
        "portfolio": {
            "balances": state.portfolio.balances,
            "equity_usd": round(state.portfolio.equity_usd, 2),
            "margin_used_pct": round(state.portfolio.margin_used_pct, 2),
            "margin_available_usd": round(state.portfolio.margin_available_usd, 2),
            "net_delta": round(state.portfolio.net_delta, 4),
            "positions_count": len(state.portfolio.option_positions),
            "positions": [
                {
                    "symbol": p.symbol,
                    "side": p.side.value,
                    "size": p.size,
                    "expiry_dte": p.expiry_dte,
                    "moneyness": p.moneyness,
                }
                for p in state.portfolio.option_positions
            ],
        },
        "vol_state": {
            "btc_iv": round(state.vol_state.btc_iv, 2),
            "btc_ivrv": round(state.vol_state.btc_ivrv, 2),
            "eth_iv": round(state.vol_state.eth_iv, 2),
            "eth_ivrv": round(state.vol_state.eth_ivrv, 2),
        },
        "candidates_count": len(state.candidate_options),
        "top_candidates": [
            {
                "symbol": c.symbol,
                "dte": c.dte,
                "delta": round(c.delta, 3),
                "premium_usd": round(c.premium_usd, 2),
                "ivrv": round(c.ivrv, 2),
            }
            for c in state.candidate_options[:3]
        ],
    }


def log_decision(
    agent_state: AgentState,
    proposed_action: dict[str, Any],
    final_action: dict[str, Any],
    risk_allowed: bool,
    risk_reasons: list[str],
    execution_result: dict[str, Any],
    additional_info: dict[str, Any] | None = None,
) -> None:
    """
    Log a complete decision cycle to JSONL file.
    
    Args:
        agent_state: Current agent state
        proposed_action: Action proposed by policy/LLM
        final_action: Final action after risk filtering
        risk_allowed: Whether risk engine allowed the action
        risk_reasons: Reasons if risk blocked the action
        execution_result: Result from execution module
        additional_info: Any additional information to log
    """
    log_entry = {
        "log_timestamp": datetime.utcnow().isoformat(),
        "state": _compress_agent_state(agent_state),
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
    
    if additional_info:
        log_entry["additional"] = additional_info
    
    log_file = _get_log_file_path()
    
    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        print(f"Warning: Failed to write log entry: {e}")


def log_error(
    error_type: str,
    error_message: str,
    context: dict[str, Any] | None = None,
) -> None:
    """Log an error entry."""
    log_entry = {
        "log_timestamp": datetime.utcnow().isoformat(),
        "type": "error",
        "error_type": error_type,
        "error_message": error_message,
        "context": context or {},
    }
    
    log_file = _get_log_file_path()
    
    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        print(f"Warning: Failed to write error log: {e}")


def get_recent_logs(
    count: int = 10,
    date_str: str | None = None,
) -> list[dict[str, Any]]:
    """
    Retrieve recent log entries.
    
    Args:
        count: Number of entries to retrieve
        date_str: Optional date string (YYYYMMDD) to read from specific file
    
    Returns:
        List of log entry dicts (most recent last)
    """
    log_dir = _ensure_log_dir()
    
    if date_str:
        log_file = log_dir / f"agent_decisions_{date_str}.jsonl"
    else:
        log_file = _get_log_file_path()
    
    if not log_file.exists():
        return []
    
    entries = []
    try:
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        print(f"Warning: Failed to read logs: {e}")
        return []
    
    return entries[-count:]


def print_decision_summary(
    proposed_action: dict[str, Any],
    risk_allowed: bool,
    risk_reasons: list[str],
    execution_result: dict[str, Any],
) -> None:
    """Print a human-readable summary of the decision to console."""
    print("\n" + "=" * 60)
    print(f"Proposed Action: {proposed_action.get('action', 'N/A')}")
    print(f"Reasoning: {proposed_action.get('reasoning', 'N/A')}")
    
    if proposed_action.get("params"):
        print(f"Parameters: {proposed_action['params']}")
    
    if risk_allowed:
        print("Risk Check: PASSED")
    else:
        print(f"Risk Check: BLOCKED - {', '.join(risk_reasons)}")
    
    exec_status = execution_result.get("status", "N/A")
    print(f"Execution Status: {exec_status}")
    
    if execution_result.get("message"):
        print(f"Execution: {execution_result['message']}")
    
    if execution_result.get("errors"):
        print(f"Errors: {execution_result['errors']}")
    
    print("=" * 60 + "\n")
