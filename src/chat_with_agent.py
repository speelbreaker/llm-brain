"""
Chat with the options trading agent using recent decision logs.
Uses OpenAI via Replit AI Integrations to answer questions about the agent's behavior.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI

from src.config import settings


def _get_openai_client() -> OpenAI:
    """Get OpenAI client configured for Replit AI Integrations."""
    api_key = os.environ.get("AI_INTEGRATIONS_OPENAI_API_KEY")
    base_url = os.environ.get("AI_INTEGRATIONS_OPENAI_BASE_URL")
    
    if not api_key or not base_url:
        raise RuntimeError(
            "Chat requires Replit AI Integrations to be configured. "
            "The AI_INTEGRATIONS_OPENAI_API_KEY and AI_INTEGRATIONS_OPENAI_BASE_URL "
            "environment variables are not set."
        )
    
    return OpenAI(api_key=api_key, base_url=base_url)


LOG_DIR = Path("logs")
LOG_PATTERN = "agent_decisions_*.jsonl"


@dataclass
class LogEntry:
    raw: Dict[str, Any]


def find_latest_log_file() -> Path:
    """Find the most recent agent_decisions_YYYYMMDD.jsonl file in logs/."""
    paths = sorted(
        LOG_DIR.glob(LOG_PATTERN),
        key=lambda p: p.name,
    )
    if not paths:
        raise FileNotFoundError(
            f"No log files matching {LOG_PATTERN} found in {LOG_DIR}. "
            "Run the agent first so it creates logs."
        )
    return paths[-1]


def load_recent_entries(log_file: Path, limit: int = 20) -> List[LogEntry]:
    """Load the last `limit` JSONL entries from the given log file."""
    lines: List[str] = []
    with log_file.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    lines = lines[-limit:]
    entries: List[LogEntry] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            entries.append(LogEntry(raw=data))
        except json.JSONDecodeError:
            continue
    return entries


def compress_entry(entry: LogEntry) -> Dict[str, Any]:
    """Build a compact, LLM-friendly summary of a single log entry."""
    d = entry.raw
    state = d.get("state", {}) or {}
    proposed = d.get("proposed_action", {}) or {}
    final_action = d.get("final_action", {}) or {}
    risk = d.get("risk_check", {}) or {}
    exec_info = d.get("execution", {}) or {}
    cfg = d.get("config_snapshot", {}) or {}

    return {
        "timestamp": d.get("log_timestamp"),
        "spot": state.get("spot", {}),
        "portfolio": {
            "equity_usd": state.get("portfolio", {}).get("equity_usd"),
            "margin_used_pct": state.get("portfolio", {}).get("margin_used_pct"),
            "net_delta": state.get("portfolio", {}).get("net_delta"),
        },
        "candidates_count": state.get("candidates_count"),
        "top_candidates": state.get("top_candidates", []),
        "proposed_action": {
            "action": proposed.get("action"),
            "params": proposed.get("params"),
            "reasoning": proposed.get("reasoning"),
        },
        "risk_check": {
            "allowed": risk.get("allowed"),
            "reasons": risk.get("reasons"),
        },
        "final_action": {
            "action": final_action.get("action"),
            "params": final_action.get("params"),
        },
        "execution": {
            "status": exec_info.get("status"),
            "dry_run": exec_info.get("dry_run"),
            "orders": exec_info.get("orders"),
            "message": exec_info.get("message"),
        },
        "config_snapshot": {
            "dry_run": cfg.get("dry_run"),
            "max_margin_used_pct": cfg.get("max_margin_used_pct"),
            "max_net_delta_abs": cfg.get("max_net_delta_abs"),
            "max_expiry_exposure": cfg.get("max_expiry_exposure"),
        },
    }


def build_logs_context(entries: List[LogEntry]) -> Dict[str, Any]:
    """Build a compact context object summarizing recent decisions."""
    compressed = [compress_entry(e) for e in entries]
    return {
        "entries": compressed,
        "entry_count": len(compressed),
    }


def chat_with_agent(question: str, log_limit: int = 20) -> str:
    """
    Use the OpenAI API to answer a question about the agent's behavior
    based on recent log entries.
    """
    latest_log = find_latest_log_file()
    entries = load_recent_entries(latest_log, limit=log_limit)
    if not entries:
        return (
            "I couldn't find any recent log entries. "
            "Make sure the agent has been running and generating logs."
        )

    context = build_logs_context(entries)

    client = _get_openai_client()

    system_prompt = (
        "You are the user's automated BTC/ETH options trading agent running on Deribit testnet.\n"
        "You make decisions about covered calls (open, roll, close, or do nothing).\n"
        "You are given a compact summary of your recent decisions and their context from the logs.\n"
        "Your job is to answer the user's questions about:\n"
        "- why you took certain actions,\n"
        "- what your internal rules and constraints are,\n"
        "- how you've behaved recently,\n"
        "- what you might do in similar situations.\n\n"
        "Important rules:\n"
        "- This is a research/experimentation system on TESTNET only.\n"
        "- Do NOT give financial advice.\n"
        "- Do NOT claim you traded real money; it's testnet.\n"
        "- Base your answers on the provided logs and general behavior, "
        "and admit uncertainty if something isn't clear.\n"
    )

    user_content = {
        "question": question,
        "recent_decisions": context,
    }

    model_name = getattr(settings, "llm_chat_model_name", None) or getattr(
        settings, "llm_model_name", "gpt-4.1-mini"
    )

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_content, default=str)},
        ],
        max_tokens=1024,
    )

    try:
        answer = response.choices[0].message.content or ""
    except Exception:
        answer = "I had trouble reading the model's response."

    return answer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chat with your options trading agent using recent decision logs."
    )
    parser.add_argument(
        "question",
        nargs="*",
        help="Question to ask the agent (e.g. 'Why did you pick the 97k call?')",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Number of recent log entries to include (default: 20)",
    )
    args = parser.parse_args()

    if not args.question:
        print("Please provide a question, e.g.:")
        print('  python -m src.chat_with_agent "Why did you pick the 97k strike?"')
        return

    question = " ".join(args.question)
    answer = chat_with_agent(question, log_limit=args.limit)
    print()
    print("=== Agent's explanation ===")
    print(answer)
    print()


if __name__ == "__main__":
    main()
