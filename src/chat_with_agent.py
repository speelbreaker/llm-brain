"""
Chat with the options trading agent using real-time state and decision logs.
Uses OpenAI via Replit AI Integrations to answer questions about the agent's behavior.

Features:
- Multi-turn conversation history
- Real-time trading state context (positions, PnL, mode)
- Recent decision summaries
- Architecture documentation context
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI

from src.config import settings
from src.chat_store import chat_store
from src.status_store import status_store
from src.decisions_store import decisions_store
from src.position_tracker import position_tracker


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

DOCS_FILES = [
    "ARCHITECTURE_OVERVIEW.md",
    "HEALTHCHECK.md",
    "ROADMAP.md",
    "replit.md",
]

MAX_DOC_CHARS = 3000


@dataclass
class LogEntry:
    raw: Dict[str, Any]


def find_latest_log_file() -> Optional[Path]:
    """Find the most recent agent_decisions_YYYYMMDD.jsonl file in logs/."""
    paths = sorted(
        LOG_DIR.glob(LOG_PATTERN),
        key=lambda p: p.name,
    )
    if not paths:
        return None
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


def _summarize_decisions(limit: int = 10) -> str:
    """Summarize recent decisions from the decisions_store."""
    decisions = decisions_store.get_all()[:limit]
    if not decisions:
        return "No recent decisions recorded."
    
    action_counts: Dict[str, int] = {}
    symbols_traded: List[str] = []
    reasonings: List[str] = []
    
    for d in decisions:
        final = d.get("final_action", {})
        proposed = d.get("proposed_action", {})
        action = final.get("action") or proposed.get("action") or d.get("action", "UNKNOWN")
        action_counts[action] = action_counts.get(action, 0) + 1
        
        params = final.get("params", {}) or proposed.get("params", {}) or d.get("params", {})
        if params and "symbol" in params:
            symbols_traded.append(params["symbol"])
        
        reasoning = final.get("reasoning") or proposed.get("reasoning", "")
        if reasoning and len(reasonings) < 3:
            reasonings.append(reasoning[:100])
    
    parts = [f"{count}x {action}" for action, count in action_counts.items()]
    summary = f"Last {len(decisions)} decisions: " + ", ".join(parts)
    
    if symbols_traded:
        unique_symbols = list(dict.fromkeys(symbols_traded[:5]))
        summary += f". Symbols: {', '.join(unique_symbols)}"
    
    if reasonings:
        summary += "\n\nRecent reasoning:\n" + "\n".join(f"- {r}..." for r in reasonings)
    
    return summary


def _format_positions_summary(positions: List[Dict[str, Any]], limit: int = 5) -> str:
    """Format open positions as a compact summary."""
    if not positions:
        return "No open positions."
    
    lines = []
    for p in positions[:limit]:
        symbol = p.get("symbol", "?")
        qty = p.get("quantity", 0) or p.get("qty", 0)
        pnl = p.get("unrealized_pnl", 0) or 0
        pnl_pct = p.get("unrealized_pnl_pct", 0) or 0
        dte = p.get("dte", "?")
        lines.append(f"  - {symbol}: qty={qty}, PnL=${pnl:.2f} ({pnl_pct:+.2f}%), DTE={dte}")
    
    if len(positions) > limit:
        lines.append(f"  ... and {len(positions) - limit} more positions")
    
    return "\n".join(lines)


def _load_docs_summary() -> str:
    """Load and truncate architecture documentation files."""
    docs_parts = []
    
    for filename in DOCS_FILES:
        path = Path(filename)
        if path.exists():
            try:
                content = path.read_text(encoding="utf-8")[:MAX_DOC_CHARS]
                if len(content) >= MAX_DOC_CHARS:
                    content = content + "\n... [truncated]"
                docs_parts.append(f"### {filename}\n{content}")
            except Exception:
                pass
    
    if not docs_parts:
        return "No documentation files available."
    
    return "\n\n".join(docs_parts)


def _enrich_positions_with_live_data(
    raw_positions: List[Dict[str, Any]],
    live_positions: List[Dict[str, Any]],
    spot_prices: Dict[str, float],
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Enrich position tracker data with live mark prices and PnL from Deribit.
    Mirrors the logic in /api/positions/open endpoint.
    """
    live_by_symbol: Dict[str, Dict[str, Any]] = {}
    for p in live_positions:
        symbol = p.get("symbol")
        if symbol:
            live_by_symbol[symbol] = p
    
    enriched_positions: List[Dict[str, Any]] = []
    for pos in raw_positions:
        enriched = dict(pos)
        symbol = enriched.get("symbol")
        underlying = enriched.get("underlying", "BTC")
        spot = float(spot_prices.get(underlying, 0.0))
        live_data = live_by_symbol.get(symbol, {})
        
        if live_data:
            live_mark = float(live_data.get("mark_price") or 0.0)
            live_pnl = float(live_data.get("unrealized_pnl") or 0.0)
            entry_price_btc = float(enriched.get("entry_price") or 0.0)
            qty = abs(float(enriched.get("quantity") or 1.0))
            
            if live_mark > 0:
                enriched["mark_price"] = live_mark
                enriched["unrealized_pnl"] = live_pnl
                if entry_price_btc > 0 and qty > 0 and spot > 0:
                    notional_usd = entry_price_btc * qty * spot
                    enriched["unrealized_pnl_pct"] = (live_pnl / notional_usd) * 100.0 if notional_usd > 0 else 0.0
        
        enriched_positions.append(enriched)
    
    total_pnl = sum(float(p.get("unrealized_pnl", 0.0)) for p in enriched_positions)
    totals = {
        "positions_count": len(enriched_positions),
        "unrealized_pnl": total_pnl,
    }
    
    return enriched_positions, totals


def build_chat_context() -> Dict[str, Any]:
    """
    Collect the current trading context for the LLM:
    - /status snapshot (mode, training, llm_enabled, positions, PnL, etc.)
    - recent decisions (last N from decisions_store)
    - optional summarized docs for architecture/safety questions
    
    Note: All stores return copies to avoid race conditions with background agent.
    """
    import copy
    
    current_status = copy.deepcopy(status_store.get())
    
    state = current_status.get("state", {})
    portfolio = state.get("portfolio", {})
    spot = state.get("spot", {})
    live_positions = portfolio.get("positions", [])
    
    raw_positions_data = copy.deepcopy(position_tracker.get_open_positions_payload())
    raw_positions = raw_positions_data.get("positions", [])
    
    positions, totals = _enrich_positions_with_live_data(raw_positions, live_positions, spot)
    
    context = {
        "environment": {
            "deribit_env": getattr(settings, "deribit_env", "testnet"),
            "mode": "research" if settings.is_research else "production",
            "training_mode": settings.is_training_enabled,
            "llm_enabled": settings.llm_enabled,
            "explore_prob": settings.explore_prob,
            "dry_run": settings.dry_run,
        },
        "portfolio": {
            "equity_usd": portfolio.get("equity_usd"),
            "margin_used_pct": portfolio.get("margin_used_pct"),
            "net_delta": portfolio.get("net_delta"),
        },
        "spot_prices": spot,
        "positions": positions,
        "positions_count": len(positions),
        "total_unrealized_pnl_usd": totals.get("unrealized_pnl", 0.0),
        "recent_decisions_summary": _summarize_decisions(10),
        "positions_summary": _format_positions_summary(positions, 5),
        "docs_summary": _load_docs_summary(),
    }
    
    return context


def _build_rules_text() -> str:
    """Build a short rules summary for the system prompt."""
    return """Trading Rules:
- Prefers selling OTM calls with delta in [0.10, 0.40] and DTE in [1, 21] days.
- In training mode: allows multiple positions per underlying (up to 6), builds delta ladders.
- In live mode: one covered call per underlying, stricter risk checks.
- Rolls a short call if:
  * Close to expiry (< 3 days) and near-the-money, or
  * ITM and at risk of assignment, or
  * Better premium available at higher strike.
- Otherwise holds to expiry and lets options expire worthless (OTM).
- Risk controls: margin cap (80%), net delta cap (5.0), per-expiry exposure limits.
- Position reconciliation runs each loop to detect local/exchange divergence."""


def chat_with_agent(question: str, log_limit: int = 20) -> str:
    """
    Use the OpenAI API to answer a question about the agent's behavior
    based on real-time state, positions, and recent decisions.
    
    Maintains multi-turn conversation history via chat_store.
    """
    ctx = build_chat_context()
    
    history = chat_store.get_history()
    
    env = ctx["environment"]
    system_prompt = f"""You are an assistant embedded inside an options trading bot dashboard.

The user is the bot owner. Your job is to:
- Explain what the bot is currently doing and why.
- Explain trading rules (when it opens, rolls, or closes positions).
- Interpret positions, PnL, training mode, and risk limits.
- Give high-level advice on architecture, safety, and coding WHEN ASKED, based only on the docs provided below.
- Answer questions about recent decisions based on the decision log.

Important:
- This is a research/experimentation system on TESTNET only.
- Do NOT give financial advice.
- Do NOT claim to trade real money; it's testnet.
- Be concise but thorough. Use the context provided.

Current Runtime State:
- Deribit Environment: {env.get('deribit_env', 'testnet').upper()}
- Mode: {env.get('mode', 'research')}
- Training Mode: {'ENABLED' if env.get('training_mode') else 'DISABLED'}
- LLM Decisions: {'ENABLED' if env.get('llm_enabled') else 'DISABLED (rule-based)'}
- Exploration Probability: {env.get('explore_prob', 0):.0%}
- Dry Run: {'YES' if env.get('dry_run') else 'NO'}

Portfolio:
- Equity: ${ctx['portfolio'].get('equity_usd') or 0:,.2f}
- Margin Used: {ctx['portfolio'].get('margin_used_pct') or 0:.2f}%
- Net Delta: {ctx['portfolio'].get('net_delta') or 0:.4f}

Spot Prices: {ctx['spot_prices']}

Open Positions ({ctx['positions_count']}):
{ctx['positions_summary']}

Total Unrealized PnL: ${ctx['total_unrealized_pnl_usd']:,.2f}

Recent Decisions:
{ctx['recent_decisions_summary']}

{_build_rules_text()}

Project Documentation (truncated):
{ctx['docs_summary']}
"""
    
    messages: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    
    messages.append({"role": "user", "content": question})
    
    client = _get_openai_client()
    
    model_name = getattr(settings, "llm_chat_model_name", None) or getattr(
        settings, "llm_model_name", "gpt-4.1-mini"
    )

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=1024,
    )

    try:
        answer = response.choices[0].message.content or ""
    except Exception:
        answer = "I had trouble reading the model's response."
    
    chat_store.append("user", question)
    chat_store.append("assistant", answer)

    return answer


def chat_with_agent_full(question: str, log_limit: int = 20) -> Dict[str, Any]:
    """
    Same as chat_with_agent but returns full response with messages.
    Used by API to avoid race conditions.
    """
    answer = chat_with_agent(question, log_limit)
    messages = chat_store.get_history()
    return {"answer": answer, "messages": messages}


def get_chat_messages() -> List[Dict[str, str]]:
    """Get all chat messages for API response."""
    return chat_store.get_history()


def clear_chat_history() -> None:
    """Clear all chat history."""
    chat_store.clear()


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
