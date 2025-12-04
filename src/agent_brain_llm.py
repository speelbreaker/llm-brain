"""
LLM-based decision module using OpenAI Chat Completions API.
Provides AI-powered action selection for covered call strategy.
Uses Replit AI Integrations for OpenAI access.
"""
from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI

from src.config import settings
from src.models import AgentState, CandidateOption

_client: OpenAI | None = None


def _get_openai_client() -> OpenAI:
    """
    Lazily initialize the OpenAI client.
    Only creates the client when LLM mode is actually used.
    Raises RuntimeError if the required environment variables are not set.
    """
    global _client
    if _client is not None:
        return _client
    
    api_key = os.environ.get("AI_INTEGRATIONS_OPENAI_API_KEY")
    base_url = os.environ.get("AI_INTEGRATIONS_OPENAI_BASE_URL")
    
    if not api_key or not base_url:
        raise RuntimeError(
            "LLM mode requires Replit AI Integrations to be configured. "
            "The AI_INTEGRATIONS_OPENAI_API_KEY and AI_INTEGRATIONS_OPENAI_BASE_URL "
            "environment variables are not set. Please ensure the OpenAI integration "
            "is properly installed, or disable LLM mode by setting LLM_ENABLED=false."
        )
    
    _client = OpenAI(api_key=api_key, base_url=base_url)
    return _client


def _compress_state_for_llm(state: AgentState) -> dict[str, Any]:
    """
    Create a compact, LLM-friendly snapshot of the current environment.
    Avoid huge lists; stick to what matters for decisions.
    """
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
            "open_positions_summary": [
                {
                    "underlying": p.underlying,
                    "symbol": p.symbol,
                    "side": p.side.value,
                    "size": p.size,
                    "strike": p.strike,
                    "expiry_dte": p.expiry_dte,
                    "moneyness": p.moneyness,
                    "unrealized_pnl": round(p.unrealized_pnl or 0, 2),
                }
                for p in state.portfolio.option_positions
            ],
        },
        "vol_state": {
            "btc_iv": round(state.vol_state.btc_iv, 2),
            "btc_rv": round(state.vol_state.btc_rv, 2),
            "btc_ivrv": round(state.vol_state.btc_ivrv, 2),
            "eth_iv": round(state.vol_state.eth_iv, 2),
            "eth_rv": round(state.vol_state.eth_rv, 2),
            "eth_ivrv": round(state.vol_state.eth_ivrv, 2),
        },
        "risk_limits": {
            "max_margin_used_pct": settings.max_margin_used_pct,
            "max_net_delta_abs": settings.max_net_delta_abs,
            "max_expiry_exposure": settings.max_expiry_exposure,
        },
    }


def _compress_candidates_for_llm(
    candidates: list[CandidateOption],
) -> list[dict[str, Any]]:
    """Reduce each candidate to the key decision features."""
    return [
        {
            "symbol": c.symbol,
            "underlying": c.underlying,
            "strike": c.strike,
            "dte": c.dte,
            "delta": round(c.delta, 3),
            "otm_pct": round(c.otm_pct, 2),
            "premium_usd": round(c.premium_usd, 2),
            "iv": round(c.iv, 2),
            "ivrv": round(c.ivrv, 2),
            "bid": c.bid,
            "ask": c.ask,
        }
        for c in candidates
    ]


def choose_action_with_llm(
    state: AgentState,
    candidates: list[CandidateOption],
) -> dict[str, Any]:
    """
    Call the OpenAI Chat Completions API to choose an action.
    
    Returns a dict with at least:
      {
        "action": "DO_NOTHING" | "OPEN_COVERED_CALL" | "ROLL_COVERED_CALL" | "CLOSE_COVERED_CALL",
        "params": {...},
        "reasoning": "short explanation"
      }
    
    The risk engine will still validate this before executing.
    """
    compact_state = _compress_state_for_llm(state)
    compact_candidates = _compress_candidates_for_llm(candidates)
    
    system_prompt = """You are an options trading agent managing BTC/ETH covered calls for a single user.
Your job is to choose a single action from a small, discrete set and explain it briefly.
You must obey the user's risk constraints.
Never invent symbols or sizes that are not in the provided candidates or positions.
Return ONLY valid JSON matching the requested schema."""

    high_level_rules = {
        "objective": "Sell weekly covered calls to collect premium while controlling drawdowns.",
        "preferences": {
            "ivrv_min": settings.ivrv_min,
            "delta_range": [settings.delta_min, settings.delta_max],
            "dte_range": [settings.dte_min, settings.dte_max],
            "premium_min_usd": settings.premium_min_usd,
        },
        "avoid": [
            "opening new shorts immediately after large downside crashes",
            "overusing margin",
            "rolling into extremely ITM calls with huge assignment risk",
        ],
    }

    user_instruction = {
        "instruction": (
            "Given the current state and candidate options, choose exactly ONE action:\n"
            "- DO_NOTHING\n"
            "- OPEN_COVERED_CALL (on one candidate)\n"
            "- ROLL_COVERED_CALL (from an existing position into one candidate)\n"
            "- CLOSE_COVERED_CALL (on an existing position)\n\n"
            "Constraints:\n"
            "- Respect the risk limits.\n"
            "- Prefer actions consistent with the objective and preferences.\n"
            "- If nothing looks good, choose DO_NOTHING.\n\n"
            "Output format (JSON only):\n"
            "{\n"
            '  "action": "...",\n'
            '  "params": { ... },\n'
            '  "reasoning": "short explanation"\n'
            "}"
        ),
        "state": compact_state,
        "candidates": compact_candidates,
        "rules": high_level_rules,
    }

    try:
        client = _get_openai_client()
        response = client.chat.completions.create(
            model=settings.llm_model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_instruction)},
            ],
            response_format={"type": "json_object"},
            max_completion_tokens=1024,
        )
        
        model_output = response.choices[0].message.content or ""
        decision = json.loads(model_output)
        
    except json.JSONDecodeError:
        decision = {
            "action": "DO_NOTHING",
            "params": {},
            "reasoning": "Failed to parse model JSON; defaulting to no action.",
        }
    except RuntimeError as e:
        decision = {
            "action": "DO_NOTHING",
            "params": {},
            "reasoning": f"LLM configuration error: {str(e)}; defaulting to no action.",
        }
    except Exception as e:
        decision = {
            "action": "DO_NOTHING",
            "params": {},
            "reasoning": f"LLM API error: {str(e)}; defaulting to no action.",
        }

    if "action" not in decision:
        decision["action"] = "DO_NOTHING"
    if "params" not in decision:
        decision["params"] = {}
    if "reasoning" not in decision:
        decision["reasoning"] = "no reasoning provided"
    
    valid_actions = {"DO_NOTHING", "OPEN_COVERED_CALL", "ROLL_COVERED_CALL", "CLOSE_COVERED_CALL"}
    if decision["action"] not in valid_actions:
        decision["action"] = "DO_NOTHING"
        decision["reasoning"] = f"Invalid action '{decision.get('action')}'; defaulting to DO_NOTHING."

    return decision
