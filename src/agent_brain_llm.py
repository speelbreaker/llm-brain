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
from src.models import AgentState, CandidateOption, MarketContext

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


def _compress_market_context(mc: MarketContext | None) -> dict[str, Any] | None:
    """Serialize market context for LLM consumption."""
    if mc is None:
        return None
    return {
        "underlying": mc.underlying,
        "time": mc.time.isoformat(),
        "regime": mc.regime,
        "pct_from_50d_ma": round(mc.pct_from_50d_ma, 2),
        "pct_from_200d_ma": round(mc.pct_from_200d_ma, 2),
        "return_1d_pct": round(mc.return_1d_pct, 2),
        "return_7d_pct": round(mc.return_7d_pct, 2),
        "return_30d_pct": round(mc.return_30d_pct, 2),
        "realized_vol_7d": round(mc.realized_vol_7d, 2),
        "realized_vol_30d": round(mc.realized_vol_30d, 2),
        "support_level": mc.support_level,
        "resistance_level": mc.resistance_level,
        "distance_to_support_pct": mc.distance_to_support_pct,
        "distance_to_resistance_pct": mc.distance_to_resistance_pct,
    }


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
        "market_context": _compress_market_context(state.market_context),
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
    
    system_prompt = """You are an options trading agent managing covered calls on Deribit (testnet for now) for BTC and ETH.

You receive a JSON input with:
- portfolio: holdings, equity, margin usage.
- market_context: compact summary of recent price action and regime.
- candidates: possible options to trade (symbol, strike, DTE, delta, IV, IVRV, premium, etc.).
- risk_limits: hard constraints (max margin usage, max net delta, max expiry exposure).
- config: default order size and minimum IVRV.

Your job:
1. Propose ONE of the following actions:
   - DO_NOTHING
   - OPEN_COVERED_CALL
   - ROLL_COVERED_CALL
   - CLOSE_COVERED_CALL

2. Your JSON response MUST have exactly:
   {
     "action": "DO_NOTHING" | "OPEN_COVERED_CALL" | "ROLL_COVERED_CALL" | "CLOSE_COVERED_CALL",
     "params": { ... },
     "reasoning": "short explanation referencing the data you used"
   }

3. Respect risk limits strictly:
   - Never suggest a trade that would violate max_expiry_exposure, max_margin_used_pct, or max_net_delta_abs.
   - Never invent instrument symbols. Only use candidates or existing open positions.

4. Use the market_context in a simple, structured way:
   - If regime is "bull" AND 30-day return is strongly positive (e.g. > +15%) AND price is >5% above the 50-day MA:
       * Be more conservative with calls:
         - prefer lower deltas (further OTM),
         - avoid very short-dated aggressive calls unless IVRV is clearly high.
   - If regime is "bear" AND 30-day return is strongly negative (e.g. < -15%):
       * Prioritize capital preservation:
         - you may choose DO_NOTHING instead of opening new covered calls,
         - or sell smaller size / further OTM if IVRV is attractive.
   - If market_context shows a very recent dump (7-day return < -10%):
       * Be cautious about selling new calls immediately after the drop unless IVRV is substantially above the minimum and margin is comfortable.
   - If market_context shows sideways regime:
       * It is acceptable to be more assertive with covered calls within risk limits (moderate deltas and shorter DTE).

5. Use IVRV and premiums together with market_context:
   - Only open/roll calls when IVRV is at or above the configured minimum.
   - Between candidates, prefer those with a better balance of:
       * higher premium,
       * acceptable delta and DTE,
       * and lower risk of getting deep ITM given the current regime.

Be concise in reasoning but explicitly mention:
- The regime (bull/sideways/bear),
- Key return/vol metrics that influenced your choice,
- Why you picked this particular candidate or chose DO_NOTHING.

Return ONLY valid JSON matching the requested schema."""

    high_level_rules = {
        "objective": "Sell weekly covered calls to collect premium while controlling drawdowns.",
        "mode": settings.mode,
        "preferences": {
            "ivrv_min": settings.effective_ivrv_min,
            "delta_range": [settings.effective_delta_min, settings.effective_delta_max],
            "dte_range": [settings.effective_dte_min, settings.effective_dte_max],
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
            "- Use the market_context to inform your decision (regime, returns, vol).\n"
            "- Prefer actions consistent with the objective and preferences.\n"
            "- If nothing looks good or market conditions are unfavorable, choose DO_NOTHING.\n\n"
            "Output format (JSON only):\n"
            "{\n"
            '  "action": "...",\n'
            '  "params": { ... },\n'
            '  "reasoning": "short explanation mentioning regime and key metrics"\n'
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

    decision["mode"] = settings.mode
    decision["policy_version"] = "llm_v1"
    decision["decision_source"] = "llm"

    return decision
