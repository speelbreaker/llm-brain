"""Optimist/Skeptic/Arbiter loop for trading decisions.

This is a *separate* loop from the PR Supervisor debate system.
It takes a compact trading state + candidate options and returns one action.

Design goals:
- Strict JSON I/O (fail-closed to DO_NOTHING)
- Uses the same validation as agent_brain_llm.validate_llm_decision
- Safe defaults: if anything fails, do nothing.

NOTE: This is intended for testnet/research first.
"""

from __future__ import annotations

import json
import os
from typing import Any, Literal

from openai import OpenAI

from src.config import settings
from src.agent_brain_llm import (
    _compress_candidates_for_llm,
    _compress_state_for_llm,
    validate_llm_decision,
)
from src.models import ActionType, AgentState, CandidateOption

Role = Literal["optimist", "skeptic", "arbiter"]


def _get_openai_client(api_key: str, base_url: str | None = None) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=base_url)


def _role_cfg(role: Role) -> tuple[str, str]:
    """Return (provider, model) for this trading debate role."""
    provider = (os.environ.get(f"TRADING_DEBATE_{role.upper()}_PROVIDER") or "openai").strip().lower()
    model = (os.environ.get(f"TRADING_DEBATE_{role.upper()}_MODEL") or settings.llm_model_name).strip()
    return provider, model


def _call_openai_like(api_key: str, base_url: str | None, model: str, prompt: str) -> tuple[dict[str, Any], str | None]:
    try:
        client = _get_openai_client(api_key=api_key, base_url=base_url)
        from src.llm_client import chat_completions_create_with_timeout

        timeout_s = float(settings.llm_timeout_seconds)
        req_kwargs = {
            "model": model,
            "messages": [
                {"role": "system", "content": "Return ONLY valid JSON."},
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
            "max_completion_tokens": 900,
        }

        resp = chat_completions_create_with_timeout(client, timeout_s=timeout_s, req_kwargs=req_kwargs)

        content = resp.choices[0].message.content or "{}"
        data = json.loads(content)
        if not isinstance(data, dict):
            return {"summary": "invalid_json"}, "invalid_json_non_object"
        return data, None
    except Exception as e:
        return {"summary": "error"}, f"{type(e).__name__}: {e}"


def _call_gemini(api_key: str, model: str, prompt: str) -> tuple[dict[str, Any], str | None]:
    """Minimal Gemini JSON call via REST.

    model should be like: models/gemini-3-pro-preview
    """
    try:
        import httpx

        url = f"https://generativelanguage.googleapis.com/v1beta/{model}:generateContent?key={api_key}"
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.4,
                "responseMimeType": "application/json",
                "maxOutputTokens": 1024,
            },
        }
        r = httpx.post(url, json=payload, timeout=30.0)
        r.raise_for_status()
        j = r.json()
        # Extract text
        candidates = j.get("candidates") or []
        parts = (((candidates[0] or {}).get("content") or {}).get("parts") or []) if candidates else []
        text = (parts[0] or {}).get("text") if parts else None
        if not text:
            return {"summary": "invalid_json"}, "no_text"
        data = json.loads(text)
        if not isinstance(data, dict):
            return {"summary": "invalid_json"}, "invalid_json_non_object"
        return data, None
    except Exception as e:
        return {"summary": "error"}, f"{type(e).__name__}: {e}"


def _call_role(role: Role, prompt: str) -> tuple[dict[str, Any], str | None, str, str]:
    """Call the configured provider for this role.

    Returns (data, error, provider, model).
    """
    provider, model = _role_cfg(role)

    if provider in ("zai", "glm", "zhipu"):
        api_key = (os.environ.get("GLM_API_KEY") or "").strip()
        base_url = (os.environ.get("TRADING_DEBATE_ZAI_BASE_URL") or "https://api.z.ai/api/paas/v4").strip()
        if not api_key:
            return {"summary": "error"}, "GLM_API_KEY missing", provider, model
        data, err = _call_openai_like(api_key=api_key, base_url=base_url, model=model, prompt=prompt)
        return data, err, provider, model

    if provider == "gemini":
        api_key = (os.environ.get("GEMINI_API_KEY") or "").strip()
        if not api_key:
            return {"summary": "error"}, "GEMINI_API_KEY missing", provider, model
        data, err = _call_gemini(api_key=api_key, model=model, prompt=prompt)
        return data, err, provider, model

    # default: OpenAI
    api_key = (os.environ.get("OPENAI_API_KEY") or settings.openai_api_key or "").strip()
    base_url = (settings.openai_base_url or None)
    if not api_key:
        return {"summary": "error"}, "OPENAI_API_KEY missing", provider, model
    data, err = _call_openai_like(api_key=api_key, base_url=base_url, model=model, prompt=prompt)
    return data, err, provider, model

def choose_action_with_debate(
    state: AgentState,
    candidates: list[CandidateOption],
) -> dict[str, Any]:
    """Run Optimist/Skeptic/Arbiter loop and return a validated action dict."""

    compact_state = _compress_state_for_llm(state)
    compact_candidates = _compress_candidates_for_llm(candidates)

    context = {
        "state": compact_state,
        "candidates": compact_candidates,
        "preferences": {
            "ivrv_min": settings.effective_ivrv_min,
            "delta_range": [settings.effective_delta_min, settings.effective_delta_max],
            "dte_range": [settings.effective_dte_min, settings.effective_dte_max],
            "premium_min_usd": settings.premium_min_usd,
        },
        "constraints": {
            "allowed_actions": [a.value for a in ActionType],
            "must_use_existing_symbols_only": True,
            "return_one_action": True,
        },
    }

    optimist_prompt = (
        "You are OPTIMIST. Propose the single best covered-call management action. "
        "Be concise and focus on premium vs risk.\n\n"
        "Return JSON: {role, proposal:{action, params, reasoning}}\n\n"
        f"INPUT:\n{json.dumps(context)}"
    )

    optimist, optimist_err, optimist_provider, optimist_model = _call_role("optimist", optimist_prompt)
    proposal = optimist.get("proposal") if isinstance(optimist, dict) else None

    skeptic_prompt = (
        "You are SKEPTIC. Critique the optimist proposal for risk, symbol validity, "
        "and market regime. Suggest safer alternative if needed.\n\n"
        "Return JSON: {role, critique, concerns:[...], suggested:{action, params, reasoning}}\n\n"
        f"INPUT:\n{json.dumps({**context, 'optimist': optimist})}"
    )
    skeptic, skeptic_err, skeptic_provider, skeptic_model = _call_role("skeptic", skeptic_prompt)

    arbiter_prompt = (
        "You are ARBITER. Choose ONE final action. You may accept optimist proposal or skeptic suggested. "
        "You MUST return valid JSON: {action, params, reasoning}.\n\n"
        f"INPUT:\n{json.dumps({**context, 'optimist': optimist, 'skeptic': skeptic})}"
    )
    arbiter, arbiter_err, arbiter_provider, arbiter_model = _call_role("arbiter", arbiter_prompt)

    debate_debug = {
        "optimist": {"provider": optimist_provider, "model": optimist_model, "error": optimist_err},
        "skeptic": {"provider": skeptic_provider, "model": skeptic_model, "error": skeptic_err},
        "arbiter": {"provider": arbiter_provider, "model": arbiter_model, "error": arbiter_err},
        "has_optimist_proposal": isinstance(proposal, dict),
        "has_skeptic_suggested": isinstance(skeptic, dict) and isinstance(skeptic.get("suggested"), dict),
        "arbiter_has_action": isinstance(arbiter, dict) and "action" in arbiter,
    }

    decision: dict[str, Any]
    if isinstance(arbiter, dict) and "action" in arbiter:
        decision = {
            "action": arbiter.get("action"),
            "params": arbiter.get("params") or {},
            "reasoning": arbiter.get("reasoning") or "arbiter: no reasoning",
        }
    elif isinstance(skeptic, dict) and isinstance(skeptic.get("suggested"), dict):
        decision = skeptic["suggested"]
    elif isinstance(proposal, dict):
        decision = proposal
    else:
        decision = {
            "action": ActionType.DO_NOTHING.value,
            "params": {},
            "reasoning": "Debate failed to produce a usable action",
        }

    # Attach debug for observability (safe, no secrets)
    decision["debate_debug"] = debate_debug

    # Auto-fill roll symbols if the arbiter decided to roll but omitted required fields.
    # This keeps the LLM focused on "whether" to roll while we deterministically select symbols.
    try:
        action = (decision.get("action") or "").strip()
        params = decision.get("params") or {}
        if action == ActionType.ROLL_COVERED_CALL.value:
            from_symbol = (params.get("from_symbol") or "").strip()
            to_symbol = (params.get("to_symbol") or "").strip()

            # Determine underlying (best-effort)
            underlying = (params.get("underlying") or decision.get("underlying") or "").strip() or None

            if not from_symbol:
                # Pick an existing open option position for this underlying.
                pf = getattr(state, "portfolio", None)
                pos_list = getattr(pf, "positions", None) if pf is not None else None
                if isinstance(pos_list, list):
                    for p in pos_list:
                        sym = getattr(p, "symbol", None) or (p.get("symbol") if isinstance(p, dict) else None)
                        und = getattr(p, "underlying", None) or (p.get("underlying") if isinstance(p, dict) else None)
                        if sym and (underlying is None or und == underlying):
                            from_symbol = str(sym)
                            break

            if not to_symbol:
                # Pick a candidate option (exclude the current symbol if known).
                best = None
                for c in candidates or []:
                    try:
                        if underlying and getattr(c, "underlying", None) != underlying:
                            continue
                        sym = getattr(c, "symbol", None)
                        if not sym:
                            continue
                        if from_symbol and sym == from_symbol:
                            continue
                        # choose by highest premium_usd (deterministic, easy)
                        prem = float(getattr(c, "premium_usd", 0.0) or 0.0)
                        if best is None or prem > best[0]:
                            best = (prem, sym)
                    except Exception:
                        continue
                if best is not None:
                    to_symbol = str(best[1])

            if from_symbol and to_symbol:
                params = {**params, "from_symbol": from_symbol, "to_symbol": to_symbol}
                decision["params"] = params
                decision["autofill"] = {
                    "kind": "roll_symbols",
                    "from_symbol": from_symbol,
                    "to_symbol": to_symbol,
                }
                # Keep reasoning clean; dashboard can display autofill separately.
                decision["reasoning"] = (decision.get("reasoning") or "").strip() or "arbiter: no reasoning"
    except Exception:
        # Never let autofill failures break the decision path; validation will handle missing fields.
        pass

    try:
        decision = validate_llm_decision(decision, state, candidates, settings)
        decision["validated"] = True
    except Exception as e:
        if settings.llm_validation_strict:
            return {
                "action": ActionType.DO_NOTHING.value,
                "params": {},
                "reasoning": f"Debate decision rejected by validation: {e}",
                "decision_source": "debate_rejected",
                "validated": False,
                "mode": settings.mode,
                "policy_version": "debate_v1",
            }
        decision = {
            "action": ActionType.DO_NOTHING.value,
            "params": {},
            "reasoning": f"Validation warning (downgraded): {e}",
            "validated": False,
        }

    decision["decision_source"] = "debate"
    decision["policy_version"] = "debate_v1"
    return decision
