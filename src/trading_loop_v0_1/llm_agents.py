"""LLM-backed Optimist, Skeptic, and Arbiter for trading loop v0.1."""
from __future__ import annotations

import json
from typing import Any, Dict, Optional

from src.llm.router import LLMRouter, LLMRouterError

from .arbiter import arbiter_decide_from_model_output
from .types import (
    ArbiterDecisionV01,
    NoTradeDecisionV01,
    NoTradeReasonCode,
    SnapshotV01,
    arbiter_decision_schema,
    normalized_proposal_schema,
)


def optimist_propose(snapshot: SnapshotV01, *, router: Optional[LLMRouter] = None) -> Any:
    router = router or LLMRouter()
    messages = _base_messages(
        role="trading_optimist",
        snapshot=snapshot,
        extra={"task": "Propose a single NormalizedProposalV01 or return no proposal."},
    )
    try:
        return router.call("trading_optimist", messages, json_schema=normalized_proposal_schema())
    except LLMRouterError:
        return None


def skeptic_review(
    snapshot: SnapshotV01,
    optimist_raw: Any,
    *,
    router: Optional[LLMRouter] = None,
) -> Any:
    router = router or LLMRouter()
    messages = _base_messages(
        role="trading_skeptic",
        snapshot=snapshot,
        extra={
            "task": "Review the optimist proposal and return a revised NormalizedProposalV01 or no proposal.",
            "optimist_proposal": _safe_json(optimist_raw),
        },
    )
    try:
        return router.call("trading_skeptic", messages, json_schema=normalized_proposal_schema())
    except LLMRouterError:
        return None


def arbiter_decide(
    snapshot: SnapshotV01,
    optimist_raw: Any,
    skeptic_raw: Any,
    *,
    router: Optional[LLMRouter] = None,
) -> ArbiterDecisionV01:
    router = router or LLMRouter()
    messages = _base_messages(
        role="trading_arbiter",
        snapshot=snapshot,
        extra={
            "task": "Select exactly one decision for this tick.",
            "optimist_proposal": _safe_json(optimist_raw),
            "skeptic_proposal": _safe_json(skeptic_raw),
        },
    )
    try:
        raw_decision = router.call("trading_arbiter", messages, json_schema=arbiter_decision_schema())
    except LLMRouterError:
        return NoTradeDecisionV01(
            reason_code=NoTradeReasonCode.MODEL_UNAVAILABLE,
            details={"reason": "model_unavailable"},
        )
    return arbiter_decide_from_model_output(snapshot, raw_decision)


def _base_messages(role: str, snapshot: SnapshotV01, extra: Dict[str, Any]) -> list[Dict[str, str]]:
    payload = {
        "role": role,
        "snapshot": snapshot.model_dump(mode="json"),
        "context": extra,
    }
    return [
        {
            "role": "system",
            "content": "Return JSON only. Do not include commentary.",
        },
        {
            "role": "user",
            "content": json.dumps(payload, sort_keys=True, ensure_ascii=True),
        },
    ]


def _safe_json(value: Any) -> Optional[Any]:
    if value is None:
        return None
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value
