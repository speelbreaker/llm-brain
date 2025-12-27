"""Arbiter normalization + fail-closed logic for trading loop v0.1."""
from __future__ import annotations

import json
from typing import Any

from pydantic import TypeAdapter, ValidationError

from .types import (
    ArbiterDecisionV01,
    NoTradeDecisionV01,
    NoTradeReasonCode,
    NormalizedProposalV01,
    ProposalDecisionV01,
    ProposalModeV01,
    ProposalReasonCode,
    SnapshotV01,
)

_ENTRY_ACTIONS = {"ENTER_COVERED_CALL"}
_MANAGEMENT_ACTIONS = {"HOLD", "ROLL", "CLOSE", "HEDGE_STEP", "REDUCE_RISK"}


def _parse_proposal_payload(raw: Any) -> Any:
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("invalid_json") from exc
    return raw


def _normalize_proposal(raw: Any) -> NormalizedProposalV01:
    payload = _parse_proposal_payload(raw)
    if isinstance(payload, NormalizedProposalV01):
        return payload
    return NormalizedProposalV01.model_validate(payload)


def _expected_mode(snapshot: SnapshotV01) -> ProposalModeV01:
    return "MANAGEMENT" if snapshot.has_open_short_call else "ENTRY"


def _action_allowed_for_mode(mode: ProposalModeV01, action_type: str) -> bool:
    if mode == "ENTRY":
        return action_type in _ENTRY_ACTIONS
    return action_type in _MANAGEMENT_ACTIONS


def arbiter_decide(snapshot: SnapshotV01, proposal_raw: Any) -> ArbiterDecisionV01:
    """Return NO_TRADE or a single normalized proposal, fail-closed on errors."""
    try:
        if proposal_raw is None:
            return NoTradeDecisionV01(
                reason_code=NoTradeReasonCode.NO_PROPOSAL,
                details={"reason": "no_proposal"},
            )

        try:
            proposal = _normalize_proposal(proposal_raw)
        except (ValueError, ValidationError, TypeError):
            return NoTradeDecisionV01(
                reason_code=NoTradeReasonCode.INVALID_OUTPUT,
                details={"reason": "invalid_proposal_payload"},
            )

        if proposal.underlying != snapshot.underlying:
            return NoTradeDecisionV01(
                reason_code=NoTradeReasonCode.INVALID_OUTPUT,
                details={"reason": "underlying_mismatch"},
            )

        if proposal.action_type == "ENTER_COVERED_CALL" and snapshot.has_open_short_call:
            return NoTradeDecisionV01(
                reason_code=NoTradeReasonCode.CAPACITY_BLOCK,
                details={"reason": "capacity_block", "open_short_call_count": snapshot.open_short_call_count},
            )

        expected_mode = _expected_mode(snapshot)
        if proposal.mode != expected_mode:
            return NoTradeDecisionV01(
                reason_code=NoTradeReasonCode.MODE_MISMATCH,
                details={"reason": "mode_mismatch", "expected_mode": expected_mode},
            )

        if not _action_allowed_for_mode(proposal.mode, proposal.action_type):
            return NoTradeDecisionV01(
                reason_code=NoTradeReasonCode.MODE_MISMATCH,
                details={"reason": "action_not_allowed", "mode": proposal.mode},
            )

        return ProposalDecisionV01(
            proposal=proposal,
            reason_code=ProposalReasonCode.APPROVED,
        )
    except Exception:
        return NoTradeDecisionV01(
            reason_code=NoTradeReasonCode.INTERNAL_ERROR,
            details={"reason": "unexpected_exception"},
        )


def arbiter_decide_from_model_output(
    snapshot: SnapshotV01,
    decision_raw: Any,
) -> ArbiterDecisionV01:
    """Validate a model-produced decision and enforce capacity/mode constraints."""
    try:
        payload = decision_raw
        if isinstance(decision_raw, str):
            payload = json.loads(decision_raw)
        decision = TypeAdapter(ArbiterDecisionV01).validate_python(payload)
    except (json.JSONDecodeError, ValidationError, TypeError, ValueError):
        return NoTradeDecisionV01(
            reason_code=NoTradeReasonCode.INVALID_OUTPUT,
            details={"reason": "invalid_decision_payload"},
        )
    except Exception:
        return NoTradeDecisionV01(
            reason_code=NoTradeReasonCode.INTERNAL_ERROR,
            details={"reason": "unexpected_exception"},
        )

    if isinstance(decision, NoTradeDecisionV01):
        return decision

    return arbiter_decide(snapshot, decision.proposal)
