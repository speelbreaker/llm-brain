"""Single-tick runner for trading loop v0.1."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from pydantic import ValidationError

from .llm_agents import arbiter_decide, optimist_propose, skeptic_review
from .snapshot import build_snapshot
from .store import append_tick
from .types import ArbiterDecisionV01, NormalizedProposalV01, ProposalSummaryV01, SnapshotV01

ProposalFn = Callable[[SnapshotV01], Any]
SkepticFn = Callable[[SnapshotV01, Any], Any]
ArbiterFn = Callable[[SnapshotV01, Any, Any], ArbiterDecisionV01]


def _snapshot_hash(snapshot: SnapshotV01) -> str:
    payload = snapshot.model_dump(mode="json")
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _summarize_proposal(source: str, raw: Any) -> ProposalSummaryV01:
    if raw is None:
        return ProposalSummaryV01(source=source, status="NO_PROPOSAL")
    try:
        payload = raw
        if isinstance(raw, str):
            payload = json.loads(raw)
        proposal = raw if isinstance(raw, NormalizedProposalV01) else NormalizedProposalV01.model_validate(payload)
        return ProposalSummaryV01(
            source=source,
            status="VALID_PROPOSAL",
            proposal_id=proposal.proposal_id,
            action_type=proposal.action_type,
            mode=proposal.mode,
        )
    except (json.JSONDecodeError, ValidationError, TypeError):
        return ProposalSummaryV01(source=source, status="RAW_PROPOSAL")


def _optimist_stub(_: SnapshotV01) -> None:
    return None


def _skeptic_stub(_: SnapshotV01, __: Any) -> None:
    return None


def run_once(
    underlying: str = "BTC",
    window_hours: float = 24.0,
    now: Optional[datetime] = None,
    *,
    client: Any = None,
    optimist: Optional[ProposalFn] = None,
    skeptic: Optional[SkepticFn] = None,
    arbiter: Optional[ArbiterFn] = None,
) -> ArbiterDecisionV01:
    now = now or datetime.now(timezone.utc)
    snapshot = build_snapshot(underlying, window_hours=window_hours, now=now, client=client)
    optimist_fn = optimist or optimist_propose
    skeptic_fn = skeptic or skeptic_review
    arbiter_fn = arbiter or arbiter_decide

    optimist_raw = optimist_fn(snapshot)
    skeptic_raw = skeptic_fn(snapshot, optimist_raw)

    decision = arbiter_fn(snapshot, optimist_raw, skeptic_raw)

    record = {
        "timestamp": now.isoformat(),
        "underlying": snapshot.underlying,
        "snapshot_hash": _snapshot_hash(snapshot),
        "proposals_summary": [
            _summarize_proposal("optimist", optimist_raw).model_dump(mode="json"),
            _summarize_proposal("skeptic", skeptic_raw).model_dump(mode="json"),
        ],
        "arbiter_decision": decision.model_dump(mode="json"),
        "reason_code": decision.reason_code.value,
    }
    append_tick(record)
    return decision
