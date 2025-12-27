"""Type system for Trading Decision Loop v0.1."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Dict, Literal, Optional

from pydantic import BaseModel, Field, TypeAdapter
from pydantic.config import ConfigDict

UnderlyingV01 = Literal["BTC", "ETH"]
ProposalModeV01 = Literal["ENTRY", "MANAGEMENT"]
EntryActionV01 = Literal["ENTER_COVERED_CALL"]
ManagementActionV01 = Literal["HOLD", "ROLL", "CLOSE", "HEDGE_STEP", "REDUCE_RISK"]
ActionTypeV01 = Literal[
    "ENTER_COVERED_CALL",
    "HOLD",
    "ROLL",
    "CLOSE",
    "HEDGE_STEP",
    "REDUCE_RISK",
]
ProposalStatusV01 = Literal["NO_PROPOSAL", "RAW_PROPOSAL", "VALID_PROPOSAL"]


class NoTradeReasonCode(str, Enum):
    INVALID_OUTPUT = "INVALID_OUTPUT"
    CAPACITY_BLOCK = "CAPACITY_BLOCK"
    MODE_MISMATCH = "MODE_MISMATCH"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    NO_PROPOSAL = "NO_PROPOSAL"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"


class ProposalReasonCode(str, Enum):
    APPROVED = "APPROVED"


class OptionParamsV01(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instrument_name: Optional[str] = Field(
        default=None,
        description="Deribit instrument name if already resolved",
    )
    expiry_iso: Optional[str] = Field(
        default=None,
        description="ISO-8601 expiry timestamp",
    )
    strike: Optional[float] = Field(
        default=None,
        description="Strike price",
    )
    delta_target: Optional[float] = Field(
        default=None,
        description="Target delta for selection",
    )


class NormalizedProposalV01(BaseModel):
    model_config = ConfigDict(extra="forbid")

    proposal_id: str = Field(..., description="UUID for this proposal")
    underlying: UnderlyingV01
    mode: ProposalModeV01
    action_type: ActionTypeV01
    option: OptionParamsV01
    size_units: float = Field(..., ge=0.0, description="Underlying units")
    rationale: str = Field(..., max_length=1200)
    confidence: float = Field(..., ge=0.0, le=1.0)


class NoTradeDecisionV01(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Literal["NO_TRADE"] = "NO_TRADE"
    reason_code: NoTradeReasonCode
    details: Optional[Dict[str, Any]] = None


class ProposalDecisionV01(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Literal["PROPOSAL"] = "PROPOSAL"
    proposal: NormalizedProposalV01
    reason_code: ProposalReasonCode = ProposalReasonCode.APPROVED


class SnapshotV01(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timestamp: datetime
    underlying: UnderlyingV01
    window_hours: float
    spot_price: float
    has_open_short_call: bool
    open_short_call_count: int
    open_short_call_symbols: list[str]


class ProposalSummaryV01(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: Literal["optimist", "skeptic"]
    status: ProposalStatusV01
    action_type: Optional[ActionTypeV01] = None
    mode: Optional[ProposalModeV01] = None
    proposal_id: Optional[str] = None


ArbiterDecisionV01 = Annotated[
    NoTradeDecisionV01 | ProposalDecisionV01,
    Field(discriminator="decision"),
]


def arbiter_decision_schema() -> Dict[str, Any]:
    return TypeAdapter(ArbiterDecisionV01).json_schema(ref_template="#/definitions/{model}")


def normalized_proposal_schema() -> Dict[str, Any]:
    return NormalizedProposalV01.model_json_schema(ref_template="#/definitions/{model}")


def write_schema(path: Path, schema: Dict[str, Any]) -> None:
    path.write_text(json_dumps(schema), encoding="utf-8")


def write_arbiter_schema(path: Path) -> None:
    write_schema(path, arbiter_decision_schema())


def write_proposal_schema(path: Path) -> None:
    write_schema(path, normalized_proposal_schema())


def json_dumps(payload: Dict[str, Any]) -> str:
    import json

    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
