from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime, timezone
import uuid

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

class CandidateSummary(BaseModel):
    instrument_name: str
    expiry: str
    strike: float
    delta: float
    dte: int
    mark: float
    premium: float
    iv: float
    score: float
    otm_pct: float

class RiskCheckResult(BaseModel):
    name: str
    passed: bool
    reason: str
    metrics: Dict[str, float]

class DecisionTrace(BaseModel):
    trace_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    created_at: str = Field(default_factory=now_iso)
    bot_id: str
    strategy_id: str
    mode: str  # "testnet" | "paper"
    run_reason: str # "schedule" | "manual"
    underlying: str
    spot: float
    account_id: Optional[str] = None
    candidates: List[CandidateSummary] = []
    chosen: Optional[CandidateSummary] = None
    risk_checks: List[RiskCheckResult] = []
    action: Optional[Dict[str, Any]] = None  # order intent
    execution: Optional[Dict[str, Any]] = None # order result
    decision: str # "TRADE" | "NO_TRADE" | "BLOCKED"
    reason_codes: List[str] = []
    narrative: str
    errors: List[str] = []
