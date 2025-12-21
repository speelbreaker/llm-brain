"""Typed models for the supervisor loop."""

from pydantic import BaseModel, Field


class FixPlan(BaseModel):
    """Proposed fix plan from the Optimist."""
    category: str
    objectives: list[str] = Field(default_factory=list)
    approach: str = ""
    estimated_risk: str = "low"
    rationale: str = ""


class SkepticReport(BaseModel):
    """Skeptic assessment of the fix plan."""
    risk_level: str = "low"
    blockers: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    classification: str = "unknown"
    file_risk_score: int = 0
    loc_risk_score: int = 0


class LoopDecision(BaseModel):
    """Arbiter decision for the loop."""
    decision: str
    reason: str
    fix_objectives: list[str] = Field(default_factory=list)
    allowed_to_modify: list[str] = Field(default_factory=list)
    risk_level: str = "unknown"
